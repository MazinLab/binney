use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use winnow::binary::bits::{bits, take};
use winnow::error::{ContextError, ErrMode};
use winnow::prelude::*;
use winnow::token::literal;
use winnow::Bytes;

use chrono::{DateTime, NaiveDate, Utc};

use hashbrown::{HashMap, HashSet};

use indicatif::ParallelProgressIterator;

use polars::prelude::*;

use rayon::prelude::*;

use pyo3::{prelude::*, pyclass, wrap_pymodule};

use pyo3_polars::PyDataFrame;

#[derive(Debug)]
pub enum BinneyError {
    IOError(std::io::Error),
    ParseError(winnow::error::ErrMode<ContextError>),
    PolarsError(PolarsError),
    BinDirError(String),
    SerializationError(String),
}

impl From<BinneyError> for PyErr {
    fn from(error: BinneyError) -> Self {
        match error {
            BinneyError::IOError(e) => e.into(),
            BinneyError::ParseError(e) => {
                pyo3::exceptions::PyBufferError::new_err(format!("BinParsingError: {}", e))
            }
            BinneyError::PolarsError(e) => {
                pyo3::exceptions::PyException::new_err(format!("PolarsError: {}", e))
            }
            BinneyError::BinDirError(e) => pyo3::exceptions::PyIOError::new_err(e),
            BinneyError::SerializationError(e) => pyo3::exceptions::PyValueError::new_err(e),
        }
    }
}

impl From<std::io::Error> for BinneyError {
    fn from(error: std::io::Error) -> BinneyError {
        BinneyError::IOError(error)
    }
}

impl From<winnow::error::ErrMode<ContextError>> for BinneyError {
    fn from(error: winnow::error::ErrMode<ContextError>) -> BinneyError {
        BinneyError::ParseError(error)
    }
}

impl From<PolarsError> for BinneyError {
    fn from(error: PolarsError) -> BinneyError {
        BinneyError::PolarsError(error)
    }
}

/// A structure representing a column wise vector of photons.
#[derive(Debug, PartialEq)]
pub struct Photons {
    /// The REPORTED `(x, y)` array coordinates as `(x << 8) | y`
    /// It is unlikely these are the true array coordinates this
    /// is here purely for resonator tracking and is used to later
    /// apply a beammap. This is also NOT the "Reasonator ID".
    pub xy: Vec<u16>,
    /// The timestamp reported by the readout in microseconds.
    pub timestamp: Vec<i64>,
    /// The phase reported by the readout in ?? units
    pub phase: Vec<i32>,
    /// The baseline phase reported by the readout in ?? units
    pub baseline: Vec<i32>,
}

fn tableschema() -> Schema {
    let mut s = Schema::new();
    s.with_column("xy".into(), DataType::UInt16);
    s.with_column("timestamp".into(), DataType::Int64);
    s.with_column("phase".into(), DataType::Int32);
    s.with_column("baseline".into(), DataType::Int32);
    s
}

impl Photons {
    pub fn with_capacity(capacity: usize) -> Photons {
        Photons {
            xy: Vec::with_capacity(capacity),
            timestamp: Vec::with_capacity(capacity),
            phase: Vec::with_capacity(capacity),
            baseline: Vec::with_capacity(capacity),
        }
    }
}

/// Serializes one or more polars dataframes into gen2 `.bin` files
///
/// This expects a polars dataframe with the same schema provided by
/// `binney.BinDirectoryDF.read_timerange`, namely:
///
/// ```python
/// Schema({
///     'xy': UInt16,
///     'timestamp': Int64,
///     'phase': Int32,
///     'baseline': Int32,
///     'bar': Int64
/// })
/// ```
///
/// It also expects every cell in the dataframe to be non-empty and
/// every pixel to be present exactly once in the board-to-pixel
/// mapping
///
/// ### Usage
///
/// ```python
/// # Board 234 with pixels (x: 1, y: 1) and (x: 10, y: 7)
/// # Board 236 with pixel (x: 0, y: 0)
/// # Assume the year part of the timestamp is in 2022
/// # A skew of -1ms between the nominal bin edges and the serialized ones
/// serializer = binney.BinSerializer(
///     directory="./mybins",
///     board2pixel={
///         234: [(1 << 8) | 1, (10 << 8) | 7],
///         236: [0]
///     },
///     year=2022,
///     skew=-1000
/// )
/// serializer.serialize_pydataframe(mydf)
/// serializer.serialize_pydataframe(mydf)
/// ```
///
/// ### Parameters
/// `binney.BinSerializer(directory: pathlib.Path | str, board2pixel: dict[int, list[int]], year: int, skew: int) -> BinSerializer`
/// - `directory: pathlib.Path | str` The output directory for bin files
/// - `board2pixel: dict[int, list[int]]` A dictionary mapping board numbers to
///   a set of pixel xy numbers which must be uniquely mapped to a board
/// - `year: int` The year to serialize as, gen2 timestamps cover a roughly 390
///    day range, they are referenced to the microseconds from the start of the
///    year the run was taken in
/// - `skew: int` The skew between the nominal second each bin file covers
///   and the true second it covers in microseconds
#[pyclass(frozen)]
pub struct BinSerializer {
    directory: PathBuf,
    reference_time: DateTime<Utc>,
    skew: i64,
    // 64K Hashmap, Yes, It's fine
    pix2board: HashMap<u16, u8>,
    rangecache: Arc<Mutex<Vec<TimestampRange>>>,
}

#[pymethods]
impl BinSerializer {
    #[new]
    pub fn new(
        directory: PathBuf,
        board2pixel: HashMap<u8, Vec<u16>>,
        year: i32,
        skew: i64,
    ) -> Result<BinSerializer, BinneyError> {
        let mut pix2board = HashMap::new();
        for (b, ps) in board2pixel.into_iter() {
            for p in HashSet::<u16>::from_iter(ps.into_iter()).into_iter() {
                if let Some(br) = pix2board.insert(p, b) {
                    return Err(BinneyError::SerializationError(format!(
                        "Pixel {} (xy) duplicated on board {} and board {}",
                        p, b, br
                    )));
                }
            }
        }
        Ok(BinSerializer {
            directory,
            reference_time: NaiveDate::from_ymd_opt(year, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc(),
            skew,
            pix2board,
            rangecache: Arc::new(Mutex::new(vec![])),
        })
    }

    /// Serialize a `polars.DataFrame` with appropriate schema into bin files
    ///
    /// Currently this will overwrite any existing bin files in the same folder
    /// if they exist in the same time range and were created by a different
    /// BinSerializer method. It will not overwrite binfiles written by previous
    /// calls to this method by virtue of refusing to serialize subsequent
    /// dataframes if it has previously written a dataframe with a timestamp range
    /// coming within 1 second of this one. This requirement may be relaxed in
    /// a future release if it is necessary.
    pub fn serialize_pydataframe(&self, df: PyDataFrame) -> Result<(), BinneyError> {
        self.serialize_dataframe(&mut df.into())
    }
}

impl BinSerializer {
    /// See `binney.BinSerializer.serialize_pydataframe`
    pub fn serialize_dataframe(&self, df: &mut DataFrame) -> Result<(), BinneyError> {
        let ene = || BinneyError::SerializationError("Cannot have empty slot".into());
        if df.schema() != tableschema() {
            return Err(BinneyError::SerializationError(format!(
                "Got a table with schema {:#?}, expected {:#?}",
                df.schema(),
                tableschema()
            )));
        }

        let max = df.clone().column("timestamp")?.max::<i64>()?.ok_or(
            BinneyError::SerializationError("timestamp column empty".into()),
        )?;
        let min = df.clone().column("timestamp")?.min::<i64>()?.ok_or(
            BinneyError::SerializationError("timestamp column empty".into()),
        )?;

        let start_sec = ((min + self.skew) / (1000 * 1000)) + self.reference_time.timestamp();

        let range = TimestampRange::new(min, max);
        {
            let mut rangecache = self.rangecache.lock().unwrap();
            let overlapping = rangecache
                .iter()
                .map(|i| i.grow(1000 * 1000).overlaps(&range))
                .fold(false, |a, b| a | b);

            if overlapping {
                return Err(BinneyError::SerializationError(
                    "The timerange must be disjoing by at least 1 second with all previously serialized timeranges".into(),
                ));
            }

            rangecache.push(range);
        }

        let board = df
            .column("xy")?
            .u16()?
            .iter()
            .map(|xy| {
                if let Some(u) = xy {
                    self.pix2board.get(&u).cloned()
                } else {
                    None
                }
            })
            .collect::<Option<Vec<u8>>>()
            .ok_or(BinneyError::SerializationError(
                "Could not find a board mapping for all pixels in the dataframe".into(),
            ))?;

        let ldf = df.with_column(Series::new("board", board))?.clone().lazy();

        let mut schema = Schema::new();
        schema.with_column("length".into(), DataType::UInt64);

        let df = ldf
            .sort(["timestamp"], Default::default())
            .group_by_dynamic(
                col("timestamp"),
                [col("board")],
                DynamicGroupOptions {
                    index_column: "".into(),
                    every: Duration::new(500),
                    period: Duration::new(500),
                    offset: Duration::new(0),
                    label: Label::Left,
                    include_boundaries: true,
                    closed_window: ClosedWindow::Left,
                    start_by: StartBy::DataPoint,
                },
            )
            .agg([
                col("xy"),
                col("timestamp").alias("ptimestamp"),
                col("phase"),
                col("baseline"),
            ])
            .sort(["timestamp"], Default::default())
            .collect()?;

        // Do some nasty ass type coersion because polars erases type info
        let xys = df
            .column("xy")?
            .list()?
            .cast(&DataType::List(Box::new(DataType::UInt16)))?;

        let timestamps = df
            .column("ptimestamp")?
            .list()?
            .cast(&DataType::List(Box::new(DataType::Int64)))?;

        let phases = df
            .column("phase")?
            .list()?
            .cast(&DataType::List(Box::new(DataType::Int32)))?;

        let baselines = df
            .column("baseline")?
            .list()?
            .cast(&DataType::List(Box::new(DataType::Int32)))?;

        let mut current_second = start_sec;
        let mut file = std::io::BufWriter::new(File::create(
            self.directory
                .with_file_name(format!("{}.bin", current_second)),
        )?);

        let mut framenum: HashMap<u8, u16> = HashMap::new();

        for i in 0..df.shape().0 {
            let timestamp = df.column("timestamp")?.i64()?.get(i).ok_or_else(ene)?;
            let mut packets = 100;
            if (timestamp + self.skew) / (1000 * 1000)
                > current_second - self.reference_time.timestamp()
            {
                current_second =
                    (timestamp + self.skew) / (1000 * 1000) + self.reference_time.timestamp();
                file = std::io::BufWriter::new(File::create(
                    self.directory
                        .with_file_name(format!("{}.bin", current_second)),
                )?);
            }
            let board = df.column("board")?.u8()?.get(i).ok_or_else(ene)?;
            if let (AnyValue::List(p), AnyValue::List(b), AnyValue::List(t), AnyValue::List(x)) = (
                phases.get(i)?,
                baselines.get(i)?,
                timestamps.get(i)?,
                xys.get(i)?,
            ) {
                let x: Vec<u16> = x.u16()?.iter().collect::<Option<_>>().ok_or_else(ene)?;
                let t: Vec<i64> = t.i64()?.iter().collect::<Option<_>>().ok_or_else(ene)?;
                let p: Vec<i32> = p.i32()?.iter().collect::<Option<_>>().ok_or_else(ene)?;
                let b: Vec<i32> = b.i32()?.iter().collect::<Option<_>>().ok_or_else(ene)?;
                assert!(p.len() == b.len() && b.len() == t.len());

                let mut header_t = t[0] / 500;

                for j in 0..p.len() {
                    if t[j] / 500 > header_t {
                        if packets <= 99 {
                            file.write_all(&PaddingPacket.serialize())?;
                        }
                        packets = 100;
                    }
                    if packets >= 100 {
                        header_t = t[j] / 500;
                        file.write_all(
                            &HeaderPacket {
                                board,
                                frame: *framenum.get(&board).unwrap_or(&0),
                                timestamp: header_t,
                            }
                            .serialize(),
                        )?;
                        framenum.insert(board, *framenum.get(&board).unwrap_or(&0) + 1);
                        packets = 0;
                    }
                    file.write_all(
                        &DataPacket {
                            x: (x[j] >> 8) as u8,
                            y: (x[j] & 0xff) as u8,
                            timestamp: (t[j] - header_t * 500) as u16,
                            phase: p[j],
                            baseline: b[j],
                        }
                        .serialize(),
                    )?;
                }
            } else {
                return Err(BinneyError::SerializationError(
                    "Could not coerce polars lists into the appropriate type".into(),
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Photon {
    xy: u16,
    timestamp: i64,
    phase: i32,
    baseline: i32,
}

impl Photon {
    pub fn new(xy: u16, timestamp: i64, phase: i32, baseline: i32) -> Result<Photon, BinneyError> {
        if !(-(1 << 17)..=(1 << 17) - 1).contains(&phase)
            || !(-(1 << 16)..=(1 << 16) - 1).contains(&baseline)
        {
            Err(BinneyError::SerializationError("phase or baseline out of range, expected phase in [-1<<17,1<<17-1], baseline in [-1<<16,1<<16-1]".into()))
        } else {
            Ok(Photon {
                xy,
                timestamp,
                phase,
                baseline,
            })
        }
    }
}

impl From<Vec<Photon>> for Photons {
    fn from(value: Vec<Photon>) -> Self {
        Photons {
            xy: value.iter().map(|i| i.xy).collect(),
            timestamp: value.iter().map(|i| i.timestamp).collect(),
            phase: value.iter().map(|i| i.phase).collect(),
            baseline: value.iter().map(|i| i.baseline).collect(),
        }
    }
}

impl TryInto<DataFrame> for Photons {
    type Error = BinneyError;
    fn try_into(self) -> Result<DataFrame, Self::Error> {
        let xys = Series::new("xy", self.xy);
        let ts = Series::new("timestamp", self.timestamp);
        let ps = Series::new("phase", self.phase);
        let bs = Series::new("baseline", self.baseline);

        Ok(DataFrame::new(vec![xys, ts, ps, bs])?)
    }
}

impl TryFrom<DataFrame> for Photons {
    type Error = BinneyError;
    fn try_from(value: DataFrame) -> Result<Self, Self::Error> {
        Ok(Photons {
            xy: value
                .column("xy")?
                .u16()?
                .iter()
                .collect::<Option<Vec<u16>>>()
                .ok_or(BinneyError::SerializationError(
                    "cannot have empty xy positions".into(),
                ))?,

            timestamp: value
                .column("timestamp")?
                .i64()?
                .iter()
                .collect::<Option<Vec<i64>>>()
                .ok_or(BinneyError::SerializationError(
                    "cannot have empty xy positions".into(),
                ))?,

            phase: value
                .column("phase")?
                .i32()?
                .iter()
                .collect::<Option<Vec<i32>>>()
                .ok_or(BinneyError::SerializationError(
                    "cannot have empty xy positions".into(),
                ))?,

            baseline: value
                .column("baseline")?
                .i32()?
                .iter()
                .collect::<Option<Vec<i32>>>()
                .ok_or(BinneyError::SerializationError(
                    "cannot have empty xy positions".into(),
                ))?,
        })
    }
}

/// Represents a range of us gen2 timestamps
///
/// This is intentionally opaque to force you to use the provided methods for
/// operations which correctly handle counter overflows which may occur up to
/// once during a gen2 observing night. Both sides of the range are inclusive
///
/// ### Usage
///
/// ```python
/// timerange = binney.TimestampRange(10, 100)
/// assert(timerange.inside(50))
/// other = binney.TimestampRange(100, 500)
/// assert(other.overlaps(timerange))
/// ```
///
/// ### Parameters
/// `binney.TimestampRange(start: int, stop: int) -> TimestampRange`
/// - `start: int` The start time for this timerange in ticks
/// - `stop: int` The stop time for this timerange in ticks
#[pyclass(frozen, eq, hash)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimestampRange {
    start: i64,
    stop: i64,
}

#[pymethods]
impl TimestampRange {
    /// The number of gen2 ticks in a second (ticks are 1us)
    pub const TICKS_PER_SEC: u64 = 1000 * 1000;

    /// Create a new timestamp range from a start and stop time in ticks, these
    /// will be wrapped appropriately according to how the readout nominally
    /// wraps timestamps
    #[new]
    pub fn new(start: i64, stop: i64) -> TimestampRange {
        // Initilizing we truncate to 36 bits
        // I just wanna do math on Z/nZ and not nearly blow my foot off
        TimestampRange {
            start: start.rem_euclid(0xf_ffff_ffff * 500 + 500),
            stop: stop.rem_euclid(0xf_ffff_ffff * 500 + 500),
        }
    }

    /// Check if a given timestamp is inside the range
    #[inline]
    pub fn inside(&self, timestamp: u64) -> bool {
        // If the start of the range is larger than the stop we interpret that
        // as meaning that the timestamp straddles a wrapping boundary
        let timestamp = timestamp.rem_euclid(0xf_ffff_ffff * 500 + 500);
        if self.start <= self.stop {
            timestamp >= self.start as u64 && timestamp <= self.stop as u64
        } else {
            timestamp <= self.stop as u64 || timestamp >= self.start as u64
        }
    }

    /// Check if another timestamp range overlaps with this one
    #[inline]
    pub fn overlaps(&self, other: &TimestampRange) -> bool {
        self.inside(other.start as u64)
            || self.inside(other.stop as u64)
            || other.inside(self.start as u64)
            || other.inside(self.stop as u64)
    }

    /// Grow the range by `tolerance` ticks on either side, can be negative to
    /// shrink a range. This returns a new timestamp range instead of mutating
    /// the current one.
    pub fn grow(&self, tolerance: i64) -> TimestampRange {
        TimestampRange::new(self.start - tolerance, self.stop + tolerance)
    }
}

pub trait SerializePacket {
    fn serialize(&self) -> [u8; 8];

    fn serialize_into(&self, vec: &mut Vec<u8>) {
        let bytes = self.serialize();
        for byte in bytes.into_iter() {
            vec.push(byte)
        }
    }
}

#[pyclass(frozen)]
#[derive(Debug, PartialEq, Clone, Copy, Hash)]
pub struct HeaderPacket {
    pub board: u8,
    pub frame: u16,
    pub timestamp: i64,
}

impl SerializePacket for HeaderPacket {
    fn serialize(&self) -> [u8; 8] {
        let mut packet = 0xff00_0000_0000_0000u64;
        packet |= (self.board as u64).wrapping_shl(64 - 8 - 8);
        packet |= (self.frame as u64).wrapping_shl(64 - 8 - 8 - 8);
        packet |= self.timestamp as u64 & 0x0000_00ff_ffff_ffff;

        packet.to_be_bytes()
    }
}

#[derive(Debug, PartialEq, Hash, Clone, Copy)]
pub struct DataPacket {
    pub x: u8,
    pub y: u8,
    pub timestamp: u16,
    pub phase: i32,
    pub baseline: i32,
}

impl SerializePacket for DataPacket {
    fn serialize(&self) -> [u8; 8] {
        let mut packet = 0xff00_0000_0000_0000u64;
        packet |= (self.x as u64).wrapping_shl(64 - 10);
        packet |= (self.y as u64).wrapping_shl(64 - 10 - 10);
        packet |= (self.timestamp as u64 & 0x1ff).wrapping_shl(64 - 10 - 10 - 9);
        packet |= (self.phase as u64 & 0x3ffff).wrapping_shl(64 - 10 - 10 - 9 - 18);
        packet |= (self.baseline as u64 & 0x1ffff).wrapping_shl(0);

        packet.to_be_bytes()
    }
}

#[derive(Debug, PartialEq, Hash, Clone, Copy)]
pub struct PaddingPacket;

impl SerializePacket for PaddingPacket {
    fn serialize(&self) -> [u8; 8] {
        0x7fff_ffff_ffff_ffffu64.to_be_bytes()
    }
}

#[derive(Debug, PartialEq, Hash)]
pub enum BinPacket {
    Header(HeaderPacket),
    Data(DataPacket),
    Padding(PaddingPacket),
}

impl SerializePacket for BinPacket {
    fn serialize(&self) -> [u8; 8] {
        match self {
            Self::Header(h) => h.serialize(),
            Self::Data(h) => h.serialize(),
            Self::Padding(h) => h.serialize(),
        }
    }
}

type Stream<'i> = &'i Bytes;

fn stream(b: &[u8]) -> Stream<'_> {
    Bytes::new(b)
}

#[inline]
fn parse_header(input: &mut Stream<'_>) -> PResult<HeaderPacket> {
    let fields: (u8, u8, u16, i64) = bits::<_, _, ContextError<(&str, usize)>, _, _>((
        take(8usize).verify(|i| *i == 0xff),
        take(8usize),
        take(12usize),
        take(36usize),
    ))
    .parse_next(input)
    .map_err(|_| ErrMode::<ContextError>::Cut(ContextError::new()))?;

    PResult::Ok(HeaderPacket {
        board: fields.1,
        frame: fields.2,
        timestamp: fields.3,
    })
}

#[inline]
fn parse_data(input: &mut Stream<'_>) -> PResult<DataPacket> {
    let fields: (u16, u16, u16, i32, i32) = bits::<_, _, ContextError<(String, usize)>, _, _>((
        // We verify that this is not a padding packet or a header packet
        take(10usize).verify(|i| *i < 256),
        take(10usize).verify(|i| *i < 256),
        take(9usize),
        // Use an arithmetic right shift to sign extend these
        take(18usize).map(|input: i32| input.wrapping_shl(32 - 17).wrapping_shr(32 - 17)),
        take(17usize).map(|input: i32| input.wrapping_shl(32 - 16).wrapping_shr(32 - 16)),
    ))
    .parse_next(input)
    .map_err(|_| ErrMode::<ContextError>::Cut(ContextError::new()))?;

    PResult::Ok(DataPacket {
        x: fields.0 as u8,
        y: fields.1 as u8,
        timestamp: fields.2,
        phase: fields.3,
        baseline: fields.4,
    })
}

#[inline]
fn complete_packet(
    header: HeaderPacket,
    storage: &mut Photons,
    input: &mut Stream<'_>,
) -> PResult<()> {
    while let Ok(packet) = parse_data.parse_next(input) {
        storage.xy.push(packet.y as u16 | ((packet.x as u16) << 8));
        storage
            .timestamp
            .push(packet.timestamp as i64 + header.timestamp * 500);
        storage.phase.push(packet.phase);
        storage.baseline.push(packet.baseline);
    }
    let _ = literal::<&[u8], Stream<'_>, ErrMode<ContextError>>(&[
        0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    ])
    .parse_next(input);
    Ok(())
}

fn parse_packet(storage: &mut Photons, input: &mut Stream<'_>) -> PResult<HeaderPacket> {
    let header = parse_header.parse_next(input)?;
    complete_packet(header, storage, input)?;
    Ok(header)
}

/// Read from a file, potentially one that is still being written (or a pipe)
/// and optionally one that does not start with a header by providing
/// a header for photons that come before any header
///
/// Instead of using this directly you probably want `BinDirectory`
pub fn read_file(
    file: &mut File,
    header: Option<HeaderPacket>,
) -> Result<(Photons, HeaderPacket), BinneyError> {
    // Read the full file into a buffer
    let mut buffer: Vec<u8> = vec![];
    file.read_to_end(&mut buffer)?;
    let mut buffer = stream(buffer.as_ref());

    // Preallocate photon storage that is at least as large as the number of photons
    // in the file
    let mut storage = Photons::with_capacity(buffer.len() as usize / 8);

    // If a header is provided attempt to read the first bytes in the file
    // as if they are photons with the provided header
    if let Some(header) = header {
        complete_packet(header, &mut storage, &mut buffer)?;
    }
    if buffer.len() < 8 {
        // If the file does not have more than 8 bytes left and we were
        // provided a header than return any photons we may have read and
        // the provided header
        if let Some(header) = header {
            return Ok((storage, header));
        }

        // If not error out because there was never enough info to read
        // anything, report how much more we would need
        Err(ErrMode::Incomplete(winnow::error::Needed::Size(
            std::num::NonZero::new(8usize - buffer.len()).unwrap(),
        )))?;
    }

    // Parse the first header in this file and its set of photon packets,
    // then parse all the rest
    let mut header = parse_packet(&mut storage, &mut buffer)?;
    while buffer.len() >= 8 {
        header = parse_packet(&mut storage, &mut buffer)?;
    }
    Ok((storage, header))
}

fn to_dataframe(binfile: &mut File) -> Result<(DataFrame, HeaderPacket), BinneyError> {
    let (photons, header) = read_file(binfile, None)?;
    let xys = Series::new("xy", photons.xy);
    let ts = Series::new("timestamp", photons.timestamp);
    let ps = Series::new("phase", photons.phase);
    let bs = Series::new("baseline", photons.baseline);

    Ok((DataFrame::new(vec![xys, ts, ps, bs])?, header))
}

pub fn to_parquet(binfile: &mut File, parquet: &mut File) -> Result<HeaderPacket, BinneyError> {
    let (mut df, header) = to_dataframe(binfile)?;

    df.sort_in_place(
        ["xy", "timestamp"],
        SortMultipleOptions::new().with_multithreaded(false),
    )?;

    ParquetWriter::new(parquet).finish(&mut df)?;
    df.clear();

    Ok(header)
}

/// Represents a folder full of bin files and provides access methods
///
/// This is intentionally opaque to force you to use the provided methods for
/// operations which correctly handle counter overflows which may occur up to
/// once during a gen2 observing night. Both sides of the range are inclusive
///
/// Instead of using this directly you probably want `binney.BinDirectoryDF`
/// which provides methods for getting a lazy `polars.LazyFrame` which
/// supports a powerful and effecient query syntax (Requires the polars
/// feature to be enabled, ie `"pybinney[polars]"` in your `pyproject.toml`)
///
/// ### Usage
///
/// ```python
/// bindir = binney.BinDirectory("/nfs/dark/data/ScienceData/Subaru/20201006/", "./parquet-cache", True)
/// bindir.convert_all()
/// ```
///
/// ### Parameters
/// `binney.BinDirectory(bindir: pathlib.Path | str, parquet_dir: pathlib.Path | str, /, *, progress: bool = False) -> BinDirectory`
/// - `bindir: pathlib.Path | str` A folder full of gen2 format `.bin` files
/// - `parquet_dir: pathlib.Path | str`
/// - `progress: bool` Show a progress bar during large conversions (`binney.BinDirectory.convert_all`
//     and `binney.BinDirectory.convert_timeranges`)
#[pyclass(frozen, subclass)]
pub struct BinDirectory {
    files: Vec<(PathBuf, TimestampRange)>,
    parquet_dir: (
        PathBuf,
        Arc<Mutex<HashMap<(TimestampRange, usize), PathBuf>>>,
    ),
    progress: bool,
}

impl BinDirectory {
    fn convert_or_cached(
        &self,
        index: usize,
        binpath: &PathBuf,
        trange: TimestampRange,
        overwrite: bool,
    ) -> Result<PathBuf, BinneyError> {
        let (path, cache) = &self.parquet_dir;
        let ppath = path.join(format!(
            "{}-{}-{}.parquet",
            trange.start, trange.stop, index
        ));

        // Return immediately if the parquet file already exists and we don't need to update it
        if ppath.is_file() && !overwrite && ppath.metadata()?.mtime() >= binpath.metadata()?.mtime()
        {
            return Ok(ppath);
        }

        // Check the cache to see if we have already converted it
        //
        // NOTE: This check is only strictly necessary if this function is
        //       being called by several different threads for the same file
        //       implements a per file lock, hopefully avoiding TOCTOU
        //       nightmares and clobbering
        {
            let mut cache = cache.lock().unwrap();
            if cache.contains_key(&(trange, index)) {
                return Ok(ppath);
            }
            cache.insert((trange, index), ppath.clone());
        }

        // Create and wite the file, and appropriately manage the cache and FS if this step fails.
        let mut file = File::create(&ppath)?;
        if let Err(e) = to_parquet(&mut File::open(binpath)?, &mut file) {
            let mut cache = cache.lock().unwrap();
            cache.remove(&(trange, index));
            std::fs::remove_file(ppath)?;
            return Err(e);
        }
        Ok(ppath)
    }
}

#[pymethods]
impl BinDirectory {
    /// We assume that the timestamps from consecutive binfiles overlap by no more than 10 ms
    pub const OVERLAP_RANGE: i64 = 10000;

    /// Construct a new `BinDirectory` instance from a bindirectory and a parquet cache directory
    /// Optionally enable a progressbar during conversions
    #[new]
    #[pyo3(signature = (bindir, parquet_dir, /, *, progress=false))]
    pub fn new(
        bindir: std::path::PathBuf,
        parquet_dir: std::path::PathBuf,
        progress: bool,
    ) -> Result<BinDirectory, BinneyError> {
        if bindir.is_dir() && !parquet_dir.is_file() {
            if !parquet_dir.is_dir() {
                std::fs::create_dir(&parquet_dir)?
            }
            let mut files = vec![];
            let mut hbuf = [0; 8];

            // TODO: This is not fast as it could be in theory, maybe parallelize the for
            for file in std::fs::read_dir(bindir)? {
                let path = file?.path();

                // Look for files ending in .bin
                if path.is_file() && path.extension().unwrap_or(std::ffi::OsStr::new("")) == "bin" {
                    let meta = std::fs::metadata(&path)?;
                    // Look for files that are non-empty files that are mmultiples of 8 bytes
                    // TODO: Should we error if we find a .bin that is empty or not a multiple of 8?
                    if meta.len() >= 8 && meta.len() % 8 == 0 {
                        let mut f = File::open(&path)?;

                        // Parse the begining as a header to get the timestamp of the first packet
                        // Note: We do not necessarily know that the first packet has the first
                        //       timestamp in the file
                        f.read_exact(&mut hbuf)?;
                        f.rewind()?;
                        let header = parse_header(&mut stream(&hbuf))?;
                        files.push((path, header))
                    }
                }
            }

            // Find the packet with the highest timestamp
            files.sort_by_key(|(_, h)| h.timestamp);
            let lastheader = {
                let lastfile = files.last().ok_or(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "No binfiles found in directory",
                ))?;

                let mut lastfile = (File::open(&lastfile.0)?, lastfile.1);

                // For gen2 we know there should be a header every 102 packets or less, make room
                // for 256 just in case
                let mut lbuf = Vec::with_capacity(256 * 8);
                if lastfile.0.seek(SeekFrom::End(0))? < 256 {
                    lastfile.0.rewind()?;
                    lastfile.0.read_to_end(&mut lbuf)?;
                    lastfile.0.rewind()?;
                } else {
                    lastfile.0.seek(SeekFrom::End(-256 * 8))?;
                    lastfile.0.read_to_end(&mut lbuf)?;
                    lastfile.0.rewind()?;
                }

                // This is a semi recreational programming project so I reserve the right to use cursed syntax
                let mut i = lbuf.len() - 8;
                while match lbuf.get(i) {
                    Some(0xff) => false,
                    Some(_) => {
                        i -= 8;
                        true
                    }
                    None => {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "Binfile did not have a header packet before the end :( RIP",
                        ))?;
                        true
                    }
                } {}

                parse_header(&mut stream(&lbuf.as_slice()[i..i + 8]))?
            };

            let mut bd = BinDirectory {
                // See above comment
                //
                // This could be written more clearly imperatively but I wanted to learn iterators better, so:
                // - We can't clone a path so we use `std::Vec::into_iter` which consumes the vector giving us an owned
                //   file in our iterator
                // - We use rfold to go from the end back to the begining with an accumulator that has
                //   a vector that we want to eventually return, and a timestamp that starts with the
                //   last timestamp of the last file and ends up being the starting timestamp of each
                //   which we use to build ranges
                files: files
                    .into_iter()
                    .rfold((vec![], lastheader), |(mut v, h), (f, fh)| {
                        v.push((f, TimestampRange::new(fh.timestamp, h.timestamp)));
                        (v, fh)
                    })
                    .0,
                parquet_dir: (parquet_dir, Arc::new(Mutex::new(HashMap::new()))),
                progress,
            };
            bd.files.reverse();

            Ok(bd)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "bindir and or parquet dir is not a directory",
            ))?
        }
    }

    /// Convert all bin files in the bin directory into parquet files
    ///
    /// If `overwrite` is set this will overwrite existing parquet files
    /// otherwise it will only overwrite a parquet file if the corresponding
    /// bin file has changed since the parquet file was last written
    pub fn convert_all(&self, overwrite: bool) -> Result<Vec<PathBuf>, BinneyError> {
        if self.progress {
            self.files
                .par_iter()
                .progress_count(self.files.len() as u64)
                .enumerate()
                .map(|(i, (bpath, t))| self.convert_or_cached(i, bpath, *t, overwrite))
                .collect()
        } else {
            self.files
                .par_iter()
                .enumerate()
                .map(|(i, (bpath, t))| self.convert_or_cached(i, bpath, *t, overwrite))
                .collect()
        }
    }

    /// Convert a `TimestampRange` length
    pub fn convert_timerange(
        &self,
        trange: TimestampRange,
        overwrite: bool,
    ) -> Result<Vec<PathBuf>, BinneyError> {
        self.files
            .par_iter()
            .enumerate()
            .filter_map(|(i, (p, t))| {
                if t.grow(BinDirectory::OVERLAP_RANGE).overlaps(&trange) {
                    Some((i, p, t))
                } else {
                    None
                }
            })
            .map(|(i, p, t)| self.convert_or_cached(i, p, *t, overwrite))
            .collect()
    }

    /// Convert a set of `TimestampRanges`
    pub fn convert_timeranges(
        &self,
        tranges: Vec<TimestampRange>,
        overwrite: bool,
    ) -> Result<Vec<Vec<PathBuf>>, BinneyError> {
        tranges
            .par_iter()
            .map(|t| self.convert_timerange(*t, overwrite))
            .collect()
    }
}

/// The binney library provides several interfaces
#[pymodule]
fn binney(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BinDirectory>()?;
    m.add_class::<TimestampRange>()?;
    m.add_class::<BinSerializer>()?;

    m.add_wrapped(wrap_pymodule!(cli))?;

    Ok(())
}

/// Command line interface
#[pymodule]
mod cli {
    use hashbrown::HashMap;

    use super::*;
    #[derive(clap::Parser)]
    #[command(version, about)]
    struct Cli {
        /// Suppress the progress bar and any logging
        #[arg(long, short)]
        quiet: bool,

        /// Which subutility to use
        #[command(subcommand)]
        command: Commands,
    }

    #[derive(clap::Subcommand)]
    enum Commands {
        ConvertAll {
            /// A directory containing some bin files
            bin_dir: PathBuf,

            /// The directory where parquet files will be written
            parquet_dir: PathBuf,

            /// Overwrite parquet files even if the bin file has not changed
            /// since they were written
            #[arg(long)]
            overwrite: bool,
        },
        RoundTrip {
            binfile: PathBuf,
        },
    }

    /// Run the command line program
    #[pyfunction]
    pub fn main(py: Python) -> Result<(), BinneyError> {
        use clap::Parser;

        let argv = py
            .import_bound("sys")
            .unwrap()
            .getattr("argv")
            .unwrap()
            .extract::<Vec<String>>()
            .unwrap();

        let cli = Cli::parse_from(argv);

        match cli.command {
            Commands::ConvertAll {
                bin_dir,
                parquet_dir,
                overwrite,
            } => {
                BinDirectory::new(bin_dir, parquet_dir, true)
                    .unwrap()
                    .convert_all(overwrite)
                    .unwrap();
            }
            Commands::RoundTrip { binfile } => {
                let mut file = File::open(&binfile)?;
                let (mut df, _) = to_dataframe(&mut file)?;
                let mut board2pixel: HashMap<u8, Vec<u16>> = HashMap::new();
                for (i, b) in (230..=239).enumerate() {
                    let mut v = vec![];
                    for x in i * 14..(i + 1) * 14 {
                        for y in 0..146 {
                            v.push((x << 8 | y) as u16);
                        }
                    }
                    board2pixel.insert(b, v);
                }
                let serializer = BinSerializer::new("./".into(), board2pixel, 2020, 0)?;

                serializer.serialize_dataframe(&mut df)?;
            }
        }
        Ok(())
    }

    /// Hack: workaround for https://github.com/PyO3/pyo3/issues/759
    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        Python::with_gil(|py| {
            py.import_bound("sys")?
                .getattr("modules")?
                .set_item("binney.cli", m)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet() {
        #[rustfmt::skip]
        let mut input = stream(&[
            0xff, 0xe8, 0x76, 0x0b, 0x46, 0x72, 0x2c, 0xfb,
            0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xe8, 0x76, 0x0b, 0x46, 0x72, 0x2c, 0xfb,
            0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12,
            0xff, 0xe8, 0x76, 0x0b, 0x46, 0x72, 0x2c, 0xfb,
            0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12,
            0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12,
            0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12,
            0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12,
            0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ]);
        let mut storage = Photons::with_capacity(0);
        parse_packet(&mut storage, &mut input).unwrap();
        assert_eq!(storage, Photons::with_capacity(0));
        parse_packet(&mut storage, &mut input).unwrap();
        assert_eq!(
            storage,
            Photons {
                xy: vec![(99 << 8) + 44],
                timestamp: vec![131 + 48426527995 * 500],
                phase: vec![-34148],
                baseline: vec![-4334]
            }
        );
        parse_packet(&mut storage, &mut input).unwrap();
        assert_eq!(
            storage,
            Photons {
                xy: vec![(99 << 8) + 44; 5],
                timestamp: vec![131 + 48426527995 * 500; 5],
                phase: vec![-34148; 5],
                baseline: vec![-4334; 5]
            }
        );
    }

    #[test]
    fn test_header() {
        let mut input_header = stream(&[0xff, 0xe8, 0x76, 0x0b, 0x46, 0x72, 0x2c, 0xfb]);
        assert_eq!(
            parse_header(&mut input_header),
            Ok(HeaderPacket {
                board: 232,
                frame: 1888,
                timestamp: 48426527995
            })
        );
        assert_eq!(input_header, stream(&[]))
    }

    #[test]
    fn test_data() {
        let mut input_data = stream(&[0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12]);
        assert_eq!(
            parse_data(&mut input_data),
            Ok(DataPacket {
                x: 99,
                y: 44,
                timestamp: 131,
                phase: -34148,
                baseline: -4334
            })
        );

        assert_eq!(input_data, stream(&[]))
    }

    #[test]
    fn test_ranges() {
        let a = TimestampRange::new(0, 100);
        let b = TimestampRange::new(99, 1000);
        let c = TimestampRange::new(101, 102);
        let d = TimestampRange::new(
            0xf_ffff_ffff * 500 + 500 - 30,
            0xf_ffff_ffff * 500 + 500 + 30,
        );
        let e = TimestampRange::new(100, 101);
        let f = TimestampRange::new(-30, 30);

        assert_eq!(d, f);

        assert!(d.start > d.stop);
        assert!(d.stop > 0);

        assert!(a.overlaps(&b));
        assert!(b.overlaps(&c));
        assert!(!a.overlaps(&c));
        assert!(d.overlaps(&a));
        assert!(!d.overlaps(&b));
        assert!(e.overlaps(&a));
        assert!(e.overlaps(&b));
        assert!(e.overlaps(&c));
        assert!(!e.overlaps(&d));

        assert!(b.overlaps(&a));
        assert!(c.overlaps(&b));
        assert!(!c.overlaps(&a));
        assert!(a.overlaps(&d));
        assert!(!b.overlaps(&d));
        assert!(a.overlaps(&e));
        assert!(b.overlaps(&e));
        assert!(c.overlaps(&e));
        assert!(!d.overlaps(&e));

        assert!(a.inside(99));
        assert!(a.inside(100));
        assert!(!a.inside(101));
        assert!(a.inside(0));
        assert!(b.inside(99));
        assert!(!b.inside(98));

        assert!(d.inside(0xf_ffff_ffff * 500 + 500));
        assert!(d.inside(0xf_ffff_ffff * 500 + 500 - 1));
        assert!(d.inside(0xf_ffff_ffff * 500 + 500 + 1));
    }
}
