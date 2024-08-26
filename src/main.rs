use std::fs::File;
use std::io::Read;

use winnow::binary::bits::{bits, take};
use winnow::error::{ContextError, ErrMode};
use winnow::prelude::*;
use winnow::token::literal;
use winnow::Bytes;

use polars::prelude::*;

#[derive(Debug, PartialEq, Clone, Copy)]
struct HeaderPacket {
    board: u8,
    frame: u16,
    timestamp: u64,
}

#[derive(Debug, PartialEq)]
struct DataPacket {
    x: u8,
    y: u8,
    timestamp: u16,
    phase: i32,
    baseline: i32,
}

#[derive(Debug, PartialEq, Clone)]
struct Photon {
    x: u8,
    y: u8,
    timestamp: u64,
    phase: i32,
    baseline: i32,
}

#[derive(Debug)]
struct Photons {
    x: Vec<u8>,
    y: Vec<u8>,
    timestamp: Vec<u64>,
    phase: Vec<i32>,
    baseline: Vec<i32>,
}

impl Photons {
    fn with_capacity(capacity: usize) -> Photons {
        Photons {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            timestamp: Vec::with_capacity(capacity),
            phase: Vec::with_capacity(capacity),
            baseline: Vec::with_capacity(capacity),
        }
    }
}

#[derive(Debug)]
enum BinneyError<T> {
    IOError(std::io::Error),
    ParseError(winnow::error::ErrMode<T>),
}

impl<T> From<std::io::Error> for BinneyError<T> {
    fn from(error: std::io::Error) -> BinneyError<T> {
        BinneyError::IOError(error)
    }
}

impl<T> From<winnow::error::ErrMode<T>> for BinneyError<T> {
    fn from(error: winnow::error::ErrMode<T>) -> BinneyError<T> {
        BinneyError::ParseError(error)
    }
}

type Stream<'i> = &'i Bytes;

fn stream(b: &[u8]) -> Stream<'_> {
    Bytes::new(b)
}

#[inline]
fn parse_header(input: &mut Stream<'_>) -> PResult<HeaderPacket> {
    let fields: (u8, u8, u16, u64) = bits::<_, _, ContextError<(&str, usize)>, _, _>((
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
    storage: &mut Vec<Photon>,
    input: &mut Stream<'_>,
) -> PResult<()> {
    while let Ok(packet) = parse_data.parse_next(input) {
        storage.push(Photon {
            x: packet.x,
            y: packet.y,
            timestamp: packet.timestamp as u64 + header.timestamp,
            phase: packet.phase,
            baseline: packet.baseline,
        })
    }
    let _ = literal::<&[u8], Stream<'_>, ErrMode<ContextError>>(&[
        0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    ])
    .parse_next(input);
    Ok(())
}

fn parse_packet(storage: &mut Vec<Photon>, input: &mut Stream<'_>) -> PResult<HeaderPacket> {
    let header = parse_header.parse_next(input)?;
    complete_packet(header, storage, input)?;
    Ok(header)
}

fn read_file(
    filename: &str,
    header: Option<HeaderPacket>,
) -> Result<(Vec<Photon>, HeaderPacket), BinneyError<ContextError>> {
    let mut buffer: Vec<u8> = vec![];
    File::open(filename)?.read_to_end(&mut buffer)?;
    let mut buffer = stream(buffer.as_ref());
    let mut storage = Vec::with_capacity(buffer.len() as usize / 8);

    if let Some(header) = header {
        complete_packet(header, &mut storage, &mut buffer)?;
    }
    if buffer.len() < 8 {
        if let Some(header) = header {
            return Ok((storage, header));
        }
        Err(ErrMode::Incomplete(winnow::error::Needed::Size(
            std::num::NonZero::new(8usize).unwrap(),
        )))?;
    }
    let mut header = parse_packet(&mut storage, &mut buffer)?;
    while buffer.len() >= 8 {
        header = parse_packet(&mut storage, &mut buffer)?;
    }
    Ok((storage, header))
}

fn main() -> Result<(), BinneyError<ContextError>> {
    let (photons, header) = read_file("/tmp/1602050064.bin", None)?;
    let xs = Series::new("x", photons.iter().map(|i| i.x).collect::<Vec<u8>>());
    let ys = Series::new("y", photons.iter().map(|i| i.y).collect::<Vec<u8>>());
    let ts = Series::new(
        "timestamp",
        photons.iter().map(|i| i.timestamp).collect::<Vec<u64>>(),
    );
    let ps = Series::new(
        "phase",
        photons.iter().map(|i| i.phase).collect::<Vec<i32>>(),
    );
    let bs = Series::new(
        "baseline",
        photons.iter().map(|i| i.baseline).collect::<Vec<i32>>(),
    );

    let mut df = DataFrame::new(vec![xs, ys, ts, ps, bs])
        .unwrap()
        .sort(["timestamp"], Default::default())
        .unwrap();

    ParquetWriter::new(&mut std::fs::File::create("/tmp/test.parquet")?)
        .finish(&mut df)
        .unwrap();

    Ok(())
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
        let mut storage = Vec::with_capacity(10);
        parse_packet(&mut storage, &mut input).unwrap();
        assert_eq!(storage, vec![]);
        parse_packet(&mut storage, &mut input).unwrap();
        assert_eq!(
            storage,
            vec![Photon {
                x: 99,
                y: 44,
                timestamp: 131 + 48426527995,
                phase: -34148,
                baseline: -4334
            }]
        );
        parse_packet(&mut storage, &mut input).unwrap();
        assert_eq!(
            storage,
            std::iter::repeat(Photon {
                x: 99,
                y: 44,
                timestamp: 131 + 48426527995,
                phase: -34148,
                baseline: -4334
            })
            .take(5)
            .collect::<Vec<_>>()
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
}
