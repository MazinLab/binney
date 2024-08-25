use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use winnow::binary::bits::{bits, take};
use winnow::combinator::seq;
use winnow::error::InputError;
use winnow::prelude::*;
use winnow::Bytes;

#[derive(Debug, PartialEq)]
struct HeaderPacket {
    roach: u8,
    frame: u16,
    timestamp: u64,
}

#[derive(Debug, PartialEq)]
struct DataPacket {
    x: u16,
    y: u16,
    timestamp: u16,
    phase: i32,
    baseline: i32,
}

#[derive(Debug, PartialEq)]
struct Photon {
    x: u16,
    y: u16,
    timestamp: u64,
    phase: i32,
    baseline: i32,
}

type Stream<'i> = &'i Bytes;

fn stream(b: &[u8]) -> Stream<'_> {
    Bytes::new(b)
}

#[inline]
fn parse_header(input: Stream<'_>) -> IResult<Stream<'_>, (u8, u8, u16, u64)> {
    bits::<_, _, InputError<(_, usize)>, _, _>((
        take(8usize),
        take(8usize),
        take(12usize),
        take(36usize),
    ))
    .parse_peek(input)
}

#[inline]
fn parse_data(input: Stream<'_>) -> IResult<Stream<'_>, (u16, u16, u16, i32, i32)> {
    bits::<_, _, InputError<(_, usize)>, _, _>((
        take(10usize),
        take(10usize),
        take(9usize),
        // Use an arithmetic right shift to sign extend these
        take(18usize).map(|input: i32| input.wrapping_shl(32 - 17).wrapping_shr(32 - 17)),
        take(17usize).map(|input: i32| input.wrapping_shl(32 - 16).wrapping_shr(32 - 16)),
    ))
    .parse_peek(input)
}

fn main() -> std::io::Result<()> {
    // let bin = File::open("/tmp/1602050064.bin")?;
    let input_header = stream(&[0xff, 0xe8, 0x76, 0x0b, 0x46, 0x72, 0x2c, 0xfb]);
    let input_data = stream(&[0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12]);
    println!("{:?}", parse_header(input_header).unwrap());
    println!("{:?}", parse_data(input_data).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header() {
        let input_header = stream(&[0xff, 0xe8, 0x76, 0x0b, 0x46, 0x72, 0x2c, 0xfb]);
        assert_eq!(
            parse_header(input_header),
            Ok((stream(&[]), (255, 232, 1888, 48426527995)))
        );
    }

    #[test]
    fn test_data() {
        let input_data = stream(&[0x18, 0xc2, 0xc4, 0x1e, 0xf5, 0x39, 0xef, 0x12]);
        assert_eq!(
            parse_data(input_data),
            Ok((stream(&[]), (99, 44, 131, -34148, -4334)))
        );
    }
}
