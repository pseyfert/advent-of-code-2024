#![feature(associated_type_defaults)]
#![feature(test)]

extern crate test;
use log::{debug, error};
use std::path::Path;

use test::{black_box, Bencher};

pub fn main_wrap<T: Day>() -> std::process::ExitCode {
    match main_fn::<T>() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // Does this work in case of failure to set up logging?
            // You know what? I don't care.
            error!("Program failed: {e:?}");
            std::process::ExitCode::from(1)
        }
    }
}

fn main_fn<T: Day>() -> anyhow::Result<()> {
    let input = aoc_cli::setup_and_input()?;
    debug!("input path is clear, let's parse");

    let parsed: T::Parsed = T::parse(T::deser(input)?)?;
    debug!("parsed, let's process");

    T::process(&parsed)?;

    Ok(())
}

// TODO: merge Day and Parsed into one type. That way, parse can be dropped and I only keep the
// From.
// TODO: rewrite such that TryFrom is sufficient.
pub trait Day {
    type Desered: for<'a> serde::de::Deserialize<'a>;
    type Parsed: From<Self::Desered> = Self::Desered;

    fn deser(p: impl AsRef<Path>) -> anyhow::Result<Self::Desered> {
        Ok(serde_linewise::from_str(&std::fs::read_to_string(p)?)?)
    }
    fn parse(desered: Self::Desered) -> anyhow::Result<Self::Parsed> {
        debug!("deserialized, now convert");

        Ok(desered.into())
    }
    fn process(p: &Self::Parsed) -> anyhow::Result<()> {
        let part1 = Self::part1(&p)?;
        println!("Answer to part 1 {part1}.");
        let part2 = Self::part2(&p)?;
        println!("Answer to part 2 {part2}.");
        Ok(())
    }
    fn part1(_: &Self::Parsed) -> anyhow::Result<u64>;
    fn part2(_: &Self::Parsed) -> anyhow::Result<u64> {
        Ok(0)
    }

    fn bench_part2(b: &mut Bencher, p: impl AsRef<Path>) {
        let input = Self::deser(p).unwrap().into();
        b.iter(|| black_box(Self::part2(&input)))
    }
    fn bench_part1(b: &mut Bencher, p: impl AsRef<Path>) {
        let input = Self::deser(p).unwrap().into();
        b.iter(|| black_box(Self::part1(&input)))
    }
}
