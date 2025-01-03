#![feature(test)]
#![feature(associated_type_defaults)]

extern crate test;
use log::{debug, error};
use std::path::Path;

use test::{black_box, Bencher};

pub fn main_wrap<T: Day>() -> std::process::ExitCode
where
    <<T as Day>::Desered as TryInto<T>>::Error: 'static,
{
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

fn main_fn<T: Day>() -> anyhow::Result<()>
where
    <<T as Day>::Desered as TryInto<T>>::Error: 'static,
{
    let input = aoc_cli::setup_and_input()?;
    debug!("input path is clear, let's parse");

    let parsed: T = T::deser(input)?.try_into()?;
    debug!("parsed, let's process");

    T::process(&parsed)?;

    Ok(())
}

// TODO: rewrite such that TryFrom is sufficient.
pub trait Day: Sized
where
    <Self::Desered as TryInto<Self>>::Error: Send + Sync + std::error::Error,
{
    type Desered: for<'a> serde::de::Deserialize<'a> + TryInto<Self>;
    type State<'c> = ();

    fn deser(p: impl AsRef<Path>) -> anyhow::Result<Self::Desered> {
        Ok(serde_linewise::from_str(&std::fs::read_to_string(p)?)?)
    }
    fn process(p: &Self) -> anyhow::Result<()> {
        let (part1, state) = Self::inner_part1(p)?;
        println!("Answer to part 1 {part1}.");
        let part2 = Self::inner_part2(p, state)?;
        println!("Answer to part 2 {part2}.");
        Ok(())
    }
    fn inner_part1<'b>(&'b self) -> anyhow::Result<(u64, Self::State<'b>)>;
    fn inner_part2<'b>(&self, _s: Self::State<'b>) -> anyhow::Result<u64> {
        Ok(0)
    }
    fn part1(&self) -> anyhow::Result<u64> {
        let (r, _) = self.inner_part1()?;
        Ok(r)
    }
    fn part2(&self) -> anyhow::Result<u64> {
        // let's hope the optimizer can deal with this
        let (_, state) = self.inner_part1()?;
        self.inner_part2(state)
    }

    fn bench_part2(b: &mut Bencher, p: impl AsRef<Path>) {
        let input = Self::deser(p).unwrap().try_into().unwrap();
        b.iter(|| black_box(Self::part2(&input)))
    }
    fn bench_part1(b: &mut Bencher, p: impl AsRef<Path>) {
        let input = Self::deser(p).unwrap().try_into().unwrap();
        b.iter(|| black_box(Self::part1(&input)))
    }
}

#[macro_export]
macro_rules! bench_parts {
    ($DayType:ty, $input:expr) => {
        #[cfg(test)]
        mod benchmarks {
            use super::*;
            use boiler_plate::Day;
            use test::Bencher;

            #[bench]
            fn bench_wrap1(b: &mut Bencher) {
                <$DayType>::bench_part1(b, $input);
            }
            #[bench]
            fn bench_wrap2(b: &mut Bencher) {
                <$DayType>::bench_part2(b, $input);
            }
        }
    };
}

#[macro_export]
macro_rules! main {
    ($DayType:ty) => {
        fn main() -> std::process::ExitCode {
            boiler_plate::main_wrap::<$DayType>()
        }
    };
}

/// This macro is to support data types that already exist somewhere.
#[macro_export]
macro_rules! just_wrap {
    ($OutType:ident, $InType:ty) => {
        struct $OutType {
            data: $InType,
        }

        impl From<<$OutType as boiler_plate::Day>::Desered> for $OutType {
            fn from(vs: <$OutType as boiler_plate::Day>::Desered) -> $OutType {
                $OutType { data: vs.into() }
            }
        }
    };
}
