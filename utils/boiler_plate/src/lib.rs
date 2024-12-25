#![feature(associated_type_defaults)]
use log::{debug, error};
use std::path::Path;

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

    T::process(parsed)?;

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
    fn process(_: Self::Parsed) -> anyhow::Result<()>;
}
