use std::path::Path;
use log::error;

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

    let parsed: T::Parsed = T::parse(input)?;

    T::process(parsed)?;

    Ok(())
}

pub trait Day {
    type Parsed;
    fn parse(p: impl AsRef<Path>) -> anyhow::Result<Self::Parsed>;
    // where
    //     <Self as Day>::Parsed: for<'a> serde::de::Deserialize<'a>,
    // {
    //     Ok(serde_linewise::from_str(&std::fs::read_to_string(p)?)?)
    // }
    fn process(_: Self::Parsed) -> anyhow::Result<()>;
}
