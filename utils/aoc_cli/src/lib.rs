use clap::{Parser, ValueHint};
use clap_verbosity_flag::{Verbosity, WarnLevel};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about)]
#[clap(name = "whopper")]
pub struct Options {
    #[command(flatten)]
    pub verbose: Verbosity<WarnLevel>,

    #[clap(
        value_parser,
        help = "File with the input data.",
        default_value = "../input.txt",
        value_hint = ValueHint::FilePath,
        value_name = "FILE"
    )]
    pub input: PathBuf,
}

#[derive(thiserror::Error, Debug)]
pub enum SetupError {
    #[error("failed to initialize logging: {0:?}")]
    LogSystemError(#[from] log::SetLoggerError),
}

pub fn setup_and_input() -> Result<std::path::PathBuf, SetupError> {
    let args = Options::parse();
    stderrlog::new()
        .verbosity(args.verbose.log_level_filter())
        .timestamp(stderrlog::Timestamp::Off)
        .init()?;

    Ok(args.input)
}
