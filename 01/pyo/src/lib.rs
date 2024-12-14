#[allow(unused_imports)]
use log::{debug, info, warn};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use rayon::prelude::*;

static LOG: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn init_log() {
    LOG.get_or_init(|| {
        stderrlog::new()
            .verbosity(log::Level::Warn)
            .timestamp(stderrlog::Timestamp::Off)
            .init()
            .unwrap()
    });
}

#[gen_stub_pyfunction]
#[pyfunction]
fn rust_sort(mut locations: Vec<u32>) -> Vec<u32> {
    init_log();
    locations.par_sort();
    locations
}

#[gen_stub_pyfunction]
#[pyfunction]
fn distance_sum(list1: Vec<u32>, list2: Vec<u32>) -> u32 {
    init_log();
    list1
        .par_iter()
        .zip(list2.par_iter())
        .map(|(lhs, rhs)| if lhs > rhs { lhs - rhs } else { rhs - lhs })
        .sum::<u32>()
}

#[gen_stub_pyfunction]
#[pyfunction]
fn part2(list1: Vec<u32>, list2: Vec<u32>) -> u32 {
    init_log();
    debug!("{list1:?}, {list2:?}");
    list1
        .iter()
        // .par_iter()
        .map(|loc| {
            let maybe_first_pos = list2.partition_point(|x| x < loc);
            let rv = list2[maybe_first_pos..]
                .iter()
                .take_while(|x| *x == loc)
                .count() as u32
                * loc;
            debug!("returning {rv} for {loc}");
            rv
        })
        .sum::<u32>()
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    // #[rstest]
    // #[case(0, 0, 0)]
    // fn test_3_bit(#[case] comb: u64, #[case] pow: u64, #[case] result: u64) {
    //     assert_eq!(get_base_3_bit(comb, pow), result);
    // }
}

#[pymodule]
fn pyo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_sort, m)?)?;
    m.add_function(wrap_pyfunction!(distance_sum, m)?)?;
    m.add_function(wrap_pyfunction!(part2, m)?)
}

define_stub_info_gatherer!(stub_info);
