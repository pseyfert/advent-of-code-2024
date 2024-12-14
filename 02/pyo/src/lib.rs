use adjacent_pair_iterator::AdjacentPairIterator;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use log::{debug, info, warn};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use rayon::prelude::*;

#[derive(Debug, PartialEq, Eq)]
enum ReportState {
    Safe,
    Unsafe,
}

impl TryFrom<State> for ReportState {
    type Error = ();
    fn try_from(state: State) -> Result<ReportState, ()> {
        match state {
            State::Descending | State::Ascending => Ok(ReportState::Safe),
            State::Uninitialized => Err(()),
            State::Unsafe => Ok(ReportState::Unsafe),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum State {
    Uninitialized,
    Ascending,
    Descending,
    Unsafe,
}

fn check_report(report: &Vec<i32>) -> ReportState {
    report
        .adjacent_pairs()
        .fold_while(State::Uninitialized, |state, (lhs, rhs)| {
            let this_state = match rhs - lhs {
                -3..=-1 => State::Descending,
                1..=3 => State::Ascending,
                _ => {
                    return Done(State::Unsafe);
                }
            };
            match state {
                State::Uninitialized => Continue(this_state),
                State::Unsafe => panic!(),
                defined_state => {
                    if defined_state == this_state {
                        Continue(this_state)
                    } else {
                        Done(State::Unsafe)
                    }
                }
            }
        })
        .into_inner()
        .try_into()
        .unwrap()
}

#[gen_stub_pyfunction]
#[pyfunction]
fn part_one(data: Vec<Vec<i32>>) -> i32 {
    data.par_iter()
        .map(check_report)
        .filter(|state| *state == ReportState::Safe)
        .count() as i32
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(vec![7,6,4,2,1])]
    #[case(vec![1,3,6,7,9])]
    fn test_some_safe(#[case] report: Vec<i32>) {
        assert_eq!(check_report(&report), ReportState::Safe);
    }
}

#[pymodule]
fn pyo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(part_one, m)?)
}

define_stub_info_gatherer!(stub_info);
