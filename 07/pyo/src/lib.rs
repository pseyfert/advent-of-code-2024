use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn check_operators(test_value: u64, operands: Vec<u64>) -> u16 {
    let num_operands = operands.len();
    assert_ne!(num_operands, 0);
    assert_ne!(num_operands, 1);
    let num_operators = num_operands - 1;
    let combinations = 2_u16.pow(num_operators as u32);
    let working_combinations = (0..combinations)
        .into_iter()
        .filter_map(|comb| {
            let first = operands[0];
            let equation = operands[1..]
                .iter()
                .enumerate()
                .fold(first, |acc, (pow, val)| {
                    if 2_u16.pow(pow as u32) & comb > 0 {
                        acc * val
                    } else {
                        acc + val
                    }
                });
            if equation == test_value {
                Some(comb)
            } else {
                None
            }
        })
        .collect::<Vec<u16>>();
    working_combinations.len() as u16
}

fn get_base_3_bit(comb: u32, pow: u32) -> u32 {
    (comb / 3_u32.pow(pow)) % 3
}

fn merge(lhs: u64, rhs: &u64) -> u64 {
    10_u64.pow(1 + rhs.ilog10()) * lhs + rhs
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(0, 0, 0)]
    #[case(1, 0, 1)]
    #[case(2, 0, 2)]
    #[case(3, 0, 0)]
    #[case(4, 0, 1)]
    #[case(5, 0, 2)]
    #[case(6, 0, 0)]
    #[case(0, 1, 0)]
    #[case(1, 1, 0)]
    #[case(2, 1, 0)]
    #[case(3, 1, 1)]
    #[case(4, 1, 1)]
    #[case(5, 1, 1)]
    #[case(6, 1, 2)]
    #[case(7, 1, 2)]
    #[case(8, 1, 2)]
    #[case(9, 1, 0)]
    #[case(9, 2, 1)]
    fn test_3_bit(#[case] comb: u32, #[case] pow: u32, #[case] result: u32) {
        assert_eq!(get_base_3_bit(comb, pow), result);
    }

    #[rstest]
    #[case(0, 1, 1)]
    #[case(1, 1, 11)]
    #[case(1, 9, 19)]
    #[case(1, 10, 110)]
    #[case(1, 11, 111)]
    #[case(1, 99, 199)]
    #[case(1, 100, 1100)]
    #[case(66, 100, 66100)]
    fn test_merge(#[case] lhs: u64, #[case] rhs: u64, #[case] result: u64) {
        assert_eq!(merge(lhs, &rhs), result);
    }

}

#[pyfunction]
fn check_again(test_value: u64, operands: Vec<u64>) -> u32 {
    let num_operands = operands.len();
    assert_ne!(num_operands, 0);
    assert_ne!(num_operands, 1);
    let num_operators = num_operands - 1;
    let combinations = 3_u32.pow(num_operators as u32);
    let working_combinations = (0..combinations)
        .into_iter()
        .filter_map(|comb| {
            let first = operands[0];
            let equation = operands[1..]
                .iter()
                .enumerate()
                .fold(first, |acc, (pow, val)| {
                    match get_base_3_bit(comb, pow as u32) {
                        1 => acc * val,
                        0 => acc + val,
                        2 => merge(acc, &val),
                        _ => panic!("must not happen")
                    }
                });
            if equation == test_value {
                Some(comb)
            } else {
                None
            }
        })
        .collect::<Vec<u32>>();
    working_combinations.len() as u32
}

#[pymodule]
fn pyo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_operators, m)?)?;
    m.add_function(wrap_pyfunction!(check_again, m)?)
}
