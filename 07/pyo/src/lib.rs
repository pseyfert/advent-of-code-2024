use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;

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

#[pymodule]
fn pyo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_operators, m)?)
}
