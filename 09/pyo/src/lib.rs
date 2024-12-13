use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use log::{debug, info, warn};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::gen_stub_pyfunction};
use rayon::prelude::*;

fn get_file_loc_and_size(block_sizes: &[u8]) -> Vec<(u64, u8)> {
    block_sizes
        .iter()
        .scan(0_u64, |pointer, incoming_size| {
            let rv = (*pointer, *incoming_size);
            *pointer += *incoming_size as u64;
            Some(rv)
        })
        .step_by(2)
        .collect()
}

fn get_space_loc_and_size(block_sizes: &[u8]) -> Vec<(u64, u8)> {
    block_sizes
        .iter()
        .scan(0_u64, |pointer, incoming_size| {
            let rv = (*pointer, *incoming_size);
            *pointer += *incoming_size as u64;
            Some(rv)
        })
        .skip(1)
        .step_by(2)
        .collect()
}

fn checksum_files(locs: &[(u64, u8)]) -> u64 {
    locs.par_iter()
        .enumerate()
        .map(|(file_id, (loc, size))| (*loc..loc + (*size as u64)).sum::<u64>() * (file_id as u64))
        .sum()
}

#[gen_stub_pyfunction]
#[pyfunction]
fn defrag(block_sizes: Vec<u8>) -> u64 {
    let mut files = get_file_loc_and_size(&block_sizes);
    let mut spaces = get_space_loc_and_size(&block_sizes);

    debug!("{files:?}");
    for (file_loc, file_size) in files.iter_mut().rev() {
        let need_to_trim = if let Some((pos, (target_loc, space_size))) = spaces
            .iter_mut()
            .enumerate()
            .take_while(|(_, (loc, _))| loc <= file_loc)
            .find(|(_, (_, space_size))| space_size >= file_size)
        {
            *file_loc = *target_loc;
            *target_loc += *file_size as u64;
            *space_size -= *file_size;
            if *space_size == 0 {
                Some(pos)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(pos_to_trim) = need_to_trim {
            spaces.remove(pos_to_trim);
        }
    }
    debug!("{files:?}");
    checksum_files(&files)
}

#[gen_stub_pyfunction]
#[pyfunction]
fn checksum(block_sizes: Vec<u8>) -> u64 {
    // let disc_size: u64 = (&block_sizes).iter().map(|x| *x as u64).sum();
    let occupied: u64 = block_sizes.iter().step_by(2).map(|x| *x as u64).sum();
    let mut just_files: Vec<(usize, u8)> =
        block_sizes.iter().cloned().step_by(2).enumerate().collect();
    let (_, result) = block_sizes
        .iter()
        .chunks(2)
        .into_iter()
        .enumerate()
        .fold_while(
            (0_u64, 0_u64),
            |(org_pointer, acc), (left_file_id, mut chunk)| {
                if org_pointer >= occupied {
                    return Done((org_pointer, acc));
                }
                let (Some(current_file_size), Some(next_space), None) =
                    (chunk.next(), chunk.next(), chunk.next())
                else {
                    panic!()
                };
                info!("chunk: {current_file_size}, {next_space}, {org_pointer}");
                for i in std::cmp::max(0, just_files.len() - 4)..just_files.len() {
                    info!("{:?}", just_files[i]);
                }

                // treat file
                let start_file = org_pointer;
                let end_file = std::cmp::min(org_pointer + (*current_file_size as u64), occupied);
                let current_file_contribution =
                    (start_file..end_file).sum::<u64>() * (left_file_id as u64);


                // treat space
                let mut pointer = org_pointer + (*current_file_size as u64); // should use
                                                                             // difference:
                                                                             // end_file -
                                                                             // start_file

                let mut to_fill = std::cmp::min(*next_space as u64, occupied - pointer) as u8;
                // underflow?
                if to_fill > 0 && pointer >= occupied {
                    to_fill = 0;
                }



                debug!("to_fill = {to_fill} as min of {} and {} = {} - {}", *next_space, occupied - pointer, occupied, pointer);
                let mut current_space_contribution = 0;
                while to_fill > 0 {
                    let end_idx = just_files.len() - 1;
                    let (back_id, back_size) = just_files[just_files.len() - 1];
                    debug!("get something from the last file {back_id}. need {to_fill}, have {back_size}");
                    let getting = if back_size > to_fill {
                        debug!("last file large enough");
                        just_files[end_idx] = (back_id, back_size - to_fill);
                        to_fill
                    } else {
                        let rv = just_files.pop().unwrap().1;
                        debug!("pop last element: {rv}");
                        rv
                    };
                    to_fill -= getting;
                    debug!("still {to_fill} to go for this space");

                    let start_file = pointer;
                    let end_file = std::cmp::min(pointer + (getting as u64), occupied);
                    current_space_contribution +=
                        (start_file..end_file).sum::<u64>() * (back_id as u64);
                    debug!("moving pointer from {pointer} by adding {getting}");
                    pointer += getting as u64;
                }
                debug!("go to next chunk");

                // Can't handle in-file abort
                // assert_eq!(
                //     (org_pointer + (*current_file_size as u64) + (*next_space as u64)),
                //     pointer
                // );
                Continue((
                    pointer,
                    acc + current_file_contribution + current_space_contribution,
                ))
            },
        )
        .into_inner();

    result
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
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(checksum, m)?)?;
    m.add_function(wrap_pyfunction!(defrag, m)?)
}

define_stub_info_gatherer!(stub_info);
