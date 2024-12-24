#![feature(test)]

extern crate test;

pub mod binding;
pub mod read_mod;

use crate::binding::wrap;

#[cfg(test)]
mod tests {
    use super::*;
    use test::{black_box, Bencher};

    #[bench]
    fn bench_wrap(b: &mut Bencher) {
        let machines = Day13::parse(Day13::deser("../input_mod.txt").unwrap()).unwrap();

        let mut data = machines_to_raw_data(&machines);

        b.iter(|| {
            // Inner closure, the actual test
            black_box(unsafe {
                wrap(
                    data.Ax.as_mut_ptr(),
                    data.Bx.as_mut_ptr(),
                    data.Ay.as_mut_ptr(),
                    data.By.as_mut_ptr(),
                    data.Tx.as_mut_ptr(),
                    data.Ty.as_mut_ptr(),
                    data.Ax.len() as i32,
                )
            });
        })
    }
}

impl boiler_plate::Day for Day13 {
    type Desered = Vec<read_mod::Machine>;
    fn process(machines: Self::Desered) -> anyhow::Result<()> {
        let mut data = machines_to_raw_data(&machines);

        // Button A: X+94, Y+34
        // Button B: X+22, Y+67
        // Prize: X=8400, Y=5400
        //
        // Button A: X+26, Y+66
        // Button B: X+67, Y+21
        // Prize: X=12748, Y=12176
        //
        // Button A: X+17, Y+86
        // Button B: X+84, Y+37
        // Prize: X=7870, Y=6450
        //
        // Button A: X+69, Y+23
        // Button B: X+27, Y+71
        // Prize: X=18641, Y=10279
        // let mut Ax = vec![94, 1];
        // let mut Bx = vec![22, 1];
        // let mut Ay = vec![34, 1];
        // let mut By = vec![67, 1];
        // let mut Tx = vec![8400, 1000];
        // let mut Ty = vec![5400, 1000];

        println!("okay, let's look at {} machines", machines.len());
        let r = unsafe {
            wrap(
                data.Ax.as_mut_ptr(),
                data.Bx.as_mut_ptr(),
                data.Ay.as_mut_ptr(),
                data.By.as_mut_ptr(),
                data.Tx.as_mut_ptr(),
                data.Ty.as_mut_ptr(),
                data.Ax.len() as i32,
            )
        };
        println!("{r}");
        Ok(())
    }
}

struct Day13 {}

fn main() -> std::process::ExitCode {
    boiler_plate::main_wrap::<Day13>()
}

struct RawData {
    Ax: Vec<i32>,
    Bx: Vec<i32>,
    Ay: Vec<i32>,
    By: Vec<i32>,
    Tx: Vec<i32>,
    Ty: Vec<i32>,
}

fn machines_to_raw_data(input: &Vec<read_mod::Machine>) -> RawData {
    let mut Ax = Vec::with_capacity(input.len());
    let mut Bx = Vec::with_capacity(input.len());
    let mut Ay = Vec::with_capacity(input.len());
    let mut By = Vec::with_capacity(input.len());
    let mut Tx = Vec::with_capacity(input.len());
    let mut Ty = Vec::with_capacity(input.len());
    for machine in input {
        Ax.push(machine.Ax);
        Bx.push(machine.Bx);
        Ay.push(machine.Ay);
        By.push(machine.By);
        Tx.push(machine.Tx);
        Ty.push(machine.Ty);
    }
    assert_eq!(Ax.len(), Bx.len());
    assert_eq!(Ax.len(), Ay.len());
    assert_eq!(Ax.len(), By.len());
    assert_eq!(Ax.len(), Tx.len());
    assert_eq!(Ax.len(), Ty.len());
    RawData {
        Ax,
        Bx,
        Ay,
        By,
        Tx,
        Ty,
    }
}
