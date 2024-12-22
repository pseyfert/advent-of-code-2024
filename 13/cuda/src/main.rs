pub mod binding;
pub mod read_mod;

use crate::binding::wrap;
use log::error;

fn main_fn() -> anyhow::Result<()> {
    let input = aoc_cli::setup_and_input()?;

    let machines: Vec<read_mod::Machine> =
        serde_linewise::from_str(&std::fs::read_to_string(input)?)?;

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
    let mut Ax = vec![];
    let mut Bx = vec![];
    let mut Ay = vec![];
    let mut By = vec![];
    let mut Tx = vec![];
    let mut Ty = vec![];
    for machine in machines {
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
    println!("okay, let's look at {} machines", Ax.len());
    let r = unsafe {
        wrap(
            Ax.as_mut_ptr(),
            Bx.as_mut_ptr(),
            Ay.as_mut_ptr(),
            By.as_mut_ptr(),
            Tx.as_mut_ptr(),
            Ty.as_mut_ptr(),
            Ax.len() as i32,
        )
    };
    println!("{r}");
    Ok(())
}

fn main() -> std::process::ExitCode {
    match main_fn() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // Does this work in case of failure to set up logging?
            // You know what? I don't care.
            error!("Program failed: {e:?}");
            std::process::ExitCode::from(1)
        }
    }
}
