pub mod binding;
use crate::binding::wrap;

fn main() {
    let mut Ax = vec![4];
    let mut Bx = vec![4];
    let mut Ay = vec![4];
    let mut By = vec![4];
    let mut Tx = vec![4];
    let mut Ty = vec![4];
    let r = unsafe {
        wrap(
            Ax.as_mut_ptr(),
            Bx.as_mut_ptr(),
            Ay.as_mut_ptr(),
            By.as_mut_ptr(),
            Tx.as_mut_ptr(),
            Ty.as_mut_ptr(),
            4,
        )
    };
    println!("{r}");
}
