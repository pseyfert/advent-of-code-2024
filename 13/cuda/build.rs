fn main() {
    println!("cargo:rustc-link-search=/home/pseyfert/coding/advent-of-code-2024/13/cuda");
    println!("cargo:rustc-link-search=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.6/targets/x86_64-linux/lib/");
    println!("cargo:rustc-link-lib=static=mycuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudadevrt");

    let bindings = bindgen::Builder::default()
        .header("c_interface.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap();

    bindings
        .write_to_file("/home/pseyfert/coding/advent-of-code-2024/13/cuda/src/binding.rs")
        .unwrap();
}
