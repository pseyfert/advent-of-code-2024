fn main() {
    println!("cargo:rustc-link-search=/home/pseyfert/coding/advent-of-code-2024/20/cuda_rust");
    println!("cargo:rustc-link-search=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/12.2/targets/x86_64-linux/lib/");
    println!("cargo:rustc-link-lib=static=mycuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudadevrt");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo::rerun-if-changed=libmycuda.a");

    let bindings = bindgen::Builder::default()
        .header("c_interface.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap();

    bindings
        // TODO: use some cargo to set this path correctly
        .write_to_file("/home/pseyfert/coding/advent-of-code-2024/20/cuda_rust/src/binding.rs")
        .unwrap();
}
