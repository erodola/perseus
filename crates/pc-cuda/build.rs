// Links the CUDA runtime (dynamic) from CUDA_HOME (default /usr/local/cuda-13.2). The rpath only reaches this
// package's own targets: crates with binaries add their own rpath (see pc-gpu/build.rs).
fn main() {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-13.2".into());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}
