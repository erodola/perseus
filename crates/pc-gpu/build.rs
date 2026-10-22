// rpath for this package's binaries (pc-cuda/pc-kernels link cudart, but their rpath only reaches their own targets).
fn main() {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-13.2".into());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}
