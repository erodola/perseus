// Links libcublasLt from the CUDA 13.2 toolkit, with an rpath so the binaries run
// without LD_LIBRARY_PATH. No cudart: streams and buffers cross this crate as raw pointers.
fn main() {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-13.2".into());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}
