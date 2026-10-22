//! PCMM / CCMM: `Pcmm`, `Ccmm` over the `Gemm` backend seam (cuBLASLt int8 -> int32 via pc-cublas,
//! own kernel if alignment constraints bite).
pub mod gemm; // GemmBackend seam, cuBLASLt TN plans per shape (autotuned at setup)
pub mod pcmm; // PCMM engine (keys, ring switch, digit GEMMs, pack back) + the linear driver over BtsCall
pub mod ccmm; // CCMM engine (keys, ring switch 2^16 -> 2^13, CMT, batched PPMM GEMM, relin) + SinCBoot + leaf services
