//! pc-core: the CUDA-free mathematics of perseus. Everything here builds and tests on any machine.
pub mod word;      // Word trait (u32 / u64 / u16)
pub mod modmul;    // ModMul<W> strategies (Barrett64, Barrett32, Montgomery) + Shoup constants, host impls
pub mod params;    // ChainSpec -> Chain (prime search for composite/mixed chains)
pub mod scale;     // scaling-factor / CRT tables, ElemForEvalMult / ElemForEvalAddOrSub
pub mod encode;    // CKKS encode / decode (FFTSpecial), coefficient rows
pub mod keys;      // secret / error samplers, key material, KSKA seed expansion (host twin)
pub mod galois;    // automorphism exponents, slot permutations
pub mod cheb;      // Chebyshev utilities (long division, degree tables, Paterson-Stockmeyer plans)
pub mod bts_setup; // bootstrap constants (CtS/StC DFT factors, SparseB, EvalMod coefficients)
pub mod pcmm_keys; // PCMM ring-switch / pack keys (pcmm.py make_keys), weight digit planes
pub mod ccmm_keys; // CCMM ring-switch / automorphism / relin keys (ccmm.py make_keys)
pub mod error;     // shared error type
