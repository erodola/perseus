//! Raw cuBLASLt declarations (CUDA 13.2 `cublasLt.h`): only the entry points the int8 GEMM path uses.
//! Every unsafe call of the crate goes through these; the safe layer is in `lib.rs`.
#![allow(non_camel_case_types, non_upper_case_globals)]
use std::ffi::{c_char, c_int, c_void};

pub type cublasStatus_t = c_int;
pub type cudaStream_t = *mut c_void;
pub type cudaDataType_t = c_int;
pub type cublasComputeType_t = c_int;

#[repr(C)]
pub struct cublasLtContext {
    _p: [u8; 0],
}
pub type cublasLtHandle_t = *mut cublasLtContext;

// The descriptors are pointers to opaque structs (`cublasLt*Opaque_t`) owned by the library.
pub type cublasLtMatmulDesc_t = *mut c_void;
pub type cublasLtMatrixLayout_t = *mut c_void;
pub type cublasLtMatmulPreference_t = *mut c_void;

/// `cublasLtMatmulAlgo_t`: semi-opaque, trivially copyable, valid for the library version that produced it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct cublasLtMatmulAlgo_t {
    pub data: [u64; 8],
}

/// `cublasLtMatmulHeuristicResult_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct cublasLtMatmulHeuristicResult_t {
    pub algo: cublasLtMatmulAlgo_t,
    pub workspace_size: usize,
    pub state: cublasStatus_t,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

// sizes as laid out by the C header (x86_64)
const _: () = assert!(std::mem::size_of::<cublasLtMatmulAlgo_t>() == 64);
const _: () = assert!(std::mem::size_of::<cublasLtMatmulHeuristicResult_t>() == 96);

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

// library_types.h
pub const CUDA_R_8I: cudaDataType_t = 3;
pub const CUDA_R_32I: cudaDataType_t = 10;
pub const CUDA_R_32F: cudaDataType_t = 0;

// cublas_api.h
pub const CUBLAS_COMPUTE_32I: cublasComputeType_t = 72;
pub const CUBLAS_COMPUTE_32I_PEDANTIC: cublasComputeType_t = 73;
pub const CUBLAS_OP_N: i32 = 0;
pub const CUBLAS_OP_T: i32 = 1;

// cublasLtOrder_t
pub const CUBLASLT_ORDER_COL: i32 = 0;
pub const CUBLASLT_ORDER_ROW: i32 = 1;

// cublasLtMatrixLayoutAttribute_t
pub const CUBLASLT_MATRIX_LAYOUT_ORDER: c_int = 1;
pub const CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: c_int = 5;
pub const CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: c_int = 6;

// cublasLtMatmulDescAttributes_t
pub const CUBLASLT_MATMUL_DESC_SCALE_TYPE: c_int = 1;
pub const CUBLASLT_MATMUL_DESC_TRANSA: c_int = 3;
pub const CUBLASLT_MATMUL_DESC_TRANSB: c_int = 4;

// cublasLtMatmulPreferenceAttributes_t
pub const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: c_int = 1;
pub const CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES: c_int = 5;
pub const CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES: c_int = 6;
pub const CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES: c_int = 7;
pub const CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES: c_int = 8;

// cublasLtMatmulAlgoConfigAttributes_t
pub const CUBLASLT_ALGO_CONFIG_ID: c_int = 0;
pub const CUBLASLT_ALGO_CONFIG_TILE_ID: c_int = 1;
pub const CUBLASLT_ALGO_CONFIG_SPLITK_NUM: c_int = 2;
pub const CUBLASLT_ALGO_CONFIG_STAGES_ID: c_int = 6;

extern "C" {
    pub fn cublasLtCreate(h: *mut cublasLtHandle_t) -> cublasStatus_t;
    pub fn cublasLtDestroy(h: cublasLtHandle_t) -> cublasStatus_t;
    pub fn cublasLtGetVersion() -> usize;
    pub fn cublasLtGetStatusString(s: cublasStatus_t) -> *const c_char;

    pub fn cublasLtMatmulDescCreate(
        d: *mut cublasLtMatmulDesc_t,
        compute: cublasComputeType_t,
        scale: cudaDataType_t,
    ) -> cublasStatus_t;
    pub fn cublasLtMatmulDescDestroy(d: cublasLtMatmulDesc_t) -> cublasStatus_t;
    pub fn cublasLtMatmulDescSetAttribute(
        d: cublasLtMatmulDesc_t,
        attr: c_int,
        buf: *const c_void,
        size: usize,
    ) -> cublasStatus_t;

    pub fn cublasLtMatrixLayoutCreate(
        l: *mut cublasLtMatrixLayout_t,
        ty: cudaDataType_t,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> cublasStatus_t;
    pub fn cublasLtMatrixLayoutDestroy(l: cublasLtMatrixLayout_t) -> cublasStatus_t;
    pub fn cublasLtMatrixLayoutSetAttribute(
        l: cublasLtMatrixLayout_t,
        attr: c_int,
        buf: *const c_void,
        size: usize,
    ) -> cublasStatus_t;

    pub fn cublasLtMatmulPreferenceCreate(p: *mut cublasLtMatmulPreference_t) -> cublasStatus_t;
    pub fn cublasLtMatmulPreferenceDestroy(p: cublasLtMatmulPreference_t) -> cublasStatus_t;
    pub fn cublasLtMatmulPreferenceSetAttribute(
        p: cublasLtMatmulPreference_t,
        attr: c_int,
        buf: *const c_void,
        size: usize,
    ) -> cublasStatus_t;

    pub fn cublasLtMatmulAlgoGetHeuristic(
        h: cublasLtHandle_t,
        op: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: c_int,
        results: *mut cublasLtMatmulHeuristicResult_t,
        returned: *mut c_int,
    ) -> cublasStatus_t;
    pub fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const cublasLtMatmulAlgo_t,
        attr: c_int,
        buf: *mut c_void,
        size: usize,
        written: *mut usize,
    ) -> cublasStatus_t;

    #[allow(clippy::too_many_arguments)]
    pub fn cublasLtMatmul(
        h: cublasLtHandle_t,
        op: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        adesc: cublasLtMatrixLayout_t,
        b: *const c_void,
        bdesc: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        cdesc: cublasLtMatrixLayout_t,
        d: *mut c_void,
        ddesc: cublasLtMatrixLayout_t,
        algo: *const cublasLtMatmulAlgo_t,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
}
