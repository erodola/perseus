//! The `Gemm` backend seam of the matmuls: exact int8 x int8 -> int32 products (digit planes) on the compute lane.
//! Today cuBLASLt with the configuration validated exact on CUDA 13.2 (`CUBLAS_COMPUTE_32I`, column-major TN: both operands
//! K-contiguous, 256-byte aligned operands, heuristic candidates timed once at setup); an own kernel is the
//! alternative if an alignment constraint ever bites. Contract (gate `pc-check pcmm`, GEMM rows): every output
//! element equals the exact integer sum while `pairs * k * 2^14 < 2^31` (`pc_cublas::exact_k_limit`).
use crate::error::{shape, Result};
use pc_cuda::blas::lt::{self, GemmI8I32, GemmSpec, Op, Order, Prefs, Workspace};
use pc_cuda::{Compute, DBuf, Event, Stream};

/// Which implementation runs the int8 GEMMs (seam enum).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GemmBackend {
    /// cuBLASLt int8 -> int32 (pc-cublas).
    #[default]
    CublasLt,
}

fn blas(e: lt::Error) -> crate::Error {
    crate::Error::Cuda(pc_cuda::Error::Blas(e.to_string()))
}

/// D (m x n, col-major, ld m) (+)= A^T B with A stored k x m and B stored k x n (both col-major, ld k): the PCMM's
/// `G_e (2 cols x rows) += digits(X)^T W_u^T` with the digit planes written K-contiguous (col-TN).
pub fn tn_spec(m: u64, n: u64, k: u64) -> GemmSpec {
    let mut s = GemmSpec::packed(m, n, k, Op::T, Op::N, Order::Col, 1);
    s.lda = k as i64;
    s.ldb = k as i64;
    s.ldc = m as i64;
    s
}

/// `batch` strided TN products back to back (packed strides m k, k n, m n): the CCMM's PPMM over the 64 points of Y.
pub fn tn_spec_batched(m: u64, n: u64, k: u64, batch: i32) -> GemmSpec {
    let mut s = tn_spec(m, n, k);
    s.batch = batch;
    s.stride_a = (m * k) as i64;
    s.stride_b = (k * n) as i64;
    s.stride_c = (m * n) as i64;
    s
}

/// One planned GEMM shape.
pub struct I8Gemm<'lt> {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    plan: GemmI8I32<'lt>,
    /// Device ms of each heuristic candidate when tuned (empty otherwise).
    pub tuned_ms: Vec<f32>,
}

impl<'lt> I8Gemm<'lt> {
    pub fn new(lt_: &'lt lt::Lt, m: u64, n: u64, k: u64, max_algos: usize, max_workspace: usize) -> Result<Self> {
        Self::with_spec(lt_, tn_spec(m, n, k), max_algos, max_workspace)
    }
    /// A plan for any spec (`tn_spec_batched` for the CCMM).
    pub fn with_spec(lt_: &'lt lt::Lt, spec: GemmSpec, max_algos: usize, max_workspace: usize) -> Result<Self> {
        let prefs = Prefs { max_workspace, max_algos: max_algos.max(1), ..Prefs::default() };
        let (m, n, k) = (spec.m, spec.n, spec.k);
        let plan = GemmI8I32::new(lt_, spec, prefs).map_err(blas)?;
        Ok(I8Gemm { m, n, k, plan, tuned_ms: Vec::new() })
    }
    pub fn algos(&self) -> usize {
        self.plan.algos().len()
    }
    pub fn label(&self) -> String {
        self.plan.algo().label()
    }
    /// Bytes of A (stored k x m), B (k x n), C (m x n, int32) this plan reads/writes.
    pub fn lens(&self) -> (usize, usize, usize) {
        let s = self.plan.spec();
        (s.len_a(), s.len_b(), 4 * s.len_c())
    }

    /// Enqueue `c (+)= a^T b` on the compute lane.
    ///
    /// # Safety
    /// `a`, `b`, `c` point into live device buffers of at least `lens()` bytes, owned by `s` for the GEMM's span;
    /// `c` aliases neither input.
    pub unsafe fn run(&self, s: &Stream<Compute>, a: *const i8, b: *const i8, c: *mut i32, accumulate: bool, ws: &mut DBuf<'_, Compute, u8>) -> Result<()> {
        let w = Workspace { ptr: ws.vptr_mut(), bytes: ws.len() };
        self.plan.run(s.raw(), a, b, c, accumulate, w).map_err(blas)
    }

    /// Time every heuristic candidate once on the given operands (setup; contents irrelevant) and keep the fastest.
    ///
    /// # Safety
    /// As `run`.
    pub unsafe fn tune(&mut self, s: &Stream<Compute>, ev: (&Event, &Event), a: *const i8, b: *const i8, c: *mut i32, ws: &mut DBuf<'_, Compute, u8>, reps: usize) -> Result<()> {
        let n = self.plan.algos().len();
        if n < 2 {
            return Ok(());
        }
        let mut ms = Vec::with_capacity(n);
        for i in 0..n {
            self.plan.select(i).map_err(blas)?;
            if self.plan.algo().workspace > ws.len() {
                ms.push(f32::INFINITY);
                continue;
            }
            self.run(s, a, b, c, false, ws)?; // warm-up
            ev.0.record(s)?;
            for _ in 0..reps.max(1) {
                self.run(s, a, b, c, false, ws)?;
            }
            ev.1.record(s)?;
            ev.1.sync()?;
            ms.push(ev.0.elapsed_ms(ev.1)? / reps.max(1) as f32);
        }
        let best = ms.iter().enumerate().min_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i).unwrap_or(0);
        self.plan.select(best).map_err(blas)?;
        self.tuned_ms = ms;
        Ok(())
    }
}

/// Plans by (m, n, k), built at setup.
#[derive(Default)]
pub struct GemmPlans<'lt> {
    pub plans: Vec<I8Gemm<'lt>>,
}

impl<'lt> GemmPlans<'lt> {
    pub fn get(&self, m: u64, n: u64, k: u64) -> Result<&I8Gemm<'lt>> {
        self.plans
            .iter()
            .find(|g| (g.m, g.n, g.k) == (m, n, k))
            .ok_or_else(|| shape("pcmm gemm", format!("no GEMM plan for m {m} n {n} k {k} (PcmmCfg::shapes)")))
    }
    pub fn ensure(&mut self, lt_: &'lt lt::Lt, m: u64, n: u64, k: u64, max_algos: usize, max_workspace: usize) -> Result<()> {
        if self.get(m, n, k).is_err() {
            self.plans.push(I8Gemm::new(lt_, m, n, k, max_algos, max_workspace)?);
        }
        Ok(())
    }
}
