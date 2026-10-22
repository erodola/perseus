// SPRU extraction (FIDESlib Spru.cu spruExtract_): the LWE columns of the bottom composite modulus q0 = p0 p1 turned
// into the bootstrapping plaintext slots. Semantics in args.cuh SpruExtractArgs.
#pragma once
#include "arith.cuh"

namespace pc {
namespace spru {
constexpr int BX = 256;

template <class W>
__global__ void extract_k(SpruExtractArgs A) {
    const int tid = blockIdx.x * BX + threadIdx.x;
    const int c = blockIdx.y;
    const int n = A.n, h = A.h, B = A.B;
    if (tid >= N * n) return;
    const int i = tid / n, a = tid % n;
    const int ka = a * (N / n);
    const W* c0 = (const W*)A.c0 + (size_t)c * A.ct_stride;
    const W* c1 = (const W*)A.c1 + (size_t)c * A.ct_stride;
    const uint64_t p0 = A.p0, p1 = A.p1, q0 = p0 * p1;
    auto crt = [&](const W* r, int k) -> uint64_t {
        const uint64_t x0 = (uint64_t)r[k], x1 = (uint64_t)r[N + k];
        const uint64_t t = ((x1 + p1 - x0 % p1) % p1) * A.inv % p1;
        return x0 + p0 * t;
    };
    // coefficient ka of c1 * X^-i ... the negacyclic column of key index i
    uint64_t v;
    if (i <= ka) {
        v = crt(c1, ka - i);
    } else {
        v = crt(c1, N + ka - i);
        v = v ? q0 - v : 0;
    }
    if (i == 0) {
        v += crt(c0, ka);
        if (v >= q0) v -= q0;
    }
    const double ang = 2.0 * (double)v / (double)q0;  // units of pi
    double sn, cs;
    sincospi(ang, &sn, &cs);
    const int sub = B / (2 * n), b = i / B, rem = i % B, u = rem / sub, k = rem % sub;
    const int slot = k * h * n + b * n + a;
    // straight into the FFT input of encoding u: slot j sits at zeta^(5^j) -> index (t-1)/2, its conjugate at
    // (2N-t-1)/2, t = 5^j mod 2N (every index of the N-point input is written exactly once)
    const uint32_t m2n = 2u * N - 1;
    uint32_t t = 1, bb = 5, e = (uint32_t)slot;
    while (e) {
        if (e & 1u) t = (t * bb) & m2n;
        bb = (bb * bb) & m2n;
        e >>= 1;
    }
    double2* o = (double2*)A.out + ((size_t)c * 2 * n + u) * N;
    o[(t - 1) / 2] = make_double2(A.delta * cs, A.delta * sn);
    o[(2u * N - t - 1) / 2] = make_double2(A.delta * cs, -A.delta * sn);
}

// one radix-2 Stockham pass (forward, exp(-2 pi i ...)) over `batch` N-point transforms; twiddles from a table
// tw[m] = exp(-i pi m / (N/2)), m < N/2 (no double-precision trigonometry per element)
__global__ void fft_pass_k(SpruFftArgs A) {
    const int j = blockIdx.x * BX + threadIdx.x;
    const int half = N / 2;
    if (j >= half) return;
    const int e = blockIdx.y;
    const double2* I = (const double2*)A.in + (size_t)e * N;
    double2* O = (double2*)A.out + (size_t)e * N;
    const int Ns = A.ns;
    const double2 v0 = I[j], v1 = I[j + half];
    const int k = j % Ns;
    const double2 w = ((const double2*)A.tw)[k * (half / Ns)];
    const double2 t = make_double2(v1.x * w.x - v1.y * w.y, v1.x * w.y + v1.y * w.x);
    const int idx = (j / Ns) * Ns * 2 + k;
    O[idx] = make_double2(v0.x + t.x, v0.y + t.y);
    O[idx + Ns] = make_double2(v0.x - t.x, v0.y - t.y);
}

// m_k = round(Re(Y_k zeta^-k / N) * scale) reduced into limbs 0..nl of encoding e (coefficient form);
// zt[k] = exp(-i pi k / N)
template <class W>
__global__ void scatter_k(SpruScatterArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    if (k >= N) return;
    const int e = blockIdx.y;
    const double2 y = ((const double2*)A.y)[(size_t)e * N + k];
    const double2 z = ((const double2*)A.zt)[k];
    const double re = (y.x * z.x - y.y * z.y) / (double)N;
    const long long m = llrint(re * A.scale);
    const uint64_t am = m < 0 ? (uint64_t)(-m) : (uint64_t)m;
    W* out = (W*)A.out + (size_t)e * A.out_stride;
    for (int l = 0; l < A.nl; ++l) {
        const uint64_t p = A.pr[l].p;
        uint64_t r = am % p;
        if (m < 0 && r) r = p - r;
        out[(size_t)l * N + k] = (W)r;
    }
}

int fft_pass(const SpruFftArgs* a, cudaStream_t s) {
    if (a->ns <= 0 || a->ns > N / 2 || a->batch <= 0) return -1;
    fft_pass_k<<<dim3((N / 2 + BX - 1) / BX, a->batch), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}

template <class W>
int scatter(const SpruScatterArgs* a, cudaStream_t s) {
    if (a->nl <= 0 || a->batch <= 0) return -1;
    scatter_k<W><<<dim3((N + BX - 1) / BX, a->batch), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}

template <class W>
int extract(const SpruExtractArgs* a, cudaStream_t s) {
    if (a->n <= 0 || a->h <= 0 || a->B * a->h != N || (a->B % (2 * a->n)) != 0 || a->k <= 0) return -1;
    dim3 grid((N * a->n + BX - 1) / BX, a->k);
    extract_k<W><<<grid, BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace spru
}  // namespace pc
