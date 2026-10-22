// Ring-size-generic negacyclic NTT (same order and tables as ntt.cuh) for N = 2^LOGN, LOGN in 10..17.
//  LOGN <= 12: one block per row, the whole row staged in shared memory.
//  LOGN >= 13: N = R x C (R = 2^ceil(LOGN/2)): a column pass (R-point sub-transforms of stride C, the first log R
//              stages) then a row pass (C-point contiguous sub-transforms); the inverse runs them reversed.
// Every pass stages a tile of 4096 elements in shared memory and runs radix-8 rounds in registers (8 elements per
// thread, one barrier per round). Sub-transform g of a pass with P points and stride S uses, at local stage s and
// element k, the twiddle index (G + g) 2^s + (k >> (LOGP - s)) with G = N / (P S): the global m + i of ntt.cuh.
#pragma once
#include "arith.cuh"

namespace pc {
namespace nttr {
constexpr int TILE = 4096;

__host__ __device__ constexpr int cmin(int a, int b) { return a < b ? a : b; }

template <int LOGN>
struct Shape {
    static constexpr bool ONE = LOGN <= 12;
    static constexpr int LOGC = ONE ? 0 : LOGN / 2;
    static constexpr int LOGR = LOGN - LOGC;
    static constexpr int CB_COL = ONE ? 1 : cmin(1 << LOGC, TILE >> LOGR);  // columns per column-pass tile
    static constexpr int CB_ROW = ONE ? 1 : TILE >> LOGC;                   // rows per row-pass tile
};

// one padding word per 32 against shared-memory bank conflicts of strided round accesses
__device__ __forceinline__ int pad(int i) { return i + (i >> 5); }

template <class W, int LOGN, int LOGP, int LOGS, int CB, bool INV, bool LAST>
__global__ void __launch_bounds__((1 << LOGP) * CB / 8) pass(NttArgs T) {
    constexpr int P = 1 << LOGP, S = 1 << LOGS, NT = P * CB / 8, TE = P * CB;
    constexpr int G = 1 << (LOGN - LOGP - LOGS);
    constexpr int NR = (LOGP + 2) / 3;
    __shared__ W sm[TE + TE / 32];
    const int row = blockIdx.y;
    W* a = (W*)T.base + ((size_t)T.roff[row] << LOGN);
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)(INV ? T.iw : T.fw) + ((size_t)pid << LOGN);
    const W* ws = (const W*)(INV ? T.iws : T.fws) + ((size_t)pid << LOGN);
    int g0, col0;
    if constexpr (S > 1) {
        constexpr int TPG = S / CB;
        g0 = blockIdx.x / TPG;
        col0 = (blockIdx.x % TPG) * CB;
    } else {
        g0 = blockIdx.x * CB;
        col0 = 0;
    }
    // tile-linear index i: (k, c) = (i / CB, i % CB) for strided passes, (i % P, i / P) for contiguous ones, so that
    // consecutive i are consecutive addresses either way
    auto gaddr = [&](int i) -> size_t {
        if constexpr (S > 1) return (size_t)g0 * P * S + col0 + (i % CB) + (size_t)S * (i / CB);
        else return (size_t)g0 * P + i;
    };
    auto tidx = [](int k, int c) -> int {
        if constexpr (S > 1) return k * CB + c;
        else return c * P + k;
    };
    for (int i = threadIdx.x; i < TE; i += NT) sm[pad(i)] = a[gaddr(i)];
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < NR; ++rr) {
        const int r = INV ? NR - 1 - rr : rr;
        const int s0 = 3 * r;
        const int e = cmin(3, LOGP - s0);
        const int ER = 1 << e;
        const int lh = LOGP - s0 - e;  // log2 of the smallest butterfly distance of the round
        const int hmin = 1 << lh;
        const int sets = P / ER;
        for (int it = threadIdx.x; it < sets * CB; it += NT) {
            int set, c;
            if constexpr (S > 1) {
                c = it % CB;
                set = it / CB;
            } else {
                set = it % sets;
                c = it / sets;
            }
            const int base = ((set >> lh) << (LOGP - s0)) + (set & (hmin - 1));
            const int gc = S > 1 ? g0 : g0 + c;
            W x[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < ER) x[j] = sm[pad(tidx(base + j * hmin, c))];
            if (!INV) {
#pragma unroll
                for (int t = 0; t < 3; ++t) {
                    if (t < e) {
                        const int s = s0 + t, hl = ER >> (t + 1);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (j < ER && (j & hl) == 0) {
                                const int ti = ((G + gc) << s) + ((base + j * hmin) >> (LOGP - s));
                                ntt::ct_bf<W>(x[j], x[j + hl], __ldg(w + ti), __ldg(ws + ti), p);
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (int t = 2; t >= 0; --t) {
                    if (t < e) {
                        const int s = s0 + t, hl = ER >> (t + 1);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (j < ER && (j & hl) == 0) {
                                const int ti = ((G + gc) << s) + ((base + j * hmin) >> (LOGP - s));
                                ntt::gs_bf<W>(x[j], x[j + hl], __ldg(w + ti), __ldg(ws + ti), p);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < ER) sm[pad(tidx(base + j * hmin, c))] = x[j];
        }
        __syncthreads();
    }
    if constexpr (LAST) {
        const W ni = ((const W*)T.ninv)[pid], nis = ((const W*)T.ninvs)[pid];
        for (int i = threadIdx.x; i < TE; i += NT) a[gaddr(i)] = mul_shoup<W>(sm[pad(i)], ni, nis, p);
    } else {
        for (int i = threadIdx.x; i < TE; i += NT) a[gaddr(i)] = sm[pad(i)];
    }
}

template <class W, int LOGN, bool INV>
void run_chunk(const NttArgs& b, int m, cudaStream_t s) {
    using Sh = Shape<LOGN>;
    if constexpr (Sh::ONE) {
        pass<W, LOGN, LOGN, 0, 1, INV, INV><<<dim3(1, m), (1 << LOGN) / 8, 0, s>>>(b);
    } else {
        constexpr int R = 1 << Sh::LOGR, C = 1 << Sh::LOGC;
        constexpr int CBc = Sh::CB_COL, CBr = Sh::CB_ROW;
        if constexpr (!INV) {
            pass<W, LOGN, Sh::LOGR, Sh::LOGC, CBc, false, false><<<dim3(C / CBc, m), R * CBc / 8, 0, s>>>(b);
            pass<W, LOGN, Sh::LOGC, 0, CBr, false, false><<<dim3(R / CBr, m), C * CBr / 8, 0, s>>>(b);
        } else {
            pass<W, LOGN, Sh::LOGC, 0, CBr, true, false><<<dim3(R / CBr, m), C * CBr / 8, 0, s>>>(b);
            pass<W, LOGN, Sh::LOGR, Sh::LOGC, CBc, true, true><<<dim3(C / CBc, m), R * CBc / 8, 0, s>>>(b);
        }
    }
}

template <class W, int LOGN, bool INV>
int launch(const NttArgs* a, cudaStream_t s) {
    NttArgs b = *a;
    const int ch = a->chunk > 0 ? a->chunk : a->nrows;
    for (int i = 0; i < a->nrows; i += ch) {
        const int m = min(ch, a->nrows - i);
        b.roff = a->roff + i;
        b.rprime = a->rprime + i;
        run_chunk<W, LOGN, INV>(b, m, s);
    }
    return (int)cudaGetLastError();
}
template <class W, int LOGN>
int fwd_g(const NttArgs* a, cudaStream_t s) {
    return launch<W, LOGN, false>(a, s);
}
template <class W, int LOGN>
int inv_g(const NttArgs* a, cudaStream_t s) {
    return launch<W, LOGN, true>(a, s);
}
// the fastest transform of the ring size: the register-pass kernels of ntt.cuh at N = 2^16, else the generic one
template <class W, int LOGN>
int fwd(const NttArgs* a, cudaStream_t s) {
    if constexpr (LOGN == 16) return ntt::fwd<W>(a, s);
    else return fwd_g<W, LOGN>(a, s);
}
template <class W, int LOGN>
int inv(const NttArgs* a, cudaStream_t s) {
    if constexpr (LOGN == 16) return ntt::inv<W>(a, s);
    else return inv_g<W, LOGN>(a, s);
}
}  // namespace nttr
}  // namespace pc
