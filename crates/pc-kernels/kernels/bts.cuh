// Bootstrap raise kernels, generic over (W, MM); semantics in args.cuh (ModRaiseArgs, SmallIntArgs).
//  - mod_raise: FIDESlib compositeModRaise_ (each Garner term centred on its own prime, then re-reduced; the sum may
//    leave (-Q0/2, Q0/2], which is why the small-integer scaling that follows reconstructs from 3 limbs).
//  - small_int: FIDESlib smallIntDivRound_ (Garner, centre, fixed-point multiply by R and >> 192 with round half up,
//    reduce mod every output prime). Only the integer q must match FIDESlib; the residues are canonical, so the
//    reduction here uses the double-word Barrett instead of `%`. u32 words only (the launcher refuses others).
#pragma once
#include "arith.cuh"

namespace pc {
namespace bts {
constexpr int BX = 128;

template <class W, class MM>
__global__ void __launch_bounds__(BX) mod_raise(ModRaiseArgs A) {
    using Wd = WideOf<W>;
    const int t = A.r0 + blockIdx.y;
    const int c = blockIdx.z / A.nparts, part = blockIdx.z % A.nparts;
    const int i = blockIdx.x * BX + threadIdx.x;
    W* x = (W*)A.x + c * A.ct_stride + part * A.part_stride;
    const PrimeConst& qt = A.pr[t];
    const W pt = (W)qt.p;
    W acc = 0;
    for (int k = 0; k < A.d; ++k) {
        const PrimeConst& qk = A.pr[k];
        const W pk = (W)qk.p;
        W y = MM::mul(x[(size_t)k * N + i], (W)A.qhatinv[k], qk);
        // SwitchModulus(k -> t): centred lift of y, reduced mod q_t
        Wd v = y;
        if (y > (pk >> 1)) v += (Wd)pt - (Wd)(pk % pt);
        const W r = (W)(v % pt);
        acc = add<W>(acc, MM::mul(r, (W)A.qhat[k][t], qt), pt);
    }
    x[(size_t)t * N + i] = acc;
}

template <class W, class MM>
__global__ void __launch_bounds__(BX) small_int(SmallIntArgs A) {
    const SmallIntTab* tab = A.tab;
    const int part = blockIdx.y, c = blockIdx.z;
    const int idx = blockIdx.x * BX + threadIdx.x;
    const int k = tab->k, nw = tab->nw;
    const W* src = (const W*)A.src + c * A.src_ct_stride + part * A.src_part_stride;
    uint32_t v[PC_SI_MAXK];
    for (int i = 0; i < k; ++i) {
        const uint32_t qi = tab->q[i];
        uint64_t t = (uint64_t)src[(size_t)i * N + idx] % qi;
        for (int j = 0; j < i; ++j) {
            const uint64_t vj = v[j] % qi;
            t = (t + qi - vj) % qi;
            t = (t * tab->qinv[i][j]) % qi;
        }
        v[i] = (uint32_t)t;
    }
    uint32_t x[PC_SI_MAXW + PC_SI_MAXR];
    for (int i = 0; i < PC_SI_MAXW + PC_SI_MAXR; ++i) x[i] = 0;
    int nx = 1;
    x[0] = v[k - 1];
    for (int i = k - 2; i >= 0; --i) {
        uint64_t carry = v[i];
        for (int w = 0; w < nx; ++w) {
            const uint64_t t = (uint64_t)x[w] * tab->q[i] + carry;
            x[w] = (uint32_t)t;
            carry = t >> 32;
        }
        if (carry) x[nx++] = (uint32_t)carry;
    }
    bool neg = false;
    {
        bool gt = false;
        for (int w = nw - 1; w >= 0; --w)
            if (x[w] != tab->Qkh[w]) {
                gt = x[w] > tab->Qkh[w];
                break;
            }
        if (gt) {
            neg = true;
            int64_t borrow = 0;
            for (int w = 0; w < nw; ++w) {
                int64_t t = (int64_t)tab->Qk[w] - (int64_t)x[w] - borrow;
                borrow = t < 0;
                if (t < 0) t += (int64_t)1 << 32;
                x[w] = (uint32_t)t;
            }
        }
    }
    int nq = nw;
    if (tab->divide) {
        uint32_t prod[PC_SI_MAXW + PC_SI_MAXR];
        for (int i = 0; i < PC_SI_MAXW + PC_SI_MAXR; ++i) prod[i] = 0;
        for (int i = 0; i < nw; ++i) {
            uint64_t carry = 0;
            for (int j = 0; j < tab->nr; ++j) {
                const uint64_t t = (uint64_t)x[i] * tab->R[j] + prod[i + j] + carry;
                prod[i + j] = (uint32_t)t;
                carry = t >> 32;
            }
            int p_ = i + tab->nr;
            while (carry && p_ < PC_SI_MAXW + PC_SI_MAXR) {
                const uint64_t t = (uint64_t)prod[p_] + carry;
                prod[p_] = (uint32_t)t;
                carry = t >> 32;
                ++p_;
            }
        }
        {  // + 2^191 (round half up), then drop 6 words
            uint64_t carry = 1ull << 31;
            for (int i = 5; i < PC_SI_MAXW + PC_SI_MAXR && carry; ++i) {
                const uint64_t t = (uint64_t)prod[i] + carry;
                prod[i] = (uint32_t)t;
                carry = t >> 32;
            }
        }
        constexpr int DW = PC_SI_SHIFT / 32;
        nq = nw + tab->nr - DW;
        if (nq < 1) nq = 1;
        for (int i = 0; i < nq; ++i) x[i] = prod[i + DW];
    }
    W* out = (W*)A.out + c * A.ct_stride + part * A.part_stride;
    for (int o = 0; o < tab->nout; ++o) {
        const int row = tab->orow[o];
        const PrimeConst& q = A.pr[row];
        // sum_w x_w (2^32)^w: terms < 2^32 * 2^28, at most 16 of them -> < 2^64
        uint64_t acc = 0;
        for (int w = 0; w < nq; ++w) acc += (uint64_t)x[w] * tab->pow32[o][w];
        W r = MM::reduce((WideOf<W>)acc, q);
        if (neg && r) r = (W)q.p - r;
        out[(size_t)row * N + idx] = r;
    }
}

// See args.cuh SmallInt1Args.
template <class W, class MM>
__global__ void __launch_bounds__(BX) small_int1(SmallInt1Args A) {
    const int part = blockIdx.y, c = blockIdx.z;
    const int idx = blockIdx.x * BX + threadIdx.x;
    W* x = (W*)A.x + c * A.ct_stride + part * A.part_stride;
    const uint64_t q0 = A.pr[0].p;
    const uint64_t v = (uint64_t)x[idx];
    const bool neg = v > (q0 >> 1);
    const uint64_t mag = neg ? q0 - v : v;
    unsigned __int128 prod = (unsigned __int128)mag * A.mc;
    prod += (unsigned __int128)1 << (A.sh - 1);
    const uint64_t q = (uint64_t)(prod >> A.sh);
    for (int t = 0; t < A.nrows; ++t) {
        const uint64_t p = A.pr[t].p;
        const uint64_t r = q % p;
        x[(size_t)t * N + idx] = (W)((neg && r) ? p - r : r);
    }
}

template <class W, class MM>
int small_int1_l(const SmallInt1Args* a, cudaStream_t s) {
    if (a->sh < 1 || a->sh > 127 || a->nrows < 1 || a->nrows > 64 || a->K < 1 || a->nparts < 1 || a->nparts > 2) return -1;
    small_int1<W, MM><<<dim3(N / BX, a->nparts, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}

template <class W, class MM>
int mod_raise_l(const ModRaiseArgs* a, cudaStream_t s) {
    if (a->d < 1 || a->d > 4 || a->r0 < a->d || a->r1 > 64 || a->r1 <= a->r0 || a->nparts < 1 || a->nparts > 2) return -1;
    mod_raise<W, MM><<<dim3(N / BX, a->r1 - a->r0, a->K * a->nparts), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}

template <class W, class MM>
int small_int_l(const SmallIntArgs* a, cudaStream_t s) {
    if (sizeof(W) != 4) return -2;
    if (a->nparts < 1 || a->nparts > 2 || a->K < 1) return -1;
    small_int<W, MM><<<dim3(N / BX, a->nparts, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace bts
}  // namespace pc
