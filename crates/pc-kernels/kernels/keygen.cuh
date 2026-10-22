// Switching-key generation and packed-key expansion. Semantics in args.cuh (KskGenArgs, KskUnpackArgs).
// ksk_gen: one key, every stored digit and slot in one launch; 4 consecutive slots per thread (the u32 KSKA expander
// runs cooperatively over 4-lane groups, so whole warps stay active), a block owns 512 slots = 16*bits packed words.
#pragma once
#include "arith.cuh"
#include "chacha.cuh"

namespace pc {
namespace keygen {
constexpr int BX = 128;
constexpr int SLOTS = BX * 4;

template <class W, class MM>
__global__ void __launch_bounds__(BX) ksk_gen(KskGenArgs A) {
    __shared__ uint32_t sh[SLOTS];
    const int m = blockIdx.y, d = blockIdx.z;
    const int id = A.ids[m];
    const PrimeConst& q = A.pr[id];
    const W p = (W)q.p;
    const uint32_t base = (blockIdx.x * BX + threadIdx.x) * 4;
    W a4[4];
    chacha::regen_slots4<W, MM>(A.seed, threadIdx.x & 31u, base, (uint32_t)d, q, chacha::TAG_KSKA, a4);
    const size_t row = (size_t)d * A.nrows + m;
    const size_t off = row * N + base;
    const V4<W> e4 = load4<W>((const W*)A.e + off);
    const V4<W> sn = load4<W>((const W*)A.s_new + (size_t)id * N + base);
    const bool in_digit = id < A.nq && A.digit_of[id] == d;
    V4<W> so{};
    W pm = 0, pms = 0;
    if (in_digit) {
        so = load4<W>((const W*)A.s_old + (size_t)id * N + base);
        pm = ((const W*)A.pmod)[id];
        pms = ((const W*)A.pmods)[id];
    }
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        W x = e4.v[w];
        if (in_digit) {
            W t = so.v[w];
            if (A.sq) t = MM::mul(t, t, q);
            x = add<W>(x, mul_shoup<W>(t, pm, pms, p), p);
        }
        r[w] = sub<W>(x, MM::mul(a4[w], sn.v[w], q), p);
    }
    if (A.a) store4<W>((W*)A.a + off, a4[0], a4[1], a4[2], a4[3]);
    if (A.bits == 0) {
        store4<W>((W*)A.b + off, r[0], r[1], r[2], r[3]);
        return;
    }
    if constexpr (sizeof(W) == 4) {
#pragma unroll
        for (int w = 0; w < 4; ++w) sh[threadIdx.x * 4 + w] = r[w];
        __syncthreads();
        const int bits = A.bits;
        const uint32_t mask = bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        const int nw = 16 * bits;  // words of this block's 512 slots
        uint32_t* out = (uint32_t*)A.b + row * A.row_words + (size_t)blockIdx.x * nw;
        for (int j = threadIdx.x; j < nw; j += BX) {
            const uint32_t bit0 = (uint32_t)j * 32u;
            uint32_t k = bit0 / bits;
            const uint32_t o = bit0 - k * bits;
            uint64_t acc = (uint64_t)(sh[k] & mask) >> o;
            for (uint32_t filled = bits - o; filled < 32 && k + 1 < (uint32_t)SLOTS; filled += bits)
                acc |= (uint64_t)(sh[++k] & mask) << filled;
            out[j] = (uint32_t)acc;
        }
        if (blockIdx.x == gridDim.x - 1) {
            uint32_t* rowp = (uint32_t*)A.b + row * A.row_words;
            for (uint64_t g = (uint64_t)N * bits / 32 + threadIdx.x; g < A.row_words; g += BX) rowp[g] = 0;
        }
    }
}

template <class W>
__global__ void ksk_unpack(KskUnpackArgs A) {
    const int r = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t* in = A.in + (size_t)r * A.row_words;
    const uint64_t bit = (uint64_t)i * A.bits;
    const uint32_t w = (uint32_t)(bit >> 5), o = (uint32_t)(bit & 31);
    const uint64_t v = ((uint64_t)in[w] | ((uint64_t)in[w + 1] << 32)) >> o;
    const uint64_t mask = A.bits == 32 ? 0xFFFFFFFFull : ((1ull << A.bits) - 1);
    ((W*)A.out)[(size_t)r * N + i] = (W)(v & mask);
}

template <class W, class MM>
int gen_l(const KskGenArgs* a, cudaStream_t s) {
    if (a->bits < 0 || a->bits > 32 || (a->bits > 0 && sizeof(W) != 4)) return -1;
    if (a->bits > 0 && a->row_words < (uint64_t)N * a->bits / 32 + 1) return -2;
    ksk_gen<W, MM><<<dim3(N / SLOTS, a->nrows, a->ndig), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int unpack_l(const KskUnpackArgs* a, cudaStream_t s) {
    if (a->bits <= 0 || a->bits > 32 || a->row_words < (uint64_t)N * a->bits / 32 + 1) return -1;
    ksk_unpack<W><<<dim3(N / 128, a->nrows), 128, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace keygen
}  // namespace pc
