// Secret-key encryption in the NTT domain with `a` expanded from a seed (the KSKA expander rule with w13 = nonce and
// the "ENCA" tag): c0 = -a s + e + m, c1 = a. Semantics in args.cuh EncryptArgs. Four slots per thread (the u32
// expander runs cooperatively over 4-lane groups, so whole warps must stay active).
#pragma once
#include "arith.cuh"
#include "chacha.cuh"

namespace pc {
namespace encrypt {
constexpr int BX = 128;

template <class W, class MM>
__global__ void __launch_bounds__(BX) encrypt_k(EncryptArgs A) {
    const int l = blockIdx.y;
    const PrimeConst& q = A.pr[l];
    const W p = (W)q.p;
    const uint32_t gtid = blockIdx.x * BX + threadIdx.x;
    const uint32_t base = gtid * 4;
    W a4[4];
    chacha::regen_slots4<W, MM>(A.seed, threadIdx.x & 31u, base, A.nonce, q, chacha::TAG_ENCA, a4);
    const size_t off = (size_t)l * N + base;
    const V4<W> s4 = load4<W>((const W*)A.s + off);
    const V4<W> e4 = load4<W>((const W*)A.e + off);
    V4<W> m4{};
    if (A.m) m4 = load4<W>((const W*)A.m + off);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) r[w] = sub<W>(add<W>(e4.v[w], m4.v[w], p), MM::mul(a4[w], s4.v[w], q), p);
    store4<W>((W*)A.c0 + off, r[0], r[1], r[2], r[3]);
    if (A.c1) store4<W>((W*)A.c1 + off, a4[0], a4[1], a4[2], a4[3]);
}

template <class W, class MM>
int launch(const EncryptArgs* a, cudaStream_t s) {
    encrypt_k<W, MM><<<dim3(N / (BX * 4), a->nl), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace encrypt
}  // namespace pc
