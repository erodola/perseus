// EvalMod element-wise kernels generic over (W, MM): Q rows [0, n) of K ciphertexts, row index ==
// prime id. Semantics in args.cuh (TensorArgs, MdEpiArgs, WsumArgs, AddConstArgs, MulTableArgs). The composite rescale
// lives in rescale.cuh.
#pragma once
#include "arith.cuh"

namespace pc {
namespace em {
constexpr int BX = 256;

template <class W, class MM>
__global__ void tensor(TensorArgs A) {
    using Wd = WideOf<W>;
    const int row = blockIdx.y, c = blockIdx.z;
    const PrimeConst& q = A.pr[row];
    const W p = (W)q.p;
    const int i = blockIdx.x * BX + threadIdx.x;
    Wd s0 = 0, s1 = 0, s2 = 0;
    {
        const W x0 = at<W>(A.a0, c, 0, row)[i], x1 = at<W>(A.a0, c, 1, row)[i];
        const W y0 = at<W>(A.b0, c, 0, row)[i], y1 = at<W>(A.b0, c, 1, row)[i];
        s0 += (Wd)x0 * y0;
        s1 += (Wd)x0 * y1 + (Wd)x1 * y0;
        s2 += (Wd)x1 * y1;
    }
    if (A.npairs > 1) {
        const W x0 = at<W>(A.a1, c, 0, row)[i], x1 = at<W>(A.a1, c, 1, row)[i];
        const W y0 = at<W>(A.b1, c, 0, row)[i], y1 = at<W>(A.b1, c, 1, row)[i];
        s0 += (Wd)x0 * y0;
        s1 += (Wd)x0 * y1 + (Wd)x1 * y0;
        s2 += (Wd)x1 * y1;
    }
    W r0 = MM::reduce(s0, q), r1 = MM::reduce(s1, q);
    if (A.acc) {
        r0 = add<W>(r0, at<W>(A.out, c, 0, row)[i], p);
        r1 = add<W>(r1, at<W>(A.out, c, 1, row)[i], p);
    }
    at<W>(A.out, c, 0, row)[i] = r0;
    at<W>(A.out, c, 1, row)[i] = r1;
    const W r2 = MM::reduce(s2, q);
    at<W>(A.d2, c, 1, row)[i] = r2;
    if (A.coef) ((W*)A.coef)[c * A.coef_ct_stride + (size_t)row * N + i] = r2;
}

template <class W>
__global__ void md_epilogue(MdEpiArgs A) {
    const int row = blockIdx.y, c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const W p = (W)A.pr[row].p;
    const int i = blockIdx.x * BX + threadIdx.x;
    W* d = at<W>(A.dst, c, part, row);
    const W v = mul_shoup<W>(sub<W>(at<W>(A.x, c, part, row)[i], at<W>(A.t, c, part, row)[i], p), ((const W*)A.binv)[row],
                             ((const W*)A.binvs)[row], p);
    W r = add<W>(d[i], v, p);
    if (A.dbl) r = add<W>(r, r, p);
    const W* k = (const W*)A.k;
    if (k && part == 0) r = add<W>(r, k[row] >= p ? (W)(k[row] - p) : k[row], p);
    d[i] = r;
}

template <class W>
__global__ void md_epilogue_to(MdEpiToArgs B) {
    const MdEpiArgs& A = B.m;
    const int row = blockIdx.y, c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const W p = (W)A.pr[row].p;
    const int i = blockIdx.x * BX + threadIdx.x;
    const W v = mul_shoup<W>(sub<W>(at<W>(A.x, c, part, row)[i], at<W>(A.t, c, part, row)[i], p), ((const W*)A.binv)[row],
                             ((const W*)A.binvs)[row], p);
    const W a = at<W>(B.a, c, part, row)[i];
    at<W>(A.dst, c, part, row)[i] = B.sub ? sub<W>(a, v, p) : add<W>(a, v, p);
}

template <class W, bool KADD>
__global__ void wsum(WsumArgs A, const void* kc) {
    const int row = blockIdx.y, c = blockIdx.z;
    const W p = (W)A.pr[row].p;
    const int i = blockIdx.x * BX + threadIdx.x;
    W r0 = 0, r1 = 0;
    for (int j = 0; j < A.nsrc; ++j) {
        const W w = ((const W*)A.w)[j * A.rowstride + row], ws = ((const W*)A.ws)[j * A.rowstride + row];
        r0 = add<W>(r0, mul_shoup<W>(at<W>(A.src[j], c, 0, row)[i], w, ws, p), p);
        r1 = add<W>(r1, mul_shoup<W>(at<W>(A.src[j], c, 1, row)[i], w, ws, p), p);
    }
    if constexpr (KADD) {
        const W k = ((const W*)kc)[row];
        r0 = add<W>(r0, k >= p ? (W)(k - p) : k, p);
    }
    at<W>(A.out, c, 0, row)[i] = r0;
    at<W>(A.out, c, 1, row)[i] = r1;
}

// out = alpha x + tmp (rescale_combine with a separate output; x unchanged)
template <class W>
__global__ void rescale_combine_to(RescaleCombineToArgs R) {
    const RescaleCombineArgs& A = R.a;
    const int row = blockIdx.y, c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const W p = (W)A.pr[row].p;
    const int i = blockIdx.x * BX + threadIdx.x;
    at<W>(R.out, c, part, row)[i] =
        add<W>(mul_shoup<W>(at<W>(A.x, c, part, row)[i], ((const W*)A.alpha)[row], ((const W*)A.alphas)[row], p), at<W>(A.tmp, c, part, row)[i], p);
}

template <class W>
__global__ void add_const(AddConstArgs A) {
    const int row = blockIdx.y, c = blockIdx.z;
    const W p = (W)A.pr[row].p;
    const int i = blockIdx.x * BX + threadIdx.x;
    W* d = at<W>(A.x, c, 0, row);
    const W k = ((const W*)A.k)[row];
    d[i] = add<W>(d[i], k >= p ? (W)(k - p) : k, p);
}

template <class W, class MM>
__global__ void mul_table(MulTableArgs A) {
    const int row = blockIdx.y, part = blockIdx.z & 1, cc = blockIdx.z >> 1;
    const PrimeConst& q = A.pr[row];
    const int i = blockIdx.x * BX + threadIdx.x;
    at<W>(A.out, cc, part, row)[i] = MM::mul(at<W>(A.in, cc, part, row)[i], ((const W*)A.t)[(size_t)row * N + i], q);
}

template <class W, class MM>
int tensor_l(const TensorArgs* a, cudaStream_t s) {
    tensor<W, MM><<<dim3(N / BX, a->n, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int md_epilogue_l(const MdEpiArgs* a, cudaStream_t s) {
    md_epilogue<W><<<dim3(N / BX, a->n, 2 * a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int md_epilogue_to_l(const MdEpiToArgs* a, cudaStream_t s) {
    md_epilogue_to<W><<<dim3(N / BX, a->m.n, 2 * a->m.K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int wsum_l(const WsumArgs* a, cudaStream_t s) {
    if (a->nsrc > 8) return -1;
    wsum<W, false><<<dim3(N / BX, a->n, a->K), BX, 0, s>>>(*a, nullptr);
    return (int)cudaGetLastError();
}
template <class W>
int wsum_k_l(const WsumKArgs* a, cudaStream_t s) {
    if (a->w.nsrc > 8 || !a->k) return -1;
    wsum<W, true><<<dim3(N / BX, a->w.n, a->w.K), BX, 0, s>>>(a->w, a->k);
    return (int)cudaGetLastError();
}
template <class W>
int combine_to_l(const RescaleCombineToArgs* a, cudaStream_t s) {
    rescale_combine_to<W><<<dim3(N / BX, a->a.n, 2 * a->a.K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int add_const_l(const AddConstArgs* a, cudaStream_t s) {
    add_const<W><<<dim3(N / BX, a->n, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W, class MM>
int mul_table_l(const MulTableArgs* a, cudaStream_t s) {
    mul_table<W, MM><<<dim3(N / BX, a->n, 2 * a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace em
}  // namespace pc
