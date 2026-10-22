// Slot permutation sigma_k in the bit-reversed NTT order (gather form), with and without a fused add. Semantics in
// args.cuh AutomorphArgs. Aligned 32-slot blocks map to aligned 32-slot blocks, so the gathers stay coalesced.
#pragma once
#include "arith.cuh"

namespace pc {
namespace automorph {
constexpr int BX = 256;

template <class W, bool ADD>
__global__ void automorph_k(AutomorphArgs A) {
    const int row = rs_row(A.rs, blockIdx.y);
    const int c = blockIdx.z / A.nparts, part = blockIdx.z % A.nparts;
    const int o = blockIdx.x * BX + threadIdx.x;
    const uint32_t src = automorph_slot(A.k, (uint32_t)o);
    W v = at<W>(A.in, c, part, row)[src];
    if constexpr (ADD) v = add<W>(v, at<W>(A.add, c, part, row)[o], (W)A.pr[row].p);
    at<W>(A.out, c, part, row)[o] = v;
}

template <class W>
int launch(const AutomorphArgs* a, cudaStream_t s) {
    if ((a->k & 1u) == 0) return -1;
    dim3 grid(N / BX, a->rs.nq + a->rs.np, a->K * a->nparts);
    if (a->has_add) automorph_k<W, true><<<grid, BX, 0, s>>>(*a);
    else automorph_k<W, false><<<grid, BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace automorph
}  // namespace pc
