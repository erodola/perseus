// Ring-size instantiation: compiled once per (word, ring size) with -DPC_LOGN=<log n> -DPC_RING_W=<word type>
// -DPC_RING_TABLE=<exported name>; ModMul is Barrett64. Kernels sit in an anonymous namespace (no clash between TUs).
#include <cmath>
#include "../args.cuh"

namespace {
#include "../arith.cuh"
#include "../ntt.cuh"
#include "../ntt_ring.cuh"
#include "../bconv.cuh"
#include "../elem.cuh"

using W = PC_RING_W;
using MM = pc::Barrett64<W>;
}  // namespace

extern "C" const RingTable* PC_RING_TABLE() {
    static constexpr RingTable T = {
        PC_RING_ABI,
        (uint32_t)sizeof(W),
        MM::KIND,
        (uint32_t)PC_LOGN,
        &pc::nttr::fwd<W, PC_LOGN>,
        &pc::nttr::inv<W, PC_LOGN>,
        &pc::nttr::fwd_g<W, PC_LOGN>,
        &pc::nttr::inv_g<W, PC_LOGN>,
        &pc::bconv::launch<W, MM>,
        &pc::elem::add_l<W>,
        &pc::elem::scale_l<W>,
        &pc::elem::sub_mul_l<W>,
        &pc::elem::copy_l<W>,
    };
    return &T;
}
