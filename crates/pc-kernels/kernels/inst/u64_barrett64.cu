// Kernel instantiation: W = uint64_t, ModMul = Barrett64.
#define PC_W uint64_t
#define PC_MM Barrett64
#define PC_TABLE pc_table_u64_barrett64
#include "instantiate.inc"
