// Kernel instantiation: W = uint32_t, ModMul = Barrett64.
#define PC_W uint32_t
#define PC_MM Barrett64
#define PC_TABLE pc_table_u32_barrett64
#include "instantiate.inc"
