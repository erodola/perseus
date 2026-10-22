// Kernel instantiation: W = uint32_t, ModMul = Montgomery.
#define PC_W uint32_t
#define PC_MM Montgomery
#define PC_TABLE pc_table_u32_mont
#include "instantiate.inc"
