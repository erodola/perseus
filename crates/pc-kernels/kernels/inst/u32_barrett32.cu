// Kernel instantiation: W = uint32_t, ModMul = Barrett32.
#define PC_W uint32_t
#define PC_MM Barrett32
#define PC_TABLE pc_table_u32_barrett32
#include "instantiate.inc"
