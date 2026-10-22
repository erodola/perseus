// Kernel instantiation: W = uint16_t, ModMul = Barrett32.
#define PC_W uint16_t
#define PC_MM Barrett32
#define PC_TABLE pc_table_u16_barrett32
#include "instantiate.inc"
