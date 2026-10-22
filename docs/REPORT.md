# perseus (Rust + CUDA runtime) vs the C++ runtime (perseus 31ab85f + FIDESlib 797d26f)

Same machine and GPU (RTX PRO 6000 Blackwell), both sides in the same job, NUMA node 0, 8 threads; the new side is a
fresh clone of this repository built from nothing. New = the default plan per model
(`python/perseus/examples/defaults.py`); GPT-2 at 512 positions runs `gpt2_decode_python_n32_ctx512` on both sides.

| model | old | new | delta | accuracy old ; new | peak VRAM GiB old / new |
|---|---|---|---|---|---|
| GPT-2 decode s/tok (steady median, 16 tokens + encrypted argmax) | 3.783, 3.774 | 3.356, 3.364 | -11.0 % | 16/16 x2, KL 3.9e-4 / 3.6e-4 ; 15/16 (a near tie) and 16/16, KL 3.6e-4 / 4.1e-4 | 74.8 / 47.9 |
| GPT-2 decode up to a 512-token context, s/tok at 500-511 keys (within 32) | 5.30 (4.56) | 4.10 (3.83) | -22.6 % | 505/512, argmax 511/512, median KL 4.5e-4 ; 506/512, argmax 512/512, median KL 4.0e-4 | 75.9 / 64.6 |
| HEAT decode s/tok (`gpt2_heat_decode_python_n32_newrt`) | 7.567 | 4.544 | -40.0 % | 13/16, KL .0174 ; 14/16, KL .0154 | 50.3 / 47.5 |
| GPT-2 prefill128 s | 33.3 | 21.56 | -35.3 % | 125/128 both; KL 1.0e-4 ; 9.9e-5 | 80.6 / 53.0 |
| BERT SST-2 T=128 s | 37.6 | 23.91 | -36.4 % | label agrees; KL 4.9e-10 ; 1.3e-10 | 76.5 / 53.4 |
| ViT-B/16 s/img (12 images) | 131.7 | 58.97 | -55.2 % | top-1 and top-5 12/12 both; median KL 4.7e-4 ; 2.7e-4 | 90.1 / 57.5 |
| Llama-3.1-8B prefill128 s | 112.8 | 83.22 | -26.2 % | 126/128 ; 127/128, KL ~5e-5 | 91.8 / 76.6 |
| Llama-3.1-8B decode s/tok (median of 16) | 16.1 | 13.78 | -14.4 % | 16/16 both, KL ~4.4e-8 | 57.3 / 45.9 |
| Llama hand-off TTFT incl. first token s | out of memory | 114.3 | old cannot run | 9/9, maxKL 7.0e-5 | OOM / 77.0 |
| Served GPT-2 decode (client/server) s/tok | — | 3.339 | = in-process | 16/16, argmax 16/16; the client never on the GPU | — / 47.9 |

No device allocation after setup on any model (arena frozen, device free memory constant per token/forward).

## What changed and why it is faster
| change | effect |
|---|---|
| route costs measured and fed to the planner (`pc-bench bts --all` -> route table: dense shipped order 20.5 ms, StC-first 17.0, SparseB-512 17.4, SPRU 10.0 ms at K=1, realized landings 34 / 28 / 24 / 16 instead of the old 36 / 36) | 15-32 % fewer refreshes on prefill / BERT / ViT; GPT-2 prefill128 823 -> 562 |
| StC-first dense refreshes where the input fits the entry, the shipped order past it | 3 ms per refresh less and a 6-level deeper landing (GPT-2 prefill, BERT, ViT, PCMM finishes) |
| ILP placer on decode | 3 SPRU refreshes fewer per GPT-2 token |
| K-batching of independent refreshes | 10-25 % less per ciphertext at K=2-4 (prefill, encoders, Llama; decode blocks are one dependency chain) |
| op stream recorded once on a CPU twin, compiled per block (metadata checked at compile time), replayed in Rust with the GIL released; CUDA graphs for static decode segments | host issue 3-11 us/op, no Python on the critical path |
| measured landings in the compile (route-exact metadata) | refused plans fail at compile, not at replay; plans cut for the real landing level |
| one frozen device arena, sized from the loaded programs (plaintext slots per level, registers per level) | no allocation after setup, VRAM flat, 3-33 GiB less peak than the old stack on every model |
| refresh kernels at K=1 (fused basis conversion, seed-regenerated packed keys in CtS, 4-giant-step CtS, factored NTT twiddles, fused rescale) | decode -10 % on the same plan |
| PCMM / CCMM as exact cuBLASLt int8 GEMMs at the StC-first split | products 1.6-4x faster; Llama / encoder attention and linears |
| streamed Llama decode weights (device coefficient-row expander) | Llama decode runs beside its keys; 38 GB/token PCIe |
| secret-free server, CUDA-free client keygen | served = in-process speed |

## Still slower or open
| item | state |
|---|---|
| BERT KL margin | default plan 2e-11 .. 1.8e-10 (gate 1.2e-9); the shipped plan sat at 5.7e-10 .. 1.2e-9 |
| u64 chains in the client | the client packs 32-bit words only (n64 refused) |
| client links the CUDA runtime | `_pc` is one extension; a CUDA-free client module is not built |
| GPT-2 decode | device-bound at ~2.9 s fwd; the ILP default gains only 0.7 % |
| Llama prefill128 plan | not regenerable (patched lineage): no StC-first replan |

## Code size (non-blank lines, tracked source)
| | old | new |
|---|---|---|
| runtime (old: perseus C++/CUDA `_core` 23.9k + FIDESlib 41.4k (+ 1.8k lines of OpenFHE patches, OpenFHE itself not counted); new: Rust + CUDA crates without pc-tools / pc-proto, incl. 3.5k lines of crate tests dirs) | 65.3k | 53.2k (pc-gpu 16.7k, pc-kernels 7.8k, pc-py 7.8k, pc-core 6.0k, pc-program 5.3k, pc-shadow 3.8k, rest 5.8k) |
| Python package | 26.6k | 26.0k |
| tools (pc-tools bench / check) | — | 10.8k |

