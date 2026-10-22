"""Llama-3 (8B) encrypted decode on ``perseus.impl``: the GPT-2 port's kernels, packing and
runtime levers (perseus/examples/gpt2) applied to the Llama architecture.

Llama-specific: RMSNorm (impl.norm.rms_norm), RoPE folded into the selection masks (impl.attention.Rope: one
rotation, no level), GQA expanded in the K / V weights, SwiGLU with the THOR SiLU composite, a plaintext BOS prefix
(prefix.py; the encrypted decode starts at position 1), streamed coefficient-row weights (~40 GB pinned host memory,
encoded on the GPU per baby-step sum), 256-slot attention routes {256, 1}. Module map and the capture -> plan ->
planned decode recipe: docs/ARCHITECTURE.md.
"""
