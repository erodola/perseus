"""GPT-2 encrypted decode / generate rebuilt from the leaf primitives (cachemir packing, BSGS linear, K/V cache, THOR
softmax and GELU, Remez/Newton LayerNorm, tiled LM head, CutMax argmax, encrypted feedback embedding); ``ref.py``
mirrors the approximations in numpy so FHE noise and approximation error stay separable.

    .venv/bin/python -m perseus.examples.gpt2.run_decode --tokens 16 [--argmax] [--capture G | --plan P]

Modes: eager, capture (``--capture`` [+ ``--structure-only`` then ``capture_mags``]), planned (``--plan``; plan with
``make_plan.sh <graph> <name> [argmax]``). Default packing ``cachemir_complex`` (two real payloads per ciphertext);
long contexts via ``IMPL_SCOPED_NAMES`` / ``SM_PERIODIC``. Modules, packing, long-context and CutMax planning
details: docs/ARCHITECTURE.md.
"""
