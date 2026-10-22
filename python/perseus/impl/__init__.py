"""``perseus.impl``: build encrypted models from the leaf primitives.

The composite algorithms (linear, attention, norms, activations, polynomial kernels) written in Python on the
``Context`` leaf ops and the plaintext operands of ``Inference``, plus the runtime services a model needs (weight
residency ring, mask encode cache with worker staging, graph capture, planned bootstrapping). A model is an
``ImplModel`` subclass naming its stages, per-step masks and weights; perseus/examples/gpt2 is the reference
model. Module map: docs/ARCHITECTURE.md.
"""
