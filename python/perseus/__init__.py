"""perseus: Python orchestration over the Rust + CUDA core (`perseus._pc`).

Model code describes computations with the `impl` layer; GPU-free capture (fake) produces the op graph,
`plan` places bootstraps, and the Rust core compiles graph + plan into a replayable program.
"""
__version__ = "0.1.0"
