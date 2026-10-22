# Notebooks

| notebook | what it shows | needs |
|---|---|---|
| `setup_artifacts.ipynb` | every file a run reads: exported weights, token pool, fitted approximations, reference outputs, the recorded forward pass and its plan | CPU (network once for the pool) |
| `custom_encrypted_model.ipynb` | a small model written directly on the runtime's operations (`perseus._pc.Session(...).fhe`), checked against numpy | GPU |
| `client_server.ipynb` | GPT-2's planned decode served to a client process that holds the secret key and never touches the GPU | GPU (server), the GPT-2 artifacts |

`NB=setup|custom|client_server bash scripts/run_notebooks.sh` executes one headless.

The notebooks of the C++ runtime (`gpt2_torch_forward`, `gpt2_perseus_nn`, `bert_torch_forward`, `vit_torch_forward`,
`client_server_minimal`) used the `perseus.nn` module API and the `perseus._core` extension, which this runtime does
not have: the models are `perseus.impl` programs (`python/perseus/examples/`), run from the command line.
