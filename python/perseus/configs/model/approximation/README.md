# Approximation configs

Encrypted models can only add and multiply, so every nonlinear function (LayerNorm's inverse
square root, softmax, GELU and the encrypted argmax) runs as a polynomial or an iterative
approximation. Each `configs.json` holds those approximations for one model and one parameter
set: for every place the function appears, the input interval, the polynomial degree and the
number of iterations, fitted by `perseus-calibrate` on the value ranges that model actually
reaches, plus a record of how it was produced under `meta`. The runtime reads the file given by
`CONFIGS_PATH` as it is.

## What is here

| directory | model | parameters | notes |
|---|---|---|---|
| `gpt2_base_n32` | GPT-2 | 32-bit (the default) | GELU as a Chebyshev series; its `cutmax` section drives the encrypted argmax |
| `gpt2_base_n32_kc16` | GPT-2 | 32-bit | Decode-8 (at most 16 keys): softmax (Chebyshev-scaled Goldschmidt, `SM_CHEB_GS`) and GELU fitted to positions 0-15 by `scripts/utils/fit_gpt2_decode16.py`; LayerNorm and CutMax as `gpt2_base_n32` |
| `gpt2_base_n32_kc16_cm` | GPT-2 | 32-bit | `gpt2_base_n32_kc16` with `gpt2_base_n32_cm`'s CutMax (scaled normalisers) |
| `gpt2_base` | GPT-2 | 64-bit | the same fit, GELU as a plain polynomial |
| `gpt2_heat`, `gpt2_heat_n32` | GPT-2 fine-tuned for encryption with HEAT | both | |
| `gpt2_squeeze` | GPT-2 | 64-bit | fewer iterations per function |
| `vit_base`, `vit_base_n32` | ViT-B/16 | both | `vit_base_112` is for 112-pixel images |
| `vit_heat`, `vit_squeeze` | ViT-B/16 | 64-bit | as for GPT-2 |
| `bert_base`, `bert_base_n32` | BERT-base | both | fitted on SST-2 |
| `bert_heat`, `bert_squeeze` | BERT-base | 64-bit | as for GPT-2 |
| `paper/` | GPT-2, ViT | 64-bit | the exact configs behind the published numbers, kept so those results stay reproducible |

`recount_report.json`, where present, records how a config's iteration counts were derived.

## A config, its recording and its plan belong together

The same name appears in three places: the config here, the recorded forward pass (under
`graphs/`, or `.cache/graph_*`) and the plan (under `bootstrap_placements/`). The three must
always belong to the same model and config.

This matters because a mismatch goes unnoticed: a plan computed from one config's recording,
run with another config, gives a wrong answer rather than an error. Matching names are what
make a wrong pairing visible.

For the same reason the shipped plans are tied to these files. Change a value and the forward
pass must be recorded and planned again (README, steps 2 and 3).
`scripts/utils/lint_approx_config.py` checks a config before recording.
