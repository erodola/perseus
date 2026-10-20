# HEAT on the n32 Python backend: landscape and effort

Untracked working note, 2026-10-20. The state of HEAT across the four Perseus models, what is missing, and what filling it
costs. Nothing below has been run yet: we are waiting for a go from the user.

## 0. Where things stand

| model | Perseus base row (docs/RESULTS_2026-10-19.md) | HEAT on this backend | HEAT checkpoint (he-aware-training) |
|---|---|---|---|
| GPT-2 small | decode 3.78 s/token, 128-token prefill 33.1 s | decode plan only (`gpt2_heat_decode_python_n32`, 10-09/10-10 recipe); no prefill plan | `gladia/heat-gpt2-small-openwebtext` |
| BERT-base | 37.7 s for 128 tokens (`bert_base_sst2_128`) | legacy 64-bit `bert_heat` config only; no n32 config, graph or plan | `checkpoints/bert/heat/{ckpt_3000.pt,hf}`, 201 it/fwd, SST-2 |
| ViT-B/16 | 123.2 s per image at 224 (`vit_b16_224`) | legacy 64-bit `vit_heat` config only; no n32 config, graph or plan | `checkpoints/vit/{nokd,eurosat_vit80,gold}/hf`, EuroSAT at **80 px** |
| Llama-3.1-8B | 128-token prefill 111.0 s (`llama31_8b_prefill128`) | nothing | nothing; the trainer has no Llama surgery (`configs/model/llama3.yaml` is a stub) |

## 1. Findings that shape the plan

- **The GPT-2 HEAT plan is behind base.** It places 175 bootstraps vs 112 for `gpt2_decode_python_n32`, summing
  `placements` over all 15 plan files. Per block, HEAT places 9-12 and base 5-9; in the argmax (block 13), 36 vs 12.
  - Cause: the HEAT plan predates the 10-13 recipe. Its runtime line has no SM_PERIODIC, SM_FOLD, CHEB_BSGS, SPARSE_B,
    StC-first, CUTMAX_PAIR or LADDER_RADIX_BITS.
  - `gpt2_heat_n32` still uses the THOR GELU; base switched to a per-block Chebyshev.
  - HEAT's own counts are still lower:

    | | HEAT | base |
    |---|---|---|
    | LN Newton | 2 | 3 |
    | LN Goldschmidt | 3 | 12-13 |
    | softmax refinements | 1 | 2-3 |

  - So a replan should put HEAT ahead, but that is unmeasured.
- **The ViT rows do not line up.**
  - HEAT ViT is EuroSAT at 80 px, T=26 (`vit_heat` softmax calib_T 26).
  - The Perseus row is `google/vit-base-patch16-224` with the ImageNet head, at 224 px, T=197, calibrated on Tiny-ImageNet
    (`vit_base_224`).
  - A same-protocol comparison needs both arms on one task at one resolution.
  - `vit_base_224` has the widest ranges of the four models (LN Chebyshev seed up to degree ~100, softmax input_scale 128,
    refine 5), so HEAT has the most room to gain on ViT.
- **Llama: the LN iteration lever does not exist.**
  - `llama31_8b_n32*` evaluates RMSNorm as one Chebyshev (nr_iters 0, gs_iters 0, degree 18-32).
  - What HEAT can still cut:
    - softmax Goldschmidt: gs_iters_first 10-13, per-step [9, 9], refine 2.
    - range squeeze, which lowers Chebyshev degrees: RMSNorm; SiLU, which is single-Chebyshev under PF_SILU_SINGLE with
      degree 15-127 set by xmax (4.8-52); and the softmax input scale.
  - The 09-10 rule ("SiLU exact in training, it/fwd is the only currency") was set for the plaintext scale-up study. For a
    Perseus row, a squeezed SiLU range pays in levels.
- **BERT length mismatch.** HEAT BERT learned its counts on SST-2 sentences of at most 55 tokens. The Perseus row is a
  128-token text. The softmax denominator range widens with T, so the counts may not hold at 128.

## 2. Steps and effort

| # | step | work | calendar | GPU |
|---|---|---|---|---|
| 1 | GPT-2 HEAT replan (gate) | HEAT GELU to per-block Chebyshev on HEAT ranges (`scripts/utils/thor_to_cheb.py`), SM_CHEB_GS bands, capture, plan on the current recipe, 16-token decode + 128-token prefill, axis-by-axis vs base | 1-2 days | ~4 GPU-h |
| 2 | Llama what-if (gate) | HEAT-like counts and degrees on the base Llama config, fake capture + plan on CPU (`--fake`), planned-bootstrap delta to seconds at measured route costs | ~1 day | none |
| 3 | BERT | `perseus.export --checkpoint` for BERT (only the GPT-2 path exists; the runner reads `$HF_HOME/perseus/models/textattack/.../classic`), n32 HEAT config at T=128 (HEAT counts + refit seeds/bands), capture + plan, SST-2 dev for HEAT and base | 3-5 days (+1-2 if counts fail at 128 and need a short HEAT fine-tune) | ~20 GPU-h (872 x 38 s ~ 9 h per arm) |
| 4 | ViT | option A: HEAT retrain at 224 on EuroSAT + a EuroSAT-224 base arm (matches the published row); option B: Perseus ViT at 80 px (runner hard-codes the 224 resize), no retrain, smaller workload than the published row | A ~1 week, B ~3-4 days | A: 2-3 GPU-days of training + ~9 h per arm for 256 images |
| 5 | Llama-3-8B | trainer code, LoRA, data, phase 1 + 2, export, capture + replan, eval | 2-3 weeks | 3-6 GPU-days (guess) |

Total: ~4-5 weeks in sequence, ~3 with the Llama code written while BERT and ViT run.

Llama details:
- **Trainer code:**
  - learnable RMSNorm (the LN module minus the centering).
  - Llama attention wrapper with the learnable softmax (RoPE cos/sin in, GQA via repeat_kv).
  - SiLU range regularizer.
  - calibrator branch for RMSNorm (it keys on `isinstance(nn.LayerNorm)`).
  - export to the Perseus Llama config format: `layers.N.ln1` names, `kind: rmsnorm`, Chebyshev fields. The likely route
    is `examples/llama_from_primitives/calibrate.py` on the merged HF checkpoint (`LLAMA_MODEL`), with the HEAT counts
    imposed.
- **Memory:**
  - A full fine-tune of 8B in fp32 with Adam needs ~128 GB, more than one 96 GB card.
  - The trainer is single-GPU (ddp=False), so LoRA. With the adapters off, the frozen base is also the self-distill teacher.
  - T=512 is enough: the Perseus rows are a 128-token prefill and a 512-token context.
- **Data:** tokenize the OpenWebText slice for Llama. `llama3.yaml` has data_dtype uint16, but vocab 128256 needs uint32.
- **Perseus side:** the base prefill plan merges 8 lever re-plans (lineage `pr2s12456e2r0sme2` in its PLAN_CMD.txt). A HEAT
  replan has to redo that.
- **Disk:** 592 of 950 GB quota used (10-20). Keep LoRA checkpoints, not full optimizer state (~100 GB each), and run a
  janitor.
- **Unknown:** s/it. Measure it on day 1 before committing. Llama-3.2-1B fits a full fine-tune on one card and runs the same
  code, so a few days on it would de-risk 8B.

## 3. Open decisions (user)

1. GPUs. The last ruling on file is 09-15 (cards 3 and 4 only). On 10-20, cards 3 and 7 were used by other users;
   0-2 and 4-6 were idle.
2. ViT: option A (retrain at 224) or option B (80 px).
3. Llama SiLU: keep it exact (09-10 rule) or squeeze its range for the encrypted row.
4. Go on gates 1 and 2.
