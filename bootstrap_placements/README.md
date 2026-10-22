# Bootstrap plans

A plan says where an encrypted model refreshes its values (bootstraps) and how. Every directory
here holds one plan file per model stage, `block_N_placement.json`, and a `PLAN_CMD.txt` with the
recipe that regenerates it: the recorded forward pass it was computed from (`graph=`, under
`graphs/`, same name as the plan), the planner settings (`recipe=`), the runtime switches the plan
was cut for (`runtime=`, applied by `--plan`) and the approximation configs (`configs=`).
`replanned=` notes the planner settings a replan adds on top of `recipe=`; `python/tests/test_paper_plans.py`
replans every directory below (its `GATE` spells those settings out) and checks that the result is
identical to the committed files. The Llama recordings ship as a release artifact
(`graphs/RELEASE.txt`).

## Result plans

Each runs with `--plan` alone, and each is its model's default plan
(`python/perseus/examples/defaults.py`); `scripts/repro/` has one script per result.

| directory | model and task |
|---|---|
| `gpt2_decode` | GPT-2, decode with the encrypted argmax (ILP placement); block 0 also has its generation plan, `block_0_feedback_placement.json` (`run_generate`: prompt teacher-forced, then each token fed back) |
| `gpt2_decode_ctx512` | GPT-2, decode up to a 512-token context (below) |
| `gpt2_prefill128` | GPT-2, 128-token prefill, StC-first dense refreshes |
| `gpt2_prefill128_tail` | `gpt2_prefill128` + the encrypted tail (stage 12: last position, final LayerNorm and LM head; stage 13: encrypted argmax), planned by `speedrun/make_tail_plan.sh` from `graphs/gpt2_prefill128_tail` |
| `bert_sst2_128` | BERT-base, SST-2 classification of a 128-token input |
| `vit_b16_224` | ViT-B/16, one 224 x 224 image |
| `llama31_8b_decode` | Llama-3.1-8B, decode |
| `llama31_8b_prefill128` | Llama-3.1-8B, 128-token prefill (the recording plus the lever patches listed in its `lineage=` line) |
| `llama31_8b_handoff_decode` | Llama-3.1-8B, decode on the K / V handed over by `llama31_8b_prefill128` (`run_handoff`): cut for the hand-off's cache level (`KV_ENTRY_LEVEL=38`), on the refresh sites of the same recipe's plain decode plan (`PLAN_KEEP_PLACEMENTS`, see its `sites=` line), every softmax refresh one scaling step lower (`PLAN_SITE_CF=softmax:-1`) |

## Placement study (`python/`)

Encrypted GPT-2 small on the 32-bit parameters, every plan computed from
`graphs/gpt2_decode_baselines`. Fhelipe is planned with `python/perseus/examples/gpt2/make_plan.sh`
(the blocks first, then the encrypted argmax), which also removes bootstraps that turn out redundant
(`PLAN_PRUNE=1`, the default). DaCapo and Orion are planned by their released tools (see below) and
left as the tools output them. *Planned* counts the bootstraps a plan places; *executed* also counts
the 23 per token that the model code performs itself, the same for every system. Perseus' own plan
of that study (393 planned, 416 executed per token) has been replaced by `gpt2_decode`.

| directory | system | planned | executed / token |
|---|---|---|---|
| `python/dacapo` | DaCapo (its compiler, `hecate-opt`), level budget 48 | 595 | 618 |
| `python/orion` | Orion (its released solver), level budget 48, 68 bootstraps added by our repair step | 871 | 894 |
| `python/fhelipe` | Fhelipe (our reimplementation), level budget 48, repaired, redundant bootstraps removed | 561 | 584 |
| `python/dacapo_dense` | DaCapo, full-size bootstraps only, level budget 48 | 609 | 632 |
| `python/orion_dense` | Orion, full-size bootstraps only, level budget 50 (no plan exists at 48) | 816 | 839 |
| `python/fhelipe_dense` | Fhelipe, full-size bootstraps only, level budget 50 with its depth limit at 48 | 754 | 777 |
| `python/ablations/cf_fixed_{7,8,9,10}` | Perseus, 32-bit, one fixed scaling setting (correction factor) for every bootstrap | 390–396 | 413–419 |
| `python/ablations/kappa_{1,...,128}` | Perseus, 32-bit, safety margin κ on recorded magnitudes (the main plan uses κ = 2) | 393 each | 416 |

Some bootstraps only need to refresh a few distinct values, because the data repeats across the
ciphertext; the runtime has a cheaper bootstrap for those, and every plan uses it except the
*full-size bootstraps only* ones. Those run with `SPARSE_AUTO=0 SPARSE_BTS_SLOTS=0`. Each plan
records the runtime settings it was computed for and refuses to run under different ones.

In generation, the next token re-enters block 0 as the output of a bootstrap rather than as a
fresh encryption, so block 0 needs its own plan for those tokens:
`block_0_feedback_placement.json`, block 0 planned again the same way from that starting point
(`make_plan.sh ... feedback`; the DaCapo and Orion scripts rerun the tool's block-0 choices). Every
plan above has one except the ablations and `python/orion_dense`: on full-size bootstraps,
Orion's choices cannot fit block 0's first LayerNorm in the level budget from that starting
point, even after repair, so generation refuses that plan. In a *full-size bootstraps only* plan,
the few bootstraps the model code performs itself (in the encrypted argmax) are also full-size, as
they are in such a run (`PLAN_DELIBERATE_CLAMP0`, set by `make_plan.sh` and by the DaCapo and Orion
scripts).

## DaCapo

`python/dacapo` comes from the released DaCapo compiler
([corelab-src/dacapo](https://github.com/corelab-src/dacapo) at `4616402`), run on every block by
`scripts/utils/dacapo_upstream/`:

```bash
bash scripts/utils/dacapo_upstream/build_hecate.sh      # hecate-opt into .cache/dacapo_upstream (LLVM/MLIR 18, SEAL 4.0)
PLAN_PRUNE=0 bash scripts/utils/dacapo_upstream/plan.sh graphs/gpt2_decode_baselines python/dacapo   # add `dense` for full-size bootstraps only
```

`dacapo_4616402.patch` only adds a `--dacapo-plan` pipeline, which runs DaCapo's own analysis
and placement passes without compiling further, and prints where its planner puts bootstraps.
`translate.py` converts each block into DaCapo's input format; where our planner rescales
without a multiplication, it writes a multiplication by a constant, so that DaCapo sees the same
level structure. `ckks_ml48.json` is DaCapo's default configuration with the bootstrap level
limits set to our parameters (5 multiplications between bootstraps). `sites.py` maps DaCapo's
choices back to our recorded values, and `deploy.py` turns them into a plan: it tracks levels,
adds the bootstrap at each block's end that DaCapo assumes (it plans every block from a fresh
input), repairs anything that would run out of levels, and plans the encrypted argmax the way
`make_plan.sh` does.

## Orion

`python/orion` comes from the released Orion bootstrap solver
([baahl-nyu/orion](https://github.com/baahl-nyu/orion) at `be8a827`), run on each block by
`scripts/utils/orion_upstream/`:

```bash
PLAN_PRUNE=0 bash scripts/utils/orion_upstream/plan.sh graphs/gpt2_decode_baselines python/orion   # add `dense` for full-size bootstraps only
```

The solver works with a budget of 50 levels: it cannot represent a block input that has no
levels left, which is how a block starts at 48. The resulting plan is then deployed at the
budget `ML` (48 unless exported); deployed at 50, some solver seeds put a bootstrap past the
range the runtime's bootstrap handles, and the runtime refuses it. The script downloads the
solver into `.cache/orion_upstream` (or uses `ORION_SRC`) and applies `orion_be8a827.patch`,
which changes only what stops the solver on transformer models: two searches that enumerate every
path through the model and never finish, replaced by exact equivalents (counting paths by
dynamic programming, and finding a region as the nodes that are both descendants and
ancestors), and four crashes (branches that rejoin at the same point, three-way branches,
overlapping regions, which are priced as plain layers, and nodes off the paths the solver
analyses, which it leaves without a level and never bootstraps). `marks.py` gives the solver our
recorded forward pass and records where it places bootstraps, for every block including the
encrypted argmax; `deploy.py` turns those into a plan the same way the DaCapo script does.

The solver iterates over Python sets of node names, so its choices depend on Python's hash seed;
the recipe fixes `PYTHONHASHSEED=0`. Both scripts remove redundant bootstraps unless
`PLAN_PRUNE=0`; the shipped DaCapo and Orion plans are the ones without that removal.

## 512-context decode plan (32-bit)

`gpt2_decode_ctx512` decodes up to a 512-token context (the main plan binds up to 32 cached keys). It is
captured at position 0 with step-scoped names and the general softmax period, so one plan binds at every cache
length; past 32 keys the softmax refreshes on a sparse route of Gp x 512 slots that the runtime builds as the cache
grows (1024 from 33 keys, up to 8192 from 257), multi-stage past 1024 slots, which is why its softmax refreshes are
planned to land at 32. The magnitudes are the larger of position 0's and position 511's (`scripts/utils/merge_mags.py`):
the softmax refresh inputs grow up to ~800x with the cache, and factors sized at position 0 wrap. The configs
(`gpt2_base_n32_ctx512`) refit the softmax on every query position up to 520 keys, without the per-cache-length
refine rescale. The runtime records every position on the CPU twin at setup and compiles one program per structure
class (56 for 512 positions: the score period, the K bucket and the routes change with the cache), with the K/V
states in pinned host memory and every route resident; `IMPL_RESIDENT_BLOCKS` is a switch of the C++ runtime and is
not read. 512 positions (`scripts/repro/gpt2_decode_ctx512.sh`, RTX PRO 6000): top-1 506/512 against the plaintext
model (the six others at rank 1), the encrypted argmax 512/512, median KL 4.0e-4, 4.10 s per token at 500-511 keys
(3.83 s within 32), peak 66 GB on the GPU; setup 17 min (recording 512 positions).
