# Packed prefill on fused key switching

Prefill-128 on the code of the decode entry FusedUnits
([#7](https://github.com/gladia-research-group/perseus/pull/7)): fused key switching and runtime units, with outputs
byte-identical to record 1's. On top of it, a cheaper and more accurate prefill:

- a block's GELU is one Chebyshev series (the decode's series, in blocks 0, 1 and 3-10) instead of record 1's THOR
  composite and its forced mid-refresh;
- softmax's Goldschmidt reciprocal rounds are Chebyshev-scaled in every layer (the existing `SM_CHEB_GS`);
- the last block computes the last position alone once every position's K and V are written (`PF_LAST_QUERY`);
- attention packs its q·K^T and P·V reductions over head-dim windows (`PF_ATTN_PACK=3`), 34 % fewer rotation steps in
  a block's attention, and the last head group's K and V come out of one PCMM (`PF_QKV_PACK`), one PCMM finish less
  per block;
- the ILP placer plans it: 516 refreshes against record 1's 592.

On one H100, Prefill-128 takes 22.30 s against record 1's 36.04 (×1.616, 6 pairs). The KL drops too: on dev prompts
0-11 the mean is 1.7e-5 against record 1's 1.0e-2, and on 16 held-out C4 validation prompts 7.0e-6 against 4.5e-3.

## Changes against record #1

- the code of FusedUnits ([#7](https://github.com/gladia-research-group/perseus/pull/7)), on which this branch is built:
  - key-switching kernels (NTT epilogues in the ModDown and the rescale, rotation sums inside the ModDown, lazy NTT
    butterflies, fewer copies, resident tails for the CtS baby-step keys);
  - runtime units (rotation grids and chains, K-batched rotate_sums, products and refreshes), with the StC-first
    stage cache and the constant ring sized from the programs;
  - raise variants that carry their drop;
  - decode-only: the served decode's patch at setup, the planner option `PLAN_FALLBACK_CF` and the decode plan
    `gpt2_decode_python_n32_m16i_fcf`. None of these is on the prefill path. Without that PR's commits 37a9d13,
    f19831a and d4d86a2, the decode default is record 1's m16i.
- the GELU series in the blocked layout: `python/perseus/impl/activation.py`, the configs
  `configs/model/approximation/gpt2_base_n32_pf` (the decode's `gpt2_base_n32` series in blocks 0, 1, 3-10)
- `PF_LAST_QUERY`: `python/perseus/impl/blocked.py` (`attention_last`), `python/perseus/examples/gpt2/prefill.py`
- `PF_QKV_PACK`: `python/perseus/examples/gpt2/prefill.py`, `python/perseus/impl/blocked.py`
- `PF_ATTN_PACK=3` (an existing switch)
- tests: `python/tests/test_prefill_last_cpu.py` (the last-position path and `PF_ATTN_PACK` against the plain
  computation, in a float64 mirror), the switches' fields in `crates/pc-config/src/fields.rs`, a compile gate in
  `crates/pc-program/tests/shipped_plans.rs`
- the ILP placer the plan is made with: `python/perseus/plan/placer/{ilp,ir,planner,refresh,__main__}.py`,
  `python/perseus/plan/run_bootstrap_all_blocks.sh`
- the plan `bootstrap_placements/gpt2_prefill128_pf`, its capture `graphs/gpt2_prefill128_pf` and its recipe
  `speedrun/make_prefill_plan.sh`; `speedrun/server.py` serves it (`speedrun/record1.py` is unchanged)

## Ablation

H100 80GB GPUs, the speedrun harness, record 1 (89bc830) as the record arm, two prompts per run. The first four rows
are steps of one cumulative ablation, timed on intermediate trees of this work rather than on this PR's commits: 3
rounds on dev prompts 0-5, each round running every arm once on one GPU in rotated order. The last step's plan files
are this PR's. The fifth row is an A/B of the prefill changes alone, without FusedUnits' code.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Raise variants, key-switching kernels (NTT epilogues, sums inside the ModDown, resident tails), served decode's patch | prefill | 36.01 → 33.07 s, ×1.089 |
| + runtime units, stage cache sized to the programs | prefill | 33.07 → 28.29 s, ×1.169 |
| + GELU series, `SM_CHEB_GS`, `PF_LAST_QUERY`, ILP plan (502 refreshes) | prefill | 28.29 → 24.20 s, ×1.169; KL 8.5e-3 → 5.8e-6 |
| + lazy NTT and fewer copies, `PF_QKV_PACK`, `PF_ATTN_PACK=3`, this plan (516) | prefill | 24.20 → 22.32 s, ×1.084 |
| The prefill changes alone, on record 1's runtime (6 pairs, dev 0-11) | prefill | 35.90 → 28.62 s, ×1.254 |
| **This entry** (6 pairs, dev 0-11) | prefill | **36.04 → 22.30 s, ×1.616 (p 6.5e-12)** |

## Dev result

One H100 80GB HBM3, the speedrun harness at 449f7ba, record 1 (89bc830) as the record arm, 6 pairs in the official
order (AB BA BA AB AB BA) on dev prompts 0-11, two prompts per run, timed at e41a8c0. Logs in `dev_logs/` (`ab_*`).

| Track | Score | Record 1, same GPU | KL (position 127) | Argmax | Peak GPU memory |
|---|---|---|---|---|---|
| Prefill-128 | 22.30 s | 36.04 s | 1.7e-5 | 12/12 | 65.5 GiB (record 57.4) |

score.py: quality PASS (mean KL 1.67e-5, p 1.3e-31), speed ×1.616 (p 6.5e-12).

KL per prompt, one run per prompt:

| Prompts | Arm | Mean | Median | p90 | Max | Above 10⁻² |
|---|---|---|---|---|---|---|
| dev 0-11 | entry | 1.7e-5 | 7.1e-6 | 5.6e-5 | 5.7e-5 | 0/12 |
| dev 0-11 | record 1 (the A/B's record arm) | 1.0e-2 | 4.7e-3 | 2.8e-2 | 3.8e-2 | 4/12 |
| C4 validation 0-15 | entry | 7.0e-6 | 5.3e-6 | 1.5e-5 | 1.5e-5 | 0/16 |
| C4 validation 0-15 | record 1 | 4.5e-3 | 2.9e-3 | 1.1e-2 | 2.5e-2 | 2/16 |

The C4 validation prompts are 16 rows of the C4 validation shard 00000, unused by any fit or choice here
(`c4v0-15_*.json`). The argmax agrees on every request of every run above.

## Disclosures

- **Built on FusedUnits ([#7](https://github.com/gladia-research-group/perseus/pull/7)).** Its disclosures that touch
  the prefill:
  - memory: the StC-first stage cache and the constant ring are sized for the compiled programs, the CtS baby-step
    keys keep resident tails, and `KsWork::ext` holds a 7-key sum of up to 4 ciphertexts;
  - unused code: an EvalMod relin + rescale fusion behind `PC_EM_FUSE_RESCALE` (off), SPRU raise-drop variants no plan
    here asks for, and `Ops::mult_pt_add[_inplace]` (benchmarked, not wired in);
  - outputs byte-identical to record 1's with record 1's plans, before its plan commits.
- **Approximations.**
  - The GELU series of blocks 0, 1 and 3-10 is the decode's, fitted on OpenWebText by the repository's calibrator.
  - Blocks 2 and 11 keep record 1's THOR composite: their series (degree 511 and 383) needs more levels than the PCMM
    input cap allows.
  - A Chebyshev series diverges outside its domain [-X, X], where THOR degrades more gently. X is 1.25 times THOR's
    calibrated xmax, and on 2,048 C4 validation prompts no GELU input went past that xmax on record 1's noise-free CPU
    twin, so |x|/X stayed below 0.8 there.
  - `SM_CHEB_GS` is on in every layer.
- **The last block.** Block 11 computes the attention output, the MLP and the residual of position 127 only. The K
  and V of all 128 positions are still computed in every block, including block 11, and nothing else of positions
  0-126 in block 11 feeds an output or a cache. `python/tests/test_prefill_last_cpu.py` checks the last-position
  attention against the full blocked attention at position 127 (relative 1e-12) and the broadcast of that row (exact),
  so the last position's output is the full computation's in exact arithmetic.
- **Packing.**
  - `PF_ATTN_PACK` and `PF_QKV_PACK` are exact rearrangements in exact arithmetic. A float64 mirror gives a difference
    of 0 for `PF_QKV_PACK` and at most 2.1e-14 for `PF_ATTN_PACK`; the repository test asserts below 1e-12.
  - Under CKKS the op sequence differs: one more mask product on each path, other rotation trees, other refresh
    ranges.
  - With `PF_QKV_PACK`, V's last head-group block carries K's heads on its padding heads, so its refresh input range
    grows (block 5's capture: 2.61 → 5.81).
  - The 34 % (block 0: 6,608 → 4,340 rotation steps in attention) is counted on the captures.
- **Keys.** `PF_ATTN_PACK` asks for 7 more rotation keys (the window shifts by 1-7 score blocks, steps N - 512 k): 173
  rotation keys against record 1's 166, holding 16,458 MB of GPU memory against 15,962 MB (server log). The chain,
  secrets, key formats and the pinned client are record 1's. The prefill's SparseB refreshes raise at drop 1
  (`FIDESLIB_BTS_RAISE_DROP=512:1`), as record 1's decode already does.
- **Memory.** Peak GPU memory on Prefill-128 is 65.5 GiB against record 1's 57.4, most of it the StC-first stage
  cache sized for this plan's programs.
- **Plan.** `gpt2_prefill128_pf` has 516 refreshes (12 blocks 486, tail 30) against record 1's 592 (562 + 30).
  `speedrun/make_prefill_plan.sh` captures it on the CPU twin, from the repository's OpenWebText oracle rows
  (`ALL_BLOCKS_IO_DIR`) as for record 1, and plans it from that capture. Run from scratch at e41a8c0 (capture and
  plan, CPU only, 39 minutes), it reproduces all 16 plan files. The capture is the shipped one except for the names of
  four mask plaintexts per block, which embed a Python object id. The rest of each name, and every other leaf of the
  graphs, are equal. The PCMM band shifts are record 1's calibration (`gpt2_prefill128_m16s0`'s).
- **Planner.** This branch's planner is the ILP placer the prefill plan was made with. The plan uses its ILP model
  changes, the model now being the simulation, and its StC-first marking of captured refreshes. It also carries options
  no plan here sets: raise-drop search, base and hints, free hints, a tie-break and an "ms" objective. The model change
  is not behind an option: for recipes where the old ILP fell back to the min-cut, it now solves the ILP. We ran the
  repository's regeneration gate (`python/tests/test_paper_plans.py`, every GATE plan whose graph is in the
  repository) with it, after the commits that follow e41a8c0:
  - 12 plans regenerate file for file, among them record 1's `gpt2_prefill128_m16s0`;
  - `gpt2_prefill128` differs in its Config stamp only, as on the speedrun base;
  - `gpt2_prefill_python_n32_pcmmstcf` has the same placements and runtime fields, and its summary counts one more
    blind branch in blocks 2 and 9;
  - record 1's decode plan `m16i` and FusedUnits' `m16i_fcf` come out as different plans. They regenerate with
    FusedUnits' planner (f19831a), and their GATE entries say so. Prefill-128 uses neither.
- **Data.** No dev prompt went into any fit or choice. The GELU and softmax configs come from OpenWebText, and the
  capture from the OpenWebText oracle rows. The C4 validation KL above is held out.
- **Numbers.** Everything here was measured on H100s, not on the reference GPU.
