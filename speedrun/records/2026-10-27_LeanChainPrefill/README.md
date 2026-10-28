# Prefill-128 on the lean bootstrap chain, its PCMMs pre-divided at a composite level

Prefill-128 on LeanChainDecode's code ([#14](https://github.com/gladia-research-group/perseus/pull/14)), with:

- the lean bootstrap chain `n32a4` of LeanChainDecode, now on Prefill-128 too: StC-first's input stages on three
  single 27-bit primes, the key switching on 4 digits, every user level at record 3's primes and scale;
- the PCMMs (the plaintext-ciphertext matrix products of the linears) on that chain:
  - a PCMM's StC-first input stages run their own fixed split (`THIN_SPLIT_PCMM`), and its product on the chain's
    3-limb split;
  - a PCMM whose band shift is above 1 divides its inputs by the shift beforehand, exactly (a power-of-two multiply
    and a rescale) on the level above the StC-first entry, so that the shift does not land in a single 27-bit
    plaintext: 84 pre-divisions per prefill, after which those inputs carry the stamp 2^20 and share one cached last
    StC stage per input level;
- refresh pairs on the lean chain for members up to CF 12;
- a plan made by record 3's recipe on the lean chain: the softmax steps sized with a drift margin of 6 instead of 4,
  CutMax's stage 13 on the ILP placer, 478 refreshes against record 3's 460.

On one H100, Prefill-128 takes 14.84 s against record 3's 17.60 on the same GPU (×1.185, 6 pairs, p 1.7e-10). The KL
stays three orders of magnitude under the bar: 1.5e-5 on dev prompts 0-11, and 1.2e-5 and 1.1e-5 in two sets of
runs on 48 C4 validation prompts no fit used, about 1.15 times record 3's 1.0e-5 (Dev result).

Decode-8, which this entry does not claim, is LeanChainDecode's: on the same keys and inputs its logits are byte-identical
to LeanChainDecode's (dev prompts 0-1, 32 positions, CutMax's argmax 32/32).

## Changes against LeanChainDecode

- **PCMMs on a lean chain.**
  - The PCMM inputs' own StC-first stages (`THIN_SPLIT_PCMM`): `crates/pc-gpu/src/bts/stc_first.rs`.
  - The pre-division of a PCMM input of band shift above 1 on the level above the StC-first entry, and one input
    stamp per input level: `crates/pc-gpu/src/matmul/pcmm.rs` (with the per-input path `linear_per_input`, moved there
    from `crates/pc-py/src/pcmm.rs`), `crates/pc-program/src/{compile/mod,program}.rs`, `crates/pc-py/src/session.rs`.
  - The planner caps those inputs one level higher (`PLAN_PCMM_PREDIV_CAP`):
    `python/perseus/plan/placer/{__main__,planner}.py`, `python/perseus/plan/run_bootstrap_all_blocks.sh`.
  - `pc-bench pcmm`: the weights' amplitude (`--wamp`), and outputs past band shift 11 lowered exactly before the
    decryption (`crates/pc-tools/src/check_pcmm.rs`).
- **Refresh pairs on a lean chain:** members up to CF 12, the compile and the session agreeing:
  `crates/pc-program/src/compile/mod.rs`, `crates/pc-py/src/session.rs`, `crates/pc-program/tests/units.rs`.
- **The plan and what serves it.**
  - `speedrun/make_prefill_plan.sh` with `PRESET=n32a4` (the level caps, `PLAN_PCMM_PREDIV_CAP=42`, `SM_KAPPA=6`,
    `CUTMAX_PLACER=ilp`), and `RESUME=1` keeping the later stages a stopped run completed;
    `speedrun/make_cutmax_plan.sh` with `CUTMAX_PLACER=ilp` on Prefill-128's stage 13.
  - The plans `bootstrap_placements/gpt2_prefill128_n32a4_pfbh{,_cm}` and `gpt2_prefill128_n32a4_pfbhrd{,_cm}`; the
    served plan is `gpt2_prefill128_n32a4_pfbhrd_cm` (`speedrun/server.py`). They take record 3's captures,
    `graphs/gpt2_prefill128_pfbh{,_cm}`.
  - The gate `crates/pc-program/tests/shipped_plans.rs` `gpt2_prefill_speedrun` compiles the served plan
    (BabyStepPrefill's keeps its own, `gpt2_prefill_speedrun_pfbhrd_cm`); the planner's input caps and the own split
    have unit tests (`python/tests/test_plan_placer.py`, `crates/pc-gpu`).

## Ablation

H100 80GB GPUs, the speedrun harness, two prompts per run.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| NTT column passes on 16-column blocks (LeanChainDecode) | prefill | 17.600 → 17.278 s, ×1.019 (p 0.0027) |
| `KS_ACC` (LeanChainDecode) | prefill | 17.278 → 17.367 s, ×0.995 (p 0.97, not resolved) |
| Linear-transform baby keys, 2 digits regenerated (LeanChainDecode) | prefill | 17.367 → 17.016 s, ×1.021 (p 0.0069) |
| `ks_dot_key` on row segments, shared SparseB key tails (LeanChainDecode) | prefill | 17.016 → 16.969 s, ×1.003 (p 0.23, not resolved) |
| LeanChainDecode vs record 3 | prefill | 17.58 → 16.98 s, ×1.035 (p 1.3e-5) |
| The lean chain, the pre-divided PCMMs and the plan (478 refreshes) | prefill | 16.97 → 14.84 s, ×1.144 (p 6.8e-10) |
| **This entry** vs record 3 | prefill | **17.60 → 14.84 s, ×1.185 (p 1.7e-10)** |

- The first four rows are one job on one GPU, 4 rounds in rotated order on dev prompts 0-7, each row against the row
  above (score.py, 4 pairs); their outputs are byte-identical to record 3's. The fifth is LeanChainDecode's own
  Prefill-128 A/B (6 pairs on dev prompts 0-11).
- The lean chain's row is its own A/B against LeanChainDecode's code, 6 pairs (AB BA BA on dev prompts 0-5 and 6-11, in
  two jobs, each pair on one GPU), at an earlier commit of this entry with the same code and plans (comments apart).

## Dev result

H100 80GB HBM3 GPUs, the speedrun harness at 449f7ba's harness files, timed at this PR's commit. 6 pairs in the order
AB BA BA AB AB BA on dev prompts 0-11, two prompts per run, in two jobs of 3 pairs (dev prompts 0-5 and 6-11), each on
one GPU; score.py scores the 6 pairs together. Record 3's arm is BabyStepPrefill's code with record 1's client; this
entry's client is LeanChainDecode's `perseus_lean_client.py` (pinned to LeanChainDecode's commit). Logs are in
`dev_logs/`: `pf_*` and `c4v*`.

| Track | Score | Record 3, same GPU | KL (position 127) | Argmax | Peak GPU memory |
|---|---|---|---|---|---|
| Prefill-128 | 14.84 s | 17.60 s | 1.5e-5 | 12/12 | 59.7 GiB (record 3 62.6) |

score.py: quality PASS (mean KL 1.50e-5, p 3.4e-31), speed ×1.185 (p 1.7e-10).

| Prompts | Arm | Mean | Median | p90 | Max | Above 10⁻² | Argmax |
|---|---|---|---|---|---|---|---|
| dev 0-11 | entry | 1.5e-5 | 7.7e-6 | 2.8e-5 | 8.0e-5 | 0/12 | 12/12 |
| dev 0-11 | record 3 (the A/B's record arm) | 2.0e-5 | 1.2e-5 | 3.4e-5 | 9.8e-5 | 0/12 | 12/12 |
| C4 validation 32-79 | entry, run set 1 (12 runs) | 1.2e-5 | 1.0e-5 | 2.4e-5 | 3.0e-5 | 0/48 | 48/48 |
| C4 validation 32-79 | entry, run set 2 (12 runs) | 1.1e-5 | 9.3e-6 | 2.0e-5 | 5.1e-5 | 0/48 | 48/48 |
| C4 validation 32-79 | record 3 (12 runs) | 1.0e-5 | 7.6e-6 | 2.0e-5 | 3.6e-5 | 0/48 | 48/48 |

- The C4 validation prompts are rows 32-79 of the C4 validation shard 00000; no fit or choice used them. Each run
  draws fresh keys and holds 4 prompts. The entry's set 1 ran at this PR's commit (`c4v*_entry.json`), set 2 at an
  earlier commit with the same code and plans (`c4v*_entry2.json`); record 3's runs are `c4v*_record3.json`.
- Against record 3: ×1.17 and ×1.13 the ratio of means for sets 1 and 2 (×1.15 over both), ×1.22 and ×1.07 the
  per-prompt geometric mean (95 % CI 1.06-1.40 and 0.92-1.25), higher on 28 and 23 of 48 prompts. One-sided t-test
  against 10⁻² at n = 48 on set 1: p 3.6e-148.
- Percentiles are linear interpolations over the 48 prompts.

## Disclosures

- **Keys and parameters.** Prefill-128 now runs on LeanChainDecode's chain `n32a4` (51 Q primes, dnum 4, P = record 3's
  9 special primes and 4 more, log QP 1734 against the 1747 bound; the sparse-secret keys at q_0 q_1 · P' with P'
  record 3's 9 special primes, 308 bits, record 1's modulus; LeanChainDecode's disclosures give the details).
  - **The Prefill-128 key request** has record 3's roles, its 147 rotation steps, its PCMM shape (`pcmm_t` 64) and
    its SPRU key (64 blocks), at the chain's shapes: the full keys at q_top 50, 4 digits, 13 special primes (record
    3: 53, 6, 9); the two sparse-secret keys at the bottom shape (q_top 1, 1 digit, 9 special primes), as on record
    3's chain. The keys hold 8,403.0 MB of GPU memory against record 3's 12,406.4 MB, and the server's memory plan is
    58.9 GB against 61.9 GB (server logs).
  - **The PCMM ring-switch keys** (the ternary secret on ring dimension 4096) keep record 3's modulus q_0 q_1 P:
    q_0 = 268042241 and q_1 = 265420801 are the same primes on both chains, and P = 2147352577, the 31-bit prime the
    key generation picks outside the chain's four lowest moduli, which `n32a4` shares with record 3's chain (87.0
    bits). The keys come from the same code and the same inputs.
  - **The client** is LeanChainDecode's `perseus_lean_client.py`, unchanged: it accepts this key request (chain
    `n32a4`, SPRU 64 blocks, no CCMM, the sparse-secret keys at the bottom over 9 special primes) and refuses the same
    variations as for Decode-8. This entry changes no client code and brings no client: the speedrun's
    `perseus_lean_client.py`, pinned to LeanChainDecode's commit, runs it (its dev runs did too).
- **PCMM precision on the lean chain** (pc-bench pcmm --prec on an H100, GPT-2's output linear, CF 6, 16-bit weights,
  band shifts 0-14): 13.8 bits at every shift on the shared-stamp path, 13.7-14.0 on the per-input path, against
  record 3's chain's 15.3 bits up to shift 9 and 14.7-10.6 bits at shifts 10-14 with input-scaled inputs (15.3 flat
  with scaled weights). Without the pre-division the lean chain loses about one bit per bit of band shift (4.9 bits
  at shift 11).
- **The pre-division** is exact (a multiply by 2^-k folded into the level's rescale, checked word for word in
  pc-bench) and costs one level at the inputs of the PCMMs of band shift above 1; the planner caps those inputs one
  level higher (42). 84 pre-divisions run per prefill, inside the clock.
  - The rule is the band shift, not the stamp: block 0's and block 1's MLP up-projections (band shift 1, stamps 2^21
    and 2^22) run undivided, at about 13.3 and 12.9 bits; 16 linears of band shift above 1 whose stamp is 2^20 or less
    have their inputs capped one level higher with nothing to divide. Neither shows in the range check (their finishes
    sit at 12.7× of their bands or more) or in the KL. A rule on the stamp is left for a later entry.
  - **Setup:** the PCMM inputs' own StC-first stages (`THIN_SPLIT_PCMM`, two stages) are built at setup, untimed and
    prompt-independent, next to the refreshes' own.
- **The softmax's drift margin.** With record 3's margin (4), the lean chain's plan refreshed a square of block 1's
  softmax normalisation (v_1402) at CF 2, which the range check put at 1.17× of its band on 80 C4 validation prompts
  (record 3 does not refresh it). The lean chain's plan sizes the softmax steps of blocks 0-10 with 6 (`SM_KAPPA`, the
  value the same check passed on record 3's chain). The rule was set before the check: every site at 1.5× or more on
  the 80 seen prompts, on prompts 32-79 and on prompts 128-175 of C4 validation shard 00000.
  - The held-out prompts 128-175 were captured by a job that ran 12:41-12:50 UTC on 2026-10-27, without a range
    check. The margin of 6 was committed at 12:42 UTC and the rule posted at 12:51 UTC; the first range check that
    read those prompts ran at 17:11 UTC, on the regenerated plan.
  - Result: every one of the plan's refresh sites passes: 1.57× on the 80 seen prompts (record 3: 1.57×), 2.26× on
    32-79 (2.39×) and 2.32× on the held-out 128-175 (2.40×); 0 sites under 1.5×. On 48 more prompts, 176-223, captured
    after the claim: 2.20× (record 3 2.40×).
- **Refresh pairs** on the lean chain pair members up to CF 12 (its StC-first reference CF 13 less one); record 3's
  chain is unchanged.
- **CutMax.** Stage 13 is planned on the ILP placer, as LeanChainDecode's: 8 refresh sites against record 3's 7, at
  other nodes (14 refreshes against 13), with no raise drop. Its one-hot mass on the GPU (CutMax's calibration prompts,
  C4 validation fit-shard 0-23, one row per prompt, each run with its own keys and client; four key sets per arm):

  | | key set | argmax(z) = argmax(logits) | mass ≥ 0.99 | mass < 0.5 | median mass |
  |---|---|---|---|---|---|
  | record 3 | 1 / 2 / 3 / 4 | 24/24 each | 22 / 22 / 22 / 22 | 0 each | 0.999987065 / 0.999987064 / 0.999987058 / 0.999987068 |
  | this entry | 1 / 2 / 3 / 4 | 24/24 each | 22 / 22 / 23 / 22 | 0 each | 0.999987071 / 0.999987066 / 0.999987067 / 0.999987069 |

  The mean median is 0.999987068 against 0.999987064 (+1.9 standard errors of the key-set noise). The one 23 is a
  prompt at the threshold (0.9875-0.9900 in every key set of both arms).
- **Regeneration:** from scratch, CPU only, by `PRESET=n32a4 bash speedrun/make_prefill_plan.sh` in a git clone with its
  own build, twice.
  - At an earlier commit of this entry with the same code and plans, the committed plans are that run's output byte
    for byte (68 files).
  - At this PR's commit, the rows' oracles and the captures included: every `graph.json` is byte-identical, and
    `speedrun/runner/plan_diff.py` finds 0 differing plan fields in the four plan directories; in each placement file
    only one `summary` field, the solver's time, differs. `PLAN_CMD.txt`, `pcmm_calib.json` and `plan_rot_steps.json`
    come out identical. The ILP solves of blocks 0-11 are all proven optimal.
  - Blocks 3 and 4 each take more than 25 minutes on one HiGHS thread (block 4: a 27-minute ILP and 23 minutes of
    shared-realize trials).
- **Memory:** peak GPU memory on Prefill-128 is 59.7 GiB against record 3's 62.6 GiB and LeanChainDecode's 63.3 GiB.
  In the server's memory plan the lean chain's keys (8.4 against 12.4 GB), plaintext store (7.4 against 8.9 GB) and
  SparseB tables (3.9 against 5.0 GB) are smaller, and the StC-first cache is larger (12.4 against 8.1 GB: the PCMM
  inputs' own stages, and their last stages per input level).
- **Numbers.** Everything here was measured on H100s, not on the reference GPU, the pc-bench precision included. The
  ablation quotes measured steps only.
