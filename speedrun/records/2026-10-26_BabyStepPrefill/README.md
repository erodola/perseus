# Prefill-128 with baby-step attention, refresh pairs and searched raise drops

Prefill-128 on FittedDecode's code ([#11](https://github.com/gladia-research-group/perseus/pull/11)), with:

- attention:
  - lanes rotate by baby step and giant step, so a block's attention needs fewer rotations and the plan fewer
    rotation keys (116 against 173);
  - a half-empty head group runs one softmax for both of its query chunks;
- a plan made by the repository's planner from a capture whose magnitudes are the maximum over four OpenWebText
  oracle rows. The softmax steps of blocks 0-10 and the last position's sites are sized with their own drift margins.
  The plan has 460 refreshes against record 2's 516;
- refreshes searched for raise drops on the plan's own program: 88 refreshes are raised below the chain top;
- sibling refreshes run in pairs: two real StC-first refreshes as one complex refresh, 29 pairs per prefill;
- SparseB refreshes on the dense route's shorter EvalMod series (K = 16).

On one H100, Prefill-128 takes 17.64 s against record 2's 22.31 on the same GPU (×1.265, 6 pairs, p 3e-11), and against
FittedDecode's 20.81 (×1.180). The KL stays three orders of magnitude under the bar: 1.2e-5 on dev prompts 0-11
(record 2 1.7e-5), and 1.6e-5 on 48 C4 validation prompts no fit used (record 2 9.3e-6 on the same prompts).

## Changes against FittedDecode

- **Attention.** Baby-step giant-step lane rotations in the packed attention (`PF_ATTN_BSGS`), and one softmax for both
  query chunks of a half-empty head group (`PF_SM_HPACK`).
  - Files: `python/perseus/impl/{blocked,prefill}.py`.
  - Tests: `python/tests/test_prefill_last_cpu.py`, float64 mirrors and a padding-leak test.
- **Refresh pairs** (`PC_BTS_PAIR`).
  - `crates/pc-gpu/src/bts/{evalmod,mod,stc_first}.rs`: `run_pair`.
  - `crates/pc-program/src/{bind,compile/batch,compile/mod,program,replay}.rs`: pairs matched at compile and issued as
    one at replay.
  - `crates/pc-py/src/{program,session}.rs`.
  - Tests: `crates/pc-program/tests/units.rs`, and `pc-bench stcf-pair`
    (`crates/pc-tools/src/{bts_stc_first,bin/pc-bench}.rs`).
- **SparseB's EvalMod series length** (`FIDESLIB_BTS_SPARSE_B_K`, default 24, the plan's 16):
  `crates/pc-gpu/src/bts/sparse_b.rs`.
- **The planner.**
  - Options: a step's own drift margin (`PLAN_MAG_SAFETY_STEPS`); a stopped chain resumed (`PLAN_RESUME`); the exact
    keep switching off the re-cut passes and the prune through its config.
  - Files: `python/perseus/plan/placer/{planner,place,refresh}.py`.
  - Tests: `python/tests/test_plan_placer.py`.
- **Fields.** The new fields are in `crates/pc-config` (with `tests/legacy.rs` and `python/tests/test_fake_shadow.py`).
- **The plans.**
  - `bootstrap_placements/gpt2_prefill128_pfbh` and `gpt2_prefill128_pfbh_cm`: the program.
  - `gpt2_prefill128_pfbhrd` and `gpt2_prefill128_pfbhrd_cm`: its raise drops; `pfbhrd_cm` is the served plan.
  - Captures: `graphs/gpt2_prefill128_pfbh{,_cm}`.
  - Recipe: `speedrun/make_prefill_plan.sh`. It captures over four rows, plans, applies CutMax's stage 13 and searches
    the drops. `RESUME=1` continues a stopped run.
  - `speedrun/make_cutmax_plan.sh` takes the capture's directory. `speedrun/server.py` serves `pfbhrd_cm`.

## Ablation

H100 80GB GPUs, the speedrun harness, two prompts per run, 6 pairs on dev prompts 0-11. Each row was timed against the
row above it, on one GPU per pair; the nodes vary between rows.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Baby-step attention, shared softmax, the merged-capture plan with its margins (460 refreshes) | prefill | 20.79 → 18.75 s, ×1.109 (p 4.6e-12) |
| … of which the plan alone (same code, the single-row plan of the same 460 refreshes against this one) | prefill | 19.01 → 18.69 s, ×1.017 (p 1.1e-4) |
| + raise drops, refresh pairs, SparseB on K = 16 | prefill | 19.15 → 17.97 s, ×1.066 (p 3.2e-8) |
| **This entry** vs record 2 | prefill | **22.31 → 17.64 s, ×1.265 (p 3e-11)** |
| This entry vs FittedDecode | prefill | 20.81 → 17.64 s, ×1.180 (p 1.3e-11) |

## Dev result

One H100 80GB HBM3, the speedrun harness at 449f7ba's harness files. 6 rounds in rotated order (record 2,
FittedDecode, this entry), on dev prompts 0-11, two prompts per run, timed at this PR's commit. Logs are in
`dev_logs/`.

| Track | Score | Record 2, same GPU | KL (position 127) | Argmax | Peak GPU memory |
|---|---|---|---|---|---|
| Prefill-128 | 17.64 s | 22.31 s | 1.2e-5 | 12/12 | 62.6 GiB (record 2 65.5) |

score.py: quality PASS (mean KL 1.16e-5, p 4.8e-34), speed ×1.265 (p 3e-11).

| Prompts | Arm | Mean | Median | p90 | Max | Above 10⁻² | Argmax |
|---|---|---|---|---|---|---|---|
| dev 0-11 | entry | 1.2e-5 | 8.5e-6 | 3.2e-5 | 4.0e-5 | 0/12 | 12/12 |
| dev 0-11 | record 2 (the A/B's record arm) | 1.7e-5 | 9.2e-6 | 3.8e-5 | 7.6e-5 | 0/12 | 12/12 |
| C4 validation 0-15 | entry | 9.7e-6 | 8.0e-6 | 1.6e-5 | 3.8e-5 | 0/16 | 16/16 |
| C4 validation 0-15 | record 2 | 7.0e-6 | 5.8e-6 | 1.4e-5 | 1.5e-5 | 0/16 | 16/16 |
| C4 validation 32-79 | entry | 1.6e-5 | 9.6e-6 | 3.1e-5 | 2.0e-4 | 0/48 | 48/48 |
| C4 validation 32-79 | record 2 | 9.3e-6 | 7.9e-6 | 2.0e-5 | 4.3e-5 | 0/48 | 48/48 |

- The C4 validation prompts are rows of the C4 validation shard 00000.
- Prompts 0-15 entered the choice of the drift margins: their CPU-twin ranges, together with C4 validation fit-shard
  prompts.
- Prompts 32-79 entered no fit. They did decide, pass or fail, between the two declared values of the softmax margin
  (Disclosures).
- On 32-79 the entry's KL is higher than record 2's, ×1.4 per prompt (95 % CI 1.16-1.69, higher on 33 of 48). Prompt
  66 accounts for the largest value, 2.0e-4 against record 2's 1.2e-5 on it.
- The approximation changes behind the rise are SparseB's shorter series and the margins' higher CFs (Disclosures).

## Disclosures

- **Baby-step attention.**
  - The score block order within a score ciphertext is permuted (block a·W + 2g + p ↔ key offset 2a + p + 16g), and
    every validity mask follows it.
  - Chunk 1's P·V sums take a masked cyclic lane shift, one more level on that output.
  - Rotation keys: 116 against 173. 15 live rotated K/V copies instead of 63.
- **Shared softmax.** The half-empty head group's two query chunks share one softmax: chunk 1's scores run on the
  group's empty heads, with per-half constants. Chunk 0's P·V padding heads then carry chunk 1's probabilities, which
  reach only out_proj's zero rows; `test_padding_heads_do_not_leak` checks this with garbage on every padding head.
- **Exactness.** Both attention changes are exact in exact arithmetic. Under CKKS they give another op order and
  another noise (the KL above).
- **The plan's capture.** The magnitudes and PCMM band shifts are the maximum over rows 4096, 8192, 12288 and 16384 of
  the repository's OpenWebText calibration pool. That is a fixed rule, as for FittedDecode's decode capture.
- **Drift margins.** Two sets of sites see one value per capture row where the run sees others: block 11's last-query
  attention and the tail, and the softmax steps whose rounds square their input. They are sized with their own
  safety factors (`PLAN_MAG_SAFETY_STEPS`): 8 at block 11's attention and the tail, 4 at the softmax steps of blocks
  0-10.
  - Two values of the softmax factor, 4 and 6, were sized on the CPU-twin range of 80 C4 validation prompts (fit
    shard 0-63, prompts 0-15 of shard 00000). Both passed the range check on the 48 prompts 32-79. 4 was taken because
    it raises the fewest CFs.
  - The plan without the softmax factor fails the range check (1.36×), and its per-site list on 32-79 was printed in
    job logs. Nothing was chosen on it: both candidates were already running.
  - No dev prompt was used.
  - A step name matches as a prefix: `blk<b>.attn.softmax` also covers `blk<b>.attn.softmax_v`. No refresh of this plan
    sits there.
- **The thinnest site** is block 11's PCMM finish of the MLP's down projection, at 1.57× below its band top on the
  seen 80 prompts.
  - Block 11's PCMM band shifts come from four last-position values, one per calibration row, so this class sits at the
    bottom of the range check.
  - On 896 further C4 validation prompts (128-1023, used by no gate or choice), float64 GPT-2 puts its largest input at
    207.5 against the band top of 491.5. The largest on any of the 1,024 prompts is 313.4 (prompt 4 of the seen set).
  - No fix is applied in this PR. Four times the calibration (two more shift bits, from the one-value-per-row argument
    alone) would widen the margin, at about 2 bits of precision at those finishes.
- **Raise drops.** The served plan keeps its program's placements, fired hints and realize anchors exactly
  (`PLAN_KEEP_PLACEMENTS_EXACT`). It raises 88 refreshes below the chain top, by the planner's search over the dense,
  SparseB-512 and SPRU routes (drops 1-5 on SparseB's base drop 1).
  - The dense raise variants 2 and 4 are new: record 2 builds drops 1, 3 and 5 only.
  - The drops have no timing of their own. The ablation row times drops, pairs and K = 16 together.
  - They cost memory: the dense route's tables grow from 3.3 to 7.7 GB and the StC-first cache from 6.4 to 8.1 GB
    (server logs). Together with the pairs' slab, peak GPU memory is 10.6 GiB above the same plan without them
    (62.9 against 52.3 GiB).
- **Refresh pairs.**
  - **Arithmetic:** per member, the member's own single real refresh, plus one conjugation key switch at the entry.
    Measured precision per member against its single: +0.000 bits (pc-bench stcf-pair, 16 ordered pairs × 4 seeds).
  - **Not bit-identical:** pairing changes the order of refreshes, so which refresh fills a StC-first stage-cache entry
    changes, at the 1e-13 to 1.5e-12 relative level.
  - **Setup:** pairs are matched at compile from the plan alone. The replay's only choice, one pair or two singles,
    reads public metadata.
  - **On this plan:** 29 pairs per forward (the warm-up's log).
- **SparseB on K = 16** is an approximation change. The prefill's SparseB refreshes use the dense route's K = 16,
  degree-14 EvalMod series instead of K = 24, degree 20: two relinearized products less per refresh, same depth,
  landings and placements.
  - The noise-free approximation floor is 2^-29.6 against 2^-35.3 of q0. It is absolute, so it costs relative precision
    only at the bottom of a site's band.
  - The overflow-integer range is the dense route's.
- **Key request.** `bts.pair`, `bts.sparse_b_k` and the attention switches are load-bearing, so the key request's
  plan-contract hash changes. The keys are the main secret's rotation keys at the same modulus: 116 rotation keys,
  12,406.4 MB of GPU memory against record 2's 173 and 16,458.4 MB (server logs). No new secret; the chain and the
  pinned client are record 2's. The decode key request is FittedDecode's (unset fields are not serialized).
- **Regeneration.** `speedrun/make_prefill_plan.sh` from scratch in a clone of this PR's commit, CPU only, about two
  hours on 32 cores: the four rows' oracles, their PCMM calibrations and captures, the merged capture, then pfbh,
  pfbh_cm, pfbhrd and pfbhrd_cm. The job was preempted twice and continued with `RESUME=1`.
  - All 68 plan files come out with 0 plan fields different, and `pcmm_calib.json`, `plan_rot_steps.json` and
    `PLAN_CMD.txt` come out byte-identical.
  - The capture files differ only where they embed the temporary directory names (two config stamps) and where four
    mask plaintext names per block embed a Python object id. Every value and level is equal.
  - `PLAN_CMD.txt`'s planner line names the planner's last commit in seven characters.
- **Plan directories in this PR that the server does not serve:**
  - `gpt2_prefill128_pfbh`: the program; the drop pass's base;
  - `gpt2_prefill128_pfbh_cm`: the program with CutMax's stage 13, without drops;
  - `gpt2_prefill128_pfbhrd`: the drops without CutMax's stage 13; `pfbhrd_cm`'s base.
  `make_prefill_plan.sh` writes all four; their capture is shared.
- **Graphs and pairs at run time.** Every entry run's server log has the warm-up line `refresh pairs (as one, one by
  one) (29, 0)`. None of the 25 server logs of the A/B and the C4 validation runs reports a stage-cache eviction while
  graphs exist.
- **Memory.** Peak GPU memory on Prefill-128 is 62.6 GiB against record 2's 65.5 and FittedDecode's 66.8. The
  attention changes save more than the drops and pairs cost.
- **Setup.** Session setup is 22.9 s and total setup 27.2 s (server log), plus the warm-up's graph capture.
- **Numbers.** Everything here was measured on H100s, not on the reference GPU. Part of the pairs' gain replaces
  batches whose cost differs on the reference.
