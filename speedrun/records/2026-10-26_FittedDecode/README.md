# Decode-8 fitted to its 16 positions, planned on merged captures, CutMax in 12 refreshes

Decode-8 on the records' code (FusedUnits and PackedPrefill), with:

- softmax and GELU approximations fitted to what Decode-8 computes: at most 16 keys, positions 0-15 (record 2 runs the
  general decode approximations, fitted to 1,024-token windows);
- a plan made by the repository's planner from a capture whose magnitudes are the maximum over four OpenWebText oracle
  rows at all 16 positions. The planner gains an exact ILP tie-break, raise drops searched per refresh site, and
  head-sum packs priced on their coefficient payload. The plan has 120 refreshes per token against record 2's 146;
- CutMax's normalisers scaled so that each outer iteration needs one refresh (stage 13: 12 refreshes, was 25);
- kernels and runtime:
  - EvalMod's relinearization and the rescale after it run as one exact conversion;
  - the composite rescale's correction is generated inside the NTT's column pass;
  - a plaintext product consumed by the add after it runs as one pass;
  - the captured graphs upload their constants once instead of copying them at every launch.

On one H100, Decode-8 runs at 3.471 s/token against record 2's 4.542 on the same GPU (×1.308, 6 pairs, p 2.8e-10) and
record 1's 5.291 (×1.524). The KL drops by about forty times. On 16 C4 validation prompts the mean is 8.0e-6, against
record 2's 3.1e-4; on 48 held-out C4 validation prompts never used for any choice it is 2.5e-6.

The same code also changes Prefill-128, which this entry does not claim: the prefill server replays its stages as CUDA
graphs captured on the public warm-up, CutMax's stage 13 takes 13 refreshes instead of 26, and the kernel changes apply;
on one H100 it took 20.75 s against record 2's 22.27 (6 pairs), with the KL at record 2's level (1.7e-5 on dev 0-11).
The Prefill-128 disclosures below give the details.

## Changes against record #2

- **Decode-8's approximations.**
  - The configs `configs/model/approximation/gpt2_base_n32_kc16` and `gpt2_base_n32_kc16_cm` (the latter with CutMax's
    normalisers), and a row in the configs `README.md`.
  - The fit: `scripts/utils/fit_gpt2_decode16.py`.
- **The planner.**
  - Code: `python/perseus/plan/placer/{__main__,emit,ilp,ir,planner}.py`, `python/perseus/plan/equiv.py`,
    `python/perseus/plan/run_bootstrap_all_blocks.sh`.
  - Its fields: `crates/pc-config/{src/fields.rs,src/lib.rs,tools/gen_fields.py}`, with `tests/legacy.rs` and
    `python/tests/test_fake_shadow.py`.
  - Tests: `python/tests/test_plan_{ilp,placer}.py`; the regeneration gate `python/tests/test_paper_plans.py`.
- **SparseB routes.** One route per slot count, its raise variants side by side: `crates/pc-py/src/session.rs`.
- **The decode capture** over four oracle rows and positions 0-15: `speedrun/make_decode_capture.sh`,
  `python/perseus/impl/magtape.py`, `python/perseus/examples/gpt2/{capture_mags.py,make_plan.sh}`.
- **The plan and what serves it.**
  - The plan `bootstrap_placements/gpt2_decode_python_n32_kc16_cm`, and `gpt2_decode_python_n32_kc16`, its blocks 0-12.
  - Its captures `graphs/gpt2_decode_python_n32_kc16{,m,_cm}`.
  - `speedrun/server.py` serves it.
  - The example decode default for 16 tokens: `python/perseus/examples/defaults.py`,
    `python/perseus/examples/gpt2/run_decode.py`.
  - Gates: `crates/pc-program/tests/shipped_plans.rs`, `python/tests/test_example_defaults.py`.
- **CutMax's scaled normalisers.**
  - Code: `python/perseus/examples/gpt2/{head,ref,capture_mags}.py`, `python/perseus/impl/{poly,config}.py`.
  - The fit: `python/perseus/calibrate/approximations/cutmax.py`, its driver `scripts/utils/fit_cutmax_scaled.py`, and
    the calibration windows `scripts/utils/cutmax_rows.py`.
  - The configs `gpt2_base_n32_cm` and `gpt2_base_n32_pf_cm`; stage 13 by `speedrun/make_cutmax_plan.sh`.
  - Tests in `python/tests/test_gpt2_primitives_cpu.py`.
  - The plans `gpt2_decode_python_n32_m16i_fcf_cm` (record 2's plan with this stage 13, and the example decode default
    at other lengths) and `gpt2_prefill128_pf_cm` (Prefill-128's), with their captures `graphs/*_cm`.
- **EvalMod relinearization + rescale as one exact conversion.** `PC_EM_FUSE_RESCALE=1` is on the plans' `runtime=`
  lines, and on `speedrun/make_prefill_plan.sh`'s.
  - Files: `crates/pc-gpu/src/{bts/evalmod,ops,relin}.rs`, `crates/pc-kernels/kernels/{args,em}.cuh`,
    `crates/pc-kernels/src/{args.rs,bin/selftest.rs}`, `crates/pc-tools/src/{bts_all,bts_stc_first}.rs`.
  - `crates/pc-config`: the kernel switches are marked `[kernel]` and left out of the plan-contract hash.
- **The rescale's correction inside the NTT's column pass** (`rescale_fwd`):
  `crates/pc-kernels/kernels/{args,ntt,rescale}.cuh`, `crates/pc-kernels/kernels/inst/instantiate.inc`,
  `crates/pc-kernels/src/`, `crates/pc-gpu/src/{rescale,ntt,bts/evalmod}.rs`, `crates/pc-gpu/tests/rescale_oop.rs`.
- **Fused plaintext product + add units:** `crates/pc-program/src/{bind,program,replay,schedule}.rs`,
  `crates/pc-program/tests/units.rs`.
- **Graph constants uploaded once:** `crates/pc-gpu/src/{ops,graph}.rs`, `crates/pc-program/src/replay.rs`.
- **Prefill-128's server** replays its stages as CUDA graphs captured on the public warm-up: `speedrun/server.py`, plus
  a comment in `python/perseus/impl/prefill.py`.

## Ablation

H100 80GB GPUs, the speedrun harness, two prompts per run, 6 pairs on dev prompts 0-11.

- The first five rows are the steps of our integration branch, in order. Each was timed against the step before it, on
  one GPU per pair; the nodes vary between rows.
- The sixth row is the plan's A/B against that branch's head.
- The last two rows are this PR's commit.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Fused plaintext product + add | decode | 4.569 → 4.524 s/token, ×1.010 (p 0.0014); outputs byte-identical |
| CutMax's scaled normalisers (record 2's plan otherwise) | decode | 4.538 → 4.310 s/token, ×1.053 (p 1.1e-6); logits byte-identical |
| EvalMod relinearization + rescale as one conversion | decode | 4.324 → 4.277 s/token, ×1.011 (p 0.0052); same-keys C4 validation KL ×0.84 (95 % CI 0.64-1.09) |
| Rescale correction inside the NTT | decode | 4.280 → 4.268 s/token, ×1.003, a 0.3 % gain (p 0.11, not resolved); outputs byte-identical |
| Graph constants uploaded once | decode | 4.261 → 4.210 s/token, ×1.012 (p 0.0065); outputs byte-identical |
| The approximations fitted to 16 positions, the planner and the merged capture (this plan) | decode | 4.203 → 3.48 s/token, ×1.208 (p 2.2e-9) |
| **This entry** vs record 2 | decode | **4.542 → 3.471 s/token, ×1.308 (p 2.8e-10)** |
| This entry vs record 1 | decode | 5.291 → 3.471 s/token, ×1.524 (p 2.5e-10) |

## Dev result

One H100 80GB HBM3, the speedrun harness at 449f7ba's harness files. 6 rounds in rotated order (record 1, record 2,
this entry), on dev prompts 0-11, two prompts per run, timed at this PR's commit. Record 2's arm is FusedUnits' code.
Logs are in `dev_logs/`: `ab_*` for the A/B, plus `c4v*` and `dev12-15`.

| Track | Score | Record 2, same GPU | KL (positions 8-15) | Argmax (scored) | Peak GPU memory |
|---|---|---|---|---|---|
| Decode-8 | 3.471 s/token | 4.542 s/token | 4.2e-6 | 96/96 | 56.5 GiB (record 2 52.8) |

score.py: quality PASS (mean KL 4.15e-6, p 6.1e-30), speed ×1.308 (p 2.8e-10).

| Prompts | Arm | Mean | Median | p90 | Max | Above 10⁻³ | Argmax (scored) |
|---|---|---|---|---|---|---|---|
| dev 0-15 | entry | 3.9e-6 | 3.3e-6 | 8.1e-6 | 8.6e-6 | 0/16 | 128/128 |
| dev 0-11 | record 2 (the A/B's record arm) | 1.5e-4 | 1.2e-4 | 1.9e-4 | 6.8e-4 | 0/12 | 96/96 |
| C4 validation 0-15 | entry | 8.0e-6 | 5.3e-6 | 1.6e-5 | 2.4e-5 | 0/16 | 128/128 |
| C4 validation 0-15 | record 2 | 3.1e-4 | 1.6e-4 | 4.8e-4 | 2.4e-3 | 1/16 | 128/128 |
| C4 validation 32-79 (held out) | entry | 2.5e-6 | 1.8e-6 | 3.8e-6 | 1.3e-5 | 0/48 | 384/384 |

- The C4 validation prompts are 16 rows of the C4 validation shard 00000. No fit used them, but during this work they
  compared eight candidate variants of the approximations and plans, so they are not a clean held-out set for this
  plan. Prompts 32-79 of the same shard are: no fit, choice or diagnosis used them. They ran on this PR's tree before
  its last two commits, which serve the same bytes (`c4v32-39_entry.json` to `c4v72-79_entry.json`).
- Prompt 2 was inspected in a diagnosis (the head-sum CF). Without it, the means are 7.0e-6 for the entry and 1.8e-4
  for record 2.
- Record 2's C4 validation run is its own, from its record folder.
- Dev prompts 12-15 come from one more entry run (`dev12-15_entry.json`).

## Disclosures

- **Approximations (Decode-8 only).** `configs/model/approximation/gpt2_base_n32_kc16` is fitted to Decode-8's
  geometry: at most 16 keys, positions 0-15. The fit is `scripts/utils/fit_gpt2_decode16.py`, from public data with
  fixed seeds.
  - **Softmax, every layer.** The score range over the fit windows' rows (all cache lengths 1-16), widened by 15 % of
    its width above and 5 % below. THOR's delta2 / delta1 / degree-8 exp series follow from that range. The reciprocal
    rounds are Chebyshev-scaled Goldschmidt:
    - round 0's band covers three times the fit rows' head sums;
    - the refine rounds' bands are exact for at most 16 keys;
    - the round counts come from the design rule, with layer 7's counts picked among equal-target splits by the
      planner's refresh count.
  - **GELU, every block.** X is 1.3 times the largest |input| over positions 0-15 of the fit windows. Each block uses
    one Chebyshev series of degree 63 to 255.
  - **Unchanged:** LayerNorm. CutMax's scaled normalisers and the EvalMod conversion's one rounding are the other two
    approximation changes against record 2 (below); every other change is byte-identical.
  - **Validity:** at most 16 keys. Other lengths keep the general decode plan (`perseus/examples/defaults.py`).
- **Data.**
  - **Fit:** 16-token windows of OpenWebText (parquet shard 0042) and of C4 *validation* shard 00001, through float64
    GPT-2.
  - **Never used for fitting:** the dev prompts, C4 validation shard 00000 (measurement only) or the hidden set.
  - **What looked at what:**
    - before the first commit, a float64 mirror ran once on the dev prompts' first 16 tokens, to show that record 1's
      decode KL is refresh noise; no parameter came from it;
    - C4 validation shard 00000 was inspected when the GELU safety factor 1.3 was picked, as a margin check;
    - dev prompts 0, 2 and 3 were looked at in diagnostics after the commits, and nothing changed after them.
- **Robustness** (the CPU twin, the shipped configs, 16,384 held-out document starts of C4 validation shard 00000):
  - GELU's margin is the thinnest spot: its input stays at most 0.799 of X (0.785 over all 50,257 first tokens); for
    record 2 it is 0.45-0.56; a GELU input past X would break the series;
  - the softmax scores stay at most 0.784 of the range's top; on 3 of the 16,384 prompts (layer 9) a score passes the
    range's low side, up to 1.157 of the clip, with ordinary KL;
  - the Goldschmidt reciprocal's residual stays at most 0.28.
- **The plan's capture.** The magnitudes are the per-node maximum over four rows of the repository's OpenWebText
  calibration pool, at positions 0-15. The rows are at offsets 4096, 8192, 12288 and 16384 (the script's `ROWS`).
  Position 1 keeps the capture's statistics for its K/V push.
  - The rows follow a fixed rule; none was chosen after looking at a result.
  - Some refresh inputs grow with the position: block 2's q·K^T sum goes from 0.045 at position 0 to 13.4 at
    position 15. That is why record 2's position-0 capture is not used.
- **Range check.** Every refresh site was checked at every position 0-15, on 64 C4 validation fit-shard prompts, the 16
  C4 validation prompts 0-15, 2,048 random pairs and all 50,257 first tokens. Every input stays inside its CF band; the
  smallest margin is 1.96× (block 13, v_25).
- **Regeneration.** From scratch, CPU only, in a copy of the tree:
  - the position-0 capture, the merged capture and stage 13's capture come out byte-identical;
  - `kc16_cm`'s 15 placement files are equal to the committed ones in every field but the solver's timings
    (`perseus.plan.equiv`); the planner chain takes about 50 minutes on 32 cores (one HiGHS thread);
  - `gpt2_decode_python_n32_m16i_fcf_cm` and `gpt2_prefill128_pf_cm` regenerate file for file through
    `speedrun/make_cutmax_plan.sh`.
  - The repository's regeneration gate (`python/tests/test_paper_plans.py`, every GATE plan whose graph is in the
    repository) at this commit:
    - `gpt2_decode_python_n32_kc16` (blocks, argmax stage, feedback) and 13 more plans regenerate file for file;
    - `gpt2_prefill128` differs in its Config stamp only, as on the base;
    - record 2's decode plans `m16i` and `m16i_fcf` are retired from the gate, as PackedPrefill disclosed: they were
      made with the planner before PackedPrefill's and regenerate with that one (f19831a);
    - the Llama plans' graphs are not in the repository.
- **Planner.**
  - The exact ILP tie-break (`PLAN_ILP_TIEBREAK`) runs on one HiGHS thread. Both solves are proven optimal, or there
    is no plan.
  - Head-sum packs are priced on their coefficient payload (`PLAN_PACK_CF`), only on routes whose EvalMod sees
    coefficients.
  - The argmax stage is planned on the min-cut (`PLAN_ARGMAX_MINCUT`).
  - Kept placements (`PLAN_KEEP_PLACEMENTS`) are an option no shipped plan sets.
  - Which GATE plans regenerate with it: see Regeneration above.
- **Raise drops.**
  - SPRU drops up to 12 at 9 sites: equal to drop 0 within ±0.4 bits over 5 seeds.
  - SparseB drops 1-5: the variants of one slot count are one route (`FIDESLIB_BTS_RAISE_DROPS=1,2,3,4,5`). Each
    variant reserves its own precomputation, about 1.2 GB each.
- **Head-sum packs.** The packed head-sum refresh carries the sum on its imaginary part; its CF used to be sized for the
  real payload. Pricing it on the imaginary payload lowers the decode KL by an order of magnitude at equal speed. The
  head-sum sites are part of the range check above.
- **Block 12.** The planner places block 12's one refresh after its rotate_sum, at CF 20. It rates that site like the
  CF-10 site after the product (equal predicted error). A per-block run of the same recipe chose the CF-10 site: on
  the same 16 C4 validation prompts its KL was about half (×2.1, 16/16 prompts). This PR ships what the recipe
  regenerates.
- **CutMax.**
  - Stage 13 runs one refresh per outer iteration: 25 → 12 refreshes per token.
  - **Outputs:** the logits frame is unchanged. When CutMax alone was added to record 2's plan, the logits were
    byte-identical on one key bundle and identical inputs. The z frame differs. Its argmax agrees with the decrypted
    logits' argmax on every scored position above (128/128 on dev 0-15, 128/128 on C4 validation 0-15).
  - **The fit:** bands fitted by `scripts/utils/fit_cutmax_scaled.py` on 19,454 logit rows of the CPU twin, with
    record 1's decode approximations:
    - positions 8-15 of 16-token windows of OpenWebText shard 0042 and of C4 validation shard 00001;
    - positions 0-15 of 256 more windows of each;
    - position 127 of 1,024 shard-00001 prompts.
  - **Stage 13's magnitudes:** taken over the 275 windows of `graphs/gpt2_decode_python_n32_cm/cutmax_rows.json`,
    which include the four oracle rows at positions 0-15. No dev prompt was used.
- **EvalMod relinearization + rescale as one exact conversion.**
  - A Regen relinearization followed by the program's rescale runs as one centred exact conversion.
  - It gives the same value up to one rounding, and is as accurate as record 2's relinearize + rescale: RMS error ratio
    1.000 on the StC-first, SparseB-512 and SparseB-256 routes.
  - It is not byte-identical.
  - It supersedes FusedUnits' "unused" note about this switch.
- **Contract hash.** The plan-contract hash no longer covers the perseus-core kernel switches (`bts.em_fuse_rescale`,
  `bts.lt_tails`, marked `[kernel]`). They change neither the keys, nor the op stream, nor its levels.
- **Key request.** The Decode-8 key request differs from record 2's in one field only, the plan-contract hash: this
  plan's load-bearing Config differs (the fitted approximations, the raise drops). Every requested key (roles, shapes,
  the SPRU and PCMM entries, the public key) is the same, so the key set and its security are record 2's; a key
  bundle made for record 2's plan is refused by this server, as it must be. The Prefill-128 key request is
  byte-identical to record 2's.
- **Rescale correction in the NTT, fused product + add, graph constants.**
  - All three are byte-identical; the savings are memory traffic, launches and copies.
  - The graphs' constants: 3,907 4-KB copies per decode position become one upload inside the existing position-0
    capture (about 12 ms). The entry's time ratio is the same over the unscored positions 0-7 and the scored 8-15.
  - `Ops::mult_pt_add` is now wired into the runtime.
- **Setup.** Record 2's precompute of every position's patch (about 25 s), plus the raise variants' tables. Session
  setup is 18.3 s against 15.5 s (server log).
- **Keys and parameters.**
  - Record 2's chain (log N 16, log QP 1706), secrets, key roles and formats, and pinned client.
  - The key request asks for the same 124 switching keys and 87 rotation keys, which hold 10,008.3 MB of GPU memory
    (server log).
- **Memory.** Peak GPU memory on Decode-8 is 56.5 GiB against record 2's 52.8, above the 48 GB the speedrun README
  names for the current record. The server's memory plan (56.9 GB
  against 53.1 GB) has four more SparseB raise variants (about 1.2 GB each) and a larger dense route (6.2 GB against
  3.3 GB). A smaller plaintext store (14.9 GB against 17.8 GB) and StC-first cache partly offset them.
- **Prefill-128.** This PR changes Prefill-128 too:
  - **CUDA graphs in the prefill server.** The warm-up on the public seeded prompt (untimed setup, before `out/ready`)
    captures the 14 stage programs as CUDA graphs; each request replays them on its own ciphertexts. Outputs are
    byte-identical to eager execution. Setup grows by about 7.5 s (2.5 s of session setup for the larger constant
    rings, 4.9 s of warm-up and capture), teardown by about 1.5 s; peak GPU memory by 2.7 GiB (0.5 GB of constant
    slots in the arena, 2.3 GiB of instantiated graphs); host memory 36.0 vs 32.5 GiB. These figures come from the
    server's own log, `run.json`'s `peak_gpu_mib` and the dev runner's host-memory record, not from the harness's
    score. An 8-request session ran every request at the same speed with `outputs ok`.
  - **CutMax on Prefill-128.** Stage 13 of `gpt2_prefill128_pf_cm` = `gpt2_prefill128_pf`'s stages 0-12 byte for byte
    and CutMax's scaled normalisers (13 refreshes, was 26); the logits frame is unchanged; KL-neutral by construction
    (its bands fitted as for Decode-8, including the exact logits at position 127 of 1,024 C4 validation shard-00001
    prompts).
  - **Kernels.** EvalMod relinearization + rescale as one conversion changes the prefill's outputs within rounding
    (same keys, 16 C4 validation prompts: KL ratio ×0.95, CI 0.85-1.07); the rest is byte-identical.
  - **Memory.** Peak GPU memory on Prefill-128 is 66.8 GiB against record 2's 65.5 (the graphs, partly offset by
    CutMax's smaller stage 13).
  - **Keys.** The prefill key request is byte-identical to record 2's (173 rotation keys, 16,458.4 MB of GPU memory,
    server log).
- **Numbers.** Everything here was measured on H100s, not on the reference GPU. The records' own transfers were
  ×1.160 → ×1.123 (decode) and ×1.616 → ×1.471 (prefill).
