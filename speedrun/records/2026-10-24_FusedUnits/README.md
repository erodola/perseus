# Fused key switching, runtime units and the shipped order's CF

Record 1's decode, with its arithmetic unchanged except for five constants, run faster:

- key switching moves less memory: the ModDown and the rescale fold their NTT epilogues, a rotate-and-sum group sums
  inside the ModDown, the NTT butterflies reduce lazily, and the rotate_sum and CtS giant steps lose their copies;
- the planned runtime issues the batches the compiler forms as units: rotation grids and chains that share one
  ModUp, K-batched rotate_sums, products and refreshes;
- the served decode's per-position patch is built once at setup, from public data only, instead of being recorded
  on the CPU twin inside each position's clock.

All three leave the outputs byte-identical to record 1's with record 1's plan. The plan change is separate. Six
refreshes of the decode plan have their input past the StC-first entry, so the runtime runs them in the shipped order
(CtS, then EvalMod on the plaintext coefficients). Five of them had a CF sized for the StC-first order's slot
magnitudes, and the planner now gives them the shipped order's CF. That changes no level, route or landing, and the
speed is unchanged (×1.000, 3 pairs). It lowers the decode KL: on 16 held-out C4 validation prompts, one run each, the
mean KL is 3.1e-4 against the record's 7.3e-4, with 1 prompt of 16 above 10⁻³ against 2.

On one H100, Decode-8 runs at 4.554 s/token against record 1's 5.281 (×1.160, 6 pairs).

## Changes against record #1

- key-switching kernels and their callers: `crates/pc-gpu/src/{ks,ntt,ops,relin,rescale,rotate,context}.rs`,
  `crates/pc-gpu/src/bts/{cts,dense,evalmod,mod,raise,sparse,sparse_b,stc_first}.rs`, `crates/pc-kernels/kernels/`,
  `crates/pc-kernels/src/` (with the kernel selftest), `crates/pc-gpu/tests/mem_plan.rs`,
  `crates/pc-gpu/benches/ks.rs`; two runtime switches with their fields in `crates/pc-config` (`PC_LT_TAILS`, on;
  `PC_EM_FUSE_RESCALE`, off)
- runtime units: `crates/pc-program/src/{bind,compile/batch,program,replay,schedule}.rs`,
  `crates/pc-py/src/{program,session}.rs`, `crates/pc-gpu/src/bts/batch.rs`, `python/perseus/impl/prefill.py` (the
  siblings schedule), tests in `crates/pc-gpu/tests/rotate_grid.rs` and `crates/pc-program/tests/units.rs`
- the StC-first stage cache and the constant ring sized from the compiled programs: `crates/pc-py/src/session.rs`,
  `crates/pc-gpu/src/bts/stc_first.rs`
- the served decode's patch at setup: `python/perseus/impl/driver.py` (`PlannedRun.precompute`),
  `python/perseus/serve/gpt2_decode.py`
- raise variants carry their raise drop (a gain error at drops 2-5, which no shipped plan uses; SPRU gets drop
  variants a plan can select): `crates/pc-gpu/src/bts/{sparse,sparse_b,spru,stc_first}.rs`,
  `crates/pc-shadow/src/route.rs`
- tools that follow the API changes: `crates/pc-tools/src/{bts_sparse,bts_stc_first,replay_g0}.rs`,
  `crates/pc-tools/src/bin/pc-check/check_keys.rs`, `crates/pc-config/tools/gen_fields.py`
- the planner option `PLAN_FALLBACK_CF`: `python/perseus/plan/placer/{__main__,emit,planner}.py`,
  `python/perseus/plan/run_bootstrap_all_blocks.sh`, its field in `crates/pc-config/src/fields.rs`
- the decode plan `bootstrap_placements/gpt2_decode_python_n32_m16i_fcf` = m16i's recipe + `PLAN_FALLBACK_CF=1`, the
  default in `python/perseus/examples/defaults.py`; gated in `python/tests/test_paper_plans.py` (regeneration),
  `python/tests/test_example_defaults.py` and `crates/pc-program/tests/shipped_plans.rs` (compile, and each
  shipped-order site still past the entry)

## Ablation

H100 80GB GPUs, the speedrun harness, record 1 (89bc830) as the record arm. Each row is its own A/B against the
record on one node, on a tree with that change alone: 6 pairs in the official order on dev prompts 0-11, two prompts
per run, unless the row says otherwise.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Raise variants carry their drop | both | not timed; outputs bit-identical at drops 0-1 |
| Served decode's patch at setup | decode | 5.282 → 5.113 s/token, ×1.033 (p 4.2e-6) |
| Key-switching kernels | decode | 5.284 → 4.747 s/token, ×1.113 (p 1e-10) |
| Runtime units, with the stage cache sized to the programs | prefill | 36.07 → 31.36 s, ×1.150 (p 3.1e-10); both arms' KL is record 1's (bit-identical outputs), so score.py prints quality FAIL for both; decode not timed alone |
| Shipped order's CF (plan) | decode | 5.295 → 5.294 s/token, ×1.000 (3 pairs, dev 0-5, p 0.34) |
| **This entry** | decode | **5.281 → 4.554 s/token, ×1.160 (p 1e-8)** |

## Dev result

One H100 80GB HBM3, the speedrun harness at 449f7ba, record 1 (89bc830) as the record arm, 6 pairs in the official
order (AB BA BA AB AB BA) on dev prompts 0-11, two prompts per run, timed at d4d86a2. Logs in `dev_logs/` (`ab_*`).

| Track | Score | Record 1, same GPU | KL (positions 8-15) | Argmax (scored) | Peak GPU memory |
|---|---|---|---|---|---|
| Decode-8 | 4.554 s/token | 5.281 s/token | 1.7e-4 | 96/96 | 52.8 GiB (record 47.8) |

score.py: quality PASS (mean KL 1.67e-4, p 2.8e-10), speed ×1.160 (p 1e-8).

KL per prompt, one run per prompt and fresh keys per run:

| Prompts | Arm | Mean | Median | p90 | Max | Above 10⁻³ | Argmax (scored) |
|---|---|---|---|---|---|---|---|
| dev 0-15 | entry | 2.0e-4 | 1.4e-4 | 5.4e-4 | 5.6e-4 | 0/16 | 127/128 |
| dev 0-11 | record 1 (the A/B's record arm) | 7.1e-4 | 4.9e-4 | 1.1e-3 | 2.7e-3 | 2/12 | 96/96 |
| C4 validation 0-15 | entry | 3.1e-4 | 1.5e-4 | 4.8e-4 | 2.4e-3 | 1/16 | 128/128 |
| C4 validation 0-15 | record 1 | 7.3e-4 | 5.8e-4 | 1.6e-3 | 1.7e-3 | 2/16 | 128/128 |

Dev prompts 12-15 come from one more entry run (`dev12-15_entry.json`). Its one scored argmax miss is prompt 13,
position 15. The C4 validation prompts are 16 rows of the C4 validation shard 00000, unused by any fit or choice here
(`c4v*.json`). The entry is above 10⁻³ on C4 prompt 2 (2.4e-3, against the record's 1.7e-3 on that prompt), driven by
one position (9.2e-3 at position 10); the record's run on that prompt peaks at one position too (5.4e-3 at 12).

## Disclosures

- **Plan.** `gpt2_decode_python_n32_m16i_fcf` is record 1's plan with five CF constants changed, by the planner from
  the record's own recipe and capture plus one option. Its `PLAN_CMD.txt` has the full recipe, and the regeneration
  gate reproduces it file for file at this commit. Six dense sites run the shipped order: the down-projection hints of
  blocks 1, 2, 4, 10 and 11 (CF 12, 13, 10, 10, 9 → 2) and the argmax's `v_2` (CF 5, unchanged). The planner prices
  the shipped order's CF on the plaintext coefficients that order's EvalMod sees. Those coefficients stay within 0.021
  at every position on 1,024 C4 and OpenWebText prompts, on the noise-free CPU twin (a measurement only), against a
  band top of 0.12 at CF 2. No prompt data enters the CF. We found the mechanism by decrypting the decode's
  intermediate ciphertexts on dev prompt 0, and an in-process sweep of CF 2/4/7/9 on dev prompts 0-5 was flat; the
  shipped CF is the planner's rule, not that sweep's pick. At CF 2 the refresh's power-of-two post-factor is 1, so
  five scaling kernels per token are skipped.
- **Setup.** The served decode precomputes every position's plan patch (the dynamic step values, the empty-cache
  inputs and the plaintexts that change between positions) at setup, from the CPU twin on the session's public rows.
  It depends only on the position, the token count, the plan and the weights, never on the prompt. Setup grows by
  about 25 s. Each position still stages its patch inside its own clock. Nothing moves into positions 0-7. Positions
  1-7, and position 0 of every request after the first, run as fast as positions 8-15. A session's first position is
  slower in both arms (the first replay), and faster here than in record 1.
- **Memory.** Peak GPU memory on Decode-8 is 52.8 GiB against record 1's 47.8. Three items:
  - the StC-first stage cache and the constant ring are sized for the compiled programs;
  - the CtS baby-step keys keep the tail digits of their regenerated `a` resident (`PC_LT_TAILS`);
  - `KsWork::ext` holds a 7-key sum of up to 4 ciphertexts, +0.66 GB per executor.
- **Unused here.** An EvalMod relin + rescale fusion is in the code behind `PC_EM_FUSE_RESCALE`, off by default and
  for every route. SPRU raise-drop variants are built only for a plan that asks for one; this entry's plans do not.
  `Ops::mult_pt_add` and `mult_pt_add_inplace` are benchmarked (`crates/pc-gpu/benches/ks.rs`), not wired into the
  runtime.
- **Bit identity.** With record 1's plans, the code before the plan commits gives outputs byte-identical to record 1's
  on one key bundle and one set of input ciphertexts. This holds on both tracks, and over two-request decode sessions.
  The check was a chain of pairwise comparisons, one per change.
- **Commits after the timed one.** The dev result was timed at d4d86a2. The later commits fix a test's field count,
  make every EvalMod plan default to the unfused relin + rescale, and add this folder. The second commit touches only
  the classic sparse route, which no plan here uses. With it, the outputs are byte-identical to d4d86a2's, checked on
both tracks over two-request sessions on one key bundle.
- **Keys and parameters.** Record 1's: the same chain (log N 16, log QP 1706), secrets, key formats, key request and
  pinned client. No client code changes.
- **Numbers.** Everything here was measured on H100s, not on the reference GPU. The H100 has far less integer
  throughput per byte of memory bandwidth than the RTX PRO 6000, so the kernel gains may come out differently there.
