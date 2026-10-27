# Decode-8 on a lean bootstrap chain, its softmax centred on reference keys, faster key-switching kernels

Decode-8 on BabyStepPrefill's code ([#12](https://github.com/gladia-research-group/perseus/pull/12)), with:

- a leaner chain for the bootstraps, `n32a4`: StC-first's three input stages run on three single 27-bit primes instead
  of three 54-bit prime pairs, the key switching uses 4 digits instead of 6, and P has 4 more special primes so that it
  still exceeds the widest digit by 8 bits. Every user level keeps record 3's primes and scale. A refresh's StC-first,
  SparseB and SPRU routes cost 0.84, 0.82 and 0.81 of record 3's (pc-bench on an RTX 5090, graph replay, one
  ciphertext; not a harness number), and every evaluation key is a third smaller;
- the softmax of blocks 0-7, 9 and 10 centred on one public reference key per head, so its exponentials stay at most 1
  and the head sums keep a known floor: 28 normalisation rounds per token instead of 36, and 8 fewer SparseB refreshes;
- kernels, byte-identical to record 3's outputs:
  - the NTT's column passes run on 16-column blocks;
  - a key-switching dot can add its inner product onto its outputs, so the giant-step Horner of the bootstraps' linear
    transforms rotates straight onto its accumulator;
  - a linear transform's baby key regenerates 2 digits of `a` and reads the others from resident tails;
  - single-ciphertext key switches expand `a` with one ChaCha block per thread, on warp-sized row segments.

The plan is replanned with record 3's recipe on the lean chain and the centred softmax's captures: 114 refreshes per
token against record 3's 120.

On one H100, Decode-8 runs at 2.739 s/token against record 3's 3.475 on the same GPU (×1.269, 6 pairs, p 9.3e-9). The
KL stays two orders of magnitude under the bar: 7.5e-6 on dev prompts 0-11, and on 48 C4 validation prompts no fit
used 5.7e-6 and 6.6e-6 in two sets of runs, about 1.8 times record 3's (Dev result).

Prefill-128, which this entry does not claim, keeps record 3's chain, plan and outputs (byte-identical); the kernel
changes make it faster: on one H100 it took 16.98 s against record 3's 17.58 (×1.035, 6 pairs, p 1.3e-5).

## Changes against record 3

- **The lean chain `n32a4`.**
  - The chain: per-level prime groups of one or two primes, single-prime bottom levels, dnum re-partitioned, P sized to
    the widest digit with record 3's P first: `crates/pc-core/src/{params,scale}.rs`, `crates/pc-core/tests/params_chain.rs`;
    `pc-params` lists it (`crates/pc-tools/src/bin/pc-params.rs`).
  - The rescale by one prime or two, on the device and in the shadow: `crates/pc-gpu/src/rescale.rs`,
    `crates/pc-shadow/src/{fides,route}.rs`.
  - The bottom-shape keys (the sparse-secret keys) over record 3's 9 special primes only:
    `crates/pc-gpu/src/{context,keygen,keys,ks}.rs`, `crates/pc-kernels/kernels/args.cuh`, and the client's keygen in
    `crates/pc-client/src/{client,ntt,spec}.rs`.
  - StC-first's input stages on single primes (thin plaintexts, split factors, the split top):
    `crates/pc-gpu/src/bts/{stc_first,staged}.rs`.
  - The chain as a runtime config field (`CHAIN_PRESET`): `crates/pc-config/{src/fields.rs,tools/gen_fields.py}` with
    `tests/legacy.rs` and `python/tests/test_fake_shadow.py`; `crates/pc-py/src/{session,shadow,word64}.rs`.
  - The compile's runtime shadow on the configured chain (its limb count, and a refresh input's room above q0):
    `crates/pc-program/{Cargo.toml,src/compile/check.rs,src/compile/mod.rs}`, `crates/pc-program/tests/unit.rs`.
  - A PCMM at the lean chain's 3-limb split: `crates/pc-gpu/src/matmul/pcmm.rs`, `crates/pc-kernels/kernels/pcmm.cuh`,
    `crates/pc-tools/src/check_pcmm.rs`. No shipped plan uses it (Disclosures).
  - Benches: `crates/pc-tools/src/bts_all.rs`.
- **The centred softmax.**
  - Code: `python/perseus/impl/{attention,config}.py`, `python/perseus/examples/gpt2/{model,ref}.py`; the softmax paths
    that do not centre refuse a centred config: `python/perseus/impl/{blocked,ccmm_mha,ccmm_mha_paged,prefill,sinc_softmax}.py`.
  - The config `configs/model/approximation/gpt2_base_n32_kc16r_cm` and its fit `scripts/utils/fit_gpt2_decode16_ref.py`.
  - Tests: `python/tests/test_gpt2_primitives_cpu.py`.
- **Kernels.**
  - NTT column passes on 16-column blocks, 8-column blocks for chunks of at most 9 rows:
    `crates/pc-kernels/kernels/{coeffrows,ntt,rescale}.cuh`.
  - `c0_mode | KS_ACC` (ABI 16): `crates/pc-kernels/kernels/{args,ks}.cuh`, `crates/pc-kernels/src/args.rs`,
    `crates/pc-gpu/src/{ks,bts/cts}.rs`.
  - `LT_REGEN_DIGITS` 4 → 2: `crates/pc-gpu/src/bts/cts.rs`.
  - `ks_dot_key` at K = 1 on warp-sized row segments: `crates/pc-kernels/kernels/{chacha,ks}.cuh`.
  - One `KeyTails` for every SparseB raise variant and slot count: `crates/pc-gpu/src/bts/{cts,sparse_b}.rs`,
    `crates/pc-py/src/session.rs`, `crates/pc-tools/src/{bts_all,bts_sparse,replay_g0}.rs`.
  - Tests: `crates/pc-kernels/src/bin/selftest.rs`.
- **The planner.** `PLAN_ILP_EXIT_CAP`: the final block may not exit deeper than a given level.
  `python/perseus/plan/placer/{__main__,ilp,planner}.py`, `python/perseus/plan/run_bootstrap_all_blocks.sh`,
  `python/tests/test_plan_ilp.py`.
- **The plan and what serves it.**
  - The served plan `bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm` and its blocks
    `gpt2_decode_python_n32a4_kc16r`.
  - The centred softmax's captures `graphs/gpt2_decode_python_n32_kc16r{,m,_cm}` and its plan on record 3's chain,
    `bootstrap_placements/gpt2_decode_python_n32_kc16r{,_cm}`, whose recipe and captures the served plan takes.
  - The recipe `speedrun/make_n32a4_decode_plan.sh`; `speedrun/make_cutmax_plan.sh` takes the chain's level cap and,
    with `CUTMAX_PLACER=ilp`, plans stage 13 on the ILP placer.
  - `speedrun/server.py` serves it and names each track's chain; the example decode default for 16 tokens
    (`python/perseus/examples/defaults.py`).
  - Gates: `crates/pc-program/tests/shipped_plans.rs`, `python/tests/{test_example_defaults,test_paper_plans}.py`.
- **Prefill-128's gates** now compile the plan the speedrun serves, `gpt2_prefill128_pfbhrd_cm`
  (`crates/pc-program/tests/shipped_plans.rs` `gpt2_prefill_speedrun`, `crates/pc-program/tests/units.rs`
  `refresh_pairs_only_reorder`), as asked in BabyStepPrefill's review.
- **The client.** `speedrun/harness/perseus_lean_client.py` (pinned to this PR's last code commit) and its key-request
  policy test `python/tests/test_perseus_lean_client.py`.

## Ablation

H100 80GB GPUs, the speedrun harness, two prompts per run. The first five rows are one job on one GPU: each change on
top of the row above it, 4 rounds in rotated order on dev prompts 0-7, each row scored against the row above (score.py,
4 pairs). The kernel rows give byte-identical outputs. The lean chain's row is its own A/B against the fifth row's
tree, 6 pairs (AB BA BA AB AB BA) on dev prompts 0-11 on one GPU, run at an earlier commit of this entry with the same
code and plans. The last rows are the Dev result's A/Bs at this PR's commit.

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| NTT column passes on 16-column blocks | decode | 3.476 → 3.397 s/token, ×1.023 (p 0.0072) |
| `KS_ACC`: the giant-step Horner onto its accumulator | decode | 3.397 → 3.403 s/token, ×0.998 (p 0.77, not resolved) |
| Linear-transform baby keys: 2 digits regenerated, the rest resident | decode | 3.403 → 3.331 s/token, ×1.021 (p 0.0019) |
| `ks_dot_key` on row segments at K = 1, shared SparseB key tails | decode | 3.331 → 3.313 s/token, ×1.006 (p 0.0028) |
| The centred softmax and its plan on record 3's chain (112 refreshes) | decode | 3.313 → 3.112 s/token, ×1.064 (p 0.00017) |
| The lean chain and its plan (114 refreshes) | decode | 3.117 → 2.743 s/token, ×1.136 (p 1.4e-7) |
| **This entry** vs record 3 | decode | **3.475 → 2.739 s/token, ×1.269 (p 9.3e-9)** |
| This entry vs record 3 | prefill | 17.58 → 16.98 s, ×1.035 (p 1.3e-5); not claimed, byte-identical outputs |

- `KS_ACC` is not measurable on H100 (×0.998, p 0.77) and within the noise of a pc-bench A/A on an RTX 5090. It stays
  because it removes the separate add after each giant step's rotation, its outputs are byte-identical, and the next row's
  segment kernel supports its mode.
- The segment kernel and the shared tails were measured three times on H100 against the row above, each with 4-6
  pairs on its own GPU: ×1.006 (p 0.0028, this table), ×1.012 (p 0.007, 6 pairs) and ×1.016 (p 0.00079, 6 pairs, dev
  prompts 12-23).

## Dev result

H100 80GB HBM3 GPUs, the speedrun harness at 449f7ba's harness files, timed at this PR's commit. 6 pairs in the order
AB BA BA AB AB BA on dev prompts 0-11, two prompts per run, in two jobs of 3 pairs (dev prompts 0-5 and 6-11), each on
one GPU; score.py scores the 6 pairs together. Record 3's arm is BabyStepPrefill's code with record 1's client; this
entry's client is its own (`perseus_lean_client.py` pinned to this PR's commit). Logs are in `dev_logs/`: `ab_*`
(Decode-8), `pf_*` (Prefill-128) and `c4v*`.

| Track | Score | Record 3, same GPU | KL (positions 8-15) | Argmax (scored) | Peak GPU memory |
|---|---|---|---|---|---|
| Decode-8 | 2.739 s/token | 3.475 s/token | 7.5e-6 | 96/96 | 49.0 GiB (record 3 56.5) |
| Prefill-128 (not claimed) | 16.98 s | 17.58 s | 1.4e-5 (position 127) | 12/12 | 63.3 GiB (record 3 62.6) |

score.py, Decode-8: quality PASS (mean KL 7.48e-6, p 1.8e-26), speed ×1.269 (p 9.3e-9). Prefill-128: quality PASS
(mean KL 1.42e-5, p 4.3e-33), speed ×1.035 (p 1.3e-5).

| Prompts | Arm | Mean | Median | p90 | Max | Above 10⁻³ | Argmax (scored) |
|---|---|---|---|---|---|---|---|
| dev 0-11 | entry | 7.5e-6 | 6.3e-6 | 1.5e-5 | 1.8e-5 | 0/12 | 96/96 |
| dev 0-11 | record 3 (the A/B's record arm) | 3.4e-6 | 2.3e-6 | 5.2e-6 | 1.1e-5 | 0/12 | 96/96 |
| C4 validation 32-79 | entry, run set 1 (12 runs) | 5.7e-6 | 3.9e-6 | 1.3e-5 | 2.9e-5 | 0/48 | 384/384 |
| C4 validation 32-79 | entry, run set 2 (12 runs) | 6.6e-6 | 5.0e-6 | 1.2e-5 | 4.1e-5 | 0/48 | 384/384 |
| C4 validation 32-79 | record 3, run set 1 (6 runs) | 2.5e-6 | 1.9e-6 | 4.1e-6 | 1.3e-5 | 0/48 | 384/384 |
| C4 validation 32-79 | record 3, run set 2 (6 runs) | 4.5e-6 | 3.6e-6 | 6.8e-6 | 2.6e-5 | 0/48 | 384/384 |

- The C4 validation prompts are rows 32-79 of the C4 validation shard 00000. No fit used them, and they decided
  nothing on their own; the refresh-noise model that kept the centred softmax's 4.6 over 5.0 ran on 2,048 prompts of
  the same shard (Disclosures).
- Each harness run draws fresh keys, and whole runs come out a few times higher or lower than others in both arms, so
  each arm has two run sets. The entry's set 1 ran at this PR's parent commit (the same code and plans; one script
  comment differs) with its own client (`c4v*_entry.json`); set 2 ran the same code and plans at an earlier commit of its development,
  with the client at another earlier commit whose client code is this PR's (`c4v*_entry2.json`; `client_perseus` names
  that commit). Record 3's set 1 is FittedDecode's (its record folder), set 2 is `c4v*_record3.json`.
- The entry's mean is about ×1.8 record 3's (6.1e-6 against 3.5e-6 over both sets; ×1.3 to ×2.7 between run sets).
  That is the cost of the lean chain's refresh precision (Disclosures). The worst prompt is still 25 times under the
  bar; one-sided t-test against 10⁻³ at n = 48 on set 1: p 1.5e-108.
- Percentiles are linear interpolations over the 48 prompts.

## Disclosures

- **Keys and parameters: the lean chain `n32a4`.**
  - **Q:** record 3's chain `n32` without three of its six StC-first input primes (prime ids 5, 6 and 7): q0's pair,
    one single 27-bit level (s1), one pair that StC-first walks as two single-prime levels, and record 3's top 46
    primes, unchanged. 51 Q primes against 54. Every user, EvalMod and CtS level keeps its primes and its scale.
  - **dnum 4** (digits of 13, 13, 13 and 12 primes) against 6.
  - **P:** record 3's 9 special primes and 4 more of about 27.2 bits (158334977, 158072833, 155189249, 154533889): 360.5
    bits against the widest digit's 351.8 (record 3: 251.7 against 243.7, the same 8-bit margin). Two of the four new
    special primes, 155189249 and 154533889, are Q primes of record 3's chain that `n32a4` drops; Q and P stay disjoint.
  - **log QP 1734** against the 1747 the HE standard allows at N = 2^16 (record 3: 1706). 13 bits of headroom.
  - **Main secret:** unchanged, uniform ternary, N = 2^16. Every key under it is at a modulus of at most Q·P.
  - **Sparse-secret keys:** the h = 32 encapsulation key (dense to sparse) and the SPRU block key switch over record
    3's 9 special primes only, P'. They live at q_0 q_1 · P' (308 bits), record 1's modulus. From the same secret and
    randomness they are word for word the keys record 3's chain generates, so record 1's estimator verdict carries
    over: `pc-client`'s test `keygen_relations_and_bundle` runs on both chains, and the client refuses a sparse key over
    more special primes or above the bottom level.
  - **PCMM ring-switch keys (Prefill-128):** unchanged; `n32a4` keeps record 3's first four primes.
  - **Prefill-128** stays on record 3's chain: its plan's runtime line names no chain, and its key request is
    byte-identical to record 3's.
  - **Key request (Decode-8):** record 3's roles and rotation steps (124 switching keys, 87 rotation keys) on chain
    `n32a4`, at its shapes: 6,778.9 MB of GPU memory against record 3's 10,008.3 MB (server logs). The server's memory
    plan is 49.2 GB against record 3's 56.9 GB.
- **The client.** `speedrun/harness/perseus_lean_client.py` is `perseus_core_client.py` for both chains: Decode-8's
  key request on `n32a4` and Prefill-128's on `n32`. It refuses any other chain, a SPRU key other than h = 64 in 2
  blocks, CCMM, a sparse-secret key above the bottom level (`q_top` > 1) or over more than the 9 special primes, and a
  key request with a repeated JSON key. `pc-client` also refuses a sparse key that is not at the chain's bottom shape.
- **Refresh precision on `n32a4`.** Measured per route with pc-bench (every CF 2-20, A 0.01-64, 2-8 seeds): dense and
  SparseB unchanged; StC-first 0.4-0.5 bits more maximum error with the same RMS; SPRU 0.2-0.6 bits at CF 10-12. The
  planner keeps record 3's accuracy tables. The cost shows in the KL: about ×1.8 record 3's mean on C4 validation
  prompts 32-79 (Dev result). The centred softmax alone, on record 3's chain, is indistinguishable from record 3
  there.
- **The plan.** Record 3's recipe (ILP with the exact tie-break, one HiGHS thread) on the centred softmax's merged
  captures (the same four OpenWebText pool rows, positions 0-15), with the sparse refreshes capped at level 48,
  `n32a4`'s deepest user level.
  - Stage 12 is planned alone at block 11's exit with its exit capped one level higher (`PLAN_ILP_EXIT_CAP=46`):
    CutMax's stage 13 takes its first refresh two levels below its entry.
  - CutMax's stage 13 is planned on the ILP placer (`CUTMAX_PLACER=ilp`; record 3 plans it on the min-cut). On the
    lean chain it needs one refresh more than record 3's, and the ILP's quality weight puts it in CutMax's last
    iteration at input level 24 (v_404, CF 15, the 1-slot route) instead of the min-cut's CF-20 square at its end;
    that keeps CutMax's one-hot mass at record 3's (below).
  - The ILP also takes the recipe's raise drops into stage 13, which the min-cut keeps out: v_404 is raised two levels
    below the chain top (raise drop 2), and so is record 3's own hint v_54, which now lands two levels deeper. The
    one-hot table below is measured on this plan.
  - Refreshes per stage: 8, 8, 8, 8, 10, 8, 8, 8, 8, 8, 8, 10, 1, 13; 114 per token against record 3's 120 (the
    centred softmax alone on record 3's chain: 112).
  - **Regeneration:** from scratch, CPU only, in a git clone of this PR's commit with its own build: the three
    captures (`run_decode --fake`, `speedrun/make_decode_capture.sh`, `capture_mags --rows`), the plan on record 3's
    chain (`make_plan.sh` with its recipe, `speedrun/make_cutmax_plan.sh`) and `speedrun/make_n32a4_decode_plan.sh`,
    each as the plans' `PLAN_CMD.txt` lines say. Stage 13's capture `graphs/gpt2_decode_python_n32_kc16r_cm`, which
    those lines do not spell out, is stages 0-12 of `graphs/gpt2_decode_python_n32_kc16rm` and block 13 (with its
    `capture_config.json` and `capture_env.json`) of the position-0 capture `graphs/gpt2_decode_python_n32_kc16r`,
    copied, then `python scripts/utils/cutmax_rows.py graphs/gpt2_decode_python_n32_cm/cutmax_rows.json rows.npy` and
    `python -m perseus.examples.gpt2.capture_mags graphs/gpt2_decode_python_n32_kc16r_cm --argmax --rows rows.npy
    --blocks 13 --plan bootstrap_placements/gpt2_decode_python_n32_kc16r --configs
    configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json`, as record 3's `capture13=` line does for its
    own. Every `graph.json` comes out byte-identical (the captures' configs
    differ only in the capturing tree's paths), and `speedrun/runner/plan_diff.py` finds 0 differing plan fields in
    the four plan directories (only the solver's `summary`), with identical `PLAN_CMD.txt` files. On 32 cores the
    captures take about 6 minutes, the plan on record 3's chain about 28 and `make_n32a4_decode_plan.sh` about 20 (one
    HiGHS thread).
  - **Range check:** every refresh site's input at every position 0-15, on the CPU twin, against its CF band: C4
    validation fit-shard prompts 0-63, prompts 0-79 of shard 00000, 2,816 random token pairs and 7,233 first tokens.
    All 114 sites stay inside their bands; the smallest margin is 1.96× (block 13, v_25), record 3's own smallest
    site and margin. Stage 13's added refresh (v_404) sits at 101×.
  - **CutMax's one-hot mass** (on the GPU, each run with its own keys and client, CutMax's calibration prompts C4
    validation fit-shard 0-11, 96 scored positions per key set; four key sets per arm):

    | | key set | argmax(z) = argmax(logits) | mass ≥ 0.99 | mass < 0.5 | median mass |
    |---|---|---|---|---|---|
    | record 3 | 1 / 2 / 3 / 4 | 96/96 each | 75/96 each | 0 each | 0.999986219 / 0.999986147 / 0.999986187 / 0.999986103 |
    | this entry | 1 / 2 / 3 / 4 | 96/96 each | 75/96 each | 0 each | 0.999986167 / 0.999986247 / 0.999986202 / 0.999986150 |

    - The mean median is 0.999986191 for this entry and 0.999986164 for record 3 (+2.7e-8, 0.82 standard errors of
      the key-set noise); the median moves by up to 1.4e-7 between key sets of one arm. Every key set is equal to
      record 3's on the other three counts. Key sets 1-3 of this entry ran on the claim's tree, byte-identical to this
      PR's code; key set 4 on this PR's commit.
    - With stage 13 on the min-cut placer instead (its added refresh at CF 20 near CutMax's end), one key set gave a
      median of 0.999968679: that is why the plan takes stage 13 from the ILP.
    - Noise-free, CutMax is record 3's: the chain changes no approximation, and the cutmax section of the config is
      record 3's (make_cutmax_plan.sh checks it).
- **Approximations: the softmax of blocks 0-7, 9 and 10 is centred on a reference key per head**
  (`gpt2_base_n32_kc16r_cm`, `ref_keys`).
  - Each head subtracts the score of one key before the exponential: the first key, the query's own key, or the key
    before it. The choice is fixed per layer and head and is the same for every prompt.
  - The exponentials are then at most 1. The reference key's own term keeps each head sum at least e^(-U/δ2), so the
    first reciprocal round runs on its exact band, [e^(-U/δ2)/16, 1.05].
  - The THOR normalisation rounds per layer follow from the largest centred score U*: δ2 = 2^⌈log2(U*/4.6)⌉, so a
    refreshed exponential vector stays within e^4.6 of its head sum.
  - The references, the round counts, U = 1.25 U* and the exponential's domain (its lower end widened by 5 % of the
    range) were calibrated on 16-token windows of OpenWebText shard 0042 (record 3's OpenWebText windows; no dev
    prompt) by `scripts/utils/fit_gpt2_decode16_ref.py`. It reproduces the config byte for byte on the machine that
    made it; on another platform the structure, the references and every integer are equal, and 239 floats differ by
    at most 2e-15.
  - One choice looked at C4 data: the criterion's 4.6 was kept over 5.0 (one bit of δ2) after a refresh-noise model's
    KL on 2,048 prompts of C4 validation shard 00000 doubled at 5.0.
  - Blocks 8 and 11 keep record 3's softmax, which is shallower there. Rounds per token: 36 → 28.
  - **Under encryption:** key 0 is subtracted from every key at the K/V push (the cache holds k_j - k_0, and k_0 is
    kept from the first token). The heads centred on their own or the previous key subtract that key's score after
    q·K^T, with a selector at the group mask's level and rotations by 1 to 8 slots that record 3 already has. No
    level, key type or rotation step is added. Every position runs the same operations.
  - **Validity:** at most 16 keys; past 16 the computation refuses, so a longer cache takes another plan, as record
    3's config does. The softmax paths that do not centre (the real packing, prefill, CCMM) refuse a centred config.
  - **Accuracy:** noise-free (float64 mirror of the encrypted computation), the centred softmax's KL equals record 3's
    within 1 % in mean, p90 and p99 on about 12,400 16-token windows no choice used (FineWeb, C4 validation shard
    00001, mid-document C4 validation windows). Through the harness its KL is indistinguishable from record 3's (Dev
    result).
- **Kernels.** All four kernel changes give byte-identical outputs (record 3's logits frames on both tracks, the same
  keys and inputs; the selftest checks every changed kernel against the reference kernel or the host).
  - **Memory:** a linear transform's baby key with 5 or more digits now keeps 4 resident digits of `a` instead of 2.
    They are expanded from the keys' public seeds at setup (untimed, prompt-independent). Alone, the larger tails
    raise the peak GPU memory on record 3's chain by 3.9 GiB (Decode-8) and 3.4 GiB (Prefill-128). One `KeyTails`
    shared by every SparseB variant and slot count gives back the copies each variant held (record 3's included):
    together, the server's memory plan is 0.2 GB (Decode-8) and 0.7 GB (Prefill-128) above record 3's.
  - On the lean chain a key has at most 4 digits, under the tails' threshold: Decode-8 holds no tails, and the baby
    keys' 2-digit split changes nothing there. Its ablation row was measured on record 3's chain, where Prefill-128
    runs.
  - Peak GPU memory: Decode-8 49.0 GiB (record 3 56.5: the lean chain's smaller keys), Prefill-128 63.3 GiB (record
    3 62.6).
- **The PCMM at the lean chain's 3-limb split** is in the code but no shipped plan uses it: Prefill-128 stays on record
  3's chain, where the PCMM is unchanged (4 limbs, byte-identical outputs). On `n32a4` it loses precision at the
  call-factor stage's single-prime plaintext; a later entry places that factor at a composite level.
- **Build.** `crates/pc-program/Cargo.toml` makes `pc-core` a non-optional dependency of `pc-program`: the compile's
  runtime shadow builds the configured chain.
- **FittedDecode's plan options** (from its review): FittedDecode added `PC_EM_FUSE_RESCALE=1` to the `runtime=` lines
  of record 2's plans (`gpt2_decode_python_n32_m16i_fcf{,_cm}`, `gpt2_prefill128_pf{,_cm}`), removed the planner
  options `PLAN_ILP_FREE_HINTS` and `PLAN_RAISE_HINTS`, which no shipped plan set, and added a `SUMMARY_DRIFT` entry
  for `gpt2_prefill_python_n32_pcmmstcf` (`bts_quality.blind_branches`) to the regeneration gate. Its README did not
  list them.
- **Numbers.** Everything here was measured on H100s, not on the reference GPU, except the pc-bench route costs above
  (RTX 5090).
