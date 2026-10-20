# GPT-2 n32 decode: speed-lever ledger (sorted by status)

Untracked working note: the single sorted view of every speed lever. The clean-A/B protocol, queue tooling and the
dated evidence log are in `/data02/users/azirilli/perseus-deps/notes/ab_pending.md`. Earlier versions of both notes:
`perseus-deps/notes/speed_levers_archive/` (the pre-housekeeping copies of 2026-10-12 17:30 included).

Last sorted: **2026-10-12 17:30** (perseus-01 housekeeping, user request). perseus-7e (model side) is gone since ~13:20.

## 0. Baseline and protocol

- **Current: perseus 4c6f3bf / FIDESlib ac29212 = origin/dev (pushed 2026-10-13 15:50), tag `decode-e2e-3.84s`: 3.84 s/token e2e**
  (decode 3.29 + argmax 0.55, GPU 6+7 concurrent lane). Per-block Chebyshev GELU default; composite GELU = branch
  `composite-gelu`, tag `composite-gelu-3.98s`, plan `gpt2_decode_python_n32_thor`.
- **Hashes:** on 2026-10-13 the user's commits since 10-11 were reworded (minimal messages) and force-pushed: perseus dev
  = 4c6f3bf, FIDESlib dev = ac29212. Hashes in older rows are pre-rewrite; maps in perseus-deps/notes/msg_rewrite_2026-10-13/.
- Protocol (user): isolation (private worktree + switch) -> correctness gate -> paired A/B -> commit with the fastest as
  the default -> push (FIDESlib first, so the perseus gitlink resolves) + tag `decode-e2e-<s>s` -> new baseline.
- Timing rules (user, 2026-10-13): machine declared quiet 08:10 ("start the A/Bs on GPU 4"); 10:28 "you can use 6 and 7
  for a silent machine": lanes clean/ab_lane4c.sh (GPU 4, node 3), ab_lane6c.sh (GPU 6, node 1), ab_lane7c.sh (GPU 7, node 4).
- Clean pair = interleaved arms, `clean/pairq.py` filter: foreign CPU on the arm's cores <= 1 core and steady GPU util
  >= 88 % in both arms (box load from other users: 70-120 most of 10-12).
- Trail (e2e): 9.7 decode-only (10-08) -> 6.80 (10-09) -> 5.29 (004d664) -> land3 4.98 -> land4 4.77 -> land6 ~4.77
  -> land7h 4.40 -> land8 ~4.33 -> land9 ~4.28 (shipped plan) -> land10 4.09 -> land11 3.99 -> **Chebyshev GELU 3.84**.

## 1. Landed (newest first)

| date | commit | lever | measured |
|---|---|---|---|
| 10-13 | perseus 9fd3d7e (pushed, tag decode-e2e-3.84s; composite state = branch composite-gelu / tag composite-gelu-3.98s) | GELU as one Chebyshev series per block (perseus-7c's design, 7-10 levels vs THOR ~12; 184 vs 191 bts/token); plan-bound configs; switch back --plan bootstrap_placements/gpt2_decode_python_n32_thor | **-0.141** (queue78, 3 clean pairs, 17/18); KL ~4-6x (median ~3-4e-4), accepted by the user as a structural change |
| 10-13 | perseus 225e49a (pushed, OPTION, off by default) | SM_CHEB_GS=1 + plan gpt2_decode_python_n32_smcheb: Chebyshev-scaled Goldschmidt softmax on exact per-kc bands (215 vs 290 Goldschmidt levels; keeps the softmax accurate past 32 cached tokens) | -0.008 (queue76, 4 clean pairs); KL ~2-3x the main plan within 32 tokens -> kept off (user) |
| 10-13 | perseus f42d214 / FIDESlib 33d9cd5 (pushed) | encode's max-coefficient readback issued before the encode's completion event (race fix: the compact-dump decision could read a stale value; prerequisite for ENCODE_PINNED); prefill keeps the host wait (d323076) | -0.010 (queue69b, 3 clean concurrent pairs, 14/18), correctness fix |
| 10-13 | land11 perseus 2b48db7 / FIDESlib 5dd5241 (origin/dev, tag decode-e2e-3.99s) | 512-slot route raised one composite level lower (main plan, `FIDESLIB_BTS_RAISE_DROP=512:1`, landing 26; switch back `gpt2_decode_python_n32_top512`) | **-0.065** land10 (5 clean pairs -0.058..-0.070), **-0.060** on upstream 284cb95 (3 clean pairs) |
| 10-13 | same | device-encode without the host wait restored (`FIDESLIB_ENCODE_NOSYNC`, default on; upstream f12c06e had removed it) | **-0.137** (queue62, 2 clean + 2 flagged pairs, +0.11..+0.14 for the wait); KL 64 tok equal, reload check 2560/2560 bit-exact |
| 10-12 | **land10, PUSHED** perseus 403d919 / FIDESlib b13d880, tag `decode-e2e-4.09s` | StC-first main plan: `bootstrap_placements/gpt2_decode_python_n32` runs its dense refreshes StC-first (`FIDESLIB_BTS_STC_FIRST=2` via the plan's runtime line); `..._shipped` = the old order (switch back), `..._stcf_accurate` = residual + K/V push shipped. Accuracy accepted by the user: KL ~1e-4 (was ~1e-6), near-tie top-1 flips, generation can diverge early | **-0.187** tag pair, **-0.226** queue58 (2 clean pairs, 10/12): **4.09-4.10 s/token** |
| 10-12 | land9 (perseus-7e; pushed inside land10) perseus 86f6d5f / FIDESlib ce02e64 + 7b4a8e3 | host-sync bundle by default: no host wait after device encodes (`FIDESLIB_ENCODE_NOSYNC`), aux release normalized on the device (`FIDESLIB_AUX_NOSYNC`), scratch growth retired off the host (`FIDESLIB_SCRATCH_RETIRE`) | -0.098, 3 clean pairs GPU 4 (perseus-7e); token ~95 % GPU-busy after it |
| 10-12 | land9 perseus 258ea92 | CutMax iteration-0 range 5 -> 7 sigma (config): fixes the deterministic token-3 argmax miss | accuracy fix, argmax time unchanged |
| 10-12 | land8 (perseus-01; pushed) perseus 0ac3f8d + FIDESlib a8be774 | linear baby-step sums in one fused kernel (`LIN_PT_DOT`), bit-exact | -0.044 / -0.066, 8 pairs (45/56) |
| 10-12 | perseus 6f10dad + FIDESlib 089e180 (local, land6 = land5 + this) | per-step masks erased without a host dump and kept across tokens by default (`IMPL_NODUMP_ERASE`, `IMPL_KEEP_MASKS`; FIDESlib ForgetDevicePlaintextDump) | -0.051 s/token, 22/28 (on land5) |
| 10-12 | perseus bc6cb9f + FIDESlib 147c6c9 (local, land5 = land4 + this) | lazy-BSGS Paterson-Stockmeyer in EvalMod: the products that only feed the final sum share one relinearization (`FIDESLIB_LAZY_PS`, default 1), exact | -0.039 s/token, 20/28 |
| 10-12 | perseus dae6772 + FIDESlib 59e3dd4 (local, land4 = land3 + these) | input encryption on the GPU by default (`FIDESLIB_GPU_ENCRYPT`, 0 = host): OpenFHE's EncryptZeroCore on the device, ChaCha12 randomness, pk resident; encode_input 104-113 -> 2 ms/token | -0.093 s/token, 22/28, every pair negative |
| 10-12 | (same, land4: FIDESlib de2a942, perseus 17700fd) | decrypt a device ciphertext in an empty host container instead of one built by a dummy secret-key encryption (`FIDESLIB_FAST_DECRYPT`, default 1), exact | -0.020 s/token e2e, 18/28; argmax -0.026 every pair |
| 10-12 | (same, land4) | API wrappers write their result once instead of cloning an operand (`FIDESLIB_NOCLONE`, default 2; 0 = clone-first) | -0.061 s/token, 24/28, every pair negative, bit-exact |
| 10-12 | perseus 3fc14e0 + FIDESlib 908b22c (local, land3, on upstream 9a65ce9 / d3a73e1) | faster key-switch base conversions by default: ModUp (`DecompAndModUpConv`) and ModDown conversions with the matrix tile + digit limbs in shared memory, 4-target ILP, compile-time lazy reductions (`FIDESLIB_BCONV2=2`; 1 = ModUp only, 0 = original), bit-exact | -0.079 s/token, 23/28 (level 1 alone -0.063, 24/28), GPU-6 lane, 4 clean pairs each |
| 10-12 | **perseus-wt/land7h** = land6 + the four rows below (perseus a91e2d3, FIDESlib 840bfd6; local, unpushed) | landing A/B land5 vs land7g (= land7h minus masks): **-0.269 s/token**, 43/44, 4 clean pairs GPU 4 (4.69-4.71 -> 4.42-4.43 s/token); lane 6 -0.251 24/28; full stack on efdb6c3 -0.309, 22/22 | **new baseline land6 vs land7h: -0.280 s/token, 44/44, 4 clean pairs GPU 4: 4.67-4.72 -> 4.40-4.41 s/token** |
| 10-12 | perseus 263baf0 + a91e2d3, FIDESlib 530876b (local, land7h) | EXACT BUNDLE: realize shared deg-2 operands once (`IMPL_REALIZE_SHARED`), softmax head sums on one complex ladder (`SM_HRS_PACK`), CutMax cascade refreshes paired on one SPRU (`CUTMAX_PAIR`), ModRaise without the NTTs the exact scaling overwrites (`FIDESLIB_MODRAISE_COEFF`), dead graph ops (`IMPL_DEAD_OPS`); plans ctl_b vs bnd_b | **-0.091 s/token** paired median, 18/28, 4 clean pairs GPU 4 (decode -0.043, argmax -0.050), floor -0.133, KL equal |
| 10-12 | perseus 263baf0 + a91e2d3 (local, land7h) | shared-operand realize anchors (planner `PLAN_SHARED_REALIZE`: a deg-2 value read only by products / key switches is realized once; supersedes raw `IMPL_REALIZE_SHARED`, composes with CHEB_ALIGN); plans ctl_b vs shr_b | **-0.038 s/token**, 32/44, 4 clean pairs GPU 4, floor -0.028 |
| 10-12 | FIDESlib 840bfd6 (local, land7h) | SparseB 512-route baby step 16 instead of 32 (`FIDESLIB_BTS_SPARSE_B_BSTEP=16`, env only: one giant step, half the Q+P rotation temporaries) | **-0.072 s/token**, 34/44, 4 clean pairs GPU 7, floor -0.063 |
| 10-12 | perseus 263baf0 + a91e2d3, FIDESlib db73d19 (local, land7h) | hoisted radix-8 rotate-and-sum ladders (`LADDER_RADIX_BITS=3`, `Ciphertext::rotateSumHoisted`: each group of 3 rungs = one ModUp, 7 key dots in the extended basis, one ModDown; +23 rotation keys) | **-0.124 s/token**, 43/44, 4 clean pairs GPU 4 |
| 10-11 | perseus efdb6c3 (local) | all 12 weight blocks resident across tokens (`IMPL_RESIDENT_BLOCKS`, =0 restores evict/reload), possible after upstream's memory work | -0.194 s/token, 13/14, both reps |
| 10-11 | perseus eb56de3 + FIDESlib 24e3ef1 / 97afd0c / f47c47f (local) | barrier bundle: no per-launch error polls (`FIDESLIB_ERRCHECK=0`), single-GPU OpenMP regions inlined (`FIDESLIB_OMP_REGIONS=0`), SmallInt copy via device tables (`FIDESLIB_SMALLINT_NOSYNC`), SPRU without its device fences (`FIDESLIB_SPRU_NOSYNC`), K/V caches resident (`IMPL_KV_OFFLOAD=0`) | bundle -0.155, 11/14 (parts alone ~-0.07: barrier synergy); with resident blocks -0.202 vs upstream behaviour (contended) |
| 10-11 | 004d664 (pushed) | Tier A from Sylph: Q rides the K/V push refresh, exit cap 40 on the residual stream (`QKV_LANES`) | -0.21 over 21 tokens, KL 10x better |
| 10-10 | a73a023 (pushed) | planner audit fix (sim input levels, P3c drift, `PLAN_DRIFT_RECUT`), no runtime change | placements identical |
| 10-10 | e78f236 (pushed) | refresh envelope 50 instead of 48 (`PLAN_REFRESH_ENV_CAP=50`, cutmax/tail at 48) | -0.23, 13/14 (KL ~2e-6 -> ~1e-5) |
| 10-10 | 4417875 (pushed) | baby-step giant-step Chebyshev evaluation (`CHEB_BSGS`), closure-free | -0.38, 14/14, KL better |
| 10-09 | (110c69a headline) | periodic softmax scores (`SM_PERIODIC`), softmax fold (`SM_FOLD`), SparseB on the 512-slot route, K/V lanes (`KV_LANES`); also LN_CHEB, SM_GS_FIRST | 6.80 -> 6.40 -> 6.17 -> 5.93 -> ~5.86 e2e in landing order |
| 10-08 | d5db400 (pushed) | softmax reciprocal on the denominator (`SM_DEN_RECIP`), real-payload dense bootstrap (`FIDESLIB_BTS_REAL`, `PLAN_REAL_ROUTE`), GELU 1/xmax fold (`GELU_FOLD`) | decode -1.7..-2.2, -0.6..-0.8, -0.15..-0.2 |
| 10-07/08 | (shipped plan) | level-aware ModRaise (per-site raise drops) | -3.5 % (10.1 -> 9.75 then) |
| 09-01 | FIDESlib b2d5405 | skip the redundant per-digit source INTT (`FIDESLIB_KS_DIGIT_INTT`) | -1.8 % bootstrap |

## 2. Built, waiting for a clean A/B (base: origin/dev f42d214; GPUs 6+7 concurrent pairs)

| # | lever | switch (B arm) | where | evidence so far | expected |
|---|---|---|---|---|---|
| 4 | StC-first accurate variant as the default (instead of the full StC-first plan) | `--plan bootstrap_placements/gpt2_decode_python_n32_stcf_accurate` | committed in land10 | 1 clean pair -0.051 vs shipped; KL ~3e-5, 16-token generation identical to shipped | a user choice: accuracy vs ~0.14 s |
| 5 | NUMA interleave of host memory | `ARM_NUMA_MEM=--interleave=3,4,5` | env only (GPU-4 lane) | +0.072 contended (10-11) | ~0 |
| 6 | Bootstrap CUDA graphs | `FIDESLIB_BTS_GRAPH=1` | archived (landg), old base; needs a rebase | bit-exact; replay = eager latency | ~0 (deprioritized) |
| 7 | [LANDED 10-13, see section 1] Chebyshev GELU (perseus-7c): one Chebyshev series per block at <= THOR's 4.7e-4 error on 1.25 x xmax (degree 63-127 = 7-8 levels for 10 blocks, 191 = 9 for block 10, 383/511 = 10 for blocks 2, 11) instead of the THOR composite (~12 levels). Plan w7c_gelucheb: 149 planned, -10 dense + 3 one-slot per token. GATE FAILED (top-1 2/8, KL 0.3-3); the series are exact on the GPU without bootstraps (1e-10) and the fake backend agrees, so the defect is in the planned refreshes (in-series / down-output sites, exit-36 blocks); variant w7c_gelucheb_nog (no refresh inside gelu/down, 148 planned) gating. NOT exact (approximation change): user decision even if it passes | `--configs .../gpt2_base_n32_gelucheb/configs.json --plan bootstrap_placements/w7c_gelucheb[_nog]` | perseus-wt/w7c (uncommitted) | CPU: both approximations detonate outside their range (THOR 1.10 x, this 1.5 x) | -10 dense ~ -0.13 s minus ~+0.04 s of extra coefficient multiplies, if it can be made to pass |
| 8 | FUSED_LN_VAR=1 on the SPRU route (perseus-7c): the SPRU extracts CKKS coefficient 0 = the slot mean, so fold_bootstrap(square, 1, 768) replaces the 5-ladder all-slot variance sum; PROBED on GPU 4: value correct (rel err 2e-6 vs 1.5e-5 for today's broadcast refresh) but the ladder it removes costs only 2.5 ms per LayerNorm and the Newton chain then runs at ~2x the limbs | env `FUSED_LN_VAR=1` + a new capture and plan | norm.py path exists; probe lnvar/probe.py done; capture not made (35 min GPU) | 11.9 vs 14.3 ms synced per LN for the fused vs today's path | <= -0.03 s/token (small) |

**Open security item (pre-existing, recorded 10-13, parked by the user: "drop the security hazard"):** the encapsulation dense->sparse key (h=32, `sparse_atob`) and the SPRU dense->block key (h=64) are RLWE samples under a sparse secret at the full QP (log 1705.34: Q 1453.69 + P 251.65, perseus-9e's qp.cpp) instead of q0*P (log ~308) where they are applied. Lattice Estimator rough (core-SVP): main dense 2^94.6 (calibration), encaps 2^94.0, SPRU 2^92.0; at q0*P no feasible attack. Fix candidate: generate/publish both keys modulo q0*P only. Notes: perseus-deps/notes/ab_pending.md 2026-10-13 12:05; scripts scratchpad 194e8501.../sec/est.py (the Sage env was removed).

## 3. Ideas, not implemented (expected gain; why not yet)

| idea | expected | state |
|---|---|---|
| Fused Chebyshev baby combination (perseus-7c): the BSGS evaluates sum_k c_k T_k per giant group as one scalar multiply + add per coefficient (378 const mults + 716 adds per token in the THOR GELUs, 809 + 1218 in the Chebyshev-GELU graph, plus LN_CHEB seeds and softmax series); one kernel per group reading the m babies and writing the combination would make them one pass, exact | -0.03..-0.05 s/token (more with the Chebyshev GELU) | not built; same shape as LIN_PT_DOT (resident babies instead of plaintexts) |
| Single-prime rescale for plaintext / constant products (perseus-7c): the ct keeps the 54-bit composite scale, the pt operand is encoded at ~2^27 and the product rescales ONE prime (FIDESlib has `rescaleOne`; no odd-level scale grid, `sfAtLimb` aborts off-grid) | planner what-if PLAN_PT_PRIME=1: 150 vs 155 planned (-5, ~-0.08 s) | parked: multi-day FIDESlib + planner + runtime scale bookkeeping for -3 % |
| SparseB raise drop by one composite level (512 landing 26 is count-free: what-if 155 = control; 28 -> 156): the 55 512-route calls and the ~47 ms/block of softmax ops downstream run at 2 fewer limbs | -0.06..-0.09 s/token, exact | perseus-01 (worktree perseus-wt/sbr, plan gpt2_decode_python_n32_sbr1 with FIDESLIB_BTS_RAISE_DROP=512:1; probe + 32-token gate queued) |
| CutMax normalizers through the SPRU slot mean (same trick as FUSED_LN_VAR: rotsum_all before each of the 11 1-slot refreshes) | ~-0.03..-0.05 | after the LN probe |
| Softmax reciprocal with better Goldschmidt seeds at equal precision (290 iterations + 33 exp squarings/token drive the 43 softmax 512-slot refreshes; refine rounds take 6-10 iterations) | -0.15..-0.35 | PARKED by the user (13:50); a CPU study with the plaintext mirror (`ref.py`) is the first step |
| Softmax / CutMax iteration trims | -0.15 per 2 iterations/round | off-limits (accuracy) unless the user lifts it |
| BSGS giant steps summed in the extended basis (one ModDown per accumulator) [PARKED by the user 10-13] | ~-0.015..-0.02 | measured: ModDown 108 us of a 296 us non-bootstrap rotation, ~150-200 shareable/token |
| Kernel side: per-digit special-limb NTT/INTT (843 ms kernel-sum/token in 32x9 launches), `hoistedRotateDotKSKRegen4_` (ncu: latency/occupancy-bound, coalescing worth little), `dotProductLtBatchedPt3___` | uncertain | perseus-7e's ncu pass, unfinished; merged digit loops were neutral before |
| KS_FLAT megakernel re-implementation | -1.5 % (August base) | patch lost; rebuild from memory note perseus-megakernel-launch-flattening-2026-08-31 |
| NTT row-group ILP | ~0 | banked patch (perseus-deps/notes/bootstrap_campaign_2026-10-07/fideslib_patches), neutral at bootstrap level |
| GELU as one Chebyshev (degree 95/127, 8 composite levels) instead of the THOR composite (p1 deg 31 + p2 deg 27, ~12 levels); CPU float64 at xmax 19.65: THOR max err 4.7e-4, deg 95 1.1e-6, deg 127 1.2e-10 (perseus-7c, 10-12 evening) | fewer MLP dense refreshes; planner count pending | not exact (approximation change): user decides; isolation in perseus-wt/w7c; per-block (22:10): degree 63-127 (7-8 levels) for 10 blocks, 191 (9) block 10, 383/511 (10) blocks 2 and 11 (xmax 94/78) at <= THOR error on 1.25 x xmax; THOR detonates at 1.10 x xmax, these survive 1.5 x; capture/plan/gate running |
| Plaintext/constant products rescaled by one prime, pt operand at 2^27, ct scale stays 2^54 (`PLAN_PT_PRIME` what-if, perseus-7c) | -5 planned refreshes/token | parked: odd-limb scale bookkeeping in FIDESlib/OpenFHE for a small count |
| Speculative decoding on the 32 token lanes | 2-3x per generated token | needs the multi-position forward (prefill32 detonates at block-0 ln_2) |
| 10-13 | spru16 (SPRU block key h=16, FIDESLIB_SPRU=16) | INSECURE: May MitM on the block key (N/h)^(0.25(h-1)) = 2^45 (h=32: 2^85); only h >= 64 holds at N=2^16 (Coron-Koestler 4.5); A/B stopped, never landed |
| 10-13 | FIDESLIB_ENCODE_PINNED (pinned staging ring, perseus-7e) on land11c | neutral: -0.013/+0.016/-0.010/+0.009 (queue70) |
| 10-13 | FIDESLIB_MODUP_MERGE=2 on land11 | +0.02 loss (queue63, 2 clean pairs) |
| 10-13 | FIDESLIB_PW_FUSE=1 on f42d214 | neutral +0.002 (queue71) |
| 10-13 | FIDESLIB_FUSED_RESCALE=1 (incl. the lazy-last-square variant that keeps the plan's landings) | +0.051 LOSS (queue77, 3 clean pairs, 1/18 faster), KL worse; archived refs/archive/wt/fres |
| 10-13 | FIDESLIB_NOCLONE=3 on f42d214 | neutral -0.000 (queue73, 4 clean concurrent pairs, 12/24) |
| 10-13 | Fold masks / constants / ladders adjacent to refreshes into the bootstrap's linear stages | census: 6 exact sites/token, what-if 155+25+11 unchanged (upper bound 66 free products: argmax 25 -> 24); ~1-2 ms/token, no refresh saved (perseus-deps/notes/btsfold_2026-10-13) |
| 10-13 | PLAN_KS_REALIZE=1 re-cut on the StC-first recipe | plan infeasible at runtime: bts_depth_error level 50 > 48 at blk0.softmax (queue74); best old -0.022 |
| 10-13 | Rational rescaling (+3 levels) | perseus-9e: dnum 7 +0.120 s/token, dnum 8 not buildable on the 27-tower chain, dnum 9 segfaults; QP bar 1705 (perseus-deps/notes/rr_2026-10-13/PROBE.md) -> CLOSED (user) |

## 4. Closed (do not redo)

- **10-13 FAILED (user): rational rescaling (rr-probe = perseus-9e, perseus-deps/notes/rr_2026-10-13/PROBE.md).** Cheddar-style RR on 27-bit mains (pair 2^19.58 + 2^21.39, app levels ~2^40): +1 level saves 1 bootstrap, +3 saves 12 = -0.203 s/token at measured route costs (dense 16.37 / 512 15.49 / 1-slot 8.70 ms). The 2 extra primes force more key-switch digits: dnum 7 MEASURED +0.120 s/token (queue81, GPU 4, isolated build); at today's QP (<= 1705, no wider sparse-key modulus) RR needs ~8 digits = ~+0.24 -> net loss ~+0.06. Only dnum 6 with digits by bits at QP 1746.3 (bar lift + q0*P sparse-key fix) would net -0.17..-0.19: not pursued. Gotchas: dnum 8 unbuildable on the 27-tower chain (OpenFHE estimate), dnum 9 kernels broken, PLAN_DELIB_LANDING_REMAP chains overlapping pairs.
- **10-12 evening (perseus-7c, CPU what-ifs on the StC-first recipe, control 155 planned):** dense landing 26 / 24 -> 155 / 156 (only
  22 pays: 143, the per-block residual refresh); 1-slot landing 14 / 12 -> 155 (not binding), 18 / 20-24 -> 160 / 164 (a lower SPRU
  key level costs refreshes); folding the GELU output-scale plaintext + its 0.98 constant -> 155 (0); dense raise drops on StC-first
  (Context.cu:754 keeps no variants; the planner's slack ignores PLAN_STC_FIRST_ENTRY=46: re-enabled they cost +11 refreshes, with the
  entry respected <= -0.02 s); CutMax dense pairing (already complex-packed, CUTMAX_PAIR). Production route costs (light trace, 10-12
  22:00): dense 16.1, 512 route 15.6, SPRU 8.6 ms per call; bootstraps 2134 of ~4300 ms; softmax ops 567, LN ops 364, linears 428.
- **10-12 (this session):** AKS +0.029..+0.059 and a predictable keygen error (deprecated, refused without
  `FIDESLIB_AKS_ALLOW_INSECURE`); LAR +0.016 (11/28); #4 CUDA graphs for the static steps (host side exhausted after
  the sync bundle, token ~95 % GPU-busy); StC-first on SparseB (math: <= -0.02 and +3 input levels per 512 refresh);
  route swap to classic-sparse StC-first (what-if: 4 more refreshes cancel -1.65 ms/call); EvalMod series 14:6 on
  SparseB (-1.6 ms/call but KL ~10x worse, <= -0.07; dense K16 degree 14 r5 is already depth-optimal; SPRU has no
  Chebyshev); LN_MOMENTS (perseus-7e: 40 vs 29 one-slot and 52 vs 40 dense refreshes); StC-first per-site shipped mix
  (superseded by the accurate variant); CHEB_ALIGN +0.005; STC_SINGLE + RAISE_DROP +0.051; BTS_DIM1 (auto optimal);
  SparseB baby step 8 / 4 (worse than 16); LT_CHUNK (worse); 16-slot route (k=16 costs the same as 512); ENCODE_NOSYNC
  alone (superseded by the bundle); CUDA_DEVICE_MAX_CONNECTIONS=32 +0.001; CUDA_SCALE_LAUNCH_QUEUES=4x +0.032; dense
  keys `OPENFHE_KSKA_SEED=0` +1.5; ChaCha8 regen (dots -2 %).
- **Earlier:** SHARED_STREAM=2 +0.53, NTT_CLUSTER=1 +0.58, SCRATCH_ASYNC=1 +0.21, PT_STREAM +0.40..+0.55, pinned
  pointer-table ring +0.11, FIDES_NTT_M32=4 (wrong), LEVEL_BUDGET 5:3 +12 %, ring 2^15 (infeasible at 128-bit),
  single-prime levels (accuracy), TC_NTT +12.6, KS slots (no effect), EvalRound path (net loss), EvalMod depth
  (accuracy lever, not speed).

## 5. Off-limits (user rulings)

Multi-GPU; model- or token-specific levers (e.g. the LN position-0 sink); softmax/CutMax iteration trims; nothing more
from Sylph; argmax and feedback stay on the server (no client-side argmax).

## 6. Worktrees (`/data02/users/azirilli/perseus-wt`, after the 2026-10-12 17:45 cleanup)

Only **stcf9** remains: land10 = origin/dev (perseus 403d919, FIDESlib b13d880), built (deps_n32_stcf9); the A-arm
base for the next A/Bs. Every other worktree is archived at `refs/archive/wt/<name>` (perseus repo; FIDESlib side in
the FIDESlib repo): 60 by the disk-cleanup session, plus sbstc, lnm9, stcf8, land9 and land by perseus-01 at 17:45.
Manifest and restore command: `perseus-deps/notes/wt_archive_2026-10-12/MANIFEST.md`. Restore = `git worktree add
<path> refs/archive/wt/<name>` in both repos, then rebuild (scratchpad `<name>/build.sh` recipes).
