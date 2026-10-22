# perseus architecture notes

The crate map, the seams and the bootstrap routes, then design notes for the modules whose headers point here: the
as-built facts that are too long for a module header. Level/degree rule deviations: `docs/QUIRKS.md`; Config fields
and their legacy env names: `docs/ENV_INVENTORY.md`; the client/server wire format: `docs/WIRE.md`; measured times
against the C++ runtime: `docs/REPORT.md`.

# Overview

**Crates** (dependency order; `pc-core` and `pc-shadow` build and test without CUDA):
| crate | what it is |
|---|---|
| pc-core | chain presets + prime search, scale/CRT tables, host encode/decode, key sampling + KSKA/KSKB expanders, galois, bootstrap constants (DFT factors, SparseB, EvalMod table, Chebyshev long division), PCMM/CCMM key material, `Word`/`ModMul` + host strategies |
| pc-shadow | the one level/degree/scale/tag engine: `Shadow` rules (FIDESlib FLEXIBLEAUTO), `Route` landings, EvalMod program builder, pack-tag algebra, graph loader + planner bookkeeping (`sim`), named `Quirk`s |
| pc-config | typed `Config` (250 fields) / `PlanConfig` (84), presets, legacy env bridge, load-bearing stamp |
| pc-cuda | cudart FFI, `Device` selection (opt-in guard `PERSEUS_GPU_GUARD`), `Stream<Lane>`/`Event`/`GraphExec`, the one-arena allocator |
| pc-kernels | CUDA templates over (Word, ModMul), one `KernelTable` per instantiation (u32 x {Barrett64, Barrett32, Montgomery}, u64 x Barrett64, u16 compile-only) |
| pc-cublas | cuBLASLt int8 -> int32 GEMM FFI |
| pc-gpu | device runtime: `Context`, `Executor` lanes, NTT / basis conversion / key switching, leaf ops, rescale, plaintext store + device encoders, keys/keygen/encrypt/decrypt, bootstrap routes + staged API + batching + CUDA graphs, PCMM / CCMM |
| pc-program | graph + plan -> `Program` (resolve, plan application, compile-time level check, K-batching, buffer assignment), replay, weight residency ring, KV store |
| pc-py | PyO3 module `perseus._pc`: Config, Session, eager `Fhe`/`Ct`, Shadow, Tag, Program, matmul services |
| pc-tools | `pc-params`, `pc-check` (oracle cross-checks), `pc-bench` |

**Seams and their implementations** (bold = default):
| seam | type | implementations |
|---|---|---|
| word | `Word` / `DevWord` | **u32**, u64 (n64 chain), u16 (compile-only) |
| modular product | `ModMul` / `DevStrategy` | **Barrett64**, Barrett32, Montgomery (within 1.3 % of each other; Shoup for constant operands everywhere) |
| key switching | `ks::KeySwitch`, `ks::KskSource` | **Hybrid dnum 6**; **Regen** (in-kernel `a` from the seed), Cached |
| key layout | `keys::KskLayout` | **PackedSeeded 28-bit `b`**, Dense |
| rotation | `rotate::Rotator` | Plain, Hoisted, LadderFused (rotate-and-sum) |
| rescale | `rescale::RescaleKind` | CompositeOnePass, Sequential |
| encode | `encode_dev::EncoderKind` | HostFft, **DeviceFft**, CoeffRows, CoeffRowsDevice (streamed Llama weights), Eval |
| plaintext residency | `pt_store::PtStorePolicy` | **StageAhead**, ResidentAll |
| bootstrap route | `bts::BtsRoute`, `bts::RouteKind` | Dense, StcFirst (Plain / Folded), Sparse (SparseB, classic multi-stage for 8192 slots), Spru |
| mod raise | `bts::raise::Raise` | **Encapsulated** (per-drop variants) |
| EvalMod | `bts::evalmod::EmMode` | Complex, Real (real-payload route) |
| matmul GEMM | `matmul::gemm::GemmBackend` | **CublasLt** |
| execution | Python `ops` | GPU eager (`_pc.Fhe`), planned (`Program.run`), Shadow-backed numpy fake, fake capture (graph v2) |
| placer (Python) | `plan.placer` | **min-cut**, ILP, Orion, DaCapo, Fhelipe baselines |

**Routes** (n32 chain, K = 1, one ciphertext, session (packed) keys, ms; `pc-bench bts --all --k 1 --keys packed`):
| route | slots | ms | used by |
|---|---|---|---|
| dense (shipped order) | 32768 | 20.3 | fallback |
| StC-first | 32768 | 17.0 | GPT-2 / Llama dense refreshes |
| SparseB | 512 | 17.4 | GPT-2 tH-periodic values |
| SparseB | 256 | 16.3 | Llama |
| SPRU (s = 1, h = 64) | 1 | 10.0 | constants, CutMax |
| classic multi-stage | 8192 | — | Llama hand-off decode |

# Rust crates

## pc-cuda memory (`crates/pc-cuda/src/mem.rs`)
- Buffers are typed by their element, `DBuf<'a, S, T: Pod>`, with lengths and offsets in elements. Ciphertext data
  uses `T = W` (the chain's word: u16/u32/u64); constant tables use their own element types (`DevPrimeConst`, `u32`
  row tables, `f64`), so there is no byte-level view to get wrong at call sites; kernels receive `*const c_void` and
  read their instantiation's W.
- Type-level safety:
  * a `DBuf<'a, S, T>` borrows the arena (`'a`): the arena cannot be freed while any buffer exists, and a buffer
    cannot outlive its storage;
  * `S` is the lane whose stream orders every access; launch wrappers take `&Stream<S>` with `&DBuf<S>` (read) /
    `&mut DBuf<S>` (write), so a kernel on lane T cannot touch a lane-S buffer and one launch cannot read and write
    the same buffer through two handles;
  * moving a buffer to another lane consumes it (`handoff`, `Ready::join`): the event is recorded on the old lane
    and the new lane waits;
  * host reads (`download`) take the owning lane's stream and synchronize that stream only.

## pc-core ccmm_keys

CCMM keys (Cheon-Kang-Lee ePrint 2025/1957 batch ct x ct matmul) on the four bottom moduli q0..q3 of the chain (the
`stc_high` output, limbs 0..3) plus one 31-bit special prime P; every key is hybrid with one digit per q prime.

| key | switch | role |
|---|---|---|
| `fwd` | ring 2^16 secret s -> s'(X^8), s' ternary of degree NS = 2^13 | ring switch of the 16 input ciphertexts into 128 small-ring column ciphertexts |
| `back` | s'(X^8) -> s | the switch back |
| `autos[t]`, t < 128 | sigma_h(s') -> s', h = 2 KY t + 1 | the CMT's automorphisms; t = 0 is the identity and still switches (as FIDESlib Ccmm.cu) |
| `rlk` | s'^2 -> s' | relinearization of the product's a a' term |

Layout (u32 residues): a key is `[digit j < 4][prime i < 5 (q0..q3, P)][a, b][n]` in coefficient form; `autos` are
128 of them back to back. The device reads the NTT form (`eval_form`, negacyclic NTT natural -> bit-reversed with the
`NttTab` order). The numpy original used a SHAKE-256 stream; the port samples from ChaCha20 (distribution only).

## pc-core keys: the KSKB (u64) lane

s[13] = (digit & 0xF) | hi32(p) << 4, s[14] = lo32(p), s[15] = "KSKB"; 8 coefficients per ChaCha12 block
(little-endian word pairs), escalation stride n/8, accept v < floor((2^64-1)/p) p. Valid for digit < 16 and p < 2^60.
The device twin (`chacha.cuh::expand_slot` at W = u64) is compared against `kskb_expand` by pc-kernels-selftest; the
keystream is pinned by `tests/kskb_frozen.rs` (a change invalidates every serialized u64 key). The u32 lane (KSKA) is
value-identical to FIDESlib `expand_coeff` and checked against every dense `a` of the d3 dump.

## pc-core pcmm_keys

PCMM keys (ePrint 2024/1284 Alg. 2; Sylph arXiv 2601.18511). The split ciphertext (2 limbs q0, q1 after the PCMM's
rescale) is key-switched from the ring 2^16 secret `s` to `s'(X^16)` with `s'` a ternary secret of degree N1 = 2^12
("fwd"), its coefficients regrouped into MLWE pieces of degree np = 2t (rank r = N1 / np), multiplied by the integer
weights, packed back into ring-2^12 ciphertexts with r small-ring key switches ("pack", one per MLWE component
s_l -> s'), composed into ring 2^16 under s'(X^16) and switched back to `s` ("back"). Every key switch has one digit
per q prime and one special prime P (31-bit, NTT-friendly for 2^17, distinct from the chain's split moduli).

Layout (u32 residues, the device's): a key is `[digit j < 2][prime i < 3 (q0, q1, P)][a, b][n]`; the pack keys are `r`
of them back to back. `PcmmKeys` holds them in coefficient form; `eval_*` give the NTT form the device reads
(negacyclic NTT, natural order in, bit-reversed out, tables `psi^brev` as `NttTab`).

## pc-gpu CCMM (`crates/pc-gpu/src/matmul/ccmm.rs`)
CCMM: batch ciphertext x ciphertext matrix products on SinC operands (Cheon-Kang-Lee, ePrint 2025/1957, Alg. 2-4),
a port of FIDESlib `Ccmm.cu` (`ccmmUploadKeys`, `ccmmTranspose`, `ccmmMul`) without static buffers, device syncs or
per-call allocation: every buffer is carved from the MemPlan (`Ccmm::bytes`), the PPMM's int8 GEMM is one batched
cuBLASLt plan made (and autotuned) at setup.

A matrix is 16 ciphertexts at the split of the CCMM (limbs 0..3 = q0..q3, eval form, after `stc_high(skip 1)`:
SinC_{1024, 2^16}). Per matrix: coefficient form, key switch of `a` from s to s'(X^8) ("fwd") over {q0..q3, P},
split by the stride-8 ring switch into 128 column ciphertexts of R_{2^13} (`X = [part][prime][col][NS]`; column
j holds coefficients i + 128 u = row i, batch coefficient u of R_64). Products (CKL Alg. 4): CMT (Alg. 3:
monomial twist, TWEAK, automorphism key switch per column, inverse TWEAK) of the right operand unless it is given
transposed, the four PPMMs over R_64 as one batched int8 GEMM per prime (64 NTT points of Y, [B;A] [B'|A']:
256 x 256 x 128, balanced base-256 digits, 16 digit pairs summed per exponent in int32, exact), recombination,
the two CMTs of the products, relinearization (s'^2 -> s'), the ring switch back ("back") and the chain NTT.
Inputs are read before any output is written, so outputs may overwrite inputs (`mul_inplace`).

Leaf services the Python callers (`impl/ccmm.py`, through `sinc.py` / `ccmm_attn.py` / `ccmm_mha.py` /
`ccmm_paged.py`) reach: `Ccmm::{mul, mul_inplace, transpose}` (bts.ccmm_mul / ccmm_transpose), `sinc_state` +
`split_nf` + `sinc_boot` (bts.begin / stc_first_input / set_scale + mod_raise .. stc_first_output + stc_high skip 6),
and the metadata / small ops in `leaf` (set_scale, noise_factor, scale_noise_factor, set_slots, clone, mult_int,
wsum_raw); stc_high, merge_monomial and export/import_limbs are the existing `Staged::stc_high` and
`crate::limbs` functions.

## pc-gpu CUDA graphs (`crates/pc-gpu/src/graph.rs`)
What a replay must not see change, and how it is held:
* per-call constants pushed through `Ops::ring` (the raise adjust, iteration scalars): the capture runs inside a
  `ConstRing` lease, so the pinned slots holding the captured values (one per (CF, input scale) = per key) leave
  the rotation until the entry is dropped;
* route caches filled on demand (`StageCache` of the StC-first route): a miss is refused during the capture
  (`CaptureLock`); the first call of a key runs eagerly before the capture, which warms the cache; an eviction
  during a later eager run invalidates every entry (their tables may have been rebuilt elsewhere);
* everything else a refresh reads (keys, plaintexts, EvalMod constants, work buffers) is setup-carved.
Address stability: gathering into `BatchSlots` (a gather + scatter is two copy launches per member) beats
re-capturing per input address (a capture + instantiate of a refresh is milliseconds of host time).

## pc-gpu CtS / StC stage (`crates/pc-gpu/src/bts/cts.rs`)
Seam `LinearTransform`, implementation BSGS collapsed-FFT stage: one CtS or StC stage (FIDESlib
`LinearTransform` with hoisted baby-step rotations, `ltStep`), for K ciphertexts, bit-exact with FIDESlib's
single-GPU path (generalized to any giant-step count and a final offset rotation):

1. ModUp(c1) of the input (one decomposition shared by every baby step);
2. `cts_fused`: the baby rotations (never materialized) times the stage's plaintexts, summed per giant step j in
   the extended basis Q u P (two giant steps per launch; G > 2 runs several launches over the same ModUp);
3. giant steps, Horner from the last (FIDESlib's loop `j = g-1 .. 0`): acc = res_{g-1}; for j < g-1:
   ModDown(c1 of acc), ModUp, rotate by `b * stride` with c0 kept extended (`ks_dot` c0 mode 2), acc = res_j + that;
4. the stage end on acc = res_0: (a) offset rotation `rotOut[0]` != 0 (the last StC stage pays the accumulated
   LT-friendly offset): ModDown(c1), ModUp, rotate with c0 extended, full ModDown (degree 2 output);
   (b) `fuse`: ModDown fused with the rescale (exact HPS conversion from P u {q_top-1, q_top}, FIDESlib
   `modDownRescale`; degree 1 output two limbs lower); (c) plain ModDown (degree 2 output).
   The result is written into the caller's ciphertext buffer.

Plaintexts: Q rows 0..=top then the special rows, one width `ptw` per stage (FIDESlib FIDESLIB_LT_COMPACT: a
diagonal polynomial in X^(2^s) has a block-constant NTT row, stored at width N / 2^s, mask `0x7F000000 | s`).
Built from dumped rows (oracle gates) or encoded at setup from slot values (`pc_core::bts_setup` diagonals).

## pc-gpu EvalMod (`crates/pc-gpu/src/bts/evalmod.rs`)
Seam `ModEval`, implementation: the EvalMod program executor (moved onto
pc-gpu). The program is `pc_shadow::evalmod` (FIDESlib btsEvalMod / approxModReduction / evalChebyshevSeries with
lazy Paterson-Stockmeyer / double angle replayed on metadata, the device ops recorded with their integer
constants), peephole-fused (`EmProgram::fused`); here every op runs once for a batch of `kk` ciphertexts:

* complex payload (dense route): conj(x) by key switch; Re = x + conj, Im = (x - conj) X^{3N/2}; both chains run the
  same program, as ONE batch of 2K when 2K fits a launch (`ks::DOT_KMAX`, the KS plans' k_max), else one after
  the other; out = Re + Im X^{N/2}; EvalMod's integer post-factor is 1 on composite chains.
* real payload (`Route::real`): x + conj(x), one chain (FIDESlib approxModReductionReal).

Relinearization: tensor (d2 into the key-switching workspace and the ModUp coefficient buffer), ModUp, inner
product with the relin key (`KskSource::Regen`: `a` regenerated in-kernel, the Q rows fused with the ModDown
epilogue; `Cached`: one dot + `md_epilogue`), the program's x2 / + constant epilogue. Rescales are the composite
one-pass kernels with a preceding MulConst folded in (exact; FIDESlib's EvalMod does ModDown then a separate
rescale, so the rescale is NOT fused into the ModDown); on a composite-degree-1 chain (n64) the same
kernels drop one prime (`ndrop` = d).

## pc-gpu PCMM (`crates/pc-gpu/src/matmul/pcmm.rs`)
PCMM: y = x W at the StC-first split (ePrint 2024/1284 Alg. 2; Sylph, arXiv 2601.18511), a port of FIDESlib
`Pcmm.cu` (`pcmmUploadKeys`, `pcmmMultiPlanes`) without static buffers, device syncs or per-call allocation:
every buffer is carved from the MemPlan (`Pcmm::bytes`), the int8 GEMMs are cuBLASLt plans made at setup.

Per input (a split ciphertext: limbs 0..3, eval form): coefficient form, exact rescale by q3 then q2, key switch
of `a` from s to s'(X^16) over {q0, q1, P} ("fwd"), the switched (a, b) regrouped as MLWE rows (A: N1 columns, B:
np = 2t columns; `rows_in = nch n_in` rows). Then Y = W X mod q_i for both parts as balanced base-256 digit
planes: 4 x nW int8 GEMMs summed per exponent in int32 (exact under `exact_k_limit`), recombined mod q_i. Per
output: the A rows packed back into 16 ring-2^12 ciphertexts with r small-ring key switches ("pack"), the B rows
added, composed into ring 2^16 under s'(X^16), switched back to s ("back"), lifted by q2 q3 into limbs 0..3
(limbs 2, 3 zero) and returned to eval form. Inputs are consumed (read into the GEMM operands) before any output
is written, so outputs may overwrite inputs (`product_inplace`).

Scale: the output carries `x`'s split scale over the integer weights' scale (`nf_out = nf_split / w_scale`); the
driver (`linear`) stamps `w_scale 2^shift` into the inputs' scale before their StC (perseus `scale_noise_factor`),
so the refresh sees |y| / 2^shift, and multiplies the refreshed outputs by 2^shift (perseus `mult_int`).

## pc-gpu SPRU (`crates/pc-gpu/src/bts/spru.rs`)
The s = 1 route by SPRU (FIDESlib `Spru.cu`: `spruSetup` / `spruBootstrap`): decryption in the exponent over a
block-binary key s' (h blocks of B = N/h coefficients, one 1 per block, s'_0 = 1) instead of a mod raise.

exp(2 pi i m / q0) = prod_b sum_j s'_(bB+j) exp(2 pi i C_(bB+j) / q0), C the LWE columns of the bottom ciphertext
under s'. The n = 2 components (m0, m1) = the coefficients X^0 and X^(N/2) are one complex slot. Steps:
1. FLEXIBLEAUTO adjust to integers m = v sf(L) 2^-correction, drop to the bottom composite level;
2. dense -> block key switch at the bottom (`atob`, bottom-only shape);
3. INTT, extraction of the 2n bootstrapping plaintexts E_u (pc-kernels `spru_extract`), device encode at the top;
4. acc = sum_u cs_u E_u (cs_u = Enc(S_u), the bootstrapping key ciphertexts at the top), one rescale;
5. trace (Accumulate over the h n period: rotate_sum rounds of `acc_bstep - 1` hoisted rotations);
6. product over the h blocks (log2 h rotations + relinearized products + rescales);
7. 2 Im x = (conj x - x) i, then the mask recombination m0 + i m1 (two plaintext products, no rescale: the
   output is degree 2, like every route's raw exit).

The output is the slot mean (a 1-slot fold) in every slot, 2 * levels(product) + 2 levels below the top.
All device state is carved at setup (`Spru::bytes` -> MemPlan); nothing allocates per call.

## pc-gpu SparseB (`crates/pc-gpu/src/bts/sparse_b.rs`)
SparseB precomputation of one sparse route (FIDESlib `SparseB.cu` + `RawCiphertext.cu::buildSparseB`, eprint
2026/1023 Alg. 4/5): the repetition-aware depth-1 CtS and StC of `slots` = n/2 complex slots repeated r = N / n
times, as one BSGS stage each (s = 2n / r diagonals, baby step min(s, 16)), followed by the block partial sums.

Route (one precomputation per (slots, raise drop)):
  raise to top L - d drop (seam `Raise`, encapsulated) -> fold: Accumulate(4, slots, N/2 / slots) ->
  CtS: BSGS over the P diagonals (plain ModDown, degree 2) + Accumulate(4, s, r/4) ->
  EvalMod: conj-add + the K = 24 degree-20 series + 5 double angles (`EmMode::Real`: the conj-add of
  `SparseBCoeffsToSlots` runs as the real EvalMod's input step, the same key switch and add) -> rescale ->
  StC: BSGS over the Q diagonals + Accumulate(4, n, r/2) + rotate(n/2)-add -> x 2^correction.

Plaintexts are encoded at setup straight at the level they are used (`PtSource::Slots`, exact integer rounding at
sf(top)); FIDESlib encodes them at the chain top and re-levels on first use (two roundings), so the device words
differ from FIDESlib's by the rounding of that second step (decrypted stages agree to the bootstrap noise).

## pc-gpu StC real payload (`crates/pc-gpu/src/bts/stc.rs`)
SlotsToCoeffs of the shipped order (FIDESlib `btsStCEnter` / `btsStCStage`): the three BSGS stages of
`pc_core::bts_setup` (OpenFHE `EvalSlotsToCoeffsPrecompute` in FIDESlib's LT-friendly layout, scaleDec = q0 / sf(0)
folded into the last stage), encoded at `pc_shadow::route::stc_stage_levels` (the EvalMod output's effective level
+ d e), run by the same `LtStage` as CtS. Stages 0..n-2 end with the fused ModDown-rescale; the last one pays the
accumulated LT-friendly offset (rotOut[0]) and keeps degree 2 (the planner's landing convention).

Real payload (FIDESLIB_BTS_REAL, `stcRealA0`): stage 0 with the entry of every diagonal that reads coefficient 0
halved. A real slot vector's coefficients satisfy m_{N-k} = -m_k, so one EvalMod chain on the coefficients
0..N/2-1 plus this stage give w with slots = w + conj w (the finish adds the conjugate). In OpenFHE's order the
input's coefficient 0 sits in StC input slot 0 and diagonal `ij` multiplies the input rotated by j - offset
(j = ij mod g, offset = (numRotations + 1) / 2 - 1), so the halved entry is slot (offset - j) mod slots of the
BSGS diagonal (before the LT-friendly pre-rotation). Here the diagonal's slot value is halved directly; FIDESlib
edits the encoded integers (it keeps every other slot bit-identical to the shipped plaintext).

## pc-gpu StC-first route (`crates/pc-gpu/src/bts/stc_first.rs`)
The dense route in StC-first order (`RouteKind::StcFirst`; FIDESlib FIDESLIB_BTS_STC_FIRST, Lattigo's
"DecodeThenModUp" / slim order): SlotsToCoeffs runs on the INPUT ciphertext, before the raise, at limbs
`entry .. 2d-1` instead of the ~20-28 limbs of the post-EvalMod ciphertext, and EvalMod's output is the bootstrap
output (the landing moves up by the StC depth, FAKECAP_STC_DROP = 6 levels on n32):

  begin -> stc_first_input (drop to the entry, StC stages; the ciphertext is now at the "split": its coefficients
  carry the slots, where a PCMM/CCMM consumes it) -> mod_raise (prescaled in mode 2) -> fold -> CtS x4 -> EvalMod
  -> stc_first_output (x 2^CF = corFactor << deg, slot restore)

Modes (FIDESLIB_BTS_STC_FIRST): 1 = every StC stage fixed at setup, the last one carrying 2^-deg (the shipped StC
plaintexts hold scaleDec ~ 2^deg, which the shipped order applies after EvalMod); 2 (shipped) = the last stage
also carries ModRaise's FLEXIBLEAUTO adjust (2^-cf, the input scale's ratio to its level's, the raise target's
scale), so it sits at limb 2d-1 and the raise takes its prescaled path: the entry needs one composite level less.
The mode-2 last stage depends on the call (CF, input scale, raise variant): it is re-levelled on the device per
key into a bounded LRU cache (`StageCache`, slots from the MemPlan; FIDESlib's `stc_first_last` map, capped at 24
and erased smallest-key-first). The stage is built at a reference CF (`StcFirstSetup::cf_ref`, 20) unless the
call asks for its own (`own_cf`: the PCMM refresh, perseus PF_PCMM_CF / `btsStcFirstInput(.., ownCf)`): a smaller CF
is the exact integer 2^(cf_ref - cf) put on the input first, so one entry per input scale serves every site, at
the cost of cf_ref - cf bits of plaintext precision (which the PCMM's integer products expose).

Re-levelling (FIDESlib `relevelPlaintext`): the integers of a plaintext reconstructed from its 4 bottom limbs,
times `factor sf(new) / sf(old)`, rounded, reduced into every row of the new level (`small_int` kernel), NTT. Here
the source is the stage's diagonals encoded once at 2^SRC_LOG2 (setup, host), so a cache fill is one `small_int`
launch over every diagonal plus one NTT launch; the per-fill table goes through a pinned buffer and an async copy,
ordered on the compute lane (an event guards the pinned buffer's reuse; no device-wide sync).

Inputs deeper than the entry take the shipped order (FIDESlib `btsBegin`: `inLimb >= stc_first_entry`) and land
where the shipped dense route does; this route carries the shipped StC stages for that (and for `stc_high`).
Sparse StC-first (FIDESlib `stc_first_mask`) and the single-LT route are not ported (n32 ships
FIDESLIB_BTS_STC_FIRST_ROUTES=dense).

## pc-gpu bootstrapping (`crates/pc-gpu/src/bts/`)
- seam `BtsRoute`: one implementation per route family, dispatched by `RouteKind`; `landing` (output metadata), `run`
  (the whole refresh of K-batched ciphertexts), `staged` (the stage API, when the route exposes it), `mem` (device
  bytes it needs, `MemPlan::reserve`d before `allocate`).
- `Staged`: FIDESlib's staged bootstrap (`BootstrapStages.cuh`) 1:1, K-batched over `&mut [Ct]`:
  `begin -> [stc_first_input] -> mod_raise -> fold -> cts_stage(0..n_cts) -> eval_mod ->
  (stc_first_output | stc_enter -> stc_stage(0..n_stc) -> finish)`, plus `stc_high` (StC outside a bootstrap).
  Every per-call quantity lives in a `BtsState` value (no globals, no scopes); `run_staged` is the sequence.
- `BtsPrecomp` per (slots, raise drop): the linear-transform stages (CtS/StC plaintexts from `pc_core::bts_setup` at
  the levels of `pc_shadow::route`, encoded at setup, resident), the EvalMod program (`pc_shadow::evalmod`) with its
  device constants, and the raise tables. Variants = exactly the setup's drops.
- seams `Raise` (raise.rs: encapsulated, per-drop variants), `LinearTransform` (cts.rs: BSGS collapsed-FFT stage,
  shared by CtS and StC), `ModEval` (evalmod.rs: the EvalMod program executor).
- Routes: dense (shipped order: raise, CtS, EvalMod, StC, finish), sparse (512/256 slots, SparseB), SPRU (s = 1,
  sparse secret), StC-first (StC on the input side), classic multi-stage sparse (8192 slots); `fold_bootstrap` and
  sparse auto routing in fold.rs; K-batching across ciphertexts in batch.rs.

## pc-gpu key switching (`crates/pc-gpu/src/ks.rs`)
- ModUp of K ciphertexts' polynomial: copy its live Q rows into the coefficient workspace (`Executor::bconv_work`),
  one INTT launch, then one plain fast-base-conversion launch over every live digit of the K ciphertexts (digit
  limbs -> the other live Q limbs and P) and one NTT launch over all targets. Digits above the ciphertext's top are
  dropped (band-limited digit count: 6 digits at the top of the n32 chain, 2 at level 40).
- Inner product with R keys in one launch (`ks_dot`, K <= 8 ciphertexts x R keys, each key element read once),
  optionally adding `P * c0` (c0 mode 1) and writing at the automorphism's slot (FIDESlib's "dot then sigma"), or
  with the key's `a` regenerated in-kernel (`ks_dot_regen`, relinearization only: K <= 4, no automorphism).
- ModDown P -> Q: INTT of the special rows, one plain conversion launch, NTT of the Q rows, then
  `(x - conv) * P^-1` (in place, into another ciphertext, or fused into an accumulate epilogue).
  `(P c0 + d - conv) P^-1 = c0 + (d - conv) P^-1` exactly, so adding c0 before or after the ModDown is the same
  residue; the automorphism is applied before the ModDown (it does not commute with the plain conversion).
- Plans (row tables, conversion job tables) are built per level at setup for `k_max` ciphertexts ordered by
  ciphertext, so a K < k_max launch runs a prefix. The workspaces (`KsWork`: ModUp digits, the extended-basis dot
  output, the ModDown conversion) belong to the `Executor`; nothing is global.

## pc-gpu keys (`crates/pc-gpu/src/keys.rs`)
Device key material: the `KskLayout` seam, switching keys (`DevKsk`), the key set (`KeySet`: relin,
conj, rotations by signed step, bootstrap encapsulation keys as named fields) and the device secret.

Storage is carved from the arena at setup: size it with `KeySetSpec::bytes` (+ `DevSecret::bytes`) and
`MemPlan::reserve("keys", ..)` before `MemPlan::allocate`, carve with `KeySet::carve` before `Arena::freeze`, then
fill it with `keygen` (device generation) or `import_material` (host material: oracle dumps, saved keys).

Row geometry of a key (`KskShape`): digits `0..ndig`, each holding the Q rows `0..=q_top` then the `kp` special
rows, in that order ("slot" = position in that list). A full key has `q_top = nq - 1` and `ndig = dnum`, so slot
== prime id and the dense layout is exactly the `[digit][prime id][N]` form `KsArgs::kb/ka` read with
`key_digit_stride = rows * N`. A banded key (`q_top < nq - 1`: rotation keys used only below a level, and the
bottom-only dense->sparse encapsulation key of OpenFHE patch 0013) drops the Q rows above `q_top` and every
digit that starts above it; its special rows sit at slot `q_top + 1 + j` (`KskShape::slot_of`).

Relation stored (OpenFHE `KeySwitchHYBRID::KeySwitchGenInternal`, eval form, slot m = prime id t of digit d):
  b = -a s_new + e + [t in digit d] (P mod q_t) s_old,   a = KSKA(seed, w13 = d, p_t)  (pc_core::keys spec)
Roles: relin s_old = s^2, s_new = s; rotation by r (Galois g = 5^r) s_old = s, s_new = sigma_{g^-1}(s) (the
oracle's convention: the key-switch output is then mapped by sigma_g); conj likewise with g = 2N - 1;
dense->sparse s_old = s, s_new = s_sparse; sparse->dense s_old = s_sparse, s_new = s.

## pc-gpu mod raise (`crates/pc-gpu/src/bts/raise.rs`)
Seam `Raise`: the mod raise of a bootstrap (FIDESlib `Bootstrap.cu::ModRaise` with the sparse-secret
encapsulation and the exact post-raise constant of `SmallInt.cu`), bit-exact with FIDESlib:

1. bring the input to the composite bottom: pay a pending rescale; unless `prescaled`, multiply by the FLEXIBLEAUTO
   adjust `(sf_target / nf) (q_drop / sf(top)) 2^-correction` and rescale; drop to top limb d - 1 (the scale is
   then nominally the target's, sf(target top));
2. dense -> sparse key switch at the bottom (`enc_dense_to_sparse`, OpenFHE patch 0013's bottom-only key);
3. coefficient form of the two bottom limbs, composite raise to the third limb (`mod_raise` kernel: each Garner
   term centred on its own prime, as FIDESlib `compositeModRaise_`), the exact scaling of the small-integer
   coefficients by `c = cts0_const * 2^t` reconstructed from those 3 limbs into every limb up to the target
   (`small_int` kernel; FIDESlib `smallIntScalarMultiply` on the raised ciphertext kept in coefficient form,
   FIDESLIB_MODRAISE_COEFF) and one NTT of every limb; the scale bookkeeping gains 2^t;
4. sparse -> dense key switch at the raised level (`enc_sparse_to_dense`).

Composite degree 1 (the n64 chain): the bottom is q0 alone, so step 3 is one kernel (`small_int1`):
row 0 centred, scaled by `c` exactly (c's mantissa times |x| in 128 bits, shifted with round half up) into every
limb up to the target; no `mod_raise` pass and no multi-limb Garner. Not FIDESlib-bit-exact (no n64 oracle): the
rounding is the exact one of x * c with c as a double, where FIDESlib's composite path rounds through its x87
reciprocal of 1 / c.

`Encapsulated` is the only implementation (a GHS-style raise would be another variant of the seam). The per-drop
variants differ only in the target top limb `L - d * raise_drop` and the constant of step 3.

## pc-gpu rotations (`crates/pc-gpu/src/rotate.rs`)
- `Plain`: `rotate(x, r)` = ModUp(c1) -> `ks_dot` with `P * c0` added and the output written at the automorphism's
  slot (FIDESlib `hoistedRotationFused`: dot, then sigma, in the extended basis) -> ModDown. The key for step r is
  OpenFHE's index g = 5^r mod 2N; the kernel scatters by g^-1 = 5^(2N - r) (the "5^(2N - r) twist",
  `pc_core::galois` convention). `conjugate` is the same with g = 2N - 1. Step 0 is a copy.
- `Hoisted`: `rotate_hoisted(x, steps)` shares one ModUp between the steps and runs the dots of as many keys per
  launch as the extended workspace holds; each output is ModDowned separately, so the outputs are bit-identical to
  plain rotations (FIDESlib `rotate_hoisted(ext = false)`).
- `rotate_sum(x, steps)` = x + sum_k rot(x, steps[k]) with the rotations summed in the extended basis and ONE
  ModDown (FIDESlib `rotateSumHoisted`, the perseus `rotate_sum` node): not bit-identical to a sum of plain
  rotations, which is why the ladder keeps the C++ op stream.
- `LadderFused`: `rotate_and_sum(x, start, stop)` = the perseus C++ ladder (`src/bindings/ops.cu`): gaps
  g = |start|, 2g, ... < stop in radix groups of `radix_bits` (LADDER_RADIX_BITS); a group with all its keys is one
  `rotate_sum` over j * g (0 < j < 2^n), any other group (and radix_bits < 2) runs rung by rung as
  acc = x + rot(x, g) / acc += rot(acc, g) — the same add / inplace_add stream, so captured plans match.
- Key tables: every op takes a `KeyGroup` built at setup whose keys are exactly the op's nonzero steps in order (a
  hoisted group, a radix group) or contain the step (single rotations, conjugation); `RotKeys` looks groups up.

## pc-gpu staged bootstrap (`crates/pc-gpu/src/bts/staged.rs`)
The staged bootstrap as a call object (`BtsCall`): what PCMM / CCMM work is interleaved into, and what the
perseus Python drives stage by stage (`impl/bootstrap.py`, `impl/pcmm.py`, `impl/ccmm.py` over FIDESlib's
`BootstrapStages.cuh`). One call = one `BtsState` + the phase it is at; every transition is checked against the
stage order

  begin -> [stc_first_input -> Split] -> mod_raise -> fold -> cts_stage(0..n_cts) -> eval_mod ->
      ( stc_first_output | stc_enter -> stc_stage(0..n_stc) -> finish ) -> Done

and runs on K-batched ciphertexts (`&mut [Ct]`, each a batch of k <= `BtsSetup::k_max`, all with the call's
metadata). The external-op pattern (PCMM / CCMM at the split):

  in = BtsCall::begin(route, cx, xs, &route_spec, CallOpts { own_cf: true, .. })   // state from the INPUT
  in.stc_first_input(cx, xs)            // xs now at the split (coefficients carry the slots)
  ... external op writes ys (same split metadata) ...
  fin = in.finish_call(drop)?           // the input's state under the finish's raise variant, phase Split
  fin.at_split(cx, ys)?; fin.run_to(cx, ys, Phase::Done)?   // mod_raise .. stc_first_output
  rec.external_bootstrap(&fin.external(inputs, &ys[0].meta, desc, None))   // program capture

`run_interleaved` issues several calls stage-major (as perseus `_finish_interleaved_impl`: independent chains side by
side). `BtsRecorder` is the capture hook (perseus `level_hint` + `record_external_bootstrap`): the PCMM caps its
inputs at the StC-first entry (`stc_first_cap`) and records each output as a deliberate bootstrap of its inputs.

## pc-program apply_plan (plan lowering rules)

Resolved nodes in planner order -> ops on program vars, with the plan's decisions applied as the perseus C++ wrapper
applies them at run time (`maybe_apply_planned_bootstrap_after`, `inner_bootstrap_planned`, `bootstrap_hint`,
`maybe_apply_planned_rescale_after`):
* every recorded op except the code's own `bootstrap()` passes the op-exit hook: a placement on its output refreshes it
  (fire-once) on the site's route, realized iff the var is a `rescale_after` anchor (an unbound plan realizes every
  planted landing); any other anchor realizes a degree-2 output in place;
* a hint fires iff its output is in `hint_fire` (bound plans), else by its threshold; a fired hint is the var's
  refresh and consumes a placement on the same var;
* unfired hints, level_hints and clones alias their input; `level_reduce` drops (C++ `max(cur, t)`);
* captured deliberate refreshes keep their captured landing (route inferred from it), PCMM/CCMM finishes become
  `Finish` ops.

## pc-program bind

`bind` puts a compiled `Program` on a device runtime once, at setup (before `Arena::freeze`):
* every `VarId` -> its compiler-assigned slot (`Program.mem`); each slot is one full-layout ciphertext buffer (pc-gpu
  keeps every ciphertext at the full Q u P row layout, so the compiler's limb-exact slot sizes are a lower bound, not
  the carve: one slot = `CtBuf::words(lay, 1)`), plus one scratch ciphertext for the in-place ops without an in-place
  kernel;
* every `PtRef` -> a `PtStore` key `(name, level, Eval)`; `resolve_pts` turns the keys into device row addresses
  (missing names = one error listing them) and reports whether any address moved (a captured graph must then be
  re-captured);
* every rotation step list / conjugation / relinearization -> a setup-built `KeyGroup`;
* every refresh site -> its `Route` and the route family the runtime must hold (`Routes`);
* every scalar -> an exact `f64` (v2 graphs carry it; a v1 graph's 6-digit literal must come from the driver's table
  keyed by graph node, else bind fails naming them);
* every streamed weight row (`PtForm::CoeffStream`, read by a `CoeffSum`) -> its row in the block's weight frame: the
  block's linears (first-use order) laid out back to back, each whole (`StreamLayout`; the ring uploads a linear's
  pinned rows in one copy), the row counts from the runtime's `StreamSource`;
* the per-op argument table (`Step`): kind, slot indices, the compiled output metadata.

Batched ops (`Kind::Batched`) are bound member by member: the members live in separate slots; a K-batched launch would
need them adjacent (compiler slot adjacency is not implemented).

## pc-program kv (KV choreography)

```text
for layer i:   join(i)            compute waits for layer i's reload (stream wait on its Ready event)
               prefetch(i + 1)    H2D of the next layer's page on Copy, overlapping layer i's compute
               ... run block i (reads / appends layer i's cache through page(i)) ...
               store(i)           Copy waits for an event recorded on compute after block i, then D2H
               finalize(i - 1)    layer i-1 is parked on the host (its device page may be refilled)
```
Ordering is carried by the two streams and events only: a page moves between lanes as a lane-typed buffer
(`DBuf::handoff`: the receiving lane waits an event recorded behind the last access; `Ready::join` for a reload), the
D2H of a page and the next H2D into the same device page are both on `Copy` (issue order). The host never waits
(`wait_stored` exists for checkpoints and tests).

## pc-program replay (what a captured graph must not see change)

The first run of a `Bound` executes eagerly (it warms the route caches and produces that token's result), then the same
step sequence is captured, without executing, into one `GraphExec` held by the `Bound` (= keyed by (block, variant));
later runs launch it.
* ciphertext slots, key groups, bootstrap work buffers, route tables: setup-carved, address-stable;
* plaintext rows: `Bound::resolve_pts` before each run; a moved address drops the graph (re-captured);
* streamed weight rows: the block's weight frame (`Bound::stream_base`, set by `residency::StreamRing::adopt` before
  each run) is part of a captured graph's key like the plaintext rows;
* per-call constants pushed through `Ops::ring` (scalar rows, raise adjusts): the capture runs inside a `ConstRing`
  lease, the leased pinned slots keep the captured values until the graph is dropped;
* caches filled on demand (the StC-first `StageCache`): locked during the capture (`CaptureLock`), a miss would be an
  error (the eager run filled them).

Token-boundary work stays outside the graph: the driver writes the inputs (`Bound::input_mut`) before `run` and reads /
decrypts the outputs (`Bound::output`) after it.

## pc-program residency (the weight ring)

```text
ring.stage(store, ex, sh, seq + 1, next, src)   // host encode overlaps the block still on the device
replayer.run(cur, rt)                           // block `seq`
ring.adopt(store, ctx, ex, seq + 1, next)        // compute waits the uploads; Bound::resolve_pts
```
Frames: block sequence number `seq` uses frame `seq % n_in_flight`; staging a frame first evicts what the frame held
(the block `n_in_flight` earlier, finished on compute before the uploads start: `PtStore::stage` hands the slots to
`Copy` behind an event recorded on compute). Evicted slots are refilled lowest-first in staging order, so a block's
rows land in the same slots every token once the ring cycles (a moved address makes the replay re-capture that block
once; `ReplayStats::invalidations` counts it).

Keys: the program's per-block plaintexts (`pt_N`, encoded at use) are namespaced by block in `bind`; masks and named
slot plaintexts are shared names (the store keeps one copy, a frame does not evict a key the next block also reads).

## pc-py CCMM service

A captured CCMM (impl/ccmm.py `Ccmm.apply`) is, in the graph, one external refresh per output sharing the operands' 32
ciphertexts (16 of `a`, then 16 of `b` — given as is, or the prepared K / V of the paged prefill); the compiler merges
them into one `Finish` op whose descriptor is the v2 `args.external` dict (`{op: "ccmm", site, out, n_out,
b_transposed, merge, prepared_b, shift, amax, bmax, out_scale}`). The service runs what ccmm.py `_device` runs, with
every scale an argument:

SinCBoot state (`sinc_state`, btsBegin on `a`'s metadata) -> `stc_high(out 50, skip 1, factor f, nf_ref)` of both
operands -> `Ccmm::mul` -> (merge: `merge_monomial(e, o)` X^1024, pairs) -> `sinc_boot` (restamp at the split's scale,
HalfBTS, StC stage 0) -> x 2^shift,

with the operand factors `fa, fb = sqrt(p bmax / amax), sqrt(p amax / bmax)`, `p = out_scale 2^-shift / c_unit` (x 1/2
merged). `c_unit` (output value per unit factor product) is calibrated on the device at setup from probe matrices
(`Session.ccmm_probe`, the secret still present: ccmm.py `_calibrate`), `nf_ref` / `nf_split` from a token at the
bootstrap output level (`Session.ccmm_prepare`). The CCMM keys are made from the host secret at setup
(`CcmmKeys::generate`).

## pc-py PCMM service

The descriptor is `{op: "pcmm", linear, out, n_in, n_out, t, shift, raise_drop}`. The weights are not in the graph: the
driver declares every linear at session setup (`Session(pcmm=[(name, t, n_in, n_out, planes, w_scale, shift)])`: the
MemPlan sizes the engine for those shapes and carves one resident `WeightBuf` per linear — GPT-2's 48 linears are
~250 MB at t = 32) and uploads the digit planes once (`Session.pcmm_upload`). The PCMM keys are made on the host from
the secret's coefficients at setup (`PcmmKeys::generate`), before the driver drops the secret.

A finish = `linear()`: the inputs copied into the service's own ciphertexts (the StC-first input stages end at the
split in place; the program's slots stay as compiled), the product, one staged refresh per output (finish groups of
one: the shipped plans' PF_PCMM_BATCH is off), x 2^shift, the outputs copied into their slots at the raw landing the
compile pinned.

## pc-py Session (ownership, lifetimes, threads)

One `Session` owns, in one object and with no process-global state: the `Device` guard, the ONE arena
(`MemPlan::allocate`), the `Executor` (lanes + workspaces), the `Context`, the eager `CtPool`, the key set (generated on
the device from a `KeySource`), the key-switching plans and key groups, the op layer (`Ops`), the plaintext store
(`PtStore`) and the encryptor / decryptors.

Lifetimes. Every pc-gpu object borrows the arena (`'a`). The arena is boxed (stable address) and owned by the same
`Dev` as its borrowers, so the borrows are extended to `'static` once, and `Dev`'s field order is the drop order:
everything carved from the arena first, then the arena, then the device guard (`Drop for Dev` syncs the compute lane
before any of it goes). Ciphertext handles (`Ct`) hold an `Rc<Dev>` next to their pool slot (slot first), so a `Dev`
outlives every slot it handed out.

Threads. `Session`, `Fhe` and `Ct` are `unsendable`: PyO3 refuses them on any thread but the creating one. Device calls
that block (setup, decrypt, encrypt, refresh, sync) run with the GIL released (`nogil`): the closure runs on the calling
thread while other Python threads may run, and none of them can reach these objects (the unsendable check), so the
`Rc` / `RefCell` state is never touched concurrently.

# Tools

## crates/pc-tools/src/bin/pc-bench.rs — pc-bench usage

pc-bench: bootstrap benchmarks on fresh keys.

  pc-bench bts      --route dense [--k 1,2,4,8] [--reps 10] [--drop 0] [--cf 6] [--real] [--cts-alt 5]
      device ms per call and per ciphertext (CUDA events on the compute lane, median of `reps` after 2 warm-ups)
  pc-bench bts-prec --route dense [--cf 2..8] [--amps a,b,..] [--real] [--iters 1] [--out docs/acc_n32.json]
      precision on random real payloads (uniform in [-A, A], max anchored to A, every slot), encrypted at the top
      level: rel_err = max |out - in| / A over the slots, bits = -log2(rel_err) (the FIDESlib
      bts_accuracy_sweep methodology); with `--old <bts_accuracy_n32.json>` the old dense row of the same (cf, A).

  pc-bench bts|bts-prec --route 1|256|512 ...                   the sparse routes (SPRU, SparseB): src/bts_sparse.rs
  pc-bench bts|bts-prec --classic --route 8192 ...              the classic multi-stage sparse route: src/bts_classic.rs
  pc-bench ccmm [--prec] [--reps 10] [--k 1,8]                     CCMM per call, SinCBoot, QK^T / PV precision: src/check_ccmm.rs
  pc-bench pcmm [--prec] [--shapes ..] [--k 1,2,4]                PCMM per linear / precision: src/check_pcmm.rs
  pc-bench stcf|stcf-prec|stc-high ...                          the StC-first route, stc_high: src/bts_stc_first.rs
  pc-bench replay --graph <dir> [--plan <dir>] [--block block_0]  a block program replayed on g0: src/replay_g0.rs
  pc-bench modmul [--k 1,8] [--reps 20] [--u64]                  u32 x {Barrett64, Barrett32, Montgomery} on n32 in one
      process: device ms of mult_pt / tensor / mult_scalar / add / NTT / rescale / rotate / mult+relin, and the output
      words compared across strategies (seam contract); --u64 adds (u64, Barrett64) on n64: src/word_seam.rs

Keys: a fresh secret (dense ternary + sparse h=32), every CtS/StC rotation, conjugation, relinearization and both
encapsulation keys generated on the device (dense layout with resident `a`).

## crates/pc-tools/src/bin/pc-check.rs — pc-check usage

pc-check: cross-checks against the FIDESlib oracle dumps (`perseus-oracle/fx/dumps/<name>`).

  pc-check bts  <dump> [--k 1,2,4,8] [--graph]   the d3 finish gates (bit-exact): CtS x4 stages, EvalMod + StC-first
                                                 output, per K; --graph also replays the whole finish as one CUDA graph
  pc-check load <dump>                           a g0 dump: inventory, every file's size validated against params
  pc-check keys <dump>                           a g0 dump's keys + secret in pc-gpu's device layout
  pc-check leaf [--ops a,b] [--k 1,8] <g0>       pc-gpu leaf ops replayed from the g0 dump (words, decrypted
                                                 slots, bookkeeping); see src/leaf.rs
  pc-check stcf <d3|g0> [--k 1,2,4,8] [--graph]   StC-first route + staged call object (src/bts_stc_first.rs)
  pc-check leaf-ks --ops <list> <g0> [--k 1,8] [--source cached|regen|both]
                                                 key-switching ops replayed from the g0 records (src/check_leaf_ks.rs)
  pc-check pcmm [--shapes ..] [--twin ..]        PCMM digit GEMM vs exact, whole product vs a host twin (src/check_pcmm.rs)
  pc-check ccmm [--seed s] [--dump dir]           CCMM transpose / products vs a host twin of the numpy CKL reference (src/check_ccmm.rs)
  pc-check spru <g0> [--h 64]                    the s = 1 route (SPRU) on g0's dumped s = 1 bootstraps (src/bts_sparse.rs)
  pc-check sparse-classic [--route 8192] [--old <dir>]   the classic multi-stage sparse route (src/bts_classic.rs)
  pc-check n64 [--levels 0,9,18] [--chain n64|n32] [--mm barrett64|barrett32|mont]
                                                 the word seam end to end, no oracle: keygen, sk/pk roundtrips, leaf
                                                 ops vs host references; default (u64, Barrett64) on n64 (src/word_seam.rs)

Exit status 1 on any mismatch or invalid file.

## crates/pc-tools/src/bin/pc-check/check_keys.rs — pc-check keys gates

`pc-check keys <g0 dump>`: load every key of the dump and its secret into pc-gpu's device layout and
check them.

Gates (all must be 0 mismatches / within bound):
 1. chain: `Chain::generate(n32)` primes / digits == the dump's;
 2. per key (171): `b` imported into the packed layout and unpacked on the device (`ksk_unpack`) == the dumped `b`,
    every digit and limb (rows outside a bottom-only key's shape must be zero in the dump);
 3. `a` regenerated on the device from the key's seed (`ksk_expand`) == the dumped dense `a` (keys that have one);
 4. switch relation of every key with the device-regenerated `a`: max |e| <= 64 (the sparse secret is recovered
    from the sparse->dense key: row 9 of digit 0 holds the bare error, row 0 adds (P mod q0) s_sparse);
 5. agreement: a ciphertext encrypted with the dump's secret, rotated by 1 (and conjugated, and taken
    dense->sparse->dense at the bottom) with the imported key and with a key generated here from the same secret,
    decrypts to the expected slots with both (host reference key switch);
 6. keygen at scale: the whole set regenerated on the device (timed) and its relation checked.

## crates/pc-tools/src/bts_all.rs — pc-bench bts --all

Every bootstrap route in one process: the route x K x {eager, batched, graph} table and the MemPlan
assert per (route, K).

  pc-bench bts --all [--k 1,2,4,8] [--reps 10] [--routes dense,stcf,512,256,1]

One setup per K (bootstrap buffers carved for k_max = K, the decode configuration at K = 1; the executor's KS
workspace stays at 8): every route's tables and its own `BtsWork` are carved between aligned arena offsets and
must equal the MemPlan prediction (`route.mem()` / the route's `bytes`, `BtsWork::bytes(shape)`,
`BtsGraphs::bytes`). Per route and K, on K fresh inputs (sk-encrypted at the route's usual entry):
  ref     each input alone (K = 1 call) -> reference words
  eager   the K inputs as one contiguous K-batch `Ct`, `BtsRoute::run`
  batched K separate `Ct`s as one `BtsBatch` (gather -> run -> scatter)
  graph   K separate `Ct`s through `BtsGraphs` (gather -> replay -> scatter)
device ms (CUDA events on the compute lane), median of `reps` after 2 warm-ups; every output of every mode must
equal the reference words of its input (0 mismatches).
Keys: fresh (dense ternary + sparse h=32), generated on the device; SPRU's block key and key ciphertexts fresh.

## crates/pc-tools/src/bts_stc_first.rs — pc-check / pc-bench stcf

StC-first route, stc_high and the staged call object: oracle checks and benches.

  pc-check stcf <d3> [--k 1,2,4,8] [--graph]   d3 (a StC-first PCMM finish, ownCf): the output half through the
      staged API (`BtsCall` resumed after the raise: CtS x4 vs it.cts<k>, EvalMod + stc_first_output vs it.out),
      per K, eager and as one captured CUDA graph replayed twice; the call state vs the dumped BtsState.
  pc-check stcf <g0>                            g0 bootstrap #2 (dense StC-first, CF 13, real payload), the route
      built from our own tables with the dump's CtS plaintexts and keys: (a) the dumped split -> at_split ->
      mod_raise .. stc_first_output, every stage bit-exact with metadata; (b) our stc_first_input on the dumped
      input vs the dumped split (our StC-first plaintexts: integer-domain relative error) and our whole refresh
      vs the dumped output; landing / entry vs pc-shadow.
  pc-bench stcf [--k 1,2,4,8] [--reps 10] [--cf 13] [--real]   device ms per call (and per stage) of the route
  pc-bench stcf-prec [--cf 6,13] [--amps ..] [--real]          round trip input -> split -> output on random
      payloads (reference CF and own CF), landing vs pc-shadow, K=8 batch == K=1
  pc-bench stc-high                                            stc_high vs a host StC of the decrypted input

## crates/pc-tools/src/check_ccmm.rs — pc-check / pc-bench ccmm

CCMM: gates and benches of `pc_gpu::matmul::ccmm`.

  pc-check ccmm [--seed s] [--dump <dir>] [--host-only]
      (1) `Ccmm::transpose`, `Ccmm::mul` (b given as is and transposed) and `mul_inplace` on random split
      ciphertexts (16 per matrix, uniform residues at limbs 0..3) with fresh keys vs a host twin written from the
      numpy CKL reference (perseus logs/prefill/ccmm/ccmm_ref.py `Ref` over ckl_numpy.py: ring switch, recursive
      TWEAK, CMT, PPMM over R_64, relinearization, switch back), every output word per prime. `--dump` writes keys,
      inputs and the twin's outputs as .npy for the numpy cross-check (the oracle dumps' ccmm_xcheck.py).
  pc-bench ccmm --prec [--cf c] [--cases QK,PV,QKN,PVT,QKM] [--msg bits]   (2) a real QK^T (and PV): random Q (complex, two heads per batch slot) and
      K (real) in the SinC slot layout, encrypted -> stc_high(skip 1) -> ccmm_mul -> SinCBoot (HalfBTS + StC stage
      0); the product decoded from its coefficients and the SinCBoot output from its slots vs the exact Q K^T
      (old gate methodology: fitted scale, bits = log2(max |ref| / max err), batch slots 0..15).
  pc-bench ccmm [--reps 10] [--k 1,8]     (3) device ms per CCMM (16 x 16 ciphertexts = 32 heads) by phase, per
      SinCBoot ciphertext at K = 1 and 8.

## crates/pc-tools/src/check_pcmm.rs — pc-check / pc-bench pcmm

PCMM at the StC-first split: gates and benches of `pc_gpu::matmul::pcmm`.

  pc-check pcmm [--shapes gpt2.qkv,..] [--twin gpt2.qkv,..] [--seed s]
      (1a) the digit-plane GEMM (`Pcmm::gemm`, both the A rows (N1 columns) and the B rows (2t columns)) on random
      residues and random int8 planes vs the exact integer product mod q (the integer core of pcmm.py `_exact`),
      every element; (1b) the whole product (`Pcmm::product`: INTT, rescale, fwd switch, MLWE rows, GEMMs, pack,
      compose, back switch, lift, NTT) on a random split ciphertext vs a host twin written from the same
      formulas (FIDESlib Pcmm.cu semantics), every output word.
  pc-bench pcmm --prec [--cf 6] [--wbits 24]       (2) a real input: encrypt -> StC-first input stages (own CF)
      -> product -> K-batched finishes -> x 2^shift, decrypted outputs vs `_exact` of the decrypted inputs
      (old sylph methodology: complex Gaussian sigma 0.1, bits = log2(max |Y| / max err)); per-output raise drops.
  pc-bench pcmm [--shapes ..] [--k 1,2,4] [--reps 5]   (3) device ms per linear: product phases (inputs / GEMM
      / outputs) and the output finishes at K = 1, 2, 4 per call.

## crates/pc-tools/src/leaf.rs — pc-check leaf

`pc-check leaf`: per-op checks of the pc-gpu leaf ops against the g0 oracle dump (GPT-2 decode token 0, block 0).

Every dumped op of block 0 whose kind is selected is replayed through `pc_gpu::ops` from its dumped operands
(ciphertexts `cts/<v>`, plaintexts `pts/<pt>@<level>`, constants), executing the Shadow recipe exactly as the
runtime does, at K = 1 and with the operands replicated into a K = 8 batch, under both `Rescaler`s. Compared:
* words: the output's live rows vs the dumped ciphertext (bit-exact expected for every key-free op);
* decrypted: both decrypted with the dumped `sk.npy` (host NTT + balanced Garner over the bottom 6 limbs, then
  FFTSpecial), `max |ours - ref| / max |ref|`; tensor outputs (degree 2 in s, before relinearization) are
  compared this way only, against the relinearized dump;
* bookkeeping: level, degree, NoiseFactor.
Groups `rescale` and `drop` collect every case whose recipe contains that prelude, plus: the d3 `MR-entry` ->
`MR-bottom` rescale (bit-exact), composite == sequential on every case, and `drop_to_level` on every dumped
ciphertext. Group `encode`: device FFT vs host FFT vs the oracle device rows for dumped plaintexts.

ops.jsonl prints constants with 6 significant digits; the exact scalar is recovered from the dumped output
(the integer `ElemForEvalMult` / `ElemForEvalAddOrSub` value by CRT, then the double that rounds to it) and the
op is re-run with it (both constants are reported).

## crates/pc-tools/src/replay_g0.rs — pc-bench replay

`pc-bench replay`: a compiled block program replayed on the g0 oracle (GPT-2 decode token 0).

  pc-bench replay [--plan <plan dir>] [--graph <graph dir>] [--scalars <v2 capture>] [--block block_0]
                  [--g0 <dump>] [--inputs anon_1] [--reps 10] [--debug] [--oracle-bts] [--ring] [--kv]

Setup: the program compiled from the plan + graph (a v2 graph carries every scalar exactly; a v1 graph's 6-digit
literals would be refused at bind), the dump's keys (CPU-map `b` + KSKA seed, `a` regenerated in the dots), the
dump's plaintexts (device rows by `name@level`), the three decode routes (dense StC-first, SparseB 512, SPRU)
built from our tables on those keys (SPRU's block key material is fresh: the old one was never dumped), the
dump's block input ciphertexts. Then:
  1. eager replay (metadata checked per step) -> the exit var decrypted vs the oracle's block output;
  2. eager again, graph (capture), graph (launch): every output word == run 1 (graph == eager, bit-exact);
  3. timing: device ms per block (CUDA events) and host us per `run` call, eager vs graph (medians of `reps`),
     device free memory and arena use before / after (0 allocations after setup);
  4. levels vs the old op stream (`ops.jsonl` out_level / deg per graph name; the mult in-place rescale quirk);
  `--debug`: every dumped var decrypted after its step (first divergence);
  `--ring`: the block's plaintexts staged from the dumped slot values through the weight ring (host encode, Copy
      lane, Ready joins), replayed again, vs the oracle;
  `--kv`: the KV offload choreography round trip (pinned pages, Copy lane, Ready tokens) on the block's outputs.

# Python package

## python/perseus/errors.py — error taxonomy

The error taxonomy in one importable place.

C++-raised (markers translated by the bindings; all subclass _core.FHEError):
    FHEError      any runtime error from the FHE core
    PlanError     [plan_*]: strict planned-mode violations (level/weight/bts)
    MaskError     [mask_*]: strict decode-mask misses / mask generation
    LayoutError   [layout_error]: slot-layout basis mismatch (_core.set_strict_layout)

Python-raised:
    PlanContractError   plan loaded under a different env than it was captured in
    BundleError         a key bundle's manifest disagrees with the session asked to load it
    SecurityError       a trust-boundary invariant would be violated (e.g. a server session
                        that holds a secret key)

This module imports without the compiled extension: the C++ classes are then stand-ins
with the same names and hierarchy, so `except perseus.errors.FHEError` is always valid.

Two extensions can carry the classes: perseus._core (exported here when it imports) and
perseus._client (exported when only it is built). With both built, _client raises its own
FHEError (a RuntimeError subclass, not this module's); EncClient re-raises _client errors
as these classes, so `except perseus.errors.FHEError` stays valid for the client role.

## python/perseus/examples/gpt2/__init__.py — GPT-2 example

GPT-2 encrypted decode / generate rebuilt from the leaf primitives of ``perseus._core``.

The C++ runtime (src/model/gpt2, src/algorithms, src/packing/cachemir) is re-expressed in
Python on top of ``Context.{add,sub,mult,square,negate,rotate,rotate_and_sum,conjugate,
inplace_add,bootstrap,bootstrap_hint}`` and the plaintext operands of ``Inference``: the cachemir packing,
the BSGS linear, the K/V cache, the THOR softmax, the Remez/Newton LayerNorm, the THOR GELU,
the tiled LM head, the CutMax argmax and the encrypted feedback embedding.  Every module cites
the C++ it ports so the two can be read side by side, and ``ref.py`` mirrors the same
approximations in numpy so FHE noise and approximation error stay separable.

Modules (the model-independent layer lives in ``perseus.impl``: ops protocol, kernels,
layout, linear, norm, attention, activation, config, Rt, ImplModel, fake)
    weights     export zip -> padded / folded / rearranged matrices -> EncodedLinear
    block       transformer_block()
    head        final_ln, lm_head, decode_logits, cutmax_argmax, feedback_embed
    model       Gpt2Model(ImplModel): stages, per-step masks, decode_token /
                argmax_encrypted / feedback / run_decode / generate
    ref         the plaintext mirror (norm_ref, gelu_ref, softmax_ref, attention_ref, cutmax_ref)
    env         the session env (the C++ n32 decode preset, folded ln_1/ln_2) and factory

Packing. The default (``--packing cachemir_complex`` on the complex payload,
``CKKS_COMPLEX=1``) carries two real payloads per ciphertext wherever the decode has a pair:
the fused K + iV projection refreshed by ONE bootstrap, K groups paired into complex buckets
and V lanes into pair buckets (``complex_qkt`` / ``complex_softmax_v``), the output-packed up-
and down-projections and the paired LM-head tile. Every multiply by +-i is the level-free
monomial (``Context.mult_i``), and an output-packed linear carries its unpack's 1/2 in the
weights, so no lane construction costs a level. ``--packing cachemir`` is the C++ decode's
configuration: real weights and linears, the complex payload only in the K/V push
(``cache_kv_push_pair``), CutMax (``cutmax_argmax_packed``) and the feedback tile.
``--payload real`` runs on real slots only. The shipped capture and plan are the default
packing; any other configuration needs its own.

Fused reductions. Off by default (``FUSED_SM_DEN=0``, ``FUSED_LN_VAR=0``): the softmax
denominator and the LayerNorm variance are reduced by their rotation ladders. ``=1`` finishes
the ladder inside a sparse "fold" bootstrap instead (``head_reduce_sum`` / ``norm``; the C++
decode's default for the softmax). Both change the op stream, so a plan is bound to them.

Running (the GPU must be free of other processes; the drivers refuse otherwise)::

    source scripts/local_env.sh
    .venv/bin/python -m perseus.examples.gpt2.run_decode --tokens 16 [--argmax]
    .venv/bin/python -m perseus.examples.gpt2.run_generate --prompt 1 --tokens 4

The three modes, the same loop the C++ / EncGPT2 run:

    eager     reactive bootstraps plus the C++ hint sites (default)
    capture   ``--capture graphs/gpt2_decode_python_n32`` writes block_<b>/graph.json for token 0
              (blocks 0..11, 12 = final LN + LM head, 13 = CutMax, 14 = feedback) and the
              capture contract; ``bash perseus/examples/gpt2/make_plan.sh
              graphs/gpt2_decode_python_n32 gpt2_decode_python_n32`` plans it (CPU)
    planned   ``--plan bootstrap_placements/gpt2_decode_python_n32`` installs block_<b>_placement.json
              live per block (strict: the op sequence must match the capture); CutMax is planned
              too when its plan is present; ``--no-plan-argmax`` runs it eager

A capture spends most of its time decrypting every node for its magnitudes. ``--structure-only`` records the
graph without them and ``capture_mags.py`` fills them from the numpy fake running the same kernels on the same token
(perseus/impl/magtape.py: exact values, the nodes matched one by one by op type and step)::

    run_decode --tokens 1 --capture graphs/X --structure-only    (add --argmax to capture CutMax)
    capture_mags graphs/X [--argmax]

Long contexts. A plan names ciphertexts by a per-block counter, so it binds only while every token issues the same
ops: the q.K^T adds a product per 32-token cache group and SM_PERIODIC switches the softmax's path when the cache
outgrows one group, so a plan captured at token 0 stops binding at token 32. With ``IMPL_SCOPED_NAMES=1`` every step
names its ciphertexts on its own counter (the q.K^T keyed by the cache's group count) and, with ``SM_PERIODIC=0``, one
capture plans every length; the plan declares both in its ``runtime=`` line::

    run_decode --tokens 1 --capture graphs/X --structure-only --set IMPL_SCOPED_NAMES=1 --set SM_PERIODIC=0

``SM_PERIODIC=2`` keeps the periodic softmax at every length: with G groups the scores repeat every Gp tH blocks (Gp
the power of two >= G). The ops whose count depends on Gp run in sub-steps keyed by it (``gp``), so one plan from the
token-0 capture serves every length. The softmax's refreshes then need a Gp x 512-slot route, which the session builds
when the cache enters the class (``ImplModel.ensure_routes``, ``SM_ROUTE_AHEAD``); routes past 1024 slots are
multi-stage and land deeper than SparseB's, so the plan takes a landing floor on those steps::

    S="--set IMPL_SCOPED_NAMES=1 --set SM_PERIODIC=2"
    run_decode --tokens 1 --capture graphs/X --structure-only $S; capture_mags graphs/X $S
    FORBID_STEPS=.var,.mean,.qkv,.gp PLAN_STEP_LANDING=softmax:32,softmax_v:32 make_plan.sh graphs/X X
    # runtime= adds: SM_PERIODIC=2 IMPL_SCOPED_NAMES=1 FIDESLIB_BTS_SPARSE_B_MAX=1024 SPARSE_LEVEL_BUDGET=3,3
    run_decode --tokens N --plan X

Planning CutMax (block 13) is a second round, ``make_plan.sh <graph> <name> argmax``, from
the same eager capture. Its entry is taken from the tail plan's exit (level and degree)
rather than from the capture: what enters CutMax is what the planned tail produces, and a
plan made from the capture's own levels fails the strict level check at its first op. Its
hints are dissolved, which makes the refresh envelope hard: planned live they fire at the
top of the chain, where a bootstrap returns garbage silently. The eager capture's only
inexactness is the site magnitudes, which are the eager tail's: against a capture taken
under the block plans the 51 sites and their routes come out identical and 5 correction
factors differ by one or two. The feedback (block 14) is not planned; it runs eager.

Tests: ``tests/test_gpt2_primitives_cpu.py`` (numpy fake, no GPU) and
``tests/gpu/test_gpt2_primitives.py`` (each op against its C++ composite on one session).

## python/perseus/examples/llama/__init__.py — Llama-3 example

Llama-3 (8B) encrypted decode on ``perseus.impl``: the GPT-2 port's kernels, packing and
runtime levers (perseus/examples/gpt2: cachemir_complex packing, sparse routes, SparseB,
the softmax / norm switches, capture -> plan -> planned) applied to the Llama architecture.

What is Llama-specific, and where it lives
    RMSNorm     perseus.impl.norm.rms_norm: the LayerNorm pipeline minus the mean (centering
                scale, all-slot ladder, Chebyshev seed + Newton inverse sqrt); gamma / inv_out_scale
                folded into the next linears (weights.py)
    RoPE        perseus.impl.attention.Rope: in the head-interleaved layout a lane's rotary
                partner is N/2 slots away, so q and K get x cos + rotate(x, N/2) sin' with cos /
                sin' folded into the selection masks they are multiplied by anyway: one rotation,
                no level. The masks are per position (step masks)
    GQA         the 8 KV heads are expanded to the 32 query heads in the K / V weights
                (weights.expand_kv): the attention core is the MHA one
    SwiGLU      gate and up from one prepared input (output-packed), SiLU as the THOR composite
                x (p2(p1(x / xmax)) + 1/2) of perseus.impl.activation with sigmoid coefficients
                (1/xmax folded into the gate), then the product and down
    BOS prefix  position 0 carries Llama's massive activations; its K / V are public constants,
                computed exactly in plaintext and written into the caches as encryptions of
                public vectors (prefix.py); the encrypted decode starts at position 1
    weights     ~4.6k plaintexts per block, never resident: their coefficient rows (the plaintext
                polynomials / scale, float32, built once next to the slot-vector cache) sit in pinned
                host memory (~40 GB) and every baby-step sum is encoded and multiplied on the GPU in
                batches (Inference.mult_coeff_sum, FIDESlib CoeffRows); the slot vectors stay for the
                capture and the fake (weights.LlamaWeights, model.block_weights)
    routes      the attention block is tH = t H = 8 x 32 = 256 slots (GPT-2's is 512): sparse routes
                {256, 1}, the periodic softmax at every length (SM_PERIODIC=2, the Gp x 256-slot
                route built as the cache grows: model.routes_for)

Modules
    hf          the HF checkpoint, text windows, per-site statistics, the fp32 oracle logits
    calibrate   configs.json: RMSNorm seeds, SiLU composites, THOR softmax with data-sized
                divisions (configs/model/approximation/llama31_8b_n32)
    config      configs.json -> perseus.impl config structs
    ref         the plaintext mirror (HF model with the approximations patched in): KL of the
                approximations alone
    weights / prefix / block / model / env / run_decode

Running (the GPU free of other processes)::

    source scripts/local_env.sh
    W=<token window .npy>; G=graphs/llama31_8b_decode_python_n32; P=llama31_8b_decode_python_n32
    # capture (structure only) -> magnitudes from the numpy fake -> plan -> planned decode
    .venv/bin/python -m perseus.examples.llama.run_decode --tokens 1 --window $W --kv-offload         --set KV_ARENA_GB=56 --set FIDESLIB_BTS_RAISE_DROPS=1,3,5 --capture $G --structure-only
    .venv/bin/python -m perseus.examples.llama.capture_mags $G --window $W
    bash perseus/examples/llama/make_plan.sh $G $P
    .venv/bin/python -m perseus.examples.llama.run_decode --tokens 8 --window $W --kv-offload         --set KV_ARENA_GB=56 --plan bootstrap_placements/$P

The planned decode runs ~19.5 s/token (Blackwell PRO 6000, 4 tokens, KL <= 1.6e-7, top-1 4/4).

## python/perseus/examples/llama/calibrate.py — Llama calibration

Calibrate the approximation configs of the Llama port on teacher-forced text windows.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m perseus.examples.llama.calibrate         --out configs/model/approximation/llama31_8b_n32/configs.json

Per site, from every position but 0 (the BOS prefix runs in plaintext, see model.py):
  norm     RMSNorm inverse sqrt: centering scale c (c * max|x| = center_target), the band of
           c^2 (mean(x^2) + eps) widened by `--norm-safety`, the output scale inv_out_scale =
           inv_y_max * sqrt(band lo) and the shallowest Chebyshev seed + Newton count meeting
           `--norm-err` (scripts/utils/refit_norm_cheb.py's fit)
  silu     THOR composite x * (p2(p1(x / xmax)) + 1/2) with p2 o p1 ~ tanh(x / 2) / 2 (the
           sigmoid's odd part), xmax = `--silu-safety` * max|gate|
  softmax  the THOR softmax fit of perseus.calibrate (approximation gpt2 knobs) on the score
           rows of the decode queries

## python/perseus/examples/llama/planned.py — planned Llama prefill

The forward's op stream depends on the prompt length and the layout only, never on the prompt's values, so it is
recorded once on the model's CPU twin (the fakecap `LlamaPrefill`, `impl.driver.Recorder`: v2 nodes + plaintext
values), compiled per block against the plan (pc-program), and bound on one planned Session that also holds the
two matmul services the blocks' `Finish` ops call inside the replay:
- PCMM: every linear's digit planes resident (Llama: 128 linears), keys from the secret at setup;
- CCMM (`pc-py/src/matmul.rs`): the attention's Q K^T / P V with their SinCBoot refreshes, keys from the secret at
  setup, `c_unit` calibrated on the device from probe matrices before the secret is dropped (impl/ccmm.py `prepare` /
  `_calibrate`), merged outputs on `w_theory` (the twin's unmerge masks use it too).

The paged prefill (PF_PAGED=1) runs page 0 (BOS + 127 tokens) through every block with the one token-0 plan; the
pages' K / V operands stay inside a block's program (a later page reads them as the CCMM's prepared operands).

## python/perseus/examples/llama/weights.py — Llama weight packing

HF Llama checkpoint -> the folded, head-rearranged, GQA-expanded matrices of the cachemir
packing and their slot vectors (EncodedLinear), cached on disk as memory-mapped arrays.

Per block (the matrices are (d_in, d_out), y = x @ W):
  q, k, v   ln_1's gamma / inv_out_scale folded into the rows (RMSNorm has no beta); K and V
            expanded from the 8 KV heads to the 32 query heads (head h reads KV head
            h // (H / H_kv)), so the attention core is plain MHA; columns head-interleaved
            (rearrange_qkv_weights). cachemir_complex: one fused K + iV linear.
  out       rows head-interleaved (rearrange_wo_weights)
  gate, up  ln_2's gamma / inv_out_scale folded into the rows, columns padded to E; the gate
            also carries the SiLU's 1/xmax (the THOR composite's input scaling, GELU_FOLD)
  down      rows padded to E
  lm head   lnf's gamma / inv_out_scale folded into the rows, vocab tiles of N columns

Slot vectors are stored as float32 / complex64: the bf16 weights are exact there and the folded
products round at ~6e-8, far under the CKKS noise. A layer is ~1 GB; the 32 layers + the LM head
~40 GB under PERSEUS_LLAMA_CACHE (default $HF_HOME/perseus/llama_slots), keyed by the checkpoint,
the configs and the builder sources.

## python/perseus/examples/rmsnorm_from_primitives.py — RMSNorm from primitives

RMSNorm built from the session's leaf primitives.

    y = x * rsqrt(mean(x^2) + eps) * gamma

is computed on a token ciphertext with the primitives `Context.square / mult`,
`Inference.sum_slots`, `Inference.eval_chebyshev` and `Inference.mult_pt`, and mirrored
in numpy with the SAME Chebyshev approximation so that the approximation error
(`rmsnorm_ref` vs `rmsnorm_exact`) is separable from the FHE noise (`rmsnorm` vs
`rmsnorm_ref`).

The chain, with the CKKS levels each step consumes:

    sq  = fhe.square(ct)                                        # x^2 slot-wise        1
    S   = inf.sum_slots(sq, inf.slots)                          # all-slot sum, bcast  0
    r   = inf.eval_chebyshev(S, coeffs, (a-eps)*d, (b-eps)*d)   # rsqrt(S/d + eps)     2 + ceil(log2 degree)
    xg  = inf.mult_pt(ct, gamma_slots)                          # x * gamma (parallel) 1
    out = fhe.mult(xg, r)                                       # ct * ct              1

Total depth: 4 + ceil(log2 degree) levels (`rmsnorm_depth`). `rmsnorm` returns the
output together with `ct.level` before and after.

Packing facts this example relies on (all in the cachemir packing, the session's default
for the gpt2 family; `Session.encrypt` -> `_core.encode_token_input`):

* A token ciphertext keeps real lane ``i`` at slot ``i * t`` with
  ``t = inf.slots // inf.size.hidDim`` -- `cachemir::encode_linear_input`
  (src/packing/cachemir/cachemir_linear_utils.cu: ``p.t = N / p.d`` with ``p.d = hidDim``,
  ``ptx[i * p.t] = x[i]``) and `decode_tokens` (``y[tok][i] = cy[i * t + tok]``, same file).
  `pack_tokens` (src/model/gpt2/gpt2_io.cu) zero-pads the ``d`` real features to
  ``hidDim`` first, so lanes ``d .. hidDim-1`` AND every off-lane slot ``i*t + k`` (k > 0)
  are exactly zero in a fresh encode.
* Because every non-lane slot is zero, the full rotate-and-add ladder
  ``sum_slots(ct, inf.slots)`` (steps 1, 2, 4, ..., slots/2) yields the sum over the ``d``
  real lanes broadcast to EVERY slot -- byte-for-byte the ladder the runtime's own
  LayerNorm variance uses (src/packing/cachemir/cachemir_norm_utils.cu,
  `compute_variance_interleaved`: "This ladder sums EVERY slot, so the result is a
  broadcast constant").
* The rotation keys 1, 2, 4, ..., slots/2 are in the gpt2 band
  (`collect_norm_rots`, src/packing/cachemir/cachemir_rot_indices.cu); `rmsnorm` checks
  them against `Context.loaded_rot_steps` before rotating.
* A plaintext vector for `mult_pt` follows the same layout: ``gamma[i]`` at slot ``i * t``
  (`pack_per_feature_vec`, cachemir_norm_utils.cu) -- `gamma_to_slots`.

Precondition: the input must be a FRESH token-basis ciphertext (what `Session.encrypt`
returns) or one rebased to it. A ciphertext that went through a cachemir linear carries
non-zero partial sums in the off-lane slots (include/slot_layout.h, "Derived" basis) and
the all-slot ladder would sum them too; `PackedCtx` does not expose the layout kind, so
this example cannot detect it.

Chebyshev convention: `rsqrt_cheb_coeffs` fits ``u ** -0.5`` on ``[a, b]`` with
`numpy.polynomial.chebyshev.chebinterpolate`; ``c0`` is NOT halved, which is both numpy's
`chebval` convention and the runtime's (`eval_chebyshev_series`,
src/primitives/polynomial.cu: ``r = sum_{k>=1} c_k T_k(y) + c_0``), so
``[0.5, 0, 0.5]`` on ``[-1, 1]`` is ``x^2``. The evaluator maps its argument onto
``[-1, 1]`` with ``y = (2x - (a + b)) / (b - a)``; feeding it the sum of squares ``S``
with the interval ``[(a-eps)d, (b-eps)d]`` gives the same ``y`` as feeding
``u = S/d + eps`` with ``[a, b]`` (`fhe_interval`), so the ``1/d`` and ``+eps`` cost no
level. The interval is a calibration window: `rmsnorm_ref` raises outside it; the FHE
side cannot check and the series diverges there.

Level budget: the runtime bootstraps reactively when ``ct.level >= fhe.level_limit()``
(include/fideslib_wrapper.h `maybe_bootstrap`; on the n32 chain both are PRIME counts and
one CKKS level is ``composite_degree`` primes). `level_budget(sess, ct)` returns the whole
levels left; under `SessionProfile.custom_n32()` a fresh encrypt lands at 34 with the
ceiling at 49 (AUTO_BTS_LEVEL; scripts/local_env.sh exports 46 instead), i.e. 7 levels
-> ``degree <= 7`` (5 levels -> ``degree <= 2`` with the ceiling at 46).

Usage (a live session; the GPU test in tests/gpu/test_primitives.py is the executable
form):

    from perseus.session import session
    from perseus.profile import SessionProfile
    from rmsnorm_from_primitives import rmsnorm, rmsnorm_ref, level_budget

    with session(profile=SessionProfile.custom_n32()) as s:
        ct = s.encrypt(x)                                   # x: 768 real features
        res = rmsnorm(s, ct, gamma, eps=1e-5, d=768, interval=(0.04, 0.16), degree=7)
        y = s.decrypt(res.ct, d=768)
        print(res.level_before, "->", res.level_after)       # 34 -> 48 (46: rescale pending)
        err = abs(y - rmsnorm_ref(x, gamma, eps=1e-5, interval=(0.04, 0.16), degree=7)).max()

## python/perseus/impl/__init__.py — perseus.impl package

``perseus.impl``: build encrypted models from the leaf primitives of ``perseus._core``.

The C++ runtime ships composites (``_core.linear``, ``_core.attention_softmax_thor``, ...);
this package is the same algorithms written in Python on ``Context.{add, sub, mult, square,
negate, rotate, rotate_and_sum, conjugate, bootstrap, bootstrap_hint}`` and the plaintext
operands of ``Inference``, plus the runtime services a model needs to run at C++ speed: the
residency ring for weights, the encode cache with worker staging for masks, graph capture and
planned bootstrapping. A model is a ``ImplModel`` subclass that names its stages, its
per-step masks and its weights; perseus/examples/gpt2 is the reference model.

    ops         the ops protocol: FheOps (session or fake), NumpyOps (plaintext mirror)
    poly        polynomial / iterative kernels (Chebyshev, power-sum, Remez, Goldschmidt,
                Newton, odd powers, ladders, 2-iteration bootstrap)
    layout      the cachemir geometry (Dims, cm_params, interleave), weight / bias slot
                vectors, every selection mask
    linear      the BSGS linear (prepare + apply, EncodedLinear)
    norm        norm(), ln_affine(), ln_shift(), layer_norm(), norm_step_masks()
    attention   KVCache, cache pushes, qkt, head_reduce_sum, softmax_thor, softmax_v, mha(),
                attention_step_masks()
    activation  THOR composite GELU
    config      configs.json -> NormCfg / SoftmaxCfg / GeluCfg / CutMaxCfg
    rt          Rt: ops + dims + step scopes + the mask discipline (cache, stage, evict)
    driver      ImplModel: modes (eager / capture / planned), the residency ring, per-token
                bookkeeping
    profile     TimedOps: per-primitive and per-step wall time
    fake        a numpy fake of the _core surface (level accounting, strict mode) for CPU tests

## python/perseus/impl/driver.py — ImplModel driver

``ImplModel``: the model-independent driver of a primitives model — one session, the
three execution modes, the residency ring and the per-token bookkeeping. A model subclasses
it and supplies its stages (``token_stages``), its per-step masks (``step_masks``) and its
weights; see perseus/examples/gpt2/model.py.

Modes (the same loop the C++ / EncGPT2 runs):
    eager     reactive bootstraps plus the C++ hint sites (default)
    capture   ``set_capture(graph_dir)``: block_<b>/graph.json per stage for the capture token
              (FHE_GRAPH_CAPTURE_TOKEN, default 0) plus capture_env.json (the plan contract)
    planned   ``load_plans(plan_dir)``: block_<b>_placement.json installed live per stage,
              strict (the op sequence must match the capture); a variant
              block_<b>_<variant>_placement.json replaces it while ``plan_variant`` is set
              (a stage entered in another state than the capture's, e.g. a fed-back token)

Stages. ``token_stages(x, pos)`` returns ``[(label, prefix, fn)]``: ``fn(x) -> x`` runs the
stage's ops; ``prefix`` names the block's cached weight plaintexts (``set_slot_pt`` names
``impl.<prefix>*``) or is None for a stage without streamed weights. With the runtime's
residency ring available, a stage with a prefix becomes a ``Stage(state=EncodedBlock)`` whose
plaintexts are extracted on the worker and uploaded on the side stream while the previous
stage computes, installed for the compute and evicted after it — the C++ block loader's path.
Without the ring (the fake, or the first token before the names exist) the stages run as a
plain loop with load/evict around each.

Masks. ``step_masks(pos)`` returns the per-step mask items ``[(key, build)]`` of position
``pos``; ``token_begin`` / ``token_tail`` drive ``Rt``'s adopt / stage / evict cycle around it.

## python/perseus/impl/magtape.py — magnitude tape

Capture magnitudes from the numpy fake instead of decrypting on the device.

A capture records, per graph node, the op's structure (names, levels, pack tags, the refreshes the
runtime fired: what a plan binds to) and its output's magnitudes (what the planner prices a refresh
by: max |slot|, the mean / max deviation the offset transform sees, the max plaintext coefficient
EvalMod sees). The second half is a decrypt + FFT per node on the host, which dominates a capture of
a weight-heavy model (Llama-8B: hours, the GPU idle). This module splits the two:

  1. the device capture runs with magnitudes suppressed (``Context.magnitude_capture_suppressed``):
     exact structure, no decrypts;
  2. the fake runs the same kernels on the same input with a TAPE: per leaf op, its graph op type,
     its step path and the exact (noise-free) output statistics;
  3. ``fill_graph`` walks the device graph and the tape in order and writes the tape's statistics into
     the nodes. Refresh nodes (auto / deliberate bootstraps, hints) are the runtime's decisions, not
     ops of the kernels: they are skipped on both sides and inherit their input's statistics.

The walk checks every pair (op type and step path) and stops at the first divergence, so a tape that
does not describe the device's op stream is never merged silently.

## python/perseus/impl/ops.py — ops protocol

The leaf-primitive "ops" protocol: the same kernel code (poly.py, linear.py, norm.py,
attention.py, activation.py) runs on the device session (``FheOps(sess.fhe, _pc)``: ``_pc.Fhe`` is
both the ``inf`` and the ``fhe``, ``_pc`` the module holding the data-plane functions), on the
numpy fake session (``FheOps`` over ``fake``) and on plain numpy arrays (``NumpyOps``, the
plaintext mirror). ``TimedOps`` (profile.py) wraps ``FheOps`` with per-call timing.

The plaintext caches live in Rust (``_pc.Fhe``: named slot plaintexts encoded per level on use,
the tag + level encode cache, the staging worker ring); this class only names them.

Plaintext operands. ``mult_pt(a, v)`` / ``add_pt(a, v)`` encode ``v`` at every call (the leaf
primitive). With ``key=`` they go through the runtime's caches instead:
  * ``kind="weight"`` (the default for keyed calls): ``set_slot_pt`` + ``mult_slot_pt`` — a
    named, host-resident plaintext encoded ONCE at the level of its first use, streamed
    through the residency ring as part of a block state (see driver.py);
  * ``kind="mask"``: ``mult_cached`` — the runtime's encode cache (``inf.enc_cache``, keyed by
    tag and level), the path the C++ masks take: primed / staged ahead on a worker / evicted
    per step by the runtime (see rt.py).

## python/perseus/impl/pairing.py — complex-lane pairing

Complex-lane pairing: two real payloads in one ciphertext (a + i b), so one product,
rotation or reduction serves both. Every function here is exact: its result equals the
real computation it replaces (tests/test_pairing_cpu.py, tests/gpu/test_pairing.py).

The decode's own uses (the K buckets, the V pair masks, the paired P.V, the output-packed
up- and down-projections, the fused K + iV linear) live in attention.py and linear.py; this
module holds the constructions in their general form, including the ones the decode does not
use: input pairing (slower than the real linear it replaces) and the extraction folded into a
contraction (the decode has no linear fed by two packed payloads):

  output pairing   two outputs that share an input: one complex plaintext W_a + i W_b, one
                   product, then one conjugate splits Re / Im (unpack_ri);
  input pairing    two inputs that share an output: x_a + i x_b against the CONJUGATED
                   weight W_a - i W_b, whose real part is x_a W_a + x_b W_b;
  paired products  Re((v_a + i v_b)(s_a - i s_b)) = v_a s_a + v_b s_b.

Weights are slot vectors (numpy); `key` names them for the runtime's plaintext cache
(ops.mult_pt), None encodes at every use.

## python/perseus/impl/rt.py — Rt and the mask discipline

The per-session bundle every kernel takes (``Rt``): the ops protocol, the geometry, a step
scope, and the MASK discipline of the C++ decode arm.

Masks are the plaintexts that depend on nothing but the geometry and, for some, the step
(token position): selection masks, the softmax clip/active masks, the K/V lane masks, the
LayerNorm centering scale. The C++ keeps them in the runtime's encode cache (``inf.enc_cache``,
keyed by tag AND level) and, per token: adopts the next step's masks that a worker encoded
under the previous token's argmax tail, evicts the previous step's, and primes synchronously
whatever is still missing (gpt2_decode.cu, gpt2_residency.cu). ``Rt`` does the
same through ``_pc.Fhe.{mult_cached, add_cached, prime_pt, stage_pts, adopt_pts, erase_pts}`` (1:1;
the fake's ``FakeInf`` mirrors them):

    rt.mult_mask(ct, key, build)   ct * mask (cached by tag + level; encoded on first use)
    rt.declare_step(s, items)      the per-step masks of step s: [(key, build)]
    rt.stage_step(s)               encode them on the worker at the levels their sites used
    rt.begin_step(s)               adopt step s's staged masks, evict step s-1's

A mask's level is not known before its first use, so token 0 pays synchronous encodes and
records, per SITE (the first element of the key), the levels it was used at; every later
step's masks are staged at those levels. A staged mask that is never used costs one encode;
a used mask that was not staged is primed on the spot and counted (``rt.mask_misses``).

## python/perseus/impl/sinc_mha.py — SinC multi-head packing

SinC operand packing of the CCMM attention for multi-head attention with d_head = 64 (BERT / ViT-base: 768 = 12 x 64,
padded to 16 heads), and its maps to the channel-blocked prefill layout (blocked.py, nch = 512: 8 heads per block).

Blocked source: block k = h // 8, slot j 512 + (h % 8) 64 + lane (j < 64 the head dim), Re / Im = chunk 0 / 1 of a
128-token pair (token 64 c + lane). K and V enter per chunk as real ciphertexts (prefill split).

The CKL batch CCMM multiplies 32 batch 128 x 128 matrices (sinc.py's SinC_{1024, 2^16}). A head pair (2 p, 2 p + 1)
shares batch slot l = 2 p (even, < 16: PF_CCMM_MERGE's duplicate on l + 16 stays free):
  Q: row = brev6(lane) + 64 hp (hp = h % 2), col = col_of(j, hp), Re / Im = chunk; the other head's columns stay 0
  K, V: row = brev6(lane) + 64 c (keys), col = col_of(j, hp): both heads' 64 dims side by side, real
so S = Q K^T keeps every (query of head h, key) score inside its own row half, and O = P V is valid where the row half
equals the column's hp (the cross entries are dropped on the way back). col_of(j, hp) = (j5 + 2 j4 + 4 j3) + 8 (j0 + 2 j1 +
4 j2 + 8 hp): the bits stage A leaves on slot bits 12..14 stay put.

Conversions: sinc.stage_a (slot fields [0, 6) <-> [6, 12)) then a routed stage B: per (source v, target ciphertext o)
one constant shift, factored as a rotation per source and one per target (masks from the explicit maps, numpy only).

## python/perseus/impl/smcheb.py — SM_CHEB_GS

SM_CHEB_GS: Chebyshev-scaled Goldschmidt for the softmax reciprocal rounds (SM_DEN_RECIP=1).

Round 0 keeps the calibrated seed (init_alpha, init_beta) and the band it is the minimax seed of. Refine round j >= 1
runs on the exact per-kc band of its denominator, tot = 0.5 sqrt(kc) sum(y^2) with sum(y) = 1 - e_{j-1} (e_{j-1} the
previous round's residual bound), for ANY input:
    [0.5 (1 - e)^2 / sqrt(kc), 0.5 (1 + e)^2 sqrt(kc) * TOP]       (TOP covers the CKKS noise)
The step d <- c_j d (2 - d), c_j = 2 t_j^2 / (2 t_j^2 - 1), t_j = T_{2^(j+1)}(z0), z0 = (hi + lo) / (hi - lo), leaves
the residual 1 / T_{2^k}(z0) after k levels (plain Goldschmidt: (1 / T_2(z0))^(2^(k-1))). In the packed form of
poly.goldschmidt_recip only the added constant changes, P <- P (P + conj P + 2 kappa_j); the seed is scaled by
kappa_0 and the reciprocal's imaginary seed by 1/lambda_K, so the result needs no rescaling (no extra level, no
extra product). Counts per round (recip_iters_cheb) and the mid-round residual targets (recip_targets) are per-layer
config keys, computed by design() with the count of each round the max over the design cache lengths (one op graph
for every kc); the seeds and step constants are per-kc host constants.

## python/perseus/plan/btserr.py — bootstrap accuracy model

The bootstrap accuracy model the placer plans against.

The min-cut optimises bootstrap count against a level budget; this module prices each
refresh site so the cut also respects the EvalMod band. The model:

* the bootstrap computes `C·sin(m/C)` with `C = 2^CF/2π`: a correction factor does not
  buy accuracy, it slides a fixed-quality band `[0.003, 0.03]·2^CF` (centre `0.01·2^CF`,
  ~2e-3 relative inside it) by 2x per step;
* the quantity the sine sees is the max plaintext COEFFICIENT after CtS, not the slot
  max — a broadcast constant and spread data differ by orders at the same slot max;
* sparse routing lowers the noise floor and is neutral above it (orthogonal to CF);
* the offset transform `bootstrap(ct−c)+c` costs no level and lets CF follow the residual.

`data/bts_accuracy_<chain>.json` is measured on data of known period, indexed by slot
amplitude, so a lookup keyed by `(cf, period, route, slot |m|)` already prices the
coefficient effect from what the capture records (`output_max_abs`, `pack_period`).
When a capture carries the coefficient itself (`output_max_coeff`) the period-1 column
is the coefficient-indexed law and is used instead.

Unknown period defaults are deliberately opposite: routing stays DENSE (an unproven
sparse route folds residue classes silently), while the error model prices it as
period 1 (all energy in one coefficient — the pessimistic reading).

Pure python: no GPU, no FIDESlib, no import of the runtime.

## python/perseus/plan/contract.py — plan/config contract

Plan/config contract: stamp the capture-time configuration into plan artifacts, validate at load.

A placement plan is strict against the op sequence and level schedule it was captured
under, but the knobs that shape both are configuration (LN folding flips which ops exist;
AUTO_BTS_LEVEL moves every level). A plan loaded under a different configuration dies
deep in the run with a bare [plan_weight_error]/[plan_level_error]. The contract makes
the mismatch a load-time message that names the knob instead.

The stamp (`placement.json["config"]`, written by the planner through `emit.stamp`) is
`Config.load_bearing()` of the capture: the graph dir's `capture_config.json` (fakecap v2),
else its legacy `capture_env.json` read through `Config.from_legacy_env`. Plans stamped
before the typed Config carry `placement.json["capture_env"]` (`{"env": {NAME: value|null},
"session": {...}}`); `read_stamp` returns either form and `validate_contract` checks both.

The comparison covers the fields the stamp sets plus the legacy contract's names
(`LOAD_BEARING_ENV`, compared even when unset: set<->unset changes the op sequence exactly
like a value change). Scheduling-only knobs (the arena sizes, the rotation-key band) are
deliberately NOT load-bearing: they move WHEN work happens, never what is computed.

## python/perseus/plan/equiv.py — plan equivalence

Compare a regenerated plan file against a committed one under the baseline contract.

    plan_equiv.py NEW.json REF.json      -> exit 0 if equivalent, 1 (with a reason) if not

Contract: everything the runtime consumes — every top-level key
except `summary` (placements, correction_factor, rescale_after, weight_levels, mask_levels,
hint_fire, rules, ...) — must be identical; `summary` may only GROW: every key present in
the committed plan must be present in the new one with an identical value, RECURSIVELY
(`summary` nests `final_named_levels`, `final_named_degs`, `bts_quality`, ...), so the only
permitted difference is an ADDED key at any depth. The additive-summary rule lets diagnostics such as `bts_quality.num_cf_clamped` land
without re-emitting the shipped artifacts — which must stay as they shipped, because they
are the only independent reference: re-emitted plans would only ever be compared against
themselves — while a changed placement, CF, or existing summary value still fails.

How the comparison is done. The runtime body is compared as canonical text
(`json.dumps(sort_keys=True)` of the parsed objects): key order is not semantic, and since
`repr` round-trips doubles exactly this is strictly STRICTER than parsed equality — it also
catches `1` vs `1.0` and `-0.0` vs `0.0`, which matter to a C++ reader. The summary subset
is walked structurally with type-strict leaves (`True` is not `1`).

## python/perseus/plan/placer/baselines/base.py — baseline placer seam

Shared seam for the baseline placers — the controlled-variable design.

A baseline placer reimplements another compiler's BOOTSTRAP PLACEMENT decision (WHERE
refreshes go) while sharing every other stage of plan/placer unchanged: graph ingest,
reactive-bootstrap erasure, the runtime-validated `simulate()` twin as the feasibility
oracle, `RefreshPlanner` pricing (HOW each site is refreshed), `emit.assemble()`, and
postconditions P1-P3. That makes cross-placer numbers a comparison of placement
algorithms, not of runtimes.

Baseline algorithms are magnitude-blind (their home compilers have no EvalMod wall), so a
site they pick can be `hopeless` (destructive refresh) or their plan can still be over
budget under the twin. Two policies, both honest:

  * raw (default):  PlanInfeasible, loudly — "infeasible as captured" IS the offline
                    result for that arm, not an error to hide.
  * rescue:         repair with counted, labeled bootstraps: hopeless sites move up the
                    binding-producer chain (`rescue_moved`), residual over-budget ops
                    get refreshes at their binding inputs (`rescue_added`). Every rescue
                    bootstrap counts against the baseline's total; the counts travel in
                    summary.placer_meta and the heuristic_config gains "+rescued".

Hints stay runtime-owned for every arm: the shared sim decides firing from whatever
levels the baseline's placement produces, and emit exports hint_fire so the plan binds.

## python/perseus/plan/placer/baselines/dacapo.py — DaCapo segment-DP placer

DaCapo-style segment-DP placer.

Port of the placement algorithm of corelab-src/dacapo (MIT license, (c) 2024
corelab-src; USENIX Security 2024 "DaCapo: Automatic Bootstrapping Management for
Efficient Fully Homomorphic Encryption" — `DaCapoPlanner.cpp`, `CandidateSelection.cpp`,
`BypassDetection.cpp`, `CoverageRecorder.cpp`), adapted to this repo's IR.

Upstream algorithm: cut candidates are program points with few live-out ciphertexts;
a forward DP over candidates treats each pair (from -> to) as one "bootstrapping
segment" (bootstrap all live-outs of `from`, compute to `to`), legal only while
`to` lies inside `from`'s bootstrappability coverage; segment weight comes from a
per-op latency table; bypass detection exempts live values whose consumers lie beyond
a scale threshold from the boundary bootstrap; candidate selection widens the allowed
live-out width only until the program compiles.

Adaptations (documented in summary.placer_meta.deviations):
  * program points = topo positions after each level-consuming node (the rescale
    boundaries; DaCapo's SMU boundaries are exactly the scale-management points).
  * "accumulated scale" maps to consumed prime levels: budget L is the scale budget,
    threshold*L the bypass wall; coverage passes reset every live var to fresh at the
    cut (upstream: segment args enter at waterline) and accumulate per-op costs.
  * latency table is a flat per-op-class ms table (latency_n32.json knob) instead of
    per-level tables; bootstrap dominates either way.
  * width search validates with the shared simulate() twin ("compiles" upstream).

## python/perseus/plan/placer/baselines/fhelipe.py — Fhelipe depth-boundary DP placer

Fhelipe-style depth-boundary DP placer — CLEAN-ROOM REIMPLEMENTATION.

Upstream (github.com/fhelipe-compiler/fhelipe) is GPL-3. This module was written from
the algorithm description in the PLDI 2024 paper (Krastev et al., "A Tensor Compiler
with Automatic Data Packing for Simple and Efficient Fully Homomorphic Encryption",
Sec. 6) and from prose behavioral notes; NO upstream source code was read into it.

The algorithm, as published:
  * Nodes are partitioned by multiplicative DEPTH; bootstrapping decisions are made at
    depth boundaries only — bootstrap all values crossing a chosen boundary, or none.
  * dp[i] = min over recent boundaries j of dp[j] + cost(bootstrap boundary j) + the
    shortcut values crossing (j, i] that cannot ride through un-refreshed; window depth
    is capped by the usable levels l0. Backtracking yields the boundary set.
  * Boundaries are NARROWED to chokepoints (a single value through which all of a
    depth-component's flow passes) where one exists.
  * SHORTCUT edges (skipping multiple depths, e.g. residuals) are greedily omitted from
    boundary bootstrap sets when the window's level budget can carry them.
  * A post-pass greedily prunes bootstraps that turn out to be unnecessary.

Adaptations to this repo (documented in summary.placer_meta.deviations):
  * depth = monotone accumulation of per-op level cost in composite units; hint,
    deliberate and fold sites reset it (they are runtime-owned in every arm). The shared
    sim remains the feasibility oracle.
  * ct count = 1 per var (our IR is ciphertext-level, not tensor-level).
  * shortcut omission uses a per-edge carry bound (carry + window residue <= l0)
    instead of the paper's persistent level-vector shaving; omissions decided
    independently, which the paper's tie-break approximates anyway.
  * the pruning post-pass consults the shared simulate() twin directly and runs only on a
    feasible DP plan; on an infeasible one (every block of the GPT-2 decode) the plan is
    repaired by the rescue and then pruned by the planner's own pass (PlanConfig.prune),
    which plays the published post-pass's role on the repaired plan.

## python/perseus/plan/placer/baselines/orion.py — Orion level-DAG placer

Port of the placement algorithm of baahl-nyu/orion (MIT license, (c) 2025 Austin
Ebel; `orion/core/level_dag.py` + `orion/core/auto_bootstrap.py`, the algorithm of
arXiv 2311.03470 Sec. 5.2), adapted to this repo's captured-graph IR.

Upstream algorithm, implemented here structurally 1:1:
  * every network LAYER gets a column of "run at level l" vertices; edges between
    consecutive layers price the bootstrap needed for that level transition, weighted
    by the number of ciphertexts crossing (get_num_input_cts);
  * residual regions: fork = node with >1 out-neighbour, join = first node common to the
    paths from every out-neighbour to the sink (find_residuals); each region is solved
    innermost-first into an AGGREGATE edge matrix — for every (fork level i, join
    level j) the summed per-branch shortest path, with the traversed bootstrap
    decisions stored on the cell (LevelDAG.__add__ + the `path` bookkeeping);
  * nested regions are spliced into branch paths when encountered
    (build_level_dag_from_path); the full network is one final shortest path and
    bootstraps are marked on producer outputs (mark_bootstrap_locations).

Adaptations (documented in summary.placer_meta.deviations):
  * layer = capture STEP (Orion's nodes are torch modules; our step tags are that
    granularity). State = consumed composite level AFTER the step; a bootstrap of a
    step's outputs costs #crossing vars (ciphertext count), count objective —
    upstream's own comment: the linear-layer latency term does not change bootstrap
    counts, and t_boot is a constant multiplier at fixed l_eff.
  * branch representatives: one path per fork child (upstream dedups all_simple_paths
    by first child; we take the DEEPEST-cost path per child — the binding branch, so
    the depth the region must pay is the depth the model prices; a fewest-hops
    representative routes around the deep approximation chains and leaves them
    unpriced). Branch costs sum over branches, so shared sub-steps double-count in
    the OBJECTIVE but dedupe at PLACEMENT — exactly upstream's __add__ behavior.
  * equal join levels across branches (upstream's rule; conservative-valid here since
    our runtime max-joins).
  * no free level-drop edges: our runtime places only bootstraps, never modswitches.
  * steps whose own depth exceeds l_eff are capped for the solver and left to the
    shared repair (counted in placer_meta.unfixable_steps) — upstream would declare
    the whole program unplannable ("increase your LogQ chain").

## python/perseus/plan/placer/ilp.py — exact ILP placer

Exact bootstrap placement: the stage-4 decision as a mixed-integer program.

The min-cut `Placer` is greedy at the level of the whole block: every pass is an exact
minimum cut of ITS flow network, but the network is rebuilt from the current simulation,
wires only the operands that bind now, never revisits a site an earlier pass chose, and
does not price the hints that fire as a consequence. `IlpPlacer` states the same decision
once, over the whole block, and hands it to a branch-and-bound solver (HiGHS, through
`scipy.optimize.milp`), which returns a placement together with a proven lower bound.

The model is a transcription of `sim.simulate`, case for case. With
eff = consumed + unit * [deg 2], every multi-operand ADD (any degree mix) and every MULT
is a max over its operands' eff, so the level algebra is max-plus over a handful of
integer values. Each var's eff is therefore held as a unary LADDER over the values it can
take -- one 0/1 quantity per value, "eff >= k" -- and every operation becomes logic:

  max        [max >= k]  = OR of the operands' [eff >= k]
  refresh    [post >= k] = x ? [landing >= k] : [pre >= k]     (x = refresh here)
  hint       fires       = [input eff > threshold], one rung of the input's ladder
  budget     consumed <= b  fixes the rung above b to 0 (per degree / refresh case)

ORs and ANDs of 0/1 quantities are exact at integer points under their standard
inequalities, so only the refresh decisions x are free; every logic variable follows
from them. They are still declared integer: continuous, the OR's `y <= sum(inputs)`
lets the solver's feasibility tolerance add up through fan-in and depth (1e-6 grew to
0.99 over a block and flipped a degree), and presolve recognises them as implied
integers, so declaring them costs nothing and in fact solves faster. The unary form
is what makes the relaxation tight enough to prove optimality on real blocks (a big-M
max formulation stalls at a ~25% gap).

The objective is the plan's bootstrap bill: each placed site costs what `_capacity` prices
it at (1 + miss_penalty * overshoot, + the quality pre-score), and with objective "total"
each fired hint costs the same way, so the solver may trade a hint for a placement. With
"placed" hints are free, which is the min-cut's own objective. With "ms" every refresh
(placed or hint) is priced at its route's measured latency, so a dense refresh costs what
it costs against a sparse one: the bill is bootstrap milliseconds, not bootstraps. The level-dependent pricing
forms (depth_weight, level_weight) are not linear in the decisions and are refused.

Nothing the solver says is trusted: the placement is re-simulated, and the plan is kept
only if the sim agrees with the model on every level, degree and hint decision. The
min-cut runs first, as the incumbent (a cutoff on the objective, since `milp` takes no
warm start) and as the fallback when the solver fails, times out empty-handed, or the
check disagrees.

## python/perseus/profile.py — session profiles

Typed session profiles: the env contract as an object instead of scattered exports.

The runtime is steered by env vars whose semantics-bearing subset must agree with the
plan being loaded (see perseus.plan.contract). A SessionProfile makes that subset
explicit, applies it with env-as-override semantics (an already-exported var always
wins, so run scripts keep working unchanged), and hands back InferenceOptions.

    from perseus.profile import SessionProfile
    prof = SessionProfile.gpt2_decode_n32()
    prof.apply()                      # setdefault into os.environ
    inf = _core.make_gpt2_inference(prof.options())

`extra` is a free-form env mapping for everything without a field: sparse routing, the
softmax fold, arena sizes, the rotation-key band, planner units. The subset that shapes
the executed graph is also the plan contract (perseus.plan.contract); the rest only moves
WHEN work happens.

## python/perseus/session.py — session lifecycle

A session is a live FHE context plus the model-shaped state the runtime keeps for it
(an `_core.Inference`), with a Python lifecycle around it.

    from perseus.session import session
    from perseus.profile import SessionProfile

    with session(profile=SessionProfile.custom_n32()) as s:
        model = EncSequential(...).bind(s)        # bind() accepts the session
        y = model(x)

`close()` (or leaving the block) releases what the session holds on the device —
installed weights, the KV/mask/encode caches and the loaded rotation keys — and drops
this handle; the CKKS context object itself lives until process exit (ciphertexts
reference it). A second `session()` while one is open is refused rather than silently
sharing global state; after close() a new one can be created.

# Python tests

## python/tests/test_paper_plans.py — shipped-plan regeneration test

Every shipped plan regenerates from its graph with the recipe in its PLAN_CMD.txt.

A plan whose PLAN_CMD.txt names ``tool=perseus/examples/gpt2/make_plan.sh`` (the
Python implementation's plans) is regenerated with that tool: the blocks, the argmax stage
(block 13, entered at the tail plan's exit) and, where the plan ships one, the feedback stage
(block 0 entered as a fed-back token), each stamped with the capture contract.

The main plan is always checked; the baselines and ablations (about 20 planner runs, a few
minutes) run when PERSEUS_ALL_PLANS=1; PERSEUS_PLANS=a,b runs the named plans only. The python/ DaCapo and Orion plans are regenerated with the released tools
(the old repo's scripts/utils/{dacapo,orion}_upstream/plan.sh, not ported: those plans skip).
Comparison: field by field (`PLAN_FIELDS`, the plan content the runtime binds: placements, rescale anchors,
routes, CFs, raise drops, hint pins, and the summary's final named levels/degrees), then the whole runtime body
(perseus.plan.equiv, summary additive only) with the contract stamps (`capture_env` / `config`) left out: a
replanned file carries the Config stamp, a shipped one the legacy env stamp; test_plan_contract.py checks both.

The planner gate (`PC_PLAN_GATE=1`, minutes each; `python python/tests/test_paper_plans.py [name ...]` runs them
by hand and prints per-plan diff counts and wall times) replans `GATE`: the recipes whose PLAN_CMD.txt is
incomplete (the prefill pipes recorded only part of their env) are spelled out there.

Graphs and plans live under perseus/assets (the tools run from there); a PLAN_CMD.txt
written by the old repo names its tool by the old path, mapped onto the package's copy.

## python/tests/test_placer_fidelity.py — placer fidelity tests

Port-fidelity tests for the baseline placers (Orion / DaCapo / Fhelipe).

Ground truth, vendored in tests/fixtures/placer_fidelity/:
  * graphs/          nine synthetic capture graphs (chains, fork-join, skip,
                     fan-out, diamond) exported to the real upstream tools
  * upstream.json    the site sets the real upstream Orion / DaCapo / Fhelipe
                     chose on those graphs (provenance inside the file)

The tests re-run our ports on the same graphs with the same config and assert,
per placer:
  FIDELITY  — on every case the recorded comparison found exact, our port still
              equals the real upstream tool's site set (fhelipe 9/9, dacapo 8/9,
              orion 6/9; Orion's validation basis is its objective, so its three
              site-divergent shapes are pinned as known, not asserted exact).
  NO DRIFT  — on the known-divergent cases the divergence is exactly the recorded
              one (a change in either direction fails).

A failure here means a planner/sim change moved a baseline port off the validated
upstream semantics.

## python/tests/test_plan_baselines.py — baseline placer tests

Tests for the baseline placers (perseus/plan/placer/baselines).

Synthetic cases pin the behaviors that make the benchmark honest:
  * every placer produces a P1-feasible plan on a plain deep chain;
  * Fhelipe's DP is count-optimal on a chain (exact assert);
  * a fork/fan-out gets ONE refresh at the shared producer, not one per consumer;
  * a magnitude-hopeless site is a LOUD PlanInfeasible in raw mode and a counted
    rescue_moved in rescue mode;
  * heuristic_config and placer_meta are stamped into the emitted plan.

Real-graph smoke runs when .cache/graph_n32_gpt2_base is present.

  .venv/bin/python tests/test_plan_baselines.py

## python/tests/test_plan_ilp.py — ILP placer tests

The exact placer (perseus/plan/placer/ilp.py): its model is the sim, and its answer is optimal.

* TWIN     — with the refresh decisions pinned to arbitrary placements (feasible or
             not), the MILP's levels, degrees and hint decisions equal `simulate()`'s
             on every var. This is what makes "optimal" mean optimal for the planner.
* OPTIMAL  — on the synthetic fixture graphs, no placement is cheaper than the ILP's:
             checked by enumerating every placement that could be (a site costs >= 1,
             so only placements of at most `objective` sites can beat it).
* DOMINANT — never costlier than the min-cut it starts from, on every graph here.
* PLUMBING — plan_block(placer="ilp") emits a plan through the usual postconditions,
             labelled, with the solver's status in summary.placer_meta.

python -m pytest tests/test_plan_ilp.py
