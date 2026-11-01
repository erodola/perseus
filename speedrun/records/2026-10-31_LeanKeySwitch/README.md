# Both tracks on a leaner key switch, leaner bootstraps, raise drops and a key set of their own; Decode-8 with its position plaintexts encoded at setup, a shorter CutMax and leaner LayerNorms

Record 4's code and plans (LeanChainDecode, LeanChainPrefill), with the served decode's host work trimmed, Decode-8's
SparseB refreshes on Prefill-128's EvalMod series, fewer instructions and shorter waits in the key switch's kernels, and
fewer key switches and rescales in the bootstraps, which both tracks run. The host-side and kernel changes leave what the
device computes as it was: on one key bundle and one set of input ciphertexts, the host-side changes give outputs
byte-identical to record 4's, and the kernel changes outputs byte-identical to the state before them (Prefill-128's,
whose plan was already on K = 16: byte-identical to record 4's). Three bootstrap changes compute the same refresh with
fewer roundings (two giant key switches fewer per StC-first refresh, and EvalMod's rescales fused with their
relinearizations), at the same levels and scales, so the plans stay as they are.

A second part (from `e737742`; the first push ended at `bc06718`) goes further on the same lines and adds keys. First
more exact work, byte-identical to the state before it: EvalMod's executor fusions, lazier CtS products, a third round
of key-switch kernels (hoisted rotation sums summed inside the key dots, base conversions by job kind, the key-switch
chain by programmatic dependent launch), the replay's K/V and K-batch copies gone, and a rotation's Q rows on a second
stream. Then the same maths with fewer roundings: the decode linears' giant steps summed before one ModDown and their
relinearizations fused with their rescales, and CutMax in 11 refreshes per token instead of 13, with ln_f folded into
the LM head, and Decode-8's LayerNorms take their statistics from one refresh or one complex ladder while its attention
ladders run over the 16 keys' lanes. Then the plans: refreshes whose consumers have levels to spare raise to a lower
modulus (Prefill-128's qkv and up PCMM finishes, Decode-8's LayerNorm refreshes, two K/V pushes). And keys: SparseRot
keys fold the raise's sparse → dense switch into the next hoisted rotations, StC-first's input stages switch over 6
special primes instead of 13, and EvalMod relinearizes over level keys. Every new key is an RLWE sample under the dense
main secret at a modulus of at most the main key's 1733 bits, and this entry's client (`perseus_keyset_client.py`)
checks that key by key.

### The first part

- **The position plaintexts, encoded once at setup.** Record 4's server already records every position's patch at
  setup (the plaintexts whose values change from one position to the next: masks and position constants, 94 to 97 per
  position after aliasing), but encodes them in the clock, one device encode and one host-device sync each. Here the
  1,550 distinct values the 16 positions stage are encoded once at setup into a device cache (7.4 GB), and a token copies
  its position's encodings into their slots, device to device, in place (the slots keep their addresses, so the
  captured graphs that read them stay valid).
- **CutMax issued behind the forward.** The server exported the logits (which waits for the device) before issuing
  CutMax, so the device idled while the host downloaded the logits and launched CutMax. Now CutMax is issued first,
  and both outputs are exported after it.
- **Ciphertext frames read and written in place.** Importing the 12.8 MB input frame took about 10 ms: a slice copy
  in Python, a copy into a word vector, then 25 row uploads per polynomial, each followed by a stream sync. Now the
  frame's words are checked and uploaded where they lie (one copy per polynomial), and an exported frame is downloaded
  straight into its bytes. The server reads its input without the slice copy and writes the packed outputs without
  joining them.
- **The next position's patch, ahead of its input.** Once a position is answered, the next position's dynamic steps and
  plaintext copies are applied while the client sends the next token.
- **SparseB refreshes on the degree-14 EvalMod series (K = 16).** Record 4's Decode-8 plan runs its 40 SparseB refreshes
  per token with K = 24, FIDESlib's degree-20 table; its Prefill-128 plan already sets K = 16, the dense route's degree-14
  table (`FIDESLIB_BTS_SPARSE_B_K=16`). The plan compiles to the same levels and landings; each SparseB refresh is about
  0.7 ms cheaper on the RTX PRO 6000.
- **Base conversion's pre factors in the INTT's last pass.** A fast base conversion first multiplies each source
  coefficient by its pre factor `(B / b_i)^-1 mod b_i`, a constant of the source row. Every conversion in a key switch
  reads the output of an INTT, whose last pass already multiplies each row by `N^-1`, so the INTT now multiplies by
  `N^-1 pre_i` instead and the conversion reads `x_i pre_i` as written: the ModUp (every live digit, per level), the
  ModDown (P, and P' at the bottom), and the ModDowns of the linear-transform stages and EvalMod.
- **Double-word sums reduced in 32-bit products.** The sums of 28-bit products (base conversion, the inner products,
  CtS, the tensor product, the keys' regeneration) were reduced with a 64-bit Barrett. Each is now `hi 2^32 + lo`,
  with `hi (2^32 mod p)` and `lo` reduced as 32-bit Shoup products (exact for every 64-bit sum when p < 2^31): the same
  residue in about half the instructions. Base conversion keeps its targets' constants in shared memory.
- **L2 prefetch of the next key reads.** CtS (K = 1) reads its key `b`, a resident `a` digit, a ModUp digit and the
  diagonals digit by digit, and each load waits on DRAM; a baby step now first asks L2 for the next baby step's lines.
  The K = 1 inner product asks L2 for the next digit's segment the same way, at three blocks per SM.
- **The NTT's row passes at five blocks per SM.** Their loads wait on L2 (the passes are L2-resident); at 48 registers,
  without spills, more warps hide the wait.
- **CtS stage 0 as one giant step.** Its giant rotation is the identity (16 × 2048 slots), so its giant steps'
  diagonals are summed as integers into one set of plaintexts per baby step: one product per baby step instead of
  one per giant step, and no add of the extended results. The same residues (bit-identical), and fewer plaintexts
  stored.
- **StC giant steps folded mod 2.** The StC stage whose giant rotation squares to the identity (stride N/32, 16 baby
  steps) folds its giant steps mod 2 the same way: two giant key switches fewer per StC-first refresh. Not
  bit-identical (two key-switch roundings fewer).
- **EvalMod's relinearizations next to their rescales.** A scheduling pass moves the ops between a relinearization and
  the rescale of its result to where they commute (before or after the pair), keeping one add between them, so the
  pair runs as the fused relinearization-rescale the served plans turn on (`PC_EM_FUSE_RESCALE=1`). Standalone
  rescales per refresh: StC-first 11 → 5, SparseB 13 → 6.
- **EvalMod's series without two alignment rescales.** The odd powers subtract T_1 from its first realized copy,
  brought to their level by a scalar and a limb drop instead of a rescale, and the series' giant power is realized
  before it joins the sum, so its rescale rides its relinearization. Same output level and scale. FIDESlib's program
  stays the builder's default, and the comparison test against FIDESlib still checks it.
- **The NTT, five ways.** The second pass is launched during the first pass's tail (programmatic dependent launch);
  the lazy butterflies take 2p from a register (the compiler had turned each `x - 2p` into an IMAD on the FMA pipe,
  which the row passes saturate); the row passes' lane-pair stages take each product once (each lane of a pair takes
  half the slots); their factored twiddles are read directly, as vector loads (the index map was the identity); and
  with every prime below 2^28, forward words grow to 16p, with three reductions per transform instead of one per stage.
- **CtS for the StC stages at seven blocks per SM** (`cts_fused_rq<1,4>`, 96 → 70 registers, no spills), and the
  dense routes' four-digit CtS stages read two of their four digits' `a` from memory instead of regenerating them
  (record 4's rule kept these resident tails for stages of five or more digits, which dnum 4 never has; SparseB keeps
  that rule).
- **Base conversion over a wave at 12 blocks per SM.** Launches above one wave run a 40-register body with each job's
  targets split evenly over its blocks; smaller launches keep the 46-register kernel.
- **Chains of plaintext products as one pass.** A plaintext product and the fused products that add onto its running
  sum run as one `mul_add_n` (up to 8 terms, a double-word lazy sum reduced once): a decode token's 3,049 `mul_add`
  launches become 4, plus 435 `mul_add_n`.

The kernel and bootstrap changes, per refresh on pc-bench (`bts --all --keys packed`, the lean chain, the RTX PRO
6000): the key switch's first four changes take StC-first from 13.00 to 12.43 ms, SparseB-512 from 13.06 to 12.08 ms and
SPRU from 7.34 to 6.79 ms, with outputs bit-identical; the rest, interleaved in a second job, StC-first 12.48 → 11.11 ms,
SparseB-512 12.13 → 11.67 ms, SPRU 6.82 → 6.53 ms (with EvalMod's fusion on, as the plans serve it: 12.33 → 10.81, 11.98
→ 11.32, 6.81 → 6.50; pc-bench's SparseB runs FIDESlib's K = 24 table).

### The second part

- **EvalMod's executor fusions.** The series' scaled addends ride the tensor product's pass, its three weighted sums
  run as one pass, and a sum of one source folds into its rescale (kernel ABI 19). Byte-identical.
- **CtS products reduced lazily.** The K = 1 CtS kernel keeps its baby sums in [0, 4p) and its ChaCha words
  unreduced, with 16 digit buffers per warp. Byte-identical.
- **A third round of key-switch kernels.** Small-integer scaling keeps its words in registers and runs its Garner
  steps on 32-bit words; the rescale's fused transform runs at the context's lazy level, its row pass by programmatic
  dependent launch; the forward row pass canonicalizes its lazy words with one quotient estimate; base conversions
  short of a wave split each job's targets over more blocks; the dots read the keys' `b` evict-first in L2, so the
  stream no longer evicts the ModUp digits; the K = 1 relinearization dot runs on warp segments. Byte-identical.
- **Hoisted rotation sums inside the key dot.** A sum of hoisted rotations was one extended ciphertext per rotation,
  each stored, ModDowned and added. Now the K = 1 segment dot sums the rotations' products as it goes (`KS_SUM`), and
  so does a K > 1 key dot (`ks_dot_key_sum_c`, its digits' keys in registers, at five blocks per SM), so one extended
  ciphertext is stored and ModDowned per sum. A rotation grid whose outputs sit in consecutive slots takes the
  ModDown's last step in the conversion NTT's epilogue. Byte-identical.
- **Base conversions by job kind, and the key-switch chain by dependent launch.** Plain jobs of 13 sources (a lean-chain
  digit, P) and exact jobs of 15 (P and two tops) are unrolled at compile time, the exact ones reduced in 32-bit words
  like the plain ones. The INTT's column pass lets the conversion in, which sets up its tables before it waits, and the
  conversion lets the forward column pass in; the rescale's prep and column pass do the same. Byte-identical.
- **A rotation's Q rows on a second stream.** A one-ciphertext key switch's inner products over the Q rows run on a
  forked stream while the P rows run on the main one, joined before the ModDown's NTT. The rows are disjoint, so the
  outputs are byte-identical.
- **The replay's copies.** K/V registers are lent to their slots for a block's run (842.9 → 54.9 eager copies per
  token), `x + x` runs in place, a K-batch reads and writes its operands in consecutive slots (Prefill-128's copies per
  request 33.6k → 13.1k), and a scalar adjust that folds no scale skips its copy. The server maps its input frames in
  place, verifies their checksums on a thread until the next export, and polls every 50 µs. Setup's precompute reuses
  the probe tokens' records (four CPU-twin recordings fewer). Byte-identical.
- **The decode linears and polynomials.** A giant step's rotations are summed before one ModDown, a key switch's add
  rides its ModDown, rescales are shared where their operands agree, and relinearizations are fused with their rescales
  in the program's ops (368 per decode token, 1,378 per prefill request). Same values with fewer roundings, not
  byte-identical.
- **A shorter CutMax** (Decode-8, plan `_cmfl`). The 1-slot SPRU refreshes take the vector itself, since a SPRU refresh
  returns its input's slot mean (six rotation sums dropped). The first Newton step's scaling rides the SPRU input, which
  is linear, so the step's hints refresh StC-first and the mid-chain SPRU refresh goes. One conjugate per vector
  refresh. ln_f's gain and bias fold into the LM head, so the tail exits a level earlier (stage 12 replanned) and
  CutMax's entry square comes off the logits' masked products. 13 refreshes per token (6 dense, 7 SPRU) become 11 (3
  dense, 8 SPRU). The same function in exact arithmetic, not byte-identical.
- **Leaner LayerNorms and attention ladders** (Decode-8, plan `_lh`). Decode-8 holds at most 16 keys, so a query's
  replicas, both head sums and P·V's key sum run over 16 lanes instead of 32 (`key_lanes`, exact: 12 key switches fewer
  per block). Where the plan already refreshes a LayerNorm's variance (ln_2 of blocks 1, 2, 3, 7 and 10, ln_1 of blocks
  2, 3, 4, 10 and 11), that one 1-slot refresh returns the mean and the mean square and no ladder runs (`spru`); the
  other 14 LayerNorms sum the token and its square on one complex ladder, 32 rotations instead of 22 + 35 (`zladder`);
  the variance is the mean square less the squared mean. Blocks 0-11 are recaptured and replanned with the same recipe
  (block 11 still exits at 36:1, so stages 12 and 13 are `_cmfl`'s), then the raise-drop re-pass; the K/V push raise now
  fits blocks 4 and 5 only.
- **Raise drops on deliberate refreshes.** A refresh whose consumers have levels to spare can raise d units below the
  top and land 2d levels lower, every stage on fewer limbs. A planner probe (`PLAN_RAISE_DELIB`) reports the largest
  drop each deliberate refresh could take on the final program. Prefill-128 raises its qkv and up PCMM finishes that way
  (`PF_PCMM_RAISE_OUT`, 106 of the 118 sites with slack); Decode-8 runs the raise-drop search once more on its final
  program (16 LayerNorm SPRU refreshes per token 2 to 4 levels lower) and raises the K/V push refresh of blocks 4, 5, 7
  and 10 (`KV_PUSH_RAISE`). Dense refreshes at a drop of 1 or more run a five-stage CtS from one unit higher
  (`FIDESLIB_CTS_ALT_BUDGET=5`), with the same landing and fewer limbs per stage.
- **A key set of their own.**
  - SparseRot keys (`PC_BTS_MERGE_S2D=1`). The raise's sparse → dense key switch folds into the next hoisted
    rotations (CtS stage 0, SparseB's first fold round), whose keys switch from the sparse
    secret's rotation straight to the dense secret: one ModUp and one ModDown fewer per refresh. 18 keys.
  - StC-first's input stages (tops 4, 3 and 2) switch over P'' = the first 6 special primes (`PC_BTS_STC_PS=6`) instead
    of all 13, a prefix of P that still exceeds the digit they switch by 16 bits. 48 rotation keys at q0..q4 × P''.
  - EvalMod's relinearization-rescales run over level keys whose special modulus takes the Q primes idle above the
    level (`PC_EM_LEVEL_KEYS=1`, at most 14 special primes: the level-41 relinearization in 3 digits instead of 4, the
    level-27 one in 2 instead of 3). 2 keys. Base conversion takes up to 48 sources for them (`bconv_wide`, kernel ABI
    20).
  - The client, `speedrun/harness/perseus_keyset_client.py`, is record 4's lean client with these three roles: it
    checks each key against the secret its samples are under and bounds its modulus (Disclosures).

On the reference GPU (RTX PRO 6000 Blackwell Server Edition), Decode-8 runs at 1.217 s/token and Prefill-128 at 7.084 s:
×1.259 and ×1.297 against the first part in the same jobs (6 pairs each, p 1.4e-12 and 1e-14), and ×1.475 and ×1.430
against record 4's 1.795 s/token and 10.13 s on the same GPU (paired before the LayerNorm commits, which leave
Prefill-128 byte-identical: ×1.455, p 7.2e-16, and ×1.430, p 2.3e-16). On all 32 dev prompts the mean KL is 7.5e-6 on
Decode-8 and 1.2e-5 on Prefill-128 (the first part's 3.2e-6 and 1.2e-5). The first part alone took Decode-8 to 1.530
s/token (×1.173) and Prefill-128 to 9.185 s (×1.103).

## Changes against record #4

- **The position cache.**
  - `PtStore::copy_out` / `copy_in` (a resident plaintext's rows to a buffer and back, in place) and `words_at`:
    `crates/pc-gpu/src/pt_store.rs`.
  - The session's cache, in its own device allocation, and `PtStore.cache` / `PtStore.restore`:
    `crates/pc-py/src/{session,store}.rs`.
  - `PlannedRun(pt_cache=True)`: `precompute` encodes every position's staged values into the cache, stages position
    0's plaintexts whole at setup, and `token` restores instead of encoding: `python/perseus/impl/driver.py`.
  - The served decode takes `pt_cache` (and `--pt-cache` on its own CLI): `python/perseus/serve/gpt2_decode.py`;
    `speedrun/server.py` turns it on.
- **CutMax issued before the exports:** `Gpt2Handler.run` in `python/perseus/serve/gpt2_decode.py`.
- **Frames in place.**
  - `DBuf::upload_bytes_at` / `download_bytes_into` (whole elements from or into bytes at any host alignment):
    `crates/pc-cuda/src/mem.rs`.
  - `import_ct` reads a frame (bytes or a memoryview) in place, with `CtWire::from_bytes`'s checks; `export_ct` writes
    `CtWire::to_bytes`'s frame directly: `crates/pc-py/src/client.rs`. The client crate (`crates/pc-client`) is
    unchanged.
  - `speedrun/server.py`: `read` without the slice copy, `write_packed` and `unpacked` (record 1's list framing without
    joining or slicing copies).
- **The patch ahead:** `PlannedRun.patch_ahead` (`python/perseus/impl/driver.py`), `Gpt2Handler.ahead`, and the call in
  `speedrun/server.py`'s decode loop.
- **SparseB K = 16:** the runtime lines of `bootstrap_placements/gpt2_decode_python_n32a4_kc16r{,_cm}/PLAN_CMD.txt`, and
  `speedrun/make_n32a4_decode_plan.sh`, which writes them (its placements are unchanged).
- **Base conversion's pre factors in the INTT** (kernel ABI 16 → 17).
  - The inverse column pass takes a per-row scale (`NttArgs.es` / `ess`, offset per chunk): `ntt_inv_cols{,8}` in
    `crates/pc-kernels/kernels/ntt.cuh`; `ntt::inverse_rows_prefix_scaled` / `inverse_k_scaled` in
    `crates/pc-gpu/src/ntt.rs`.
  - `BconvJob.prescaled` (the conversion reads its input as the pre-multiplied value): `bconv.cuh`, `args.cuh`,
    `crates/pc-gpu/src/bconv.rs` (`pre_factors`).
  - The tables (`N^-1 pre_i` and its Shoup word per row): the ModUp's per level and the ModDown's for P and P'
    (`crates/pc-gpu/src/ks.rs`), the linear transforms' ModDown (`MdPlan` in `crates/pc-gpu/src/bts/cts.rs`).
- **The split reduction.**
  - `PrimeConst.split` (`2^32 mod p` and its Shoup word, in the last word of a prime's constants):
    `crates/pc-core/src/modmul.rs`, `args.cuh`; the 32-bit `reduce_dw` in `arith.cuh` uses it.
  - Base conversion's own path, with its targets' constants in shared memory (`BconvJob.split`, set when every target is
    below 2^30): `bconv.cuh`, `crates/pc-gpu/src/bconv.rs`.
- **The prefetch:** `cts_fused_rq` in `crates/pc-kernels/kernels/cts.cuh`, `ks_dot_seg` in `ks.cuh`.
- **The row passes' occupancy:** `ntt_fwd_rows` / `ntt_inv_rows` in `ntt.cuh`.
- **The giant-step folds:** `fold_periodic_giants` in `crates/pc-gpu/src/bts/cts.rs` (a stage's giant steps folded mod
  the order of its giant rotation), applied to the CtS and StC stages in `bts/dense.rs`, `bts/sparse_classic.rs` and
  `bts/stc_first.rs`.
- **EvalMod.**
  - The scheduling pass that brings a relinearization next to its rescale, and the fused `Mult; Add; Rescale`:
    `crates/pc-gpu/src/bts/evalmod.rs`.
  - The lean series program (`Builder.lean`, on for the runtime, off by default): `crates/pc-shadow/src/evalmod.rs`.
- **The NTT:** `crates/pc-kernels/kernels/ntt.cuh` (the dependent launch, 2p in a register, the lane-pair stages, the
  direct twiddles, lazy level 2); `NttTables.lazy` in `crates/pc-gpu/src/context.rs` picks the lazy level.
- **CtS:** the launch bound of `cts_fused_rq<1,4>` in `cts.cuh`; the dense routes' tail threshold in
  `crates/pc-gpu/src/bts/{cts,dense,sparse_b}.rs`.
- **Base conversion over a wave:** `bconv.cuh`.
- **The product chains** (kernel ABI 17 → 18): `mul_add_n` in `em.cuh` and its table entry (`args.cuh`,
  `crates/pc-kernels/src/args.rs`); `MultPtAddChain` in `crates/pc-program/src/{bind,replay,schedule}.rs`;
  `crates/pc-gpu/src/ops.rs`.
- **Tests:** `python/tests/test_speedrun_server_framing.py` (the server's framing writes and reads `record1.pack`'s
  bytes); `crates/pc-kernels/src/bin/selftest.rs` gains prescaled and split base conversions and the row-scaled INTT
  (lazy and not, chunk offsets 0 and 5), the lazy-2 NTT at the 2^28 edge and `mul_add_n`, against the CPU, and its
  prime table now passes the new word; pc-shadow's general-vs-m2 test also checks that the lean program keeps the
  output level and scale, and pc-gpu's EvalMod buffer test runs the lean, scheduled program.

The second part:

- **EvalMod's fusions** (kernel ABI 18 → 19): the tensor pass with scaled addends and the multi-sum pass in `em.cuh`
  (`args.cuh`, `crates/pc-kernels/src/args.rs`, `inst/instantiate.inc`); their use in
  `crates/pc-gpu/src/bts/evalmod.rs`.
- **Lazy CtS products:** `cts_fused_rq` in `cts.cuh`; the sparse-input CtS identity as its own instances there too.
- **The third kernel round:** small-integer scaling in `bts.cuh`; the rescale's transform and dependent launches in
  `rescale.cuh`; the row pass's canonicalization and the column passes' dependent launches in `ntt.cuh`; base
  conversion's short-of-a-wave split, its job kinds and its dependent launch in `bconv.cuh`; the evict-first key reads
  and the relinearization dot on warp segments in `ks.cuh`.
- **Rotation sums in the key dots:** `KS_SUM` and `ks_dot_key_sum{,_c}` in `ks.cuh` (with `chacha.cuh`'s expansion for
  the K > 1 keys); their callers in `crates/pc-gpu/src/{ks,rotate}.rs`, `bts/sparse.rs`; the grid's ModDown epilogue in
  `crates/pc-gpu/src/{ks,ntt,rotate}.rs`.
- **The forked Q rows:** `crates/pc-gpu/src/{executor,ks,rotate}.rs` (the aux stream, the split dots, the joins
  before each reader).
- **The replay's copies:** `crates/pc-program/src/{bind,replay}.rs` and `compile/alloc.rs` (lent registers,
  consecutive K-batch slots), `crates/pc-py/src/program.rs`, `crates/pc-gpu/src/{ct,ops,rotate}.rs`,
  `crates/pc-cuda/src/mem.rs`; the mapped input frames in `crates/pc-py/src/client.rs` and `speedrun/server.py`; the
  precompute in `python/perseus/impl/driver.py`.
- **The decode linears:** `crates/pc-program/src/compile/{rotacc,rr,share,check,batch,mod}.rs`,
  `crates/pc-program/src/{program,bind,replay,schedule}.rs`, `crates/pc-gpu/src/{relin,rotate,ks,ops}.rs`.
- **CutMax:** `python/perseus/examples/gpt2/head.py` (`spru_mean`, `newton_fold`, `conj_once`),
  `python/perseus/impl/{config,poly,fake}.py`; the approximation configs `gpt2_base_n32_kc16r_cmf{,l}`, the plans
  `bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cmf{,l}` and their graphs; `scripts/utils/merge_lnf_tail_mags.py`
  and `speedrun/make_cutmax_plan.sh`, which write them; `speedrun/server.py` serves `_cmfl`.
- **LayerNorms and attention ladders:** `NormCfg.spru` and `NormCfg.zladder` in `python/perseus/impl/{config,norm}.py`,
  `SoftmaxCfg.key_lanes` in `python/perseus/impl/{config,attention}.py`; the approximation config
  `gpt2_base_n32_kc16r_cmfl_lh`, the plan `bootstrap_placements/gpt2_decode_python_n32a4_kc16r_lh` and its graphs
  (`graphs/gpt2_decode_python_n32_kc16r_lhm`); `speedrun/server.py` serves `_lh`; `python/tests/test_ln_spru_cpu.py`
  checks the three forms against the ladder form on the numpy fake.
- **Raise drops:** the probe in `python/perseus/plan/placer/{planner,sim}.py`; `PF_PCMM_RAISE_OUT` in
  `python/perseus/impl/pcmm.py`; `KV_PUSH_RAISE` in `python/perseus/impl/attention.py` and
  `crates/pc-program/src/compile/mod.rs`; the plans' blocks and runtime lines, written by
  `speedrun/make_prefill_pcmm_drops.sh` and `speedrun/make_n32a4_decode_rd.sh`.
- **The key set** (kernel ABI 19 → 20): SparseRot keys in
  `crates/pc-gpu/src/bts/{mod,cts,dense,raise,sparse,sparse_b,stc_first}.rs` and `cts.cuh`; the P'' stages in
  `bts/stc_first.rs` and `ks.rs`; level keys in `bts/evalmod.rs` and `crates/pc-core/src/params.rs`; `bconv_wide` in
  `bconv.cuh` and `crates/pc-gpu/src/bconv.rs`; device keys in `crates/pc-gpu/src/{keys,keygen}.rs`; the client's roles
  and shapes in `crates/pc-client/src/{client,spec,ntt}.rs`, the prime sizes it bounds moduli with in
  `crates/pc-py/src/client.rs`; the switches in `crates/pc-config` and `crates/pc-py/src/session.rs`; the served plans'
  runtime lines.
- **pc-bench:** a stray profiler-range close from the key-set commit removed (`crates/pc-tools/src/bts_all.rs`:
  `prof` was never defined, so pc-tools did not compile).
- **Planned generation's feedback:** `fb_tiles` in `python/perseus/examples/gpt2/model.py` embeds the next token with
  the unfolded LM head (with `lnf_fold` on, the head's tiles are γ-scaled; the served decode takes every position's
  input from the client and does not run the feedback).
- **Tests:** `crates/pc-kernels/src/bin/selftest.rs` gains EvalMod's two passes, `bconv_wide`, base conversion by job
  kind, `ks_dot_key_sum` and `ks_dot_key_sum_c` (K = 1, 2, 3), against the CPU; `crates/pc-client/tests/key_secrets.rs`
  recovers each new role's error from its generated keys (`b + a s_new - G s_old` small, digit by digit, with the
  secrets and gadget computed independently of the keygen), so its samples are under the secret it declares;
  `python/tests/test_cutmax_switches_cpu.py` runs CutMax's switches on the numpy fake against the record's form and the
  plaintext mirror; `crates/pc-program/tests/shipped_plans.rs` compiles Decode-8's served `_cmfl` plan (0 check errors);
  the field-count tests count the new switches.

## Ablation

The speedrun harness on the RTX PRO 6000 of the Dev result, two prompts per run (dev 0-1). Each row adds one change to
the row above (this PR's commits in order). The first four rows' five trees were timed in one job: 4 rounds, alternately
forward and backward, each row scored by `score.py` against the row above it (logs `abl_*`); those rows give
byte-identical outputs. K = 16 was timed against the row above it in its own job (4 pairs AB BA BA AB, `ab4_*`). The
kernel rows were timed in a second job of the first kind, from record 4 through K = 16 and each kernel commit (six
trees, 4 rounds, `fabl_*`; K = 16 against record 4 there: 1.794 → 1.733, ×1.035, p 1e-6); they give outputs
byte-identical to K = 16's. The bootstrap rows and the kernel rows after them were timed in a third job of the first
kind, from the row passes' commit through four groups of commits (five trees, 4 rounds, `nabl_*`), and the four groups
together in one 6-pair job per track (AB BA BA AB AB BA, `nfinal_*` and `npre_*`). The second part's first row is the
first part's final tree against the second part's exact work, linears and CutMax, one 6-pair job per track (`tA_*`,
`tApre_*`); its next rows are four groups of commits timed in one job per track on another RTX PRO 6000 of the same
machine, not isolated (other jobs on the machine's other GPUs and CPUs): five trees, 2 rounds alternately forward and
backward, each row scored by `score.py` against the row above it (`sabl_*`, `sablpre_*`; from the key set on, with this
entry's client); the LayerNorm row was timed against the commits before it on a third RTX PRO 6000 of the machine, 2
pairs A B A B (`lh_*`).

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| Position plaintexts encoded at setup, copied per token | decode | 1.792 → 1.772 s/token, ×1.011 (p 3.4e-5) |
| CutMax issued before the exports | decode | 1.772 → 1.771 s/token, ×1.000 (p 0.11, not resolved) |
| Frames imported and exported in place | decode | 1.771 → 1.762 s/token, ×1.005 (p 9.8e-6) |
| The next position's patch ahead of its input | decode | 1.762 → 1.760 s/token, ×1.001 (p 0.003) |
| SparseB refreshes on the degree-14 EvalMod series (K = 16) | decode | 1.762 → 1.733 s/token, ×1.017 (p 6e-5) |
| Base conversion's pre factors in the INTT's last pass | decode | 1.733 → 1.699 s/token, ×1.020 (p 1.1e-5) |
| Double-word sums reduced in 32-bit products | decode | 1.699 → 1.664 s/token, ×1.021 (p 3.3e-6) |
| L2 prefetch of the next key reads | decode | 1.664 → 1.650 s/token, ×1.009 (p 2.7e-4) |
| The NTT's row passes at five blocks per SM | decode | 1.650 → 1.646 s/token, ×1.003 (p 0.011) |
| CtS stage 0 as one giant step, StC giant steps folded mod 2, EvalMod's rescales next to their relinearizations and its series without two alignment rescales | decode | 1.647 → 1.613 s/token, ×1.021 (p 3.6e-6) |
| The NTT, five ways | decode | 1.613 → 1.565 s/token, ×1.031 (p 2.3e-6) |
| CtS for the StC stages at seven blocks per SM, the dense four-digit stages' resident tails | decode | 1.565 → 1.538 s/token, ×1.017 (p 7.1e-6) |
| Base conversion over a wave, chains of plaintext products as one pass | decode | 1.538 → 1.530 s/token, ×1.006 (p 9.7e-4) |
| The four groups above together | decode | 1.647 → 1.531 s/token, ×1.076 (p 4e-12) |
| The four groups above together | prefill | 9.844 → 9.187 s, ×1.071 (p 1.9e-13) |
| **The first part** vs record 4 (its Dev result) | decode | **1.795 → 1.530 s/token, ×1.173 (p 5.1e-15)** |
| **The first part** vs record 4 (its Dev result) | prefill | **10.13 → 9.185 s, ×1.103 (p 3.9e-14)** |
| EvalMod's fusions, lazy CtS products, the third kernel round, rotation sums in the K = 1 dot, the replay's copies, the decode linears, a shorter CutMax | decode | 1.530 → 1.339 s/token, ×1.143 (p 1.5e-14) |
| The same commits | prefill | 9.185 → 8.574 s, ×1.071 (p 2.3e-14) |
| A rotation's Q rows on a second stream | decode | 1.338 → 1.325 s/token, ×1.010 (p 0.039) |
| The same | prefill | 8.574 → 8.560 s, ×1.002 (p 0.036) |
| Raise drops on deliberate refreshes, the five-stage CtS | decode | 1.325 → 1.293 s/token, ×1.024 (p 0.016) |
| The same | prefill | 8.560 → 8.066 s, ×1.061 (p 5.5e-4) |
| SparseRot keys, StC-first's input stages over P'', level relinearization keys | decode | 1.293 → 1.258 s/token, ×1.028 (p 0.016) |
| The same | prefill | 8.066 → 7.838 s, ×1.029 (p 0.0019) |
| Rotation sums in the K > 1 dot, the grids' ModDown epilogue, base conversions by job kind, the key-switch chain by dependent launch | decode | 1.258 → 1.231 s/token, ×1.022 (p 0.028) |
| The same | prefill | 7.838 → 7.075 s, ×1.108 (p 6.2e-4) |
| LayerNorm statistics from one refresh or one complex ladder, 16-lane attention ladders (Decode-8 plan `_lh`) | decode | 1.232 → 1.215 s/token, ×1.014 (p 0.0077) |
| **This entry** vs record 4 (the Dev result) | decode | **1.795 → 1.217 s/token, ×1.475** (record 4 from the `efinal` job) |
| **This entry** vs record 4 (the Dev result) | prefill | **10.13 → 7.080 s, ×1.430 (p 2.3e-16)** |

- CutMax's order is not measurable here: the gap it closes (the logits' download and CutMax's launch) is a few
  milliseconds against the harness's noise. It stays because it removes an idle device gap at no cost and its outputs are
  byte-identical.
- The last group's two commits were not separated on the harness: base conversion over a wave takes about 3 ms off a
  decode token's kernel time and the product chains about 4 ms (nsys), each below the harness's resolution alone.
- Prefill-128 is not ablated per commit: the host-side commits leave its device work as record 4's (byte-identical
  outputs; about 0.3% from the in-place import, unpaired), its plan was already on K = 16, and the kernel and bootstrap
  commits are what it gains: the key switch's first four ×1.029 against record 4 (`fpre_*`, in the Dev result below),
  the rest ×1.071 (the four groups' prefill row).
- The second part's first row bundles groups that were each measured against their own base on another RTX PRO 6000
  of the same machine (harness, 2 to 4 runs per arm, not paired with each other): the third kernel round with the K = 1
  rotation sums ×1.057 on Decode-8 and ×1.019 on Prefill-128, the replay's copies ×1.015 and ×1.017, the shorter
  CutMax ×1.032 (Decode-8 only), the decode linears ×1.011; EvalMod's fusions and the lazy CtS products about 1.1% and
  0.8% of a decode token from pc-bench's per-refresh times.

## Dev result

The speedrun harness on one RTX PRO 6000 Blackwell Server Edition (96 GB; the server pinned to its NUMA node,
`NUMA=3:3,4`), 6 pairs in the order AB BA BA AB AB BA per job, dev prompts 0-1, two requests per run, scored by
`score.py`, one job per track: this entry (`e62e79e` with its client `perseus_keyset_client.py`, which pins `811eadc`:
`e62e79e` plus a pc-bench compile fix the served code does not build) against the first part (`bc06718`'s code with
record 4's lean client `24d4b3f`), logs `kfinal_*` and `kpre_*`; and, in an earlier pair of jobs on the same GPU, this
entry before the LayerNorm commits (`cc13296`, the same Prefill-128) against record 4 (`d97a6f7` with `24d4b3f`), logs
`efinal_*` and `epre_*`. GPU 5 (on GPU 4's PCIe switch) idle through all four jobs; another user's job kept GPU 6 busy
in most runs (`*_contention.tsv`).

| Track | Score | The first part, same job | Record 4, same GPU | KL | Argmax (scored) | Peak GPU memory |
|---|---|---|---|---|---|---|
| Decode-8 | 1.217 s/token | 1.532 s/token, ×1.259 (p 1.4e-12) | 1.795 s/token, ×1.475 (paired at `cc13296`: 1.234, ×1.455, p 7.2e-16) | 3.4e-6 (positions 8-15) | 96/96 | 62.5 GiB (record 4 49.0) |
| Prefill-128 | 7.084 s | 9.190 s, ×1.297 (p 1e-14) | 10.13 s, ×1.430 (paired at `cc13296`: 7.080, p 2.3e-16) | 8.4e-6 (position 127) | 12/12 | 56.6 GiB (record 4 59.3) |

score.py against the first part, Decode-8: quality PASS (mean KL 3.43e-6, p 2.2e-33), speed ×1.259 (p 1.4e-12), NEW
RECORD; the first part's arm took 1.530-1.533 s/token, against its own Dev result's 1.530; this entry's 1.216-1.218.
Prefill-128: quality PASS (mean KL 8.44e-6, p 4.3e-40), speed ×1.297 (p 1e-14), NEW RECORD; the first part's arm took
9.18-9.20 s per request, against its Dev result's 9.185; this entry's 7.08-7.09. Against record 4 at `cc13296`: Decode-8
quality PASS (mean KL 3.48e-6), speed ×1.455 (p 7.2e-16), record 4's arm 1.794-1.796 s/token against its published
1.798; Prefill-128 quality PASS (mean KL 8.96e-6), speed ×1.430 (p 2.3e-16), record 4's arm 10.12-10.13 s against its
published 10.12.

- All 32 dev prompts through this entry: Decode-8 (`dev0-15_final.json`, `dev16-31_final.json`, one session each, on
  another RTX PRO 6000 of the machine) mean KL 7.5e-6 (median 5.6e-6, max 2.2e-5), argmax 512/512 over all 16 positions;
  Prefill-128 (`dev0-31_prefill_final.json`, at `cc13296`, whose prefill this entry runs unchanged) mean KL 1.2e-5
  (median 7.4e-6, max 1.1e-4), argmax 32/32; every request within its bar. Before the LayerNorm commits, Decode-8's 32
  prompts gave mean KL 3.0e-6 (`dev0-31_cc13296.json`); KL moves 2-4× from one key draw to the next, and the LayerNorm
  A/B's own runs gave 2.4e-6 against 5.4e-6.
- The first part, same protocol (its Dev result, logs `dfinal_*` and `dpre_*`): Decode-8 1.530 s/token against record
  4's 1.795 (×1.173, p 5.1e-15), mean KL 2.96e-6, argmax 96/96, peak 54.5 GiB; Prefill-128 9.185 s against 10.13
  (×1.103, p 3.9e-14), mean KL 7.8e-6, argmax 12/12, peak 52.5 GiB. All 32 dev prompts: Decode-8 mean KL 3.2e-6
  (`dev0-31_entry.json`), Prefill-128 1.2e-5 (`dev0-31_prefill_entry.json`).
- Earlier states, same protocol: the commits up to the row passes' (the key switch's four) ×1.090 on Decode-8 (1.793 →
  1.645 s/token, p 2.4e-12, `ffinal_*`) and ×1.029 on Prefill-128 (10.13 → 9.842 s, p 2.3e-10, `fpre_*`), with another
  user's jobs keeping GPU 6 busy in Decode-8's runs 1-10 and GPU 5 (on GPU 4's PCIe switch) in its runs 8-12, and GPU 6
  through Prefill-128's job; the commits up to K = 16 ×1.036 (1.796 → 1.734 s/token, p 3.7e-9, `ab6_*`, GPUs 5-7
  idle); with another user's jobs keeping GPU 5 (on GPU 4's PCIe switch) 97-99 % busy:
  the host-side changes alone ×1.018 (1.793 → 1.760 s/token, p 2.1e-8, `ab2_*`), and an earlier state of them (the
  cache, CutMax's order and the in-place import) ×1.016 (1.795 → 1.767, p 2e-7, `ab1_*`).

## Disclosures

- **Outputs.** The host-side changes (every commit before K = 16) give outputs byte-identical to record 4's on one key
  bundle and one set of input ciphertexts (the lean client's, for dev prompts 0-1), both requests and all 16 positions:
  32 of 32 output frames equal, with record 4's own replay equal to itself (`dev_logs/bit_identity.txt`). The cached
  encodings are the encoder's own output for the same values, copied. The kernel commits give outputs byte-identical to
  K = 16's, same protocol (32 of 32): each is exact modular arithmetic in another order (the pre factor multiplied in
  with `N^-1`, the same residue reduced in two halves), a prefetch hint or a launch bound. Prefill-128 through those
  commits, and through the commits up to K = 16, gives output frames byte-identical to record 4's, same protocol (2 of
  2). After the bootstrap commits, the NTT, CtS, base conversion and product-chain commits give outputs byte-identical
  to the state before them, same protocol (Decode-8 32 of 32 at each group, Prefill-128 2 of 2 across the four), and CtS
  stage 0's fold gives the same residues (pc-bench's refresh outputs bit-identical). In the second part, the exact
  commits give outputs byte-identical to the commits before them where they join this entry, same protocol (Decode-8 32
  of 32, Prefill-128 2 of 2 each, `dev_logs/bit_identity.txt`): EvalMod's fusions and the lazy CtS products against the
  first part; the third kernel round, the K = 1 rotation sums and the replay's copies against the decode linears'
  commit; the forked Q rows against CutMax's commits; and the K > 1 rotation sums, the grids' epilogue, base conversions
  by job kind and the dependent-launch chain against the key set. Each is exact modular arithmetic in another order, on
  disjoint rows, or in place of a copy. The LayerNorm commits change Decode-8 only: Prefill-128's outputs stay
  byte-identical across them (2 of 2).
- **Fewer roundings in the bootstraps.** Three bootstrap commits change what the device computes, not what it means:
  the StC fold mod 2 drops two key switches whose rotations are the identity on the slots (two roundings fewer), the
  EvalMod scheduling fuses six or seven more rescales per refresh into their relinearizations' ModDown (one rounding
  instead of two), and the lean series program brings T_1 to level by a scalar and a limb drop instead of a rescale.
  Each refresh's output level and scale are unchanged, and so are the plans. Refresh precision, pc-bench's grid in the
  served configuration (SparseB at K = 16, fused EvalMod; 6 seeds, packed keys): rms bits within ±0.02 at every point (StC-first
  and SparseB-512, CF 5, 8, 10 and 13, amplitudes 0.5, 1 and 4; mean bits within ±0.13), output levels unchanged
  (`dev_logs/precision_grid_*.log`; the later commits are byte-identical, so the grid compares the state before the
  bootstrap commits with them). KL in the
  Dev result.
- **Fewer roundings in the decode linears.** Summing a giant step's rotations before one ModDown, adding inside a key
  switch's ModDown and fusing a relinearization with its rescale each round once where the first part rounded two or
  more times; a shared rescale rounds the same value once for its consumers. Same levels, scales and plans.
- **CutMax.** The same function in exact arithmetic with two refreshes fewer: a SPRU refresh's output is its input's
  slot mean (so the rotation sums before it go), SPRU is linear (so the first Newton step's scale moves through it), and
  ln_f's affine map folds into the LM head's (so the tail exits a level earlier). The fold is record 1's, which every
  block's ln_1 and ln_2 already use (`fold_lnf`, switched on by the approximation config's `lnf_fold`): at setup, from
  the weights at `$WEIGHTS_PATH`, the LM head's encoded tiles take W diag(γ) and ln_f's β becomes the input shift β/γ,
  so the encrypted tail adds a plaintext and skips the product by γ and its rescale; the weights file is unchanged,
  nothing depends on the prompt, and the fed-back token's embedding keeps the unfolded matrix. The plan's stages 12 and
  13 are replanned on the planner's own search (`_cmfl`, blocks 0-11 as record 4's `_cm` with the raise drops below),
  with record 4's capture statistics for the tail (`merge_lnf_tail_mags.py`). `test_cutmax_switches_cpu.py` checks each
  switch against the record's form on the numpy fake. KL in the Dev result.
- **LayerNorms.** `key_lanes` is exact (lanes 16-31 of the head sums are zero, as the padded heads' already are). `spru`
  and `zladder` compute the same statistics in another order: the variance as the mean square less the squared mean
  (record 4 and the first part run a mean ladder and then a variance ladder), from a refresh the plan already makes or
  from one complex ladder. The inverse square root's polynomials and the refresh points are the first part's;
  `test_ln_spru_cpu.py` checks all three forms against the ladder form (1e-8 on the fake). KL in the Dev result.
- **Raise drops and the five-stage CtS.** A dropped raise lands its refresh 2d levels lower, where the consumers' levels
  allow it (the planner's probe on the final program); the five-stage CtS lands where the drop does. Refresh precision,
  pc-bench's grid (StC-first): at most 0.15 rms bits less from a drop (d = 5), 0.3-0.4 bits less from the five-stage
  CtS. KL in the Dev result.
- **SparseB at K = 16.** The EvalMod series of the SparseB refreshes changes, as on Prefill-128 since record 4: the dense
  route's degree-14 table over (-16, 16) instead of the degree-20 one over (-24, 24) (the integer after the fold has the
  dense route's distribution). Same levels, landings and plan. Refresh precision, pc-bench's grid on the lean chain (4
  seeds, packed keys), at each amplitude's best CF: 17.98 → 17.77 bits (A 0.5, CF 5), 17.32 → 16.08 (A 4, CF 10),
  17.88 → 17.61 (A 16, CF 10); at most 1.2 bits less at the CFs the plan uses. The plan's accuracy tables are record
  4's (measured at K = 24). KL in the Dev result.
- **New untimed work.** At setup, the 1,550 distinct position plaintexts are encoded into the cache (2.7 s), and
  position 0's plaintexts are staged. Both depend on the position only: the values are the ones record 4's `precompute`
  already records at setup from the CPU twin on public placeholder rows. Nothing depends on the prompt.
- **Unscored positions.** No work moves into positions 0-7. Position 0 of a session's first request gets faster
  (1.87-1.93 s against record 4's 3.78-3.91 in the `efinal` A/B), because staging every plaintext of position 0
  moves from its clock to setup; a later request's position 0 is a steady token in both.
- **Memory.** The cache takes 7.4 GB of device memory in its own allocation (sized to the values it holds), the
  giant-step folds store 3.4 GiB fewer plaintexts, and the dense four-digit stages' resident tails take 1.6 GiB: peak
  GPU memory 54.5 GiB against record 4's 49.0. Host memory is unchanged (server peak 20.8-21.2 GB in both arms). The
  kernel commits' scale tables (two words per row of each INTT they fold into) leave the peak where K = 16 had it.
  Prefill-128's peak is 52.5 GiB against record 4's 59.3. With the second part, the peaks are 62.5 GiB (Decode-8)
  and 56.6 GiB (Prefill-128): the new keys, the K-batches' longer slab and the raise variants the drops build.
- **The new keys, at 128 bits each.** Every switching key here is pc-client's hybrid key: for digit d and each prime t
  it stores, `b = -a s_new + e + [t in digit d] G_t s_old (mod t)`, `a` from a seed, `e` discrete Gaussian (σ 3.19); so
  its samples are RLWE samples under `s_new` at the modulus of the primes it stores, clean outside digit d. The three
  new roles all have `s_new` the dense uniform ternary main secret (or its image under an automorphism, a signed
  permutation of its coefficients) and a modulus inside Q × P, at most the main key's. Decode-8's request (Prefill-128's
  has the same roles, 217 keys, and record 4's PCMM ring-switch keys, unchanged):

  | Role | Keys | Shape (q_top, digits, special primes) | Samples under | Switches from | log2 modulus |
  |---|---|---|---|---|---|
  | rot | 114 | 50, 4, 13 | σ_g⁻¹(s), dense | s | 1733.26 |
  | rot over P'' (new) | 48 | 4, 1, 6 | σ_g⁻¹(s), dense | s | 304.03 |
  | sparse_rot (new) | 18 | 50, 4, 13 | σ_g⁻¹(s), dense | s_sp | 1733.26 |
  | relin_level (new) | 2 | 27, 2, 13 and 41, 3, 13 | s, dense | s² | 1114.67 and 1490.61 |
  | relin, conj, auto (SPRU), enc_sparse_to_dense | 1 each | 50, 4, 13 | s or σ(s), dense | s², s or s_sp | 1733.26 |
  | enc_dense_to_sparse | 1 | 1, 1, 9 | s_sp, ternary h = 32 | s | 307.64 |
  | spru_atob | 1 | 1, 1, 9 | the SPRU block key, binary h = 64 | s | 307.64 |

  The dense rows are at most log QP = 1733.26 ≤ 1747, the HE standard's 128-bit bound at N = 2^16 for a ternary secret;
  a smaller modulus with the same secret and error is a harder RLWE instance. The two sparse keys are record 4's,
  unchanged, at record 1's Lattice Estimator verdict. The client refuses any role it does not declare, any key under a
  sparse secret above the bottom shape or over more than 9 special primes, and any dense key whose stored primes
  multiply past 1734 bits, from the chain's own prime sizes; `crates/pc-client/tests/key_secrets.rs` recovers each new
  role's error from generated keys, so its samples are under the secret the client's table names.
- **Keys, parameters, plans, approximations, client.** Parameters are record 4's (n32a4, the same primes, dnum 4, the
  same main secret). Keys are record 4's plus the three roles above (124 → 188 switching keys for Decode-8, 153 → 217
  for Prefill-128; device key memory 6.78 → 7.73 GB and 8.40 → 9.36 GB). Decode-8 serves `_lh` (above; `KV_PUSH_RAISE`
  in blocks 4 and 5), Prefill-128 record 4's `_cm` with its PCMM finishes raised; their runtime lines gain
  `FIDESLIB_BTS_SPARSE_B_K=16` (Decode-8), `PC_BTS_MERGE_S2D=1 PC_BTS_STC_PS=6 PC_EM_LEVEL_KEYS=1`,
  `FIDESLIB_CTS_ALT_BUDGET=5` and the drop switches. CutMax's approximation changes as above; the others are record 4's.
  The client is `perseus_keyset_client.py`, pinned to `811eadc` (`e62e79e` plus a pc-bench compile fix).
- **Planned generation.** With `lnf_fold` on, `generate`'s encrypted feedback built the next token's embedding from the
  γ-scaled head; `c371e9f` builds it from the unfolded one. The served decode takes every position's input from the
  client and does not run the feedback, so this changes no timed output.
- **Scope.** The cache is on for the served 16-position decode only (`pt_cache`, off by default in `PlannedRun`): its
  size grows with the positions a run stages (about 470 MB per position here).
