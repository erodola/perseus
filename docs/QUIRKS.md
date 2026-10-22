# Shadow quirks

`pc-shadow` implements the C++ truth: FIDESlib's FLEXIBLEAUTO metadata logic (`src/CKKS/Ciphertext.cpp`,
`Plaintext.cu`) as the perseus wrapper (`include/fideslib_wrapper.h`) reaches it through FIDESlib's API layer
(`api/CryptoContext.cpp`). Where the old planner mirror (`perseus/plan/placer/sim.py`) or the numpy fakes
(`perseus/impl/fake.py`, `fakecap.py`) behave differently, the deviation is a named `Quirk`
(`crates/pc-shadow/src/quirks.rs`) that a configuration switches on. Presets: `Quirks::planner()` (what every shipped
plan was planned under), `Quirks::fake()`, `Quirks::fakecap()`.

Retiring a quirk = drop it from the preset, re-plan every shipped plan (`python python/tests/test_paper_plans.py`),
run the model gate, keep the rule with <= bootstraps/token at >= KL, delete the flag. The "depends" column is measured by
`cargo test -p pc-shadow --test planner_equiv quirk_dependence -- --nocapture`: node differences against the old
`sim.py` when the quirk alone is dropped (a block the rules refuse counts once), as `planned run / plan-free run`, per shipped plan (gpt2 decode / gpt2 prefill
/ llama prefill; 19016 / 30991 / 61754 nodes).

| quirk | who | what it does | C++ truth | from | shipped plans depend |
|---|---|---|---|---|---|
| `AddPtRealizes` | fakecap | ct(deg 2) + pt(deg 1) lands at (effective level, deg 1) whatever the plaintext's level | `addPt` rescales only when the plaintext sits exactly one level deeper; otherwise it lifts the plaintext and the sum stays degree 2 | fakecap `_add_pt` | — (planner: retired, below) |
| `PtLevelIgnored` | fake | products and sums read cipher inputs only | `multPt` asserts on a plaintext deeper than its ciphertext | fake.py | — (planner: retired, below) |
| `FakeProductRescaled` | fake | products land rescaled: (level + unit, deg 1) | (effective level, deg 2), rescale pending | fake.py `_product` | — (36,899 of 102,541 captured nodes differ under `Quirks::fake()`) |
| `FakeAddMaxDeg` | fake | ct + ct = (max level, max degree) | the FLEXIBLEAUTO mixed-degree adjust | fake.py `add`/`sub`/`inplace_add` | — |
| `FakeNegateFree` | fake | negate keeps level and degree | GPU `EvalNegate` is `multScalar(-1.0)`: (effective level, deg 2) (the CPU fallback of the same API keeps the level: OpenFHE `EvalNegateInPlace`) | fake.py `negate` | — |
| `FakeAddPtFree` | fake | ct + pt keeps level and degree | as `AddPtRealizes` | FakeInf `add_pt`/`add_slot_pt`/`add_cached` | — |
| `FakeBootstrapRealized` | fake | every refresh lands at `bts_out`, degree 1, any route | per-route landing | fake.py `bootstrap` | — |
| `CapStcEntryNominal` | fakecap | StC-first entry compares the nominal input level with `FAKECAP_STC_ENTRY` | `Bootstrap.cu` bootstrapImpl / btsBegin: effective limb (`top - d` when deg 2) `>= stc_first_entry` | fakecap `inner_bootstrap` | — |
| `CapMultPtMax` | fakecap | ct * pt lands at max(effective level, plaintext level) | refused (deeper plaintext) | fakecap `_c_mult` | — |

On the captured trajectories (102,541 nodes of the three graphs whose inputs are produced inside the block,
`tests/capture_rules.rs`) the C++ rules, the planner set and the fakecap set all reproduce every captured
(level, degree); the fake set differs on 36,899. The quirks only bite off the captured trajectory, i.e. on planned
runs, where levels move.

Not a quirk (verified equal): the mixed-degree add of sim.py (`d2eff > d1max`) is exactly FIDESlib's
`adjustForAddOrSub` on the composite grid; products `max(eff) , 2` are `adjustForMult` + `mult`; the StC-first
fallback (`PLAN_STC_FIRST_ENTRY` effective) is FIDESlib's own entry test; `fold_bootstrap`'s planner landing
(captured, else bootstrap level + unit, deg 2) is the dense landing followed by the recovery product.

FIDESlib limitation found while porting (refused by the rules, not a quirk): `Plaintext::adjustPlaintextToCiphertext`
with a degree-2 plaintext richer than its ciphertext calls `multScalar` (which rescales first) and then `rescale`
(which asserts degree >= 2): that branch cannot execute.

## Retired (the planner now runs the C++ rule)
Dropped together, the five planner quirks below leave every regenerable shipped plan identical (11 plans: gpt2 decode
+ ILP, gpt2 prefill pcmmstcf / prefill128, llama decode / dsw48, llama prefill pcmmstcf / pr2 / pr9, bert
sst2_128, vit b16_224; all nine bound fields and the runtime bodies; gpt2_prefill128 differs only in the capture
stamp, as before). Same plans = same bootstraps and KL. Against the dumped `sim.py` the planned runs still agree on
every node; plan-free runs differ where a fired hint has no landing spec (480 / 12 / 32 nodes, gpt2 decode / gpt2
prefill / llama prefill: `tests/planner_equiv.rs` `BARE_HINT_LANDINGS`).

| quirk | was | now |
|---|---|---|
| `LevelReduceCaptured` | `level_reduce` landed at the captured (level, degree) | `drop_to_level`: `max(cur, t)`, degree kept |
| `HintAccountsPending` | a hint threshold read the effective level | nominal level unless the caller accounts the pending rescale |
| `HintLandingDeg2` | a fired hint without a landing spec landed at (bootstrap level, deg 2) | the route's realized landing |
| `DeliberateLandingDeg2` | a deliberate refresh landed with degree 2 | its captured degree (`plan.delib_deg=captured` is now the only behaviour) |
| `FreshAtBootstrapLevel` | a node with no cipher input started at (bootstrap level, deg 1) | refused (no shipped graph has one) |

The planner's last two (`Quirks::planner()` is now empty): the compile encodes every `pt_` / `mask.` plaintext like the
ciphertext it meets (the Python impl's `encode_like`: the ciphertext's effective level, degree 1), and under that level
the C++ rules give exactly what the quirks gave (`addPt` with the plaintext one level below a degree-2 ciphertext
realizes it; `multPt` with the plaintext at the ciphertext's level needs no adjust). The planner now reads the
plaintext at that level (`Shadow::pt_input`) instead of the captured one. Every regenerable shipped plan replans
identical; the planned runs of `tests/planner_equiv.rs` still agree with the dumped `sim.py` on every node.

| quirk | was | now |
|---|---|---|
| `AddPtRealizes` (planner) | ct(deg 2) + pt landed at (effective level, deg 1) whatever the plaintext's level | C++ `addPt` with the plaintext at the compile's level (same landing) |
| `PtLevelIgnored` (planner) | products and sums ignored the plaintext (captured levels are deeper than a planned ciphertext and the C++ rule refused) | the plaintext at the compile's level: never deeper than its ciphertext |
