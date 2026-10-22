# Env inventory of the ported Python (M4p -> M4b input; M4q additions in section 5, M7f in section 6)

Every environment read in `python/perseus/**` (the port of the old repo's `perseus/impl`, `perseus/plan`,
`examples/{gpt2,llama}_from_primitives`, `_env/_backend/profile/session/errors`, and the planner shell drivers
`plan/run_bootstrap_all_blocks.sh`, `examples/*/make_plan.sh`), with where it is read, its default, its type, which
shipped plans carry it, and the proposed typed-`Config` field (groups:
chain / bts / mem / host / model / quirks / exec; planner knobs go to the planner's `PlanConfig`, marked).

## Counts
| set | names |
|---|---|
| read by ported Python (`os.environ.get/[]/setdefault`, `os.getenv`, `"X" in os.environ`, local wrappers `_on()`/`_env_or()`, the contract's `LOAD_BEARING_ENV` tuple) | 180 |
| read by the ported shell drivers only (`${X...}`) | 66 |
| **total read in ported code (table 1)** | **246** (466 read sites) |
| set by ported code (`env.ENV`, `RUNTIME_*`, `SessionProfile` presets) or named by shipped plans, read only by the old C++/FIDESlib (table 2) | 23 |
| **total for M4b** | **269** |
| M4q (pers/dev 1894546): new reads in the ported code (section 5a) | 23 |
| M4q: carried by the four new plans, read only by the old C++ / FIDESlib (section 5b) | 10 |
| **total after M4q** | **302** |
| M7f (pers/dev 31ab85f): new reads in the ported code (section 6a) | 16 |
| M7f: carried by the result plans / new FIDESlib defaults, read only by the old C++ (section 6b) | 14 |
| **total after M7f** | **332** |

The old `ENV_VARS.md` (~165) counted C++ + Python reads of the published tree; this inventory is the ported Python
and its shell drivers, plus the C++-only names the port sets or the shipped plans stamp. C++/FIDESlib-only reads the
port never sets are out of scope (M5/M6 cards own them).

## Method
- AST scan of every `.py` (parent-chain type inference: `int(...)`/`float(...)` -> int/float, `.split(",")` -> list,
  `not in ("", "0")`/`== "1"`/truthiness -> flag, comparison with other literals -> enum, else str; `contract` rows
  are compared verbatim against the plan stamp), regex scan of the shell drivers (`${X:-d}` default, `${X:?}`
  required, `${X+x}`/`$X` unset). Shell-local variables (`B0`, `TAIL`, `TMP`, `RECIPE`, ...) excluded.
- Shipped plans = the 39 tracked plan dirs of the old repo (`git ls-files bootstrap_placements`); "capture_env N" = N
  plans stamp the key with a non-null value in `placement.json` `capture_env.env`; "runtime=/recipe= N" = N
  `PLAN_CMD.txt` lines name it. Section 3 lists the plans where not all 39.
- Field names keep the env name 1:1 (lower-cased) for the M4 gate; only the meaningless `FIDESLIB_`/`FIDESLIB_BTS_`
  library prefixes are dropped and `PF_*` (prefill) go under `model.prefill`. `PLAN_*` that the shell maps onto a
  planner CLI flag use the flag's name. Debug/trace knobs go to `exec.debug`.
- Defaults shown as the source spells them (`None` = unset reads as absent; `(required)` = `os.environ[...]` /
  `${X:?}`; several defaults = different sites disagree, M4b must pick per site).
- Writes the port does (not reads) are in table 2's "set at" column when no ported code reads the name; dynamic
  writers (`_env.scoped_env`, the capture_mags `os.environ[k] = v` loops over `env.ENV`) are not per-name.

## 1. Env reads in the ported code (246 names, 466 read sites)

| # | name | read at (file::function) | default | type | shipped plans | proposed Config field |
|---|---|---|---|---|---|---|
| 1 | `ALL_BLOCKS_IO_DIR` | examples/gpt2/oracle_io.py::RunConfig.from_env [read via _env_or()]<br>examples/gpt2/run_decode.py::main<br>examples/gpt2/run_generate.py::main | `None` | str | — | `exec.all_blocks_io_dir` |
| 2 | `AUTO_BTS_LEVEL` | impl/fakecap.py::make_session<br>plan/contract.py::current_contract/contract_mismatches [contract] | `None` | int/str (compared verbatim) | capture_env all 39; runtime= 6 | `bts.auto_bts_level` |
| 3 | `BLOCKS_ENTRY_DEG` | examples/gpt2/make_plan.sh::(shell) | `1` | str | — | `plan.blocks_entry_deg (PlanConfig)` |
| 4 | `BLOCKS_ENTRY_LEVEL` | examples/gpt2/make_plan.sh::(shell) | `(unset)` | str | — | `plan.blocks_entry_level (PlanConfig)` |
| 5 | `BTS_ITERATIONS` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `bts.bts_iterations` |
| 6 | `BTS_LEVEL` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `16` / `36` | str | recipe= 33 | `plan.bootstrap_level (PlanConfig, CLI `--bootstrap-level`)` |
| 7 | `BTS_MAX_INPUT_LEVEL` | impl/prefill.py::LockstepOps._ceiling | `None` | int | runtime= 14 | `bts.bts_max_input_level` |
| 8 | `BTS_OUT_DEG` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 2 | `plan.bts_out_deg (PlanConfig, CLI `--bts-out-deg`)` |
| 9 | `BTS_PRECISION` | impl/driver.py::ImplModel.__init__ | `'12'` | int | — | `bts.bts_precision` |
| 10 | `CACHE_READ_LEVEL` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `17` / `36` | str | recipe= 33 | `plan.cache_read_level (PlanConfig, CLI `--cache-read-level`)` |
| 11 | `CACHE_READ_LEVEL_K` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `model.cache_read_level_k` |
| 12 | `CACHE_READ_LEVEL_V` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `model.cache_read_level_v` |
| 13 | `CHAIN` | _env.py::current_chain<br>examples/gpt2/env.py::export_env<br>examples/gpt2/env.py::open_session<br>examples/gpt2/run_decode.py::main<br>examples/gpt2/run_generate.py::main<br>plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim)/str | capture_env all 39 | `chain.chain` |
| 14 | `CHEB_ALIGN` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'0'` / `None` | flag/str (compared verbatim) | — | `model.cheb_align` |
| 15 | `CHEB_BSGS` | impl/ops.py::FheOps.__init__ | `'1'` | flag | runtime= 14 | `model.cheb_bsgs` |
| 16 | `CKKS_COMPLEX` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `chain.ckks_complex` |
| 17 | `COMPOSITE_DEGREE` | impl/driver.py::ImplModel.__init__ | `'2'` | int | — | `chain.composite_degree` |
| 18 | `CONFIGS_PATH` | examples/gpt2/capture_mags.py::main<br>examples/gpt2/run_decode.py::main<br>examples/gpt2/run_generate.py::main<br>examples/gpt2/run_prefill.py::main | `None` | str | — | `exec.configs_path` |
| 19 | `CORRECTION_FACTOR` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `bts.correction_factor` |
| 20 | `CUDA_PROFILE_POS` | examples/llama/run_decode.py::main._cuda_profile_range | `'0'` | int | — | `exec.debug.cuda_profile_pos` |
| 21 | `CUDA_VISIBLE_DEVICES` | profile.py::_device_free_gb | `'0'` | enum | — | `(stays env: GPU guard)` |
| 22 | `CUTMAX_BTS_ITERS` | examples/gpt2/head.py::cutmax_iters | `None` | int | — | `model.cutmax_bts_iters` |
| 23 | `CUTMAX_PAIR` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 13; runtime= 13 | `model.cutmax_pair` |
| 24 | `CUTMAX_PRECISE_SCOPED` | examples/gpt2/head.py::cutmax_iters | `'1'` | flag | — | `model.cutmax_precise_scoped` |
| 25 | `CUTMAX_VEC_BTS_ITERS` | examples/gpt2/head.py::cutmax_iters | `None` | int | — | `model.cutmax_vec_bts_iters` |
| 26 | `FAKECAP_BATCH` | impl/fakecap.py::CapFhe.__init__ | `'1'` | int | — | `quirks.fakecap_batch` |
| 27 | `FAKECAP_CHECK_FIELD` | impl/fakecap.py::<module> | `None` | bool | — | `exec.debug.fakecap_check_field` |
| 28 | `FAKECAP_LANDINGS` | impl/fakecap.py::_landings | `'0:32,1:14,256:22,512:22'` | str | — | `quirks.fakecap_landings` |
| 29 | `FAKECAP_SPARSE_EXACT` | impl/fakecap.py::CapFhe.inner_bootstrap | `None` | flag | — | `quirks.fakecap_sparse_exact` |
| 30 | `FAKECAP_STC_DROP` | impl/fakecap.py::CapFhe.__init__ | `'6'` | int | — | `quirks.fakecap_stc_drop` |
| 31 | `FAKECAP_STC_ENTRY` | impl/fakecap.py::CapFhe.__init__ | `'46'` | int | — | `quirks.fakecap_stc_entry` |
| 32 | `FEEDBACK_ENTRY_DEG` | examples/gpt2/make_plan.sh::(shell) | `$ENTRY_DEG` | str | — | `plan.feedback_entry_deg (PlanConfig)` |
| 33 | `FEEDBACK_ENTRY_LEVEL` | examples/gpt2/make_plan.sh::(shell) | `$ENTRY_LEVEL` | str | — | `plan.feedback_entry_level (PlanConfig)` |
| 34 | `FHE_GRAPH_CAPTURE_TOKEN` | examples/llama/run_decode.py::main [setdefault]<br>examples/llama/run_prefill.py::main [setdefault] | `'1'` | str | — | `exec.fhe_graph_capture_token` |
| 35 | `FHE_GRAPH_DIR` | impl/driver.py::ImplModel.enter_stage<br>impl/fakecap.py::CapCore.begin_subgraph_capture<br>impl/fakecap.py::CapCore.end_subgraph_capture | `(required)` / `None` | flag/str | — | `exec.fhe_graph_dir` |
| 36 | `FHE_LMHEAD_CAP` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `model.fhe_lmhead_cap` |
| 37 | `FHE_PT_COEFF_ENCODE` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `mem.fhe_pt_coeff_encode` |
| 38 | `FIDESLIB_BTS_REAL` | impl/ops.py::FheOps.bootstrap_real | `'0'` | flag | runtime= 15 | `bts.real` |
| 39 | `FIDESLIB_BTS_STC_FIRST` | impl/fakecap.py::CapFhe.stc_first | `'0'` | int | runtime= 7 | `bts.stc_first` |
| 40 | `FIDESLIB_BTS_STC_FIRST_ROUTES` | impl/fakecap.py::CapFhe.stc_first | `'dense'` | list | runtime= 7 | `bts.stc_first_routes` |
| 41 | `FIDESLIB_BTS_STC_FIRST_STAGED_ONLY` | impl/bootstrap.py::<module><br>impl/fakecap.py::CapFhe.stc_first | `'0'` | flag | — | `bts.stc_first_staged_only` |
| 42 | `FIDESLIB_POOL_TRACE` | examples/llama/run_decode.py::main.on_token<br>examples/llama/run_prefill.py::main.on_block | `None` | flag | — | `exec.debug.fideslib_pool_trace` |
| 43 | `FIRST_ENTRY_DEG` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `$ENTRY_DEG` / `(unset)` | str | — | `plan.first_entry_deg (PlanConfig, CLI `--first-entry-deg`)` |
| 44 | `FIRST_ENTRY_LEVEL` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `$ENTRY_LEVEL` / `(unset)` | str | — | `plan.first_entry_level (PlanConfig, CLI `--first-entry-level`)` |
| 45 | `FORBID_STEPS` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 7 | `plan.forbid_steps (PlanConfig, CLI `--forbid-steps`)` |
| 46 | `FUSED_LN_VAR` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'0'` / `None` | flag/str (compared verbatim) | capture_env all 39 | `model.fused_ln_var` |
| 47 | `FUSED_SM_DEN` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'0'` / `None` | flag/str (compared verbatim) | capture_env all 39 | `model.fused_sm_den` |
| 48 | `GELU_FOLD` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 15; runtime= 15 | `model.gelu_fold` |
| 49 | `GEN_PROMPT` | examples/gpt2/run_generate.py::main | `'1'` | int | — | `exec.gen_prompt` |
| 50 | `GEN_TOKENS` | examples/gpt2/run_generate.py::main | `'4'` | int | — | `exec.gen_tokens` |
| 51 | `GPT2_CACHE` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `exec.gpt2_cache` |
| 52 | `GPT2_FOLD_LN1` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `model.gpt2_fold_ln1` |
| 53 | `GPT2_FOLD_LN2` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `model.gpt2_fold_ln2` |
| 54 | `GPT2_FOLD_LNF` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `model.gpt2_fold_lnf` |
| 55 | `GPT2_PACKING` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `model.gpt2_packing` |
| 56 | `GRAPH_DIR` | examples/gpt2/make_plan.sh::(shell)<br>examples/llama/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(required)` / `(unset)` | str | — | `plan.graph_dir (PlanConfig, CLI `--graph-dir`)` |
| 57 | `HF_HOME` | examples/gpt2/prefill.py::Gpt2Prefill.diag_weights<br>examples/gpt2/prefill.py::Gpt2Prefill.prefill_weights<br>examples/gpt2/slotcache.py::_cache_dir<br>examples/llama/run_decode.py::main<br>examples/llama/run_handoff.py::main<br>examples/llama/run_prefill.py::main<br>examples/llama/weights.py::LlamaWeights.__init__ | `'.'` / `None` / `os.path.expanduser('~/.cache/huggingface')` | str | — | `host.hf_home` |
| 58 | `IMPL_CHEB_PROG` | impl/poly.py::<module> | `'0'` | flag | — | `model.impl_cheb_prog` |
| 59 | `IMPL_CUTMAX_MARKS` | examples/gpt2/model.py::Gpt2Model.argmax_encrypted | `None` | flag | — | `exec.debug.impl_cutmax_marks` |
| 60 | `IMPL_DEAD_OPS` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 13; runtime= 13 | `model.impl_dead_ops` |
| 61 | `IMPL_KV_OFFLOAD` | impl/driver.py::ImplModel.__init__ | `'0'` | flag | — | `model.impl_kv_offload` |
| 62 | `IMPL_MASK_SUM` | impl/rt.py::<module> | `'0'` | flag | — | `model.impl_mask_sum` |
| 63 | `IMPL_PROFILE_SYNC` | impl/profile.py::<module> | `'0'` | flag | — | `exec.debug.impl_profile_sync` |
| 64 | `IMPL_PT_SUM_FUSED` | impl/ops.py::FheOps.__init__ | `'0'` | flag | — | `model.impl_pt_sum_fused` |
| 65 | `IMPL_REALIZE_SHARED` | impl/ops.py::FheOps.__init__ | `'0'` | flag | — | `model.impl_realize_shared` |
| 66 | `IMPL_SCOPED_NAMES` | impl/rt.py::Rt.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'0'` / `None` | flag/str (compared verbatim) | capture_env 14; runtime= 6 | `model.impl_scoped_names` |
| 67 | `IMPL_STC_FIRST_ARGMAX` | examples/gpt2/model.py::Gpt2Model.argmax_encrypted | `'1'` | flag | — | `model.impl_stc_first_argmax` |
| 68 | `IMPL_STC_FIRST_KV` | impl/attention.py::_kv_bts_scope | `'1'` | flag | runtime= 1 | `model.impl_stc_first_kv` |
| 69 | `KV_ARENA_GB` | examples/gpt2/run_prefill.py::main [setdefault]<br>examples/llama/run_handoff.py::main<br>examples/llama/run_prefill.py::main [setdefault] | `'0'` / `(required)` | str | — | `mem.kv_arena_gb` |
| 70 | `KV_LANES` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 14; runtime= 14 | `model.kv_lanes` |
| 71 | `LADDER_RADIX_BITS` | impl/layout.py::ladder_radix_bits<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'3'` / `None` | int/str (compared verbatim) | capture_env 13; runtime= 13 | `model.ladder_radix_bits` |
| 72 | `LIN_PT_DOT` | impl/ops.py::FheOps.__init__ | `'1'` | flag | — | `model.lin_pt_dot` |
| 73 | `LLAMA_GC_FREEZE` | examples/llama/run_decode.py::main | `'0'` | flag | — | `model.llama_gc_freeze` |
| 74 | `LLAMA_GQA` | examples/llama/env.py::rot_steps<br>examples/llama/model.py::LlamaModel.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'0'` / `None` | flag/str (compared verbatim) | — | `model.llama_gqa` |
| 75 | `LLAMA_GQA_MERGE` | impl/attention.py::<module> | `'1'` | flag | — | `model.llama_gqa_merge` |
| 76 | `LLAMA_KV_ARENA_GB` | examples/llama/run_decode.py::main<br>examples/llama/run_handoff.py::main | `'56'` / `'72'` | str | — | `mem.llama_kv_arena_gb` |
| 77 | `LLAMA_KV_LEVELS` | examples/llama/run_decode.py::main.on_token | `'0'` | flag | — | `model.llama_kv_levels` |
| 78 | `LLAMA_LINEAR_BOTTOM` | examples/llama/block.py::<module> | `'0'` | str | — | `model.llama_linear_bottom` |
| 79 | `LLAMA_MODEL` | examples/llama/hf.py::<module> | `'meta-llama/Llama-3.1-8B'` | str | — | `model.llama_model` |
| 80 | `LLAMA_PLAN_SPARSE_SLOTS` | examples/llama/make_plan.sh::(shell) | `256,1` | str | — | `plan.llama_plan_sparse_slots (PlanConfig)` |
| 81 | `LLAMA_PROFILE_STEPS` | examples/llama/run_decode.py::main.on_token | `None` | flag | — | `exec.debug.llama_profile_steps` |
| 82 | `LN_CHEB` | examples/gpt2/ref.py::norm_ref [read via _on()]<br>impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 15; runtime= 15 | `model.ln_cheb` |
| 83 | `LOGN` | plan/contract.py::current_contract/contract_mismatches [contract] | `None` | str (compared verbatim) | capture_env all 39 | `chain.logn` |
| 84 | `MAGTAPE_BATCH` | impl/magtape.py::Tape.__init__ | `'1'` | int | — | `model.magtape_batch` |
| 85 | `MAGTAPE_THREADS` | impl/magtape.py::node_stats_batch | `'8'` | int | — | `model.magtape_threads` |
| 86 | `MAX_LEVEL` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `24` / `50` | str | recipe= 33 | `plan.max_level (PlanConfig, CLI `--max-level`)` |
| 87 | `MULTI_T` | examples/gpt2/oracle_io.py::RunConfig.from_env [read via _env_or()]<br>examples/gpt2/run_decode.py::main | `'16'` / `None` | int | — | `exec.multi_t` |
| 88 | `OMP_NUM_THREADS` | examples/gpt2/env.py::export_env<br>examples/llama/calibrate.py::main<br>examples/llama/ref.py::main<br>examples/llama/weights.py::_build_coeff_file [setdefault] | `'1'` / `'16'` | int/str | — | `host.omp_num_threads` |
| 89 | `OUT_NAME` | examples/gpt2/make_plan.sh::(shell)<br>examples/llama/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(required)` / `(unset)` | str | — | `plan.out_dir (PlanConfig, CLI `--out-dir`)` |
| 90 | `PERSEUS_BTS_PROFILE` | impl/bootstrap.py::install | `'0'` | flag | — | `exec.debug.perseus_bts_profile` |
| 91 | `PERSEUS_CLIENT_EXTENSION` | _backend.py::client_extension | `None` | str | — | `(drop: no _client extension)` |
| 92 | `PERSEUS_LLAMA_CACHE` | examples/llama/weights.py::LlamaWeights.__init__ | `None` | str | — | `host.perseus_llama_cache` |
| 93 | `PERSEUS_SLOTVEC_CACHE` | examples/gpt2/slotcache.py::enabled | `'1'` | flag | — | `host.perseus_slotvec_cache` |
| 94 | `PERSEUS_SLOTVEC_CACHE_GB` | examples/gpt2/slotcache.py::default_rules | `'32'` | float | — | `host.perseus_slotvec_cache_gb` |
| 95 | `PERSEUS_SLOTVEC_CACHE_VARIANTS` | examples/gpt2/slotcache.py::default_rules | `'2'` | int | — | `host.perseus_slotvec_cache_variants` |
| 96 | `PERSEUS_STC_FIRST_ALLOW` | impl/bootstrap.py::<module> | `'1'` | flag | — | `model.perseus_stc_first_allow` |
| 97 | `PF_ACT_FOLD` | impl/activation.py::<module> | `'0'` | flag | runtime= 7 | `model.prefill.act_fold` |
| 98 | `PF_ATTN` | examples/llama/run_prefill.py::main<br>impl/ccmm.py::enabled | `''` | enum | runtime= 1 | `model.prefill.attn` |
| 99 | `PF_ATTN_PACK` | impl/prefill.py::<module> | `'0'` | int | runtime= 2 | `model.prefill.attn_pack` |
| 100 | `PF_ATTN_PAIR` | impl/prefill.py::<module> | `'1'` | flag | — | `model.prefill.attn_pair` |
| 101 | `PF_CCMM_CAP` | impl/ccmm.py::Ccmm.__init__ | `'0'` | int | — | `model.prefill.ccmm_cap` |
| 102 | `PF_CCMM_FAKE_LAND` | impl/ccmm.py::Ccmm.__init__ | `'28'` | int | — | `model.prefill.ccmm_fake_land` |
| 103 | `PF_CCMM_KMAX` | impl/ccmm_attn.py::<module> | `'24'` | float | — | `model.prefill.ccmm_kmax` |
| 104 | `PF_CCMM_LAND` | impl/ccmm.py::Ccmm.__init__ | `'0'` | int | — | `model.prefill.ccmm_land` |
| 105 | `PF_CCMM_MERGE` | impl/ccmm.py::<module><br>impl/sinc.py::rot_steps | `'0'` | flag | — | `model.prefill.ccmm_merge` |
| 106 | `PF_CCMM_QMAX` | impl/ccmm_attn.py::<module> | `'4'` | float | — | `model.prefill.ccmm_qmax` |
| 107 | `PF_CCMM_SHIFT_PV` | impl/ccmm_attn.py::<module> | `'3'` | int | — | `model.prefill.ccmm_shift_pv` |
| 108 | `PF_CCMM_SHIFT_QK` | impl/ccmm_attn.py::<module> | `'5'` | int | — | `model.prefill.ccmm_shift_qk` |
| 109 | `PF_CCMM_VERIFY` | impl/ccmm.py::Ccmm.apply | `None` | flag | — | `exec.debug.pf_ccmm_verify` |
| 110 | `PF_CCMM_VMAX` | impl/ccmm_attn.py::<module> | `'8'` | float | — | `model.prefill.ccmm_vmax` |
| 111 | `PF_CHEB_LEAF_NATIVE` | impl/poly.py::<module> | `'0'` | flag | — | `model.prefill.cheb_leaf_native` |
| 112 | `PF_DEBUG` | impl/prefill.py::<module> | `None` | bool | — | `exec.debug.pf_debug` |
| 113 | `PF_EVICT_ALL_BELOW_GB` | impl/prefill.py::<module> | `'4'` | float | — | `model.prefill.evict_all_below_gb` |
| 114 | `PF_EVICT_BELOW_GB` | impl/prefill.py::<module> | `'12'` | float | — | `model.prefill.evict_below_gb` |
| 115 | `PF_EVICT_MASKS` | impl/prefill.py::<module> | `'1'` | flag | — | `model.prefill.evict_masks` |
| 116 | `PF_EVICT_UNUSED` | impl/prefill.py::<module> | `'1'` | flag | — | `model.prefill.evict_unused` |
| 117 | `PF_GELU_JOINT` | examples/gpt2/prefill.py::<module><br>examples/llama/prefill.py::<module> | `'0'` | flag | runtime= 7 | `model.prefill.gelu_joint` |
| 118 | `PF_GLU_LEAN` | examples/llama/prefill.py::<module><br>impl/prefill.py::<module> | `'0'` | flag | — | `model.prefill.glu_lean` |
| 119 | `PF_GLU_XU` | examples/llama/prefill.py::<module> | `'0'` | flag | — | `model.prefill.glu_xu` |
| 120 | `PF_GQA` | examples/llama/prefill.py::LlamaPrefill.blocks | `''` | flag | runtime= 3 | `model.prefill.gqa` |
| 121 | `PF_JOINT_FORCE` | impl/prefill.py::<module> | `'0'` | flag | runtime= 7 | `model.prefill.joint_force` |
| 122 | `PF_KV_SHARE` | impl/prefill.py::<module> | `'0'` | flag | runtime= 2 | `model.prefill.kv_share` |
| 123 | `PF_LANE_HOIST` | impl/prefill.py::<module> | `'0'` | int | — | `model.prefill.lane_hoist` |
| 124 | `PF_LAYOUT` | examples/gpt2/prefill.py::_layout_blocked<br>examples/gpt2/run_prefill.py::main<br>examples/llama/prefill.py::_layout_blocked<br>examples/llama/run_prefill.py::main<br>examples/llama/run_prefill.py::main.on_block | `''` | enum | runtime= 5 | `model.prefill.layout` |
| 125 | `PF_LINEAR` | examples/gpt2/prefill.py::Gpt2Prefill.prefill_stages<br>examples/gpt2/prefill.py::Gpt2Prefill.preload_prefill<br>examples/gpt2/prefill.py::_use_pcmm<br>examples/gpt2/run_prefill.py::main<br>examples/llama/prefill.py::<module><br>examples/llama/run_prefill.py::main<br>impl/pcmm.py::enabled | `''` / `'pairs'` | enum | runtime= 7 | `model.prefill.linear` |
| 126 | `PF_LOCKSTEP` | examples/gpt2/prefill.py::<module> | `'0'` | flag | — | `model.prefill.lockstep` |
| 127 | `PF_LOCKSTEP_TOP` | impl/prefill.py::LockstepOps._ceiling | `None` | int | — | `model.prefill.lockstep_top` |
| 128 | `PF_MEMDBG` | examples/llama/prefill.py::_memdbg | `None` | flag | — | `exec.debug.pf_memdbg` |
| 129 | `PF_NCH` | examples/gpt2/prefill.py::Gpt2Prefill.blocks<br>examples/gpt2/run_prefill.py::main<br>examples/llama/prefill.py::LlamaPrefill.blocks<br>examples/llama/run_prefill.py::main | `'512'` | int | — | `model.prefill.nch` |
| 130 | `PF_NORM_SHARE` | examples/gpt2/prefill.py::<module><br>examples/llama/prefill.py::<module> | `'1'` | flag | — | `model.prefill.norm_share` |
| 131 | `PF_NVTX_STEPS` | impl/rt.py::<module> | `'0'` | flag | — | `model.prefill.nvtx_steps` |
| 132 | `PF_PCMM_BAND` | impl/pcmm.py::band_shift | `'0.25'` | float | — | `model.prefill.pcmm_band` |
| 133 | `PF_PCMM_CALIB` | examples/gpt2/prefill.py::<module><br>impl/pcmm.py::_record_calib<br>impl/pcmm.py::band_shift | `''` / `None` | flag/str | — | `model.prefill.pcmm_calib` |
| 134 | `PF_PCMM_CALIB_IN` | examples/gpt2/run_prefill.py::main<br>examples/llama/run_handoff.py::main [setdefault]<br>examples/llama/run_prefill.py::main<br>impl/pcmm.py::band_shift | `''` / `None` / `cal` | flag/str | — | `model.prefill.pcmm_calib_in` |
| 135 | `PF_PCMM_CAP` | impl/pcmm.py::Pcmm.__init__ | `'0'` | int | — | `model.prefill.pcmm_cap` |
| 136 | `PF_PCMM_CF` | impl/pcmm.py::Pcmm._device | `''` | str | runtime= 2 | `model.prefill.pcmm_cf` |
| 137 | `PF_PCMM_INTERLEAVE` | impl/pcmm.py::<module> | `'0'` | int | — | `model.prefill.pcmm_interleave` |
| 138 | `PF_PCMM_LAND` | impl/pcmm.py::Pcmm.__init__ | `'0'` | int | — | `model.prefill.pcmm_land` |
| 139 | `PF_PCMM_LLAMA_SCALES` | examples/llama/prefill.py::LlamaPrefill.pcmm_weights | `'0.25,1,0.25,0.5'` | list | — | `model.prefill.pcmm_llama_scales` |
| 140 | `PF_PCMM_MARGIN` | impl/pcmm.py::band_shift | `'1.5'` | float | — | `model.prefill.pcmm_margin` |
| 141 | `PF_PCMM_PHASES` | impl/pcmm.py::<module> | `'0'` | flag | — | `model.prefill.pcmm_phases` |
| 142 | `PF_PCMM_RAISE` | impl/pcmm.py::<module> | `''` | list | — | `model.prefill.pcmm_raise` |
| 143 | `PF_PCMM_SHIFT` | impl/pcmm.py::band_shift | `(required)` / `None` | int/flag | — | `model.prefill.pcmm_shift` |
| 144 | `PF_PCMM_STAGED_ONLY` | examples/gpt2/run_prefill.py::main<br>examples/llama/run_prefill.py::main | `'1'` | str | runtime= 2 | `model.prefill.pcmm_staged_only` |
| 145 | `PF_PCMM_VERIFY` | impl/pcmm.py::Pcmm.apply | `None` | flag | — | `exec.debug.pf_pcmm_verify` |
| 146 | `PF_PCMM_WBITS` | impl/pcmm.py::weight_bits | `'16'` | int | runtime= 5 | `model.prefill.pcmm_wbits` |
| 147 | `PF_POOL_TRIM` | impl/prefill.py::<module> | `'1'` | flag | — | `model.prefill.pool_trim` |
| 148 | `PF_PRE_PERIODIC` | impl/prefill.py::<module> | `'0'` | flag | — | `model.prefill.pre_periodic` |
| 149 | `PF_ROT_PRUNE` | examples/gpt2/run_prefill.py::main<br>examples/llama/run_prefill.py::main | `'1'` | flag | — | `model.prefill.rot_prune` |
| 150 | `PF_SILU_SINGLE` | impl/prefill.py::<module> | `'0'` | flag | — | `model.prefill.silu_single` |
| 151 | `PF_SILU_SINGLE_ABS` | impl/prefill.py::<module> | `'0'` | float | — | `model.prefill.silu_single_abs` |
| 152 | `PF_SILU_SINGLE_DEGS` | impl/prefill.py::<module> | `'63,95,127'` | list | — | `model.prefill.silu_single_degs` |
| 153 | `PF_SILU_SINGLE_EPS` | impl/prefill.py::<module> | `'1e-8'` | float | — | `model.prefill.silu_single_eps` |
| 154 | `PF_SILU_SINGLE_TOL` | impl/prefill.py::<module> | `'2'` | float | — | `model.prefill.silu_single_tol` |
| 155 | `PF_SINC_A2` | impl/sinc.py::<module> | `'0'` | flag | — | `model.prefill.sinc_a2` |
| 156 | `PF_SINC_BI_HOIST` | impl/sinc.py::<module> | `'0'` | flag | — | `model.prefill.sinc_bi_hoist` |
| 157 | `PF_SM_JOINT` | impl/prefill.py::<module> | `'0'` | flag | runtime= 1 | `model.prefill.sm_joint` |
| 158 | `PF_SM_PAIR` | impl/sinc_softmax.py::<module> | `'0'` | flag | — | `model.prefill.sm_pair` |
| 159 | `PF_SM_PAIR_ROUNDS` | impl/sinc_softmax.py::<module> | `''` | list | — | `model.prefill.sm_pair_rounds` |
| 160 | `PF_SM_R_ITER` | impl/sinc_softmax.py::<module> | `'0'` | flag | — | `model.prefill.sm_r_iter` |
| 161 | `PF_SM_R_ITER_P` | impl/sinc_softmax.py::<module> | `'8'` | int | — | `model.prefill.sm_r_iter_p` |
| 162 | `PF_SM_R_REFRESH` | impl/sinc_softmax.py::<module> | `'0'` | flag | — | `model.prefill.sm_r_refresh` |
| 163 | `PF_SM_R_SCALE` | impl/sinc_softmax.py::<module> | `'1'` | float | — | `model.prefill.sm_r_scale` |
| 164 | `PF_SM_UNPACK_FOLD` | impl/ccmm_attn.py::<module> | `'0'` | flag | — | `model.prefill.sm_unpack_fold` |
| 165 | `PF_STREAM` | examples/gpt2/prefill.py::Gpt2Prefill.prefill_weights | `'1'` | flag | — | `model.prefill.stream` |
| 166 | `PLAN_ACC_CHAIN` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `n32` | str | recipe= 33 | `plan.acc_chain (PlanConfig, CLI `--acc-chain`)` |
| 167 | `PLAN_ACC_TABLE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.acc_table (PlanConfig, CLI `--acc-table`)` |
| 168 | `PLAN_BASELINE_DEPTH_CAP` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 1 | `plan.baseline_depth_cap (PlanConfig, CLI `--baseline-depth-cap`)` |
| 169 | `PLAN_BASELINE_RESCUE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 2 | `plan.baseline_rescue (PlanConfig, CLI `--baseline-rescue`)` |
| 170 | `PLAN_BIND_HINTS` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.bind_hints (PlanConfig, CLI `--bind-hints`)` |
| 171 | `PLAN_BOUNDARY_REALIZE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.boundary_realize (PlanConfig, CLI `--boundary-realize`)` |
| 172 | `PLAN_BTS_MS` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.bts_ms (PlanConfig, CLI `--bts-ms`)` |
| 173 | `PLAN_CF_MAX` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `20` | str | recipe= 33 | `plan.cf_max (PlanConfig, CLI `--cf-max`)` |
| 174 | `PLAN_CF_MIN` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 6 | `plan.cf_min (PlanConfig, CLI `--cf-min`)` |
| 175 | `PLAN_DELIBERATE_CLAMP0` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `1` | str | — | `plan.deliberate_clamp0 (PlanConfig, CLI `--deliberate-clamp0`)` |
| 176 | `PLAN_DELIB_DEG` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `''` | enum | — | `plan.delib_deg (PlanConfig)` |
| 177 | `PLAN_DELIB_LANDING_REMAP` | plan/placer/ir.py::Graph.from_nodes | `''` | list | recipe= 14 | `plan.delib_landing_remap (PlanConfig)` |
| 178 | `PLAN_DEPTH_A` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.depth_a (PlanConfig, CLI `--depth-a`)` |
| 179 | `PLAN_DEPTH_B` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.depth_b (PlanConfig, CLI `--depth-b`)` |
| 180 | `PLAN_DEPTH_FORM` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.depth_form (PlanConfig, CLI `--depth-form`)` |
| 181 | `PLAN_DEPTH_WEIGHT` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.depth_weight (PlanConfig, CLI `--depth-weight`)` |
| 182 | `PLAN_DISSOLVE_HINTS` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.dissolve_hints (PlanConfig, CLI `--dissolve-hints`)` |
| 183 | `PLAN_DRIFT_RECUT` | plan/placer/place.py::Placer._run_once | `None` | flag | — | `plan.drift_recut (PlanConfig)` |
| 184 | `PLAN_EMIT_OFFSET` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.emit_offset (PlanConfig, CLI `--emit-offset`)` |
| 185 | `PLAN_ERR_HOPELESS` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.err_hopeless (PlanConfig, CLI `--err-hopeless`)` |
| 186 | `PLAN_ERR_TARGET` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.err_target (PlanConfig, CLI `--err-target`)` |
| 187 | `PLAN_HARD_ENV_CAP` | plan/placer/ilp.py::IlpPlacer._env_hard<br>plan/placer/place.py::Placer.run | `None` | str | — | `plan.hard_env_cap (PlanConfig)` |
| 188 | `PLAN_HINT_ENV_VETO` | examples/gpt2/make_plan.sh::(shell)<br>plan/placer/place.py::Placer._run_with_hint_veto | `1` / `None` | flag/str | recipe= 33 | `plan.hint_env_veto (PlanConfig)` |
| 189 | `PLAN_HINT_VETO_STEPS` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 12 | `plan.hint_veto_steps (PlanConfig, CLI `--hint-veto-steps`)` |
| 190 | `PLAN_ILP_FREE_EXIT` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.ilp_free_exit (PlanConfig, CLI `--ilp-free-exit`)` |
| 191 | `PLAN_ILP_GAP` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.ilp_gap (PlanConfig, CLI `--ilp-gap`)` |
| 192 | `PLAN_ILP_OBJECTIVE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.ilp_objective (PlanConfig, CLI `--ilp-objective`)` |
| 193 | `PLAN_ILP_TIME_LIMIT` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 2 | `plan.ilp_time_limit (PlanConfig, CLI `--ilp-time-limit`)` |
| 194 | `PLAN_KS_REALIZE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.ks_realize (PlanConfig, CLI `--ks-realize`)` |
| 195 | `PLAN_LATENCY_TABLE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.latency_table (PlanConfig, CLI `--latency-table`)` |
| 196 | `PLAN_LEVEL_UNIT` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `2` | str | recipe= 33 | `plan.level_unit (PlanConfig, CLI `--level-unit`)` |
| 197 | `PLAN_LEVEL_WEIGHT` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.level_weight (PlanConfig, CLI `--level-weight`)` |
| 198 | `PLAN_MAG_SAFETY` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `2` | str | recipe= 33 | `plan.mag_safety (PlanConfig, CLI `--mag-safety`)` |
| 199 | `PLAN_MISS_PENALTY` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.miss_penalty (PlanConfig, CLI `--miss-penalty`)` |
| 200 | `PLAN_MS_DISCOUNT` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.ms_discount (PlanConfig, CLI `--ms-discount`)` |
| 201 | `PLAN_NO_PRESCALE` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `1` | str | recipe= 33 | `plan.no_prescale (PlanConfig, CLI `--no-prescale`)` |
| 202 | `PLAN_PLACER` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `` / `(unset)` | str | recipe= 4 | `plan.placer (PlanConfig, CLI `--placer`)` |
| 203 | `PLAN_PREFER_SPARSE` | plan/btserr.py::choose_site | `'0'` | flag | — | `plan.prefer_sparse (PlanConfig)` |
| 204 | `PLAN_PRESCALE_BITS` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.prescale_bits_max (PlanConfig, CLI `--prescale-bits-max`)` |
| 205 | `PLAN_PRESCALE_REACH` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.prescale_reach (PlanConfig, CLI `--prescale-reach`)` |
| 206 | `PLAN_PRUNE` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `` / `1` | str | recipe= 37 | `plan.prune (PlanConfig, CLI `--prune`)` |
| 207 | `PLAN_QUALITY_WEIGHT` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.quality_weight (PlanConfig, CLI `--quality-weight`)` |
| 208 | `PLAN_RAISE_DROP_MAX` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 13 | `plan.raise_drop_max (PlanConfig, CLI `--raise-drop-max`)` |
| 209 | `PLAN_RAISE_DROP_SET` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `1,3,5` | str | recipe= 13 | `plan.raise_drop_set (PlanConfig, CLI `--raise-drop-set`)` |
| 210 | `PLAN_RAISE_ENV_RULE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 13 | `plan.raise_drop_env_rule (PlanConfig, CLI `--raise-drop-env-rule`)` |
| 211 | `PLAN_RAISE_LANDING_MAX` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 5 | `plan.raise_drop_landing_max (PlanConfig, CLI `--raise-drop-landing-max`)` |
| 212 | `PLAN_RAISE_ROUTES` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 12 | `plan.raise_drop_routes (PlanConfig, CLI `--raise-drop-routes`)` |
| 213 | `PLAN_REACTIVE_STRICT` | plan/placer/planner.py::plan_block | `'0'` | flag | — | `plan.reactive_strict (PlanConfig)` |
| 214 | `PLAN_REAL_ROUTE` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `0` / `1` | str | recipe= 13 | `plan.real_route (PlanConfig, CLI `--real-route`)` |
| 215 | `PLAN_REFRESH_ENV_CAP` | plan/placer/place.py::refresh_env_cap | `None` | str | recipe= 14 | `plan.refresh_env_cap (PlanConfig)` |
| 216 | `PLAN_REFRESH_ENV_CAP_ROUTES` | plan/placer/place.py::refresh_env_cap | `None` | list | recipe= 5 | `plan.refresh_env_cap_routes (PlanConfig)` |
| 217 | `PLAN_REFRESH_ENV_CAP_STEPS` | plan/placer/place.py::refresh_env_cap | `None` | list | recipe= 12 | `plan.refresh_env_cap_steps (PlanConfig)` |
| 218 | `PLAN_RESCALE_OPT` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.rescale_opt (PlanConfig, CLI `--rescale-opt`)` |
| 219 | `PLAN_SHARED_REALIZE` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `1` | str | recipe= 6 | `plan.shared_realize (PlanConfig, CLI `--shared-realize`)` |
| 220 | `PLAN_SITE_BTS_OUT_FILE` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | — | `plan.site_bts_out_file (PlanConfig, CLI `--site-bts-out-file`)` |
| 221 | `PLAN_SLIM_CUBIC` | plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` | str | recipe= 5 | `plan.slim_cubic (PlanConfig, CLI `--slim-cubic`)` |
| 222 | `PLAN_SPARSE_BTS_OUT` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `1:26,512:36` | str | recipe= 33 | `plan.sparse_bts_out (PlanConfig, CLI `--sparse-bts-out`)` |
| 223 | `PLAN_SPARSE_SLOTS` | examples/gpt2/make_plan.sh::(shell)<br>examples/llama/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `(unset)` / `512,1` | str | recipe= 33 | `plan.sparse_slots (PlanConfig, CLI `--sparse-slots`)` |
| 224 | `PLAN_STC_FIRST_ENTRY` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `None` | str | recipe= 5 | `plan.stc_first_entry (PlanConfig)` |
| 225 | `PLAN_STC_FIRST_FALLBACK` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `None` | str | recipe= 5 | `plan.stc_first_fallback (PlanConfig)` |
| 226 | `PLAN_STC_FIRST_KV_LANDING` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `None` | str | recipe= 1 | `plan.stc_first_kv_landing (PlanConfig)` |
| 227 | `PLAN_STC_FIRST_MAG_MAX` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `None` | str | recipe= 1 | `plan.stc_first_mag_max (PlanConfig)` |
| 228 | `PLAN_STC_FIRST_STEPS` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `None` | str | — | `plan.stc_first_steps (PlanConfig)` |
| 229 | `PLAN_STEP_LANDING` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `''` | str | recipe= 6 | `plan.step_landing (PlanConfig)` |
| 230 | `PYTHON` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `$_venv` / `(unset)` / `python3` | str | — | `host.python` |
| 231 | `PYTHONPATH` | examples/gpt2/make_plan.sh::(shell) | `(unset)` | str | — | `host.pythonpath` |
| 232 | `QKV_LANES` | impl/ops.py::FheOps.__init__ | `'1'` | flag | runtime= 14 | `model.qkv_lanes` |
| 233 | `RUNTIME_EXTRA` | examples/llama/make_plan.sh::(shell) | `` | str | — | `plan.runtime_extra (PlanConfig)` |
| 234 | `SM_CHEB_GS` | examples/gpt2/ref.py::softmax_ref<br>impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'0'` / `None` | flag/str (compared verbatim) | capture_env 4; runtime= 3 | `model.sm_cheb_gs` |
| 235 | `SM_DEN_RECIP` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 15; runtime= 15 | `model.sm_den_recip` |
| 236 | `SM_FOLD` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 14; runtime= 14 | `model.sm_fold` |
| 237 | `SM_GS_FIRST` | examples/gpt2/ref.py::softmax_ref [read via _on()]<br>impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim)/str | capture_env 15; runtime= 15 | `model.sm_gs_first` |
| 238 | `SM_HRS_PACK` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | flag/str (compared verbatim) | capture_env 13; runtime= 13 | `model.sm_hrs_pack` |
| 239 | `SM_PERIODIC` | impl/ops.py::FheOps.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'1'` / `None` | enum/flag/str (compared verbatim) | capture_env 14; runtime= 14 | `model.sm_periodic` |
| 240 | `SM_ROUTE_AHEAD` | impl/driver.py::ImplModel.ensure_routes | `'0'` | int | — | `model.sm_route_ahead` |
| 241 | `SPARSE_AUTO` | impl/fakecap.py::CapFhe.__init__<br>plan/contract.py::current_contract/contract_mismatches [contract] | `'0'` / `None` | int/str (compared verbatim) | capture_env all 39 | `bts.sparse_auto` |
| 242 | `SPARSE_BTS_SLOTS` | impl/fakecap.py::CapFhe.__init__<br>impl/fakecap.py::make_session<br>plan/contract.py::current_contract/contract_mismatches [contract] | `''` / `'512,1'` / `None` | list/str (compared verbatim) | capture_env all 39; runtime= 6 | `bts.sparse_bts_slots` |
| 243 | `SRC_LEVEL` | examples/gpt2/make_plan.sh::(shell)<br>plan/run_bootstrap_all_blocks.sh::(shell) | `16` / `36` | str | recipe= 33 | `plan.source_level (PlanConfig, CLI `--source-level`)` |
| 244 | `STEPS_T` | examples/gpt2/capture_mags.py::main [setdefault]<br>examples/gpt2/capture_mags_prefill.py::main [setdefault]<br>examples/gpt2/oracle_io.py::RunConfig.from_env [read via _env_or()]<br>examples/gpt2/run_decode.py::main [setdefault]<br>examples/gpt2/run_generate.py::main [setdefault]<br>examples/gpt2/run_prefill.py::main [setdefault] | `'128'` / `None` | int/str | — | `exec.steps_t` |
| 245 | `WEIGHTS_PATH` | examples/gpt2/capture_mags.py::main<br>examples/gpt2/capture_mags_prefill.py::main<br>examples/gpt2/run_decode.py::main<br>examples/gpt2/run_generate.py::main<br>examples/gpt2/run_prefill.py::main | `None` | str | — | `exec.weights_path` |
| 246 | `XDG_CACHE_HOME` | examples/gpt2/slotcache.py::_cache_dir | `None` | str | — | `host.xdg_cache_home` |

## 2. Set by the ported code or carried by shipped plans, read only by the old C++ / FIDESlib (23 names)

These are not read by any ported Python; the old `_core`/FIDESlib consumed them. M4b still needs a `Config` field for each (the shipped plans' `runtime=` / `capture_env` name them).

| # | name | set at | shipped plans | proposed Config field |
|---|---|---|---|---|
| 1 | `CUTMAX_SPARSE_BTS` | examples/gpt2/env.py::ENV<br>examples/llama/env.py::ENV | — | `model.cutmax_sparse_bts` |
| 2 | `FHE_STAGE_ARENA_GB` | examples/llama/env.py::ENV<br>profile.py::SessionProfile.gpt2_decode_n32 | — | `mem.fhe_stage_arena_gb` |
| 3 | `FIDESLIB_BTS_RAISE_DROP` | examples/gpt2/env.py::RUNTIME_LEGACY | runtime= 3 | `bts.raise_drop` |
| 4 | `FIDESLIB_BTS_RAISE_DROPS` | — | runtime= 15 | `bts.raise_drops` |
| 5 | `FIDESLIB_BTS_SHIFT` | examples/gpt2/env.py::RUNTIME_LEGACY<br>examples/gpt2/env.py::RUNTIME_N32 | runtime= 18 | `bts.shift` |
| 6 | `FIDESLIB_BTS_SPARSE_B` | examples/gpt2/env.py::RUNTIME_LEGACY<br>examples/gpt2/env.py::RUNTIME_N32 | runtime= 14 | `bts.sparse_b` |
| 7 | `FIDESLIB_BTS_SPARSE_B_MAX` | examples/llama/env.py::ENV | runtime= 6 | `bts.sparse_b_max` |
| 8 | `FIDESLIB_BTS_STC_FIRST_CF` | — | runtime= 6 | `bts.stc_first_cf` |
| 9 | `FIDESLIB_GPU_ENCODE` | examples/gpt2/env.py::ENV<br>examples/llama/env.py::ENV | — | `bts.gpu_encode` |
| 10 | `FIDESLIB_ROT_KEY_BAND` | examples/gpt2/env.py::RUNTIME_N32<br>profile.py::SessionProfile.custom_n32<br>profile.py::SessionProfile.gpt2_decode_n32<br>profile.py::SessionProfile.gpt2_decode_n64 | runtime= 14 | `mem.rot_key_band` |
| 11 | `FIDESLIB_SPRU` | examples/gpt2/env.py::RUNTIME_LEGACY<br>examples/gpt2/env.py::RUNTIME_N32 | runtime= 18 | `bts.spru` |
| 12 | `GPT2_INFERENCE_MODE` | examples/gpt2/env.py::ENV<br>examples/llama/env.py::ENV<br>profile.py::SessionProfile.custom_n32<br>profile.py::SessionProfile.gpt2_decode_n32<br>profile.py::SessionProfile.gpt2_decode_n64 | — | `exec.gpt2_inference_mode` |
| 13 | `LLAMA_PACKING` | examples/llama/env.py::ENV | — | `model.llama_packing` |
| 14 | `MALLOC_ARENA_MAX` | profile.py::SessionProfile.gpt2_decode_n32<br>profile.py::SessionProfile.gpt2_decode_n64 | — | `host.malloc_arena_max` |
| 15 | `ML` | — | recipe= 4 | `(old upstream DaCapo/Orion driver input, driver not ported)` |
| 16 | `PLAN_BTS_LEVEL` | profile.py::SessionProfile.custom_n32<br>profile.py::SessionProfile.gpt2_decode_n32 | — | `plan.bts_level (PlanConfig)` |
| 17 | `PLAN_CACHE_READ_LEVEL` | profile.py::SessionProfile.custom_n32<br>profile.py::SessionProfile.gpt2_decode_n32 | — | `plan.cache_read_level (PlanConfig)` |
| 18 | `PLAN_MAX_LEVEL` | profile.py::SessionProfile.custom_n32<br>profile.py::SessionProfile.gpt2_decode_n32 | — | `plan.max_level (PlanConfig)` |
| 19 | `PLAN_SRC_LEVEL` | profile.py::SessionProfile.custom_n32<br>profile.py::SessionProfile.gpt2_decode_n32 | — | `plan.src_level (PlanConfig)` |
| 20 | `PYTHONHASHSEED` | — | recipe= 2 | `(old upstream DaCapo/Orion driver input, driver not ported)` |
| 21 | `SPARSE_LEVEL_BUDGET` | examples/llama/env.py::ENV | runtime= 6 | `bts.sparse_level_budget` |
| 22 | `SPARSE_LN_BTS` | examples/gpt2/env.py::ENV<br>examples/llama/env.py::ENV | — | `bts.sparse_ln_bts` |
| 23 | `SPARSE_SM_BTS` | examples/gpt2/env.py::ENV<br>examples/llama/env.py::ENV | — | `bts.sparse_sm_bts` |

## 3. Shipped plans per key (where not all)

The 39 tracked plan dirs of the old repo (`bootstrap_placements/**` with a placement or PLAN_CMD.txt; every one is stamped with `capture_env`):

`gpt2_decode_python_n32`, `gpt2_decode_python_n32_dense`, `gpt2_decode_python_n32_ilp`, `gpt2_decode_python_n32_shipped`, `gpt2_decode_python_n32_smcheb`, `gpt2_decode_python_n32_stcf_accurate`, `gpt2_decode_python_n32_thor`, `gpt2_decode_python_n32_top512`, `gpt2_decode_python_n64`, `gpt2_decode_python_n64_dense`, `gpt2_heat_decode_python_n32`, `gpt2_heat_decode_python_n32_ilp`, `gpt2_prefill_python_n32_blocked`, `gpt2_prefill_python_n32_pcmmstcf`, `llama31_8b_decode_python_n32`, `llama31_8b_prefill_python_n32_blocked`, `llama31_8b_prefill_python_n32_blocked_gqa`, `llama31_8b_prefill_python_n32_blocked_gqa_attn`, `llama31_8b_prefill_python_n32_ccmm1`, `llama31_8b_prefill_python_n32_pcmmstcf`, `python/ablations/cf_fixed_10`, `python/ablations/cf_fixed_7`, `python/ablations/cf_fixed_8`, `python/ablations/cf_fixed_9`, `python/ablations/kappa_1`, `python/ablations/kappa_128`, `python/ablations/kappa_16`, `python/ablations/kappa_32`, `python/ablations/kappa_4`, `python/ablations/kappa_64`, `python/ablations/kappa_8`, `python/dacapo`, `python/dacapo_dense`, `python/fhelipe`, `python/fhelipe_dense`, `python/orion`, `python/orion_dense`, `shift/all3_veto1`, `shift/gpt2_decode_python_n32`

| key | source | plans |
|---|---|---|
| `CUTMAX_PAIR` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `GELU_FOLD` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `IMPL_DEAD_OPS` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `IMPL_SCOPED_NAMES` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `KV_LANES` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `LADDER_RADIX_BITS` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `LN_CHEB` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_CHEB_GS` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1 |
| `SM_DEN_RECIP` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_FOLD` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_GS_FIRST` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_HRS_PACK` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_PERIODIC` | capture_env | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `AUTO_BTS_LEVEL` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_pcmmstcf |
| `BTS_MAX_INPUT_LEVEL` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `CHEB_BSGS` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `CUTMAX_PAIR` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_BTS_RAISE_DROP` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_thor |
| `FIDESLIB_BTS_RAISE_DROPS` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_BTS_REAL` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_BTS_SHIFT` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `FIDESLIB_BTS_SPARSE_B` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_BTS_SPARSE_B_MAX` | runtime= | llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_BTS_STC_FIRST` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_BTS_STC_FIRST_CF` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_BTS_STC_FIRST_ROUTES` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_ROT_KEY_BAND` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `FIDESLIB_SPRU` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `GELU_FOLD` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `IMPL_DEAD_OPS` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `IMPL_SCOPED_NAMES` | runtime= | llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `IMPL_STC_FIRST_KV` | runtime= | gpt2_decode_python_n32_stcf_accurate |
| `KV_LANES` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `LADDER_RADIX_BITS` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `LN_CHEB` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_ACT_FOLD` | runtime= | gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_ATTN` | runtime= | llama31_8b_prefill_python_n32_ccmm1 |
| `PF_ATTN_PACK` | runtime= | llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1 |
| `PF_GELU_JOINT` | runtime= | gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_GQA` | runtime= | llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1 |
| `PF_JOINT_FORCE` | runtime= | gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_KV_SHARE` | runtime= | llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1 |
| `PF_LAYOUT` | runtime= | gpt2_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1 |
| `PF_LINEAR` | runtime= | gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_PCMM_CF` | runtime= | gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_PCMM_STAGED_ONLY` | runtime= | gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_PCMM_WBITS` | runtime= | llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PF_SM_JOINT` | runtime= | gpt2_prefill_python_n32_pcmmstcf |
| `QKV_LANES` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_CHEB_GS` | runtime= | gpt2_decode_python_n32_smcheb, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1 |
| `SM_DEN_RECIP` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_FOLD` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_GS_FIRST` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_HRS_PACK` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SM_PERIODIC` | runtime= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_prefill_python_n32_blocked, gpt2_prefill_python_n32_pcmmstcf, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SPARSE_BTS_SLOTS` | runtime= | llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `SPARSE_LEVEL_BUDGET` | runtime= | llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `BTS_LEVEL` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `BTS_OUT_DEG` | recipe= | python/fhelipe, python/fhelipe_dense |
| `CACHE_READ_LEVEL` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `FORBID_STEPS` | recipe= | gpt2_decode_python_n32, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `MAX_LEVEL` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `ML` | recipe= | python/dacapo, python/dacapo_dense, python/orion, python/orion_dense |
| `PLAN_ACC_CHAIN` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_BASELINE_DEPTH_CAP` | recipe= | python/fhelipe_dense |
| `PLAN_BASELINE_RESCUE` | recipe= | python/fhelipe, python/fhelipe_dense |
| `PLAN_CF_MAX` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_CF_MIN` | recipe= | gpt2_decode_python_n64, gpt2_decode_python_n64_dense, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9 |
| `PLAN_DELIB_LANDING_REMAP` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_HINT_ENV_VETO` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_HINT_VETO_STEPS` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_ILP_TIME_LIMIT` | recipe= | gpt2_decode_python_n32_ilp, gpt2_heat_decode_python_n32_ilp |
| `PLAN_LEVEL_UNIT` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_MAG_SAFETY` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_NO_PRESCALE` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_PLACER` | recipe= | gpt2_decode_python_n32_ilp, gpt2_heat_decode_python_n32_ilp, python/fhelipe, python/fhelipe_dense |
| `PLAN_PRUNE` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/dacapo, python/dacapo_dense, python/fhelipe, python/fhelipe_dense, python/orion, python/orion_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_RAISE_DROP_MAX` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_RAISE_DROP_SET` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_RAISE_ENV_RULE` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_RAISE_LANDING_MAX` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512 |
| `PLAN_RAISE_ROUTES` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_REAL_ROUTE` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_heat_decode_python_n32, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_REFRESH_ENV_CAP` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_REFRESH_ENV_CAP_ROUTES` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512 |
| `PLAN_REFRESH_ENV_CAP_STEPS` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PLAN_SHARED_REALIZE` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512 |
| `PLAN_SLIM_CUBIC` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512 |
| `PLAN_SPARSE_BTS_OUT` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_SPARSE_SLOTS` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |
| `PLAN_STC_FIRST_ENTRY` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512 |
| `PLAN_STC_FIRST_FALLBACK` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512 |
| `PLAN_STC_FIRST_KV_LANDING` | recipe= | gpt2_decode_python_n32_stcf_accurate |
| `PLAN_STC_FIRST_MAG_MAX` | recipe= | gpt2_decode_python_n32_stcf_accurate |
| `PLAN_STEP_LANDING` | recipe= | llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf |
| `PYTHONHASHSEED` | recipe= | python/orion, python/orion_dense |
| `SRC_LEVEL` | recipe= | gpt2_decode_python_n32, gpt2_decode_python_n32_dense, gpt2_decode_python_n32_ilp, gpt2_decode_python_n32_shipped, gpt2_decode_python_n32_smcheb, gpt2_decode_python_n32_stcf_accurate, gpt2_decode_python_n32_thor, gpt2_decode_python_n32_top512, gpt2_decode_python_n64, gpt2_decode_python_n64_dense, gpt2_heat_decode_python_n32, gpt2_heat_decode_python_n32_ilp, llama31_8b_decode_python_n32, llama31_8b_prefill_python_n32_blocked, llama31_8b_prefill_python_n32_blocked_gqa, llama31_8b_prefill_python_n32_blocked_gqa_attn, llama31_8b_prefill_python_n32_ccmm1, llama31_8b_prefill_python_n32_pcmmstcf, python/ablations/cf_fixed_10, python/ablations/cf_fixed_7, python/ablations/cf_fixed_8, python/ablations/cf_fixed_9, python/ablations/kappa_1, python/ablations/kappa_128, python/ablations/kappa_16, python/ablations/kappa_32, python/ablations/kappa_4, python/ablations/kappa_64, python/ablations/kappa_8, python/fhelipe, python/fhelipe_dense, shift/all3_veto1, shift/gpt2_decode_python_n32 |

## 4. Planner knobs after M4c
Every `PLAN_*` name above (and `BTS_LEVEL`, `MAX_LEVEL`, ... of the shell drivers) is a `plan.*` field of the
planner's `PlanConfig` (`_pc.PlanConfig`, 84 fields). They stay the planner's CLI/env for now (placer choice, time
limits, envelopes, dissolve/veto, pricing are planner heuristics, read where they always were). The ones that are
landing tables or rule switches are no longer read from the environment by the sim/refresh code: the planner reads
its recipe once (`planner.recipe_plan_config`: the process's `PLAN_*` env with the CLI-resolved `BTS_LEVEL`,
`PLAN_SPARSE_BTS_OUT`, `PLAN_CF_MAX` on top) into a `_pc.PlanConfig`, and `sim.PlannerShadow` (pc-shadow in planner
mode) answers from it: route landings (`plan.bootstrap_level`, `plan.sparse_bts_out`), the StC-first entry and the
shipped-order landing (`plan.stc_first_entry`, `plan.stc_first_fallback` = `BtsConfig.dense_shipped`), the deliberate
K/V landing (`plan.stc_first_kv_landing`, now checked against the Shadow's shipped-order landing), `plan.delib_deg`,
`plan.stc_first_mag_max`, `plan.stc_first_steps`, `plan.step_landing`.

## 5. M4q additions (port re-synced to pers/dev 1894546)

New env reads of the delta (BERT / ViT examples, paged Llama prefill, PCMM batching, CCMM MHA) and the names only the four new plans carry. Shipped-plans column here = the new plans `cc2` (bert_base_python_n32_cc2), `cc3` (vit_base_224_python_n32_cc3), `pr2`, `pr9` (llama31_8b_prefill_python_n32_pr2/_pr9); the 39-plan counts of sections 1-3 are unchanged. No `Config` field exists for these yet (pc-config's generated list predates them).

### 5a. Env reads added to the ported code (23 names)

| # | name | read at (file::function) | default | type | shipped plans | proposed Config field |
|---|---|---|---|---|---|---|
| 247 | `PF_CCMM_MHA_KMAX` | impl/ccmm_mha.py::(module) | `'12'` | float | — | `model.prefill.ccmm_mha_kmax` |
| 248 | `PF_CCMM_MHA_QMAX` | impl/ccmm_mha.py::(module) | `'6'` | float | — | `model.prefill.ccmm_mha_qmax` |
| 249 | `PF_CCMM_MHA_SHIFT_PV` | impl/ccmm_mha.py::(module) | `'3'` | int | — | `model.prefill.ccmm_mha_shift_pv` |
| 250 | `PF_CCMM_MHA_SHIFT_QK` | impl/ccmm_mha.py::(module) | `'6'` | int | — | `model.prefill.ccmm_mha_shift_qk` |
| 251 | `PF_CCMM_MHA_VMAX` | impl/ccmm_mha.py::(module) | `'6'` | float | — | `model.prefill.ccmm_mha_vmax` |
| 252 | `PF_MHA_DBG` | impl/ccmm_mha.py::(module)<br>impl/ccmm_mha_paged.py::softmax_pages | `'0' / None` | flag | — | `exec.debug.pf_mha_dbg` |
| 253 | `PF_MM_TSYNC` | impl/pcmm.py::(module) | `'0'` | flag | — | `exec.debug.pf_mm_tsync` |
| 254 | `PF_PAGED` | examples/llama/capture_mags_prefill.py::main<br>examples/llama/run_handoff.py::main<br>examples/llama/run_prefill.py::main | `'0'` | flag | runtime= pr2,pr9 | `model.prefill.paged` |
| 255 | `PF_PAGED_DBG` | impl/ccmm_paged.py::(module) | `'-1'` | int | — | `exec.debug.pf_paged_dbg` |
| 256 | `PF_PAGED_DEVKV` | impl/ccmm_paged.py::(module) | `'0'` | flag | — | `model.prefill.paged_devkv` |
| 257 | `PF_PAGED_RSTAT` | impl/ccmm_paged.py::(module) | `'0'` | flag | — | `exec.debug.pf_paged_rstat` |
| 258 | `PF_PAGED_R_SCALES` | impl/ccmm_paged.py::(module) | `''` | list | — | `model.prefill.paged_r_scales` |
| 259 | `PF_PAGED_STREAM` | impl/ccmm_paged.py::(module) | `'0'` | flag | runtime= pr2,pr9 | `model.prefill.paged_stream` |
| 260 | `PF_PCMM_BATCH` | impl/pcmm.py::(module) | `'4'` (31ab85f; `'0'` before) | int | runtime= pr9 | `model.prefill.pcmm_batch` |
| 261 | `PF_PCMM_BATCH_DROPS` | impl/pcmm.py::(module) | `'1'` (31ab85f; `'0'` before) | flag | runtime= pr9 | `model.prefill.pcmm_batch_drops` |
| 262 | `PF_PCMM_RAISE_UP` | impl/pcmm.py::(module) | `'' / None` | str (d[:lo-hi]) | runtime= pr9 | `model.prefill.pcmm_raise_up` |
| 263 | `PF_SILU_SINGLE_ERR` | impl/prefill.py::(module) | `'0'` | float | runtime= pr2,pr9 | `model.prefill.silu_single_err` |
| 264 | `VIT_MASSIVE` | examples/vit/weights.py::(module) | `''` | list | — | `model.vit.massive` |
| 265 | `VIT_MASSIVE_SCALE` | examples/vit/weights.py::(module) | `'0.03125'` | float | — | `model.vit.massive_scale` |
| 266 | `VIT_MODEL` | examples/vit/weights.py::(module) | `'google/vit-base-patch16-224'` | str | — | `model.vit.model` |
| 267 | `VIT_PAD` | examples/vit/prefill.py::VitPrefill.prefill | `'1'` | flag | — | `model.vit.pad` |
| 268 | `VIT_REF_CAUSAL` | examples/vit/weights.py::block0_taps<br>examples/vit/weights.py::exact_blocks | `None` | flag | — | `exec.debug.vit_ref_causal` |
| 269 | `VIT_TAPS` | examples/vit/run_prefill.py::main | `(required) / None` | int (block) | — | `exec.debug.vit_taps` |

### 5b. Carried by the new plans, read only by the old C++ / FIDESlib (10 names)

| # | name | old consumer | shipped plans | proposed Config field |
|---|---|---|---|---|
| 24 | `FIDESLIB_CCMM_ASYNC` | Ccmm.cu: no host syncs / step timings (legacy stream orders it) | runtime= pr9 | `exec.ccmm_async` |
| 25 | `FIDESLIB_DISCARD_SCRATCH` | NTT.cu / Discard.cuh: L2 discard of fully rewritten scratch (no DRAM write-back) | runtime= pr9 | `mem.discard_scratch` |
| 26 | `FIDESLIB_ENCODE_BARRIER` | CryptoContext.cpp encode: GPU-side barrier instead of the host wait | runtime= pr9 | `exec.encode_barrier` |
| 27 | `FIDESLIB_MONOMIAL_ASYNC` | Ciphertext.cpp monomial cache build: fences as GPU-side barriers | runtime= pr9 | `exec.monomial_async` |
| 28 | `FIDESLIB_PCMM_ASYNC` | Pcmm.cu: no host fences / timing marks, outputs lifted into their limbs | runtime= pr2,pr9 | `exec.pcmm_async` |
| 29 | `FIDESLIB_STC_CACHE_ASYNC` | Bootstrap.cu StC-first last-stage cache: misses re-level / evict under GPU barriers | runtime= pr9 | `exec.stc_cache_async` |
| 30 | `FIDESLIB_STC_CACHE_POOLFREE` | Bootstrap.cu: idle pool blocks + CUDA pool reserve count as free for the cache floor | runtime= pr9 | `mem.stc_cache_poolfree` |
| 31 | `FIDESLIB_STC_HIGH_ASYNC` | Bootstrap.cu stc_high: GPU-side barrier instead of a sync | runtime= pr9 | `exec.stc_high_async` |
| 32 | `FIDESLIB_STC_LAST_CACHE_FREE_GB` | Bootstrap.cu: below this free VRAM (default 8 GB) the StC last cache keeps one entry | runtime= pr9 | `mem.stc_last_cache_free_gb` |
| 33 | `IMPL_WSUM` | fideslib_wrapper.h weighted sum sum_i ts[i] cs[i] in one call (no graph recorded) | runtime= pr2,pr9 | `model.impl_wsum` |

### 5c. The new plans per key (every key they stamp or name)

| key | in sections 1-3 | cc2 / cc3 / pr2 / pr9 |
|---|---|---|
| `AUTO_BTS_LEVEL` | yes | capture_env cc2,cc3,pr2,pr9 |
| `BTS_ITERATIONS` | yes | capture_env cc2,cc3,pr2,pr9 |
| `BTS_LEVEL` | yes | recipe= pr2,pr9 |
| `BTS_MAX_INPUT_LEVEL` | yes | runtime= cc2,cc3,pr2,pr9 |
| `CACHE_READ_LEVEL` | yes | recipe= pr2,pr9 |
| `CACHE_READ_LEVEL_K` | yes | capture_env cc2,cc3,pr2,pr9 |
| `CACHE_READ_LEVEL_V` | yes | capture_env cc2,cc3,pr2,pr9 |
| `CHAIN` | yes | capture_env cc2,cc3,pr2,pr9 |
| `CHEB_BSGS` | yes | runtime= cc2,cc3,pr2,pr9 |
| `CKKS_COMPLEX` | yes | capture_env cc2,cc3,pr2,pr9 |
| `CORRECTION_FACTOR` | yes | capture_env cc2,cc3,pr2,pr9 |
| `CUTMAX_PAIR` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `FHE_LMHEAD_CAP` | yes | capture_env cc2,cc3,pr2,pr9 |
| `FHE_PT_COEFF_ENCODE` | yes | capture_env cc2,cc3,pr2,pr9 |
| `FIDESLIB_BTS_RAISE_DROPS` | yes | runtime= cc2,cc3,pr2,pr9 |
| `FIDESLIB_BTS_REAL` | yes | runtime= cc2,cc3,pr2,pr9 |
| `FIDESLIB_BTS_SHIFT` | yes | runtime= cc2,cc3,pr2,pr9 |
| `FIDESLIB_BTS_SPARSE_B` | yes | runtime= cc2,cc3,pr2,pr9 |
| `FIDESLIB_BTS_SPARSE_B_MAX` | yes | runtime= pr2,pr9 |
| `FIDESLIB_CCMM_ASYNC` | no (5a/5b) | runtime= pr9 |
| `FIDESLIB_DISCARD_SCRATCH` | no (5a/5b) | runtime= pr9 |
| `FIDESLIB_ENCODE_BARRIER` | no (5a/5b) | runtime= pr9 |
| `FIDESLIB_GPU_ENCODE` | yes | runtime= pr2,pr9 |
| `FIDESLIB_MONOMIAL_ASYNC` | no (5a/5b) | runtime= pr9 |
| `FIDESLIB_PCMM_ASYNC` | no (5a/5b) | runtime= pr2,pr9 |
| `FIDESLIB_ROT_KEY_BAND` | yes | runtime= cc2,cc3,pr2,pr9 |
| `FIDESLIB_SPRU` | yes | runtime= cc2,cc3,pr2,pr9 |
| `FIDESLIB_STC_CACHE_ASYNC` | no (5a/5b) | runtime= pr9 |
| `FIDESLIB_STC_CACHE_POOLFREE` | no (5a/5b) | runtime= pr9 |
| `FIDESLIB_STC_HIGH_ASYNC` | no (5a/5b) | runtime= pr9 |
| `FIDESLIB_STC_LAST_CACHE_FREE_GB` | no (5a/5b) | runtime= pr9 |
| `FORBID_STEPS` | yes | recipe= pr2,pr9 |
| `FUSED_LN_VAR` | yes | capture_env cc2,cc3,pr2,pr9 |
| `FUSED_SM_DEN` | yes | capture_env cc2,cc3,pr2,pr9 |
| `GELU_FOLD` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `GPT2_CACHE` | yes | capture_env cc2,cc3,pr2,pr9 |
| `GPT2_FOLD_LN1` | yes | capture_env cc2,cc3,pr2,pr9 |
| `GPT2_FOLD_LN2` | yes | capture_env cc2,cc3,pr2,pr9 |
| `GPT2_FOLD_LNF` | yes | capture_env cc2,cc3,pr2,pr9 |
| `GPT2_PACKING` | yes | capture_env cc2,cc3,pr2,pr9 |
| `IMPL_DEAD_OPS` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `IMPL_SCOPED_NAMES` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,pr2,pr9 |
| `IMPL_WSUM` | no (5a/5b) | runtime= pr2,pr9 |
| `KV_LANES` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `LADDER_RADIX_BITS` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `LN_CHEB` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `LOGN` | yes | capture_env cc2,cc3,pr2,pr9 |
| `MAX_LEVEL` | yes | recipe= pr2,pr9 |
| `PF_ACT_FOLD` | yes | runtime= cc3,pr2,pr9 |
| `PF_ATTN` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_ATTN_PACK` | yes | runtime= pr2,pr9 |
| `PF_CCMM_MERGE` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_CHEB_LEAF_NATIVE` | yes | runtime= pr2,pr9 |
| `PF_GELU_JOINT` | yes | runtime= cc3,pr2,pr9 |
| `PF_GLU_LEAN` | yes | runtime= pr2,pr9 |
| `PF_GLU_XU` | yes | runtime= pr2,pr9 |
| `PF_GQA` | yes | runtime= pr2,pr9 |
| `PF_JOINT_FORCE` | yes | runtime= cc3,pr2,pr9 |
| `PF_KV_SHARE` | yes | runtime= pr2,pr9 |
| `PF_LAYOUT` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_LINEAR` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_PAGED` | no (5a/5b) | runtime= pr2,pr9 |
| `PF_PAGED_STREAM` | no (5a/5b) | runtime= pr2,pr9 |
| `PF_PCMM_BATCH` | no (5a/5b) | runtime= pr9 |
| `PF_PCMM_BATCH_DROPS` | no (5a/5b) | runtime= pr9 |
| `PF_PCMM_RAISE` | yes | runtime= pr2,pr9 |
| `PF_PCMM_RAISE_UP` | no (5a/5b) | runtime= pr9 |
| `PF_PCMM_WBITS` | yes | runtime= pr2,pr9 |
| `PF_SILU_SINGLE` | yes | runtime= pr2,pr9 |
| `PF_SILU_SINGLE_ABS` | yes | runtime= pr2,pr9 |
| `PF_SILU_SINGLE_DEGS` | yes | runtime= pr2,pr9 |
| `PF_SILU_SINGLE_ERR` | no (5a/5b) | runtime= pr2,pr9 |
| `PF_SINC_A2` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_SINC_BI_HOIST` | yes | runtime= pr2,pr9 |
| `PF_SM_PAIR` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_SM_PAIR_ROUNDS` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_SM_R_ITER` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_SM_R_ITER_P` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_SM_R_REFRESH` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_SM_R_SCALE` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PF_SM_UNPACK_FOLD` | yes | runtime= cc2,cc3,pr2,pr9 |
| `PLAN_ACC_CHAIN` | yes | recipe= pr2,pr9 |
| `PLAN_CF_MAX` | yes | recipe= pr2,pr9 |
| `PLAN_DELIB_LANDING_REMAP` | yes | recipe= pr2,pr9 |
| `PLAN_HINT_ENV_VETO` | yes | recipe= pr2,pr9 |
| `PLAN_HINT_VETO_STEPS` | yes | recipe= pr2,pr9 |
| `PLAN_LEVEL_UNIT` | yes | recipe= pr2,pr9 |
| `PLAN_MAG_SAFETY` | yes | recipe= pr2,pr9 |
| `PLAN_NO_PRESCALE` | yes | recipe= pr2,pr9 |
| `PLAN_PRUNE` | yes | recipe= pr2,pr9 |
| `PLAN_RAISE_DROP_MAX` | yes | recipe= pr2,pr9 |
| `PLAN_RAISE_DROP_SET` | yes | recipe= pr2,pr9 |
| `PLAN_RAISE_ENV_RULE` | yes | recipe= pr2,pr9 |
| `PLAN_RAISE_ROUTES` | yes | recipe= pr2,pr9 |
| `PLAN_REAL_ROUTE` | yes | recipe= pr2,pr9 |
| `PLAN_REFRESH_ENV_CAP` | yes | recipe= pr2,pr9 |
| `PLAN_REFRESH_ENV_CAP_STEPS` | yes | recipe= pr2,pr9 |
| `PLAN_SPARSE_BTS_OUT` | yes | recipe= pr2,pr9 |
| `PLAN_SPARSE_SLOTS` | yes | recipe= pr2,pr9 |
| `PLAN_STEP_LANDING` | yes | recipe= pr2,pr9 |
| `QKV_LANES` | yes | runtime= cc2,cc3,pr2,pr9 |
| `SM_CHEB_GS` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `SM_DEN_RECIP` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `SM_FOLD` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `SM_GS_FIRST` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `SM_HRS_PACK` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `SM_PERIODIC` | yes | capture_env cc2,cc3,pr2,pr9; runtime= cc2,cc3,pr2,pr9 |
| `SPARSE_AUTO` | yes | capture_env cc2,cc3,pr2,pr9 |
| `SPARSE_BTS_SLOTS` | yes | capture_env cc2,cc3,pr2,pr9; runtime= pr2,pr9 |
| `SPARSE_LEVEL_BUDGET` | yes | runtime= pr2,pr9 |
| `SRC_LEVEL` | yes | recipe= pr2,pr9 |

## 6. M7f additions (port re-synced to pers/dev 31ab85f, the 2026-10-19 table push)

New env reads of the 1894546..31ab85f delta (PCMM/CCMM lanes and raise tables, warm-up runners, paged softmax caches)
and the names only the five result plans (`llama31_8b_prefill128`, `gpt2_prefill128`, `gpt2_decode_python_n32`,
`bert_base_sst2_128`, `vit_b16_224`) or the new FIDESlib 797d26f defaults carry. Shipped-plans column = those five
(`llama128` = llama31_8b_prefill128; the other four carry no name of this section). Changed defaults of section 5a
rows: `PF_PCMM_BATCH` `'0'` -> `'4'`, `PF_PCMM_BATCH_DROPS` `'0'` -> `'1'` (pcmm.py); `FIDESLIB_DISCARD_SCRATCH`
(5b) default 1 in FIDESlib 797d26f. `PF_PCMM_RAISE_UP` is now a spelling of `PF_PCMM_RAISE_HI=gate_up:...`; the Llama
example delta (`PF_WARMUP`, `PF_MASK_PREFETCH`, `PF_ENC_MISS_KEYS`, ...) is M9c's port (`PF_WARMUP` /
`PF_MASK_HOST_CACHE` listed in 6b because llama128 stamps them). gen_fields.py reads sections 1, 2, 5 and 6.

### 6a. Env reads added to the ported code (16 names)

| # | name | read at (file::function) | default | type | shipped plans | proposed Config field |
|---|---|---|---|---|---|---|
| 270 | `BERT_WARMUP` | examples/bert/run_bert.py::main | `'1'` | flag | — | `exec.bert_warmup` |
| 271 | `GPT2_PF_WARMUP` | examples/gpt2/run_prefill.py::_planned | `'1'` | enum (0/1/same) | — | `exec.gpt2_pf_warmup` |
| 272 | `PF_CCMM_DBG` | impl/ccmm.py::(module) | `'0'` | flag | — | `exec.debug.pf_ccmm_dbg` |
| 273 | `PF_CCMM_LANES` | impl/ccmm.py::(module) | `'0'` | int | — | `model.prefill.ccmm_lanes` |
| 274 | `PF_CCMM_RAISE` | impl/ccmm.py::(module) | `''` | list | runtime= llama128 | `model.prefill.ccmm_raise` |
| 275 | `PF_FINISH_GRAPH` | impl/pcmm.py::(module) | `'0'` | flag | — | `model.prefill.finish_graph` |
| 276 | `PF_FUSE_PROBE` | impl/pcmm.py::(module)<br>impl/ccmm.py::(module) | `''` | str | — | `exec.debug.pf_fuse_probe` |
| 277 | `PF_MM_SYNC` | impl/pcmm.py::(module) | `'0'` | flag | — | `exec.debug.pf_mm_sync` |
| 278 | `PF_PCMM_DBG` | impl/pcmm.py::(module) | `'0'` | flag | — | `exec.debug.pf_pcmm_dbg` |
| 279 | `PF_PCMM_LANES` | impl/pcmm.py::(module) | `'0'` | int | — | `model.prefill.pcmm_lanes` |
| 280 | `PF_PCMM_RAISE_HI` | impl/pcmm.py::(module) | `''` | list | runtime= llama128 | `model.prefill.pcmm_raise_hi` |
| 281 | `PF_POOL_TRIM_BELOW_GB` | impl/prefill.py::(module) | `'1e9'` | float | — | `model.prefill.pool_trim_below_gb` |
| 282 | `PF_POOL_TRIM_EVERY` | impl/prefill.py::(module) | `'1'` | int | — | `model.prefill.pool_trim_every` |
| 283 | `PF_SM_EXP_EPS` | impl/ccmm_paged.py::(module) | `'0'` | float | runtime= llama128 | `model.prefill.sm_exp_eps` |
| 284 | `PF_SM_META_CACHE` | impl/ccmm_paged.py::(module) | `'0'` | flag | — | `model.prefill.sm_meta_cache` |
| 285 | `PF_SM_R_RAISE` | impl/ccmm_paged.py::(module) (CapFhe.raise_drop -> the refresh route's raise_drop) | `'0'` | int | — | `model.prefill.sm_r_raise` |

### 6b. Carried by the result plans or set by the new drivers, read only by the old C++ / FIDESlib (14 names)

| # | name | old consumer | shipped plans | proposed Config field |
|---|---|---|---|---|
| 34 | `FIDESLIB_BTS_RAISE_NOSTC` | RawCiphertext.cu: raise variants built without StC tables (a path needing them throws) | runtime= llama128 | `bts.raise_nostc` |
| 35 | `FIDESLIB_BTS_RAISE_STC_ONUSE` | Context.cu: a staged-only StC-first route builds a missing raise variant at first use | runtime= llama128 | `bts.raise_stc_onuse` |
| 36 | `FIDESLIB_CTS_ALT_BUDGET` | RawCiphertext.cu: alternative CtS level budget (0 = off) | runtime= llama128 | `bts.cts_alt_budget` |
| 37 | `FIDESLIB_DUMP_FULL` | CryptoContext.cpp: plaintext dumps keep every limb (reload = one H2D per limb) | runtime= llama128 | `mem.dump_full` |
| 38 | `FIDESLIB_ENCODE_NOSYNC` | CryptoContext.cpp: no host wait after device encodes (set to 1 by the gpt2/bert/vit run_* drivers) | runtime= llama128 | `exec.encode_nosync` |
| 39 | `FIDESLIB_ENCODE_PINNED_RING` | DeviceEncode.cu: pinned host staging ring (slots; drivers set 256) | runtime= llama128 | `mem.encode_pinned_ring` |
| 40 | `FIDESLIB_PCMM_GEMM_MB` | Pcmm.cu: int32 product buffer budget (default 1024 MB) | runtime= llama128 | `mem.pcmm_gemm_mb` |
| 41 | `FIDESLIB_PCMM_NOSYNC` | Pcmm.cu: no fences at entry / exit, planes uploaded without blocking (drivers set 1) | runtime= llama128 | `exec.pcmm_nosync` |
| 42 | `FIDESLIB_PCMM_OUT_CHUNK` | Pcmm.cu: GEMMs + output switches in groups of k output blocks | runtime= llama128 | `mem.pcmm_out_chunk` |
| 43 | `FIDESLIB_PINNED_ANON` | CudaUtils.cu: anonymous pinned host buffers (default 1 in 797d26f) | — (default) | `mem.pinned_anon` |
| 44 | `FIDESLIB_STC_CACHE_RELEASE` | Bootstrap.cu: StC-first cache entries released after use | runtime= llama128 | `mem.stc_cache_release` |
| 45 | `FIDESLIB_STC_POW2_KEY` | Bootstrap.cu: StC-first cache keyed by power-of-two scale (drivers set 1) | runtime= llama128 | `bts.stc_pow2_key` |
| 46 | `PF_MASK_HOST_CACHE` | inference.h: evicted mask encodes kept in pinned host memory (reloaded by H2D) | runtime= llama128 | `model.prefill.mask_host_cache` |
| 47 | `PF_WARMUP` | examples/llama_from_primitives/run_prefill.py (not ported) | runtime= llama128 | `model.prefill.warmup` |

## 7. M9f additions (decode plans cut for a prefill hand-off's cache)

The planner's K/V entry seed: every K/V cache read of a decode block arrives at (`KV_ENTRY_LEVEL`, `KV_ENTRY_DEG`)
instead of the capture's level (a decode after a prefill hand-off reads the handed states), and a plan dir whose
refresh sites every block keeps. Shipped-plans column = `dsw48_ho36` / `dsw48_ho38`
(llama31_8b_decode_python_n32_dsw48_ho36 / _ho38). gen_fields.py reads 7a too.

### 7a. Env reads added to the ported code (3 names)

| # | name | read at (file::function) | default | type | shipped plans | proposed Config field |
|---|---|---|---|---|---|---|
| 286 | `KV_ENTRY_LEVEL` | plan/run_bootstrap_all_blocks.sh::(shell)<br>examples/llama/make_plan.sh::(shell) | `(unset)` | int | recipe= dsw48_ho36, dsw48_ho38 | `plan.kv_entry_level (PlanConfig, CLI `--kv-entry-level`)` |
| 287 | `KV_ENTRY_DEG` | plan/run_bootstrap_all_blocks.sh::(shell)<br>examples/llama/make_plan.sh::(shell) | `(unset)` (2) | int | — | `plan.kv_entry_deg (PlanConfig, CLI `--kv-entry-deg`)` |
| 288 | `PLAN_KEEP_PLACEMENTS` | plan/run_bootstrap_all_blocks.sh::(shell)<br>examples/llama/make_plan.sh::(shell) | `(unset)` | str | recipe= dsw48_ho36, dsw48_ho38 | `plan.keep_placements (PlanConfig, CLI `--keep-placements`)` |

## 8. M18 additions (planner knobs)

A deliberate dense refresh captured on the StC-first route whose input passed the StC-first entry: re-routed to the
shipped order (its landing) instead of repaired upstream by the cut; a per-site correction factor over the priced
choice. gen_fields.py reads 8a too.

### 8a. Env reads added to the ported code (2 names)

| # | name | read at (file::function) | default | type | shipped plans | proposed Config field |
|---|---|---|---|---|---|---|
| 289 | `PLAN_STC_FIRST_REROUTE` | plan/placer/planner.py::recipe_plan_config -> `sim.PlannerShadow.plan` | `None` | flag | recipe= vit_b16_224_m18s3 | `plan.stc_first_reroute (PlanConfig)` |
| 290 | `PLAN_SITE_CF` | plan/placer/emit.py::assemble -> `refresh.shadow.plan` | `None` | str | recipe= llama31_8b_decode_python_n32_dsw48_ho38_m18e | `plan.site_cf (PlanConfig)` |
