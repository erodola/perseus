# Perseus on perseus-core

Record 1 of both tracks, and the example of an entry that replaces the cryptographic backend: perseus-core, a
Rust + CUDA runtime by [@Davide-F5-Marincione](https://github.com/Davide-F5-Marincione), now Perseus's backend on
`main`:

- every stage is recorded once on a CPU twin, compiled against its plan into a program, and replayed in Rust with
  no Python on the critical path, from one device arena sized at setup;
- half of every switching key is regenerated on the GPU from a seed, so the bundle is 10 GB instead of 32 GB;
- the bootstrap routes are measured and fed to the planner, refreshes are batched, and the PCMM runs as int8 GEMMs.

Against record 0, Decode-8 is 15% faster and Prefill-128 38% faster, in about two thirds of the GPU memory. Its keys
and ciphertexts have their own formats, so the entry brings its own client.

## Changes against record #0

- the backend, the Rust + CUDA crates in `crates/`, and the Python package on it, `python/perseus/`
- the speedrun server, `speedrun/server.py`: Decode-8 drives Perseus's served decode
  (`perseus.serve.gpt2_decode`); Prefill-128 runs the planned prefill with the encrypted tail as two more stages
- the tail stages, in `python/perseus/examples/gpt2/prefill.py` (`tail_stages`) and
  `python/perseus/impl/blocked.py` (`token_of`), planned by `speedrun/make_tail_plan.sh` into
  `bootstrap_placements/gpt2_prefill128_m16s0_tail`
- the client, in the harness: `harness/perseus_core_client.py`
- a fix in `PlannedRun.token` (`python/perseus/impl/driver.py`): a new sequence re-records every position. The
  served decode kept the reference token's recorded stream across sequences, so from the second request on the
  CPU twin skipped that position's K/V push and every later position ran with wrong attention steps

## How a different backend is submitted

1. The server speaks the harness protocol (`harness/protocol.py`), whatever runs behind it.
2. Its first message, `keyspec`, is its key request in the backend's own terms. Here it is a
   `pc_client::KeySpec`: the chain, and every switching key the plan needs.
3. The entry adds one client file to the harness: keygen for that request (after checking the chain is the
   one it knows), encryption of the inputs, and decryption of the outputs. The KL and argmax checks and the
   clock stay the harness's, so the client file cannot change a score. It holds the secret key, so the
   maintainers review it like security code and pin it.

## Ablation

None.

## Dev result

Same GPU, harness and prompt as record 0, 2026-10-22. Logs in `dev_logs/`.

| Track | Score | Record 0 | KL | Argmax | Peak GPU memory |
|---|---|---|---|---|---|
| Decode-8 | 3.384 s/token | 3.993 | 4.6e-4 (positions 8-15) | 15/16 (position 4, unscored, a near tie) | 47.8 GiB |
| Prefill-128 | 22.22 s | 35.78 | 2.7e-3 (position 127) | correct | 57.4 GiB |

Both tracks are within their KL bars (Prefill-128 10⁻², Decode-8 10⁻³). The logs' `outputs_ok` was computed
against 10⁻³ for both tracks, before the prefill bar was raised.

## Disclosures

- A new client file, `harness/perseus_core_client.py`, runs the backend's CUDA-free client (`perseus._pc`).
- The tail stages (12: the last position into the decode layout, ln_f, the LM head; 13: CutMax) have a plan made
  for this entry, on top of the shipped prefill plan.
- During setup, the prefill server records and compiles its programs from a public prompt (seeded token ids)
  and runs one warm-up request on it.
- Half of every switching key is regenerated on the GPU from a seed in the bundle, so the bundle is about 10 GB.
