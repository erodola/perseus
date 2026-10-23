# Perseus

Record 0 of both tracks: Perseus on [FIDESlib32bits](https://github.com/gladia-research-group/FIDESlib32bits), our
32-bit version of [FIDESlib](https://github.com/CAPS-UMU/FIDESlib)
([Agulló-Domingo et al., ISPASS 2025](https://doi.org/10.1109/ISPASS64960.2025.00045)), entered through
`speedrun/server.py` at Perseus's tag `speedrun-record-0` like any fork.
Decode-8 runs the 12 blocks, the final LayerNorm, the LM head and the CutMax argmax at every position, under the
`gpt2_decode_python_n32` plan. Prefill-128 runs the channel-blocked prefill under `gpt2_prefill128`, then moves the
last position into the decode layout and runs the decode's tail and CutMax on it.

It builds with:

```bash
git clone --recurse-submodules -b speedrun-record-0 https://github.com/gladia-research-group/perseus && cd perseus
uv sync
NATIVE_SIZE=32 bash scripts/install_deps.sh   # patched OpenFHE + FIDESlib32bits
CHAIN=n32 bash scripts/local_build_core.sh    # the CUDA extension
.venv/bin/perseus-export --model openai-community/gpt2 --out .cache/models/openai-community/gpt2
```

and runs through the harness as `bash runner/entry.sh perseus ../perseus {decode,prefill} GPU`.

## Changes against record #N

None: this is record 0. Perseus changes made for the speedrun (tag `speedrun-record-0`):

- the server, `speedrun/server.py`, and the record's shared definitions, `speedrun/record0.py`
- SPRU's setup split into keygen (client) and load (server), in `third_party/FIDESlib/src/CKKS/Spru.cu`
- the last prefill position into the decode layout, `perseus/impl/blocked.py` (`token_of`)
- encryption and decryption out of Perseus's own timers, in `examples/gpt2_from_primitives/`
- OpenFHE patch 0015, `third_party/openfhe-n32/patches/`: the switching keys' seed is serialized, so the server
  regenerates half of every key on the GPU as the key owner's process does (12 GB of VRAM less)

## Ablation

None.

## Dev result

One RTX PRO 6000 Blackwell Server Edition, through the harness (`runner/entry.sh`, the server pinned to the GPU's
NUMA node), 2026-10-22, on the oracle prompt. Logs in `dev_logs/`.

| Track | Score | KL | Argmax | Peak GPU memory |
|---|---|---|---|---|
| Decode-8 | 3.993 s/token | 4.4e-4 (positions 8-15) | 16/16 | 74.5 GiB |
| Prefill-128 | 35.78 s | 2.8e-3 (position 127) | correct | 90.9 GiB |

Both tracks are within their KL bars (Prefill-128 10⁻², Decode-8 10⁻³). Prefill-128's KL comes from the
polynomial approximations, not the noise: the CPU twin, which has no encryption noise, gives the same value. The
logs' `outputs_ok` was computed against 10⁻³ for both tracks, before the prefill bar was raised.

## Disclosures

- The client's bundle carries SPRU's switching key and key ciphertexts and the prefill's PCMM keys, all made from
  the secret key on the client. The block positions of SPRU's sparse key never leave the client.
- During setup, the prefill server runs one warm-up request on a public prompt (seeded random token ids).
- Moving the last prefill position into the decode layout takes about ten masked rotations per channel block,
  inside the clock.
