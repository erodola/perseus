# Security model

What is protected, who holds what, and what is *not* claimed. The client (`perseus.client`, the Rust crate
`pc-client`) and the server (`perseus.serve`, a `perseus._pc.Session` opened with `Session.from_bundle`) implement it;
the byte-level format of what crosses between them is `docs/WIRE.md`. The single-process example drivers
(`python -m perseus.examples.*`) hold every role in one process and are for development, calibration and
benchmarking, not for deployment across a trust boundary.

## Parties

| party | holds | can |
|---|---|---|
| **client** (`perseus.client.Client`, `ClientSecret`) | the CKKS **secret key**; the tokenizer and the embedding tables | generate the keys the server asks for, encrypt inputs, decrypt outputs; runs without a GPU |
| **server** (`Session.from_bundle`) | the **key bundle** (relinearization, rotation, bootstrap and encapsulation keys; no secret key); the model weights **in the clear** | evaluate the planned model on ciphertexts; nothing else |

The secret key never leaves the client process: the bundle carries evaluation keys only, and a session opened from a
bundle has no secret (decryption on it raises).

## What crosses the wire

- **server -> client, once:** the key request (`Session.key_manifest`): the chain, its parameter hash, the plan's
  contract hash, and the list of keys the plan's programs need. Public information.
- **client -> server, once:** the key bundle for that request (tens of GB for GPT-2, most of it rotation keys).
- **client -> server, per request:** the encrypted input (block 0's input for a decode token).
- **server -> client, per response:** the encrypted outputs (the LM-head tile, and the encrypted argmax's one-hot
  when asked for).

Ciphertexts carry no plaintext metadata beyond CKKS's public parameters (level, scale, noise degree).

## What the server learns

- The **model**: weights are CKKS *plaintexts* on the server, never encrypted. Only activations are ciphertexts. This
  provides **input/output privacy for the client**, not model privacy for a model owner who is not the server.
- The **shape** of the computation: which blocks run, in what order, the packed widths, the number of tokens, timing.
- Nothing about the **values** of inputs, activations, or outputs, under the RLWE assumption at the chain's
  parameters (128-bit security).

## What is not claimed

- **No verifiable computation.** A malicious server can return any ciphertext; the client cannot tell a correct
  evaluation from a wrong one without redundancy of its own.
- **Approximate arithmetic.** CKKS is approximate; every nonlinearity is a fitted polynomial (`perseus.calibrate`).
  Correctness is statistical (top-1 agreement and KL against the plaintext model), not bit-exact.
- **IND-CPA-D.** CKKS decryption results can leak information about the secret key if decrypted values are shared
  with the server (Li-Micciancio). The client must treat decrypted outputs as sensitive with respect to the server.

## What the process writes

No code path logs or serializes key material other than the bundle the client writes for the server.

A **recording run is different**, and it is the one thing to keep off a machine that holds real inputs. The planner
prices a refresh from the size of the data it refreshes, so a recording (`--capture`) writes the magnitude of every
intermediate value into `graph.json` next to the op that produced it: a profile of the activations it ran on. Record
with the calibration inputs, not with someone's data, and treat a recorded graph as derived from its input.
