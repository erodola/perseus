# perseus wire format (version 1)

The client (`crates/pc-client`, Python `perseus.client`) holds the secret and runs without a device; the
server (`Session.from_bundle`) runs the plan on a key bundle and ciphertexts and holds no secret. Three objects cross
between them: the **key request** (server -> client, JSON), the **key bundle** (client -> server, once), and
**ciphertext frames** (both ways, per request). The session protocol (`perseus.serve`) carries them.

All integers little-endian. Every binary object starts with a 16-byte prefix:

| bytes | field |
|---|---|
| 4 | magic: `PCKB` (bundle) or `PCCT` (ciphertext) |
| 4 | u32 format version (= 1; a reader refuses any other) |
| 8 | u64 header length H |
| H | header, UTF-8 JSON |

followed by raw payloads of u32 words, each payload followed by its u64 checksum `sum64` (FNV-1a 64 over the
payload as little-endian u64 words, offset 0xcbf29ce484222325, prime 0x100000001b3; an odd trailing u32 word is
taken as one u64).

## Chain identity and plan contract

- `params_hash`: FNV-1a 64 (hex, 16 digits) over: N (u64), every prime Q then P (u64 each), per hybrid digit its
  length then its prime ids (u64 each), the level unit (u64), the word bits (u32). Both sides derive the chain from
  the preset name (`chain`, e.g. `n32`) and must get the same hash.
- `contract`: FNV-1a 64 (hex) of `Config.load_bearing().to_json()` — the load-bearing Config both sides run (the
  plan's stamp + runtime levers). The server puts its own in the key request and checks the bundle's.

## Key request (`Session.key_manifest`, JSON = `pc_client::KeySpec`)

```
{ "chain": "n32", "params_hash": "...", "contract": "...",
  "keys": [ {"role": "relin" | "conj" | "enc_dense_to_sparse" | "enc_sparse_to_dense" | "spru_atob",
             "shape": {"q_top", "ndig", "nq", "kp"}},
            {"role": "rot", "step": s, "galois": 5^s mod 2N, "shape": ...},
            {"role": "auto", "galois": g, "shape": ...}, ... ],
  "spru": null | {"h", "n": 2, "cs_level": 0, "cs_nf": sf(top)},
  "pcmm_t": [t, ...], "ccmm": bool, "public_key": bool }
```

`shape`: digits `0..ndig`, each the Q prime ids `0..=q_top` then the `kp` special primes (a full key: q_top = nq-1,
ndig = dnum; banded rotation keys and the bottom-only encapsulation key store fewer rows; `ndig` is the number of
hybrid digits whose first prime id is <= q_top). Relations (eval form, slot of prime id t in digit d):

    b = -a s_new + e + [t in digit d] (P mod q_t) s_old,     a = KSKA(seed, d, p_t)   (pc_core::keys, frozen)

| role | s_old | s_new |
|---|---|---|
| relin | s^2 | s |
| rot (g = 5^step), conj (g = 2N-1), auto (g) | s | sigma_{g^-1}(s) |
| enc_dense_to_sparse | s | s_sparse (Hamming weight 32) |
| enc_sparse_to_dense | s_sparse | s |
| spru_atob | s | s' (the SPRU block-binary key, h blocks) |

e: discrete Gaussian sigma 3.19, one integer polynomial per digit.

## Key bundle (`PCKB`)

Header (`pc_client::BundleHeader`):

```
{ "chain", "params_hash", "contract", "n",
  "bits": 28, "row_words": 57348,          // packed b: bits per coefficient, u32 words per row
  "entries": [ ... ] }                      // payload order
```

| entry `kind` | fields | payload (u32 words) |
|---|---|---|
| `ksk` | role fields + `shape`, `seed` (8 u32), `words` | `b` packed: `[digit][slot][row_words]`, value i at bits [i*bits, (i+1)*bits) of the row's LE u32 stream, then at least one zero guard word, rows padded to a multiple of 4 words (`row_words = ceil((N*bits/32 + 1)/4)*4`); slot = Q ids `0..=q_top` then P. `a` is not sent: KSKA(seed). |
| `spru_cs` | `k` (= 2n), `rows`, `level`, `nf`, `words` | the SPRU bootstrapping-key ciphertexts Enc(S_u), `[c][part][row][N]`, eval form |
| `pcmm` | `t`, `primes` (q0, q1, P), `split` (q0..q3), `fwd`, `back`, `pack`, `words` | `fwd`, `back`, `pack` back to back, `pc_core::pcmm_keys` coefficient-form layout |
| `ccmm` | `primes` (q0..q3, P), `fwd`, `back`, `autos`, `rlk`, `words` | the four arrays back to back, `pc_core::ccmm_keys` coefficient-form layout |
| `public_key` | `rows` (= nq), `words` | `(-a s + e, a)`: `[part][row][N]`, eval form over every Q id |

Entries come in this order: the small ones (pcmm, ccmm, spru_cs, public_key), then one `ksk` per requested key in
request order, so a writer streams the switching keys one at a time and a reader imports them one at a time
(`BundleReader` checks the file length against the header, and each payload's checksum). The server accepts a
bundle only if `chain`, `params_hash` and `contract` match its own request and the `ksk` list is exactly the
requested (role, shape) list. The SPRU block key's positions and the secrets are never in a bundle.

## Ciphertext frame (`PCCT`)

```
{ "params_hash", "n", "k", "rows",
  "meta": {"level": primes dropped, "deg": 1 | 2, "nf": scale, "slots": live slots, "tag": pack tag | null} }
```

payload `[c < k][part < 2][row < rows][N]` u32, eval form (bit-reversed negacyclic NTT, OpenFHE order) over Q ids
`0..rows`, `rows = nq - level`. The server exports degree 1 (a pending rescale is realized first); the client
decrypts the bottom 4 rows (2 below the third prime) with an exact centered CRT.

## Session protocol (`perseus.serve`)

Over a stream socket (Unix path or host:port), messages `u64 total | u32 header length | header JSON {"kind", ...,
"blobs": [sizes]} | blobs`; blobs are frames, or the bundle file streamed as one blob.

1. client `hello` {contract, params_hash, model-specific shape (e.g. tokens)}
2. server `manifest` {spec: the key request JSON, input: {level, deg}, ...} — the client refuses another contract
3. client `bundle` (one blob: the bundle file); server `ready` {setup / import seconds, keys}
4. client `run` {model-specific} + input frames -> server `result` {timings} + output frames; during a run the
   server may send `ask` {what, ...} + frames and wait for the client's `answer` (e.g. `ccmm_calibrate`: the
   client decrypts the CCMM probe outputs of known public matrices and returns only the fitted scalar)
5. client `bye`. Either side may send `error` {message} and close.

An `ask` that returns decrypted data would make the client a decryption oracle; the only one defined returns a
calibration constant fitted against public probe matrices and refuses outputs that do not fit them.
