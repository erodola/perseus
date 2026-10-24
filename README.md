# Encrypted GPT-2 Speedrun

> **Can we run encrypted transformer inference as fast as plaintext inference?**

Under fully homomorphic encryption (FHE), a model (GPT-2 in this case) can process an encrypted prompt without ever exposing its content to the provider, while computing roughly the same answer. However, the current state of the art suffers from extreme latency (what causes it is described in [Background](#background)).

This branch of [Perseus](https://github.com/gladia-research-group/perseus) defines the Encrypted GPT-2 Speedrun, in which we optimize (collaboratively and
competitively) encrypted transformer inference under FHE for GPT-2 (124M) on a single RTX PRO 6000.

The client encrypts the request locally under CKKS at 128-bit security; the server receives the encrypted prompt and computes on ciphertexts only.

Under this framing, the task naturally includes two tracks, which resemble the full inference-as-a-service paradigm:

- **Prefill-128.** The user's encrypted prompt (128 tokens) populates the KV caches.
- **Decode-8.** From a fixed, given cache state, the model decodes 8 tokens, at positions 8 to 15, under teacher forcing on unknown test sequences.

The result of each is an end-to-end time, and a KL divergence between the plaintext GPT-2 logits and the encrypted ones, which must stay below 10⁻² in prefill, where it is taken on the last position's logits, and below 10⁻³ in decode, where it is the average over the timed positions.

Every entry is a pull request to this branch, `speedrun`: a change to Perseus's code, and a record folder in
`speedrun/records/`. Perseus itself, its results and its documentation, is on its
[`main` branch](https://github.com/gladia-research-group/perseus/tree/main). You can also
change the cryptographic backend. The default is [perseus-core](crates),
Perseus's Rust + CUDA runtime. Modify it, or include another backend as a submodule and wire it in.

To learn how to submit an entry, see [Submitting](#submitting).

## Running the current record

This branch is Decode-8's record; Prefill-128's is the commit the client pins, the client of every run.

```bash
git clone -b speedrun https://github.com/gladia-research-group/perseus && cd perseus
git worktree add ../perseus-client 89bc830017fe12ab967ddb10c24feae65d8af6df
for d in . ../perseus-client; do (cd $d && uv venv --python 3.12 .venv &&
  uv pip install --python .venv/bin/python -e ".[hf,examples,dev]" && .venv/bin/maturin develop --release); done
(cd ../perseus-client && .venv/bin/perseus-export --model openai-community/gpt2 --out .cache/models/openai-community/gpt2)

CLIENT=../perseus-client bash speedrun/runner/entry.sh perseus-core . decode 0    # Decode-8, on GPU 0
bash speedrun/runner/entry.sh perseus-core ../perseus-client prefill 0           # Prefill-128
```

You need Linux, `bwrap`, a systemd user session, a GPU with 48 GB or more, and `HF_HOME` holding
`openai-community/gpt2`. An entry runs the same way, from its own checkout.

## Records

> [!IMPORTANT]
> **Records are ranked by time (lower is better).** A new record runs faster than the current one
> by at least 1 %, with the KL of the encrypted logits from the plaintext ones below 10⁻² in
> Prefill-128 and 10⁻³ in Decode-8, at 128-bit security ([Rules](#rules)).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="speedrun/records/progress-dark.svg">
    <img src="speedrun/records/progress-light.svg" width="100%" alt="Record progression of both GPT-2 tracks. Prefill-128: 35.78 s (record 0), 22.22 s (record 1). Decode-8: 3.993 s/token (record 0), 3.384 s/token (record 1), 3.031 s/token (record 2). Dotted lines: 1 token/s, and GPT-2 unencrypted on the same GPU, 0.77 ms and 0.65 ms/token.">
  </picture>
</p>

The chart fills in as records are accepted. The dotted lines are the bars to beat: 1 token/s, and GPT-2
unencrypted on the same GPU (at the bottom, not to scale).

### Track 1: Prefill-128

| # | Time (s) | Description | Date | Code | Contributors |
|---|---|---|---|---|---|
| 0 | 35.78 | Baseline: Perseus on [FIDESlib32bits](https://github.com/gladia-research-group/FIDESlib32bits) | 2026‑10‑22 | [`39493d5`](https://github.com/gladia-research-group/perseus/tree/39493d5a98122c1fb595c99070bf4e435df752f0) · [log](speedrun/records/2026-10-22_Perseus) | [@alexzilligmm](https://github.com/alexzilligmm)<br>[@Davide‑F5‑Marincione](https://github.com/Davide-F5-Marincione) |
| 1 | 22.22 | Perseus on [perseus‑core](crates), its Rust + CUDA runtime, with its own client | 2026‑10‑22 | [`89bc830`](https://github.com/gladia-research-group/perseus/tree/89bc830017fe12ab967ddb10c24feae65d8af6df) · [log](speedrun/records/2026-10-22_PerseusCore) | [@alexzilligmm](https://github.com/alexzilligmm)<br>[@Davide‑F5‑Marincione](https://github.com/Davide-F5-Marincione) |
|  | $`\color{gray}\textsf{0.00077}`$ | $`\color{gray}\textsf{Unencrypted GPT-2}`$ | $`\color{gray}\textsf{2026-10-24}`$ | [code](speedrun/records/plaintext.py) · [log](speedrun/records/plaintext.json) |  |

### Track 2: Decode-8

| # | s/token | Description | Date | Code | Contributors |
|---|---|---|---|---|---|
| 0 | 3.993 | Baseline: Perseus on [FIDESlib32bits](https://github.com/gladia-research-group/FIDESlib32bits) | 2026‑10‑22 | [`39493d5`](https://github.com/gladia-research-group/perseus/tree/39493d5a98122c1fb595c99070bf4e435df752f0) · [log](speedrun/records/2026-10-22_Perseus) | [@alexzilligmm](https://github.com/alexzilligmm)<br>[@Davide‑F5‑Marincione](https://github.com/Davide-F5-Marincione) |
| 1 | 3.384 | Perseus on [perseus‑core](crates), its Rust + CUDA runtime, with its own client | 2026‑10‑22 | [`89bc830`](https://github.com/gladia-research-group/perseus/tree/89bc830017fe12ab967ddb10c24feae65d8af6df) · [log](speedrun/records/2026-10-22_PerseusCore) | [@alexzilligmm](https://github.com/alexzilligmm)<br>[@Davide‑F5‑Marincione](https://github.com/Davide-F5-Marincione) |
| 2 | 3.031 | FusedUnits: fused key switching, batched rotations, more precise bootstraps | 2026‑10‑24 | [`4d2e72b`](https://github.com/gladia-research-group/perseus/tree/4d2e72be136426f84f3ed402e468693ba00630db) · [log](speedrun/records/2026-10-24_FusedUnits) | [@erodola](https://github.com/erodola) |
|  | $`\color{gray}\textsf{0.00065}`$ | $`\color{gray}\textsf{Unencrypted GPT-2}`$ | $`\color{gray}\textsf{2026-10-24}`$ | [code](speedrun/records/plaintext.py) · [log](speedrun/records/plaintext.json) |  |

### What the records changed

- **Record 2, FusedUnits** (Decode-8): key switching makes fewer passes over GPU memory, rotations of one ciphertext
  share their first step, and five bootstraps run more precisely (a third of the error).
- **Record 1**: Perseus on its own Rust + CUDA runtime instead of FIDESlib.
- **Record 0**: the baseline, Perseus on FIDESlib.

## Rules

The only rules are that new records must:

1. Change only the framework and its backend. The weights, the test prompts and the harness (the checks and the clock) are fixed. Key generation, encryption and decryption belong to the backend's client; an entry that changes the key or ciphertext formats adds its own, which the maintainers review and pin.
2. Keep 128-bit security, with the server computing on ciphertexts only. Every secret and switching key must reach 128 bits on the Lattice Estimator, at the modulus where it is used, and the server gets only the public and evaluation keys.
3. Attain a KL divergence from float64 GPT-2 below 10⁻² for Prefill-128 and 10⁻³ for Decode-8, at p < 0.01 over the official runs.
4. Keep all input-dependent work inside the clock. Loading keys, encoding weights and loading plans are untimed setup and must not depend on the prompt.
5. Run faster than the current record on the reference GPU, by at least 1 %  ([`speedrun/harness/score.py`](speedrun/harness/score.py)).

The maintainers review every PR and may decline a record that moves work into the unscored decode positions, or that makes the framework much harder to read for a small gain. Other than that, anything and everything is fair game (e.g. the backend, kernels, CKKS parameters, packing, the approximations, bootstrap placement...).

## Tasks

Both tracks use [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) (124M
parameters). The weights are fixed. The approximations are part of the challenge: an entry can refit
or replace the polynomials for LayerNorm, softmax, GELU and the argmax, as long as the outputs stay
within the KL bound above.

The current record opens the session (CKKS parameters on the 32-bit chain, keys) with
[`env.py`](python/perseus/examples/gpt2/env.py).
Its approximations are fitted per layer in
[`gpt2_base_n32`](python/perseus/configs/model/approximation/gpt2_base_n32) (Decode-8) and
[`gpt2_base_n32_thor`](python/perseus/configs/model/approximation/gpt2_base_n32_thor) (Prefill-128, with
polynomials from THOR [1]): GELU, LayerNorm and softmax in every block, and CutMax [2] for the argmax.

### Track 1: Prefill-128

The current record's server runs the planned prefill
([`PlannedPrefill`](python/perseus/impl/prefill.py)) with the plan
[`bootstrap_placements/gpt2_prefill128_m16s0_tail`](bootstrap_placements/gpt2_prefill128_m16s0_tail).
[`scripts/repro/gpt2_prefill128.sh`](scripts/repro/gpt2_prefill128.sh) times the same prefill in one process,
through [`run_prefill.py`](python/perseus/examples/gpt2/run_prefill.py).

- [`prefill.py`](python/perseus/examples/gpt2/prefill.py) runs the 128 positions through the 12 blocks, packed as complex chunk
  pairs, and keeps every block's K and V on the GPU.
- The timed pass follows one untimed warm-up on another prompt.
- The speedrun adds the encrypted tail: the last position moves into the decode layout, then the final
  LayerNorm, the LM head and CutMax run as in the decode, as two more stages (`Gpt2Prefill.tail_stages`).
  Perseus's own script runs them with `--tail`, and otherwise computes them in the clear on decrypted rows
  (`plain_logits`).

### Track 2: Decode-8

The current record's server drives Perseus's served decode,
[`perseus.serve.gpt2_decode`](python/perseus/serve/gpt2_decode.py), with the plan
[`bootstrap_placements/gpt2_decode_python_n32_m16i_fcf`](bootstrap_placements/gpt2_decode_python_n32_m16i_fcf).
[`scripts/repro/gpt2_decode.sh`](scripts/repro/gpt2_decode.sh) times the same decode in one process, through
[`run_decode.py`](python/perseus/examples/gpt2/run_decode.py), with per-token times.

- [`model.py`](python/perseus/examples/gpt2/model.py) holds the per-block K/V caches and the decode step: the 12 blocks of
  [`block.py`](python/perseus/examples/gpt2/block.py) on the new position, then the final LayerNorm, the LM head and the CutMax
  encrypted argmax in [`head.py`](python/perseus/examples/gpt2/head.py).
- The inputs are teacher-forced. Positions 0 to 7 fill the cache, and their time is reported but not
  scored. The score is the time of positions 8 to 15, divided by 8.
- Today the script decodes 16 tokens and reports the median of the steady tokens (position ≥ 1).

### Client and server

The harness runs the client and the server as separate processes, each in its own sandbox, and they
talk only through files: the client writes `in/`, the server writes `out/`
([`speedrun/harness/protocol.py`](speedrun/harness/protocol.py)). The server is the entry's `speedrun/serve.sh`, started
with `--track {prefill,decode} --inbox IN --outbox OUT`. It must:

1. write `out/keyspec`, its key request: the keys its plan needs, in its backend's terms. The client generates
   only the keys its backend declares; record 1's refuses, for example, a SPRU block key of other than 64 blocks
   or a sparse secret above the bottom level;
2. wait for `in/keys.done`, then load the client's key bundle from `in/keys/` (it holds no secret key);
3. set up, untimed (the weights from `$WEIGHTS_PATH`, plans, a warm-up on a public prompt), then write
   `out/ready` with what the client needs to encrypt, such as the input level;
4. answer every request k. Prefill-128 reads the encrypted prompt `in/k/0` and writes `out/k/0`, the last
   position's logits and argmax; Decode-8 reads the tokens `in/k/0` to `in/k/15` one at a time, from an
   empty cache, and writes `out/k/j` before token j + 1 is released;
5. exit at `in/quit`.

Every file is an 8-byte length and the payload, written under a temporary name and renamed. The current
record's server is [`speedrun/server.py`](speedrun/server.py).

## Submitting

1. **Develop on any GPU** (the current record requires at least 48 GB of VRAM). Run the current record on the same GPU to compare (see
   [Running the current record](#running-the-current-record)).
2. **Push** every submodule repository your change pins.
3. **Open a PR against this branch** with your change and a new folder `speedrun/records/<YYYY-MM-DD>_<Name>/`,
   copied from [`speedrun/records/_template`](speedrun/records/_template):
   - `README.md`, with a short write-up, an ablation table and a Disclosures section;
   - `submission.json`, with the authors, a one-line summary, the CKKS parameters (`ckks`: log N, log QP,
     the secret, every sparse key) and your numbers next to the record's on the same GPU;
   - your run logs;
   - if your backend has its own key or ciphertext formats, its client, `speedrun/harness/<name>_client.py`,
     whose class `Backend` does the keygen for your server's key request, the encryption and the
     decryption (see [`speedrun/harness/perseus_core_client.py`](speedrun/harness/perseus_core_client.py)), and which pins
     your code's last commit (`PERSEUS = "<sha>"`): add the client and the record folder in a commit after it, and
     nothing else. That commit is built as both your server and your client. It runs as
     `bash speedrun/runner/entry.sh <name> ...`.

   In the PR's description, accept Perseus's [CLA](CLA.md): "I have read the Perseus CLA and agree to it for my
   contributions." Leave the rest of `speedrun/` (`harness/`, `runner/`, `prompts/`, the other records), this
   README, `CLAUDE.md`, `.github/`, the licence and the CLA as they are.
4. **A check runs on the PR**, checking that it leaves those alone, the record folder, the fields, the main key at
   128-bit security on the HE standard, and the client's code: a PR that changes `crates/pc-client/` or
   `python/perseus/client/` must bring its own client. When the check passes, a maintainer is asked to
   review.
5. **The maintainers commit to time it** ([`speedrun/MAINTAINING.md`](speedrun/MAINTAINING.md)) within 72 hours of the check passing,
   at the commit the check names (a new push starts the review again), with this branch's harness,
   against the current record on the reference GPU and the hidden test prompts, run the Lattice Estimator on
   new sparse keys, check the [rules](#rules) and post the results on the PR. An accepted record is merged into
   this branch with a merge commit, so the commit they timed stays in its history, and becomes the one to beat.

## Background

**The NanoGPT speedrun.** This challenge is inspired by the NanoGPT speedrun [3], in which contributors
search, collaboratively and competitively, for the fastest way to train a language model to 3.28
cross-entropy loss on the FineWeb validation set with 8 NVIDIA H100 GPUs. Its record went from the
45 minutes of the llm.c GPT-2 replication to under 40 seconds. We borrow its format: fixed hardware,
a fixed quality bar, a public record history, and rules that leave everything else open. Here the
workload is encrypted inference instead of training, and the bar is 128-bit security and the KL.

**Threat model.** A client $`C`$ wants to prompt the model, and a server $`S`$ runs it. $`C`$ generates the
keys, keeps the secret key $`sk`$ and gives $`S`$ the public and evaluation keys. $`C`$ tokenizes and
encrypts the prompt locally, so $`S`$ never sees the tokens, the activations, the KV cache or the
logits, which it returns encrypted. $`S`$ is honest but curious, i.e. it follows the protocol but may try to
learn about the prompt from what it computes. Its computation must be oblivious, meaning that neither
its control flow nor its memory accesses depend on the prompt. Hardware side channels and metadata
such as the sequence length are out of scope.

**Homomorphic arithmetic.** CKKS [4] is an FHE scheme for
approximate arithmetic on real and complex vectors. If $`c_1 \leftarrow \mathrm{Enc}_{pk}(\mathbf{p}_1)`$
and $`c_2 \leftarrow \mathrm{Enc}_{pk}(\mathbf{p}_2)`$, then

```math
\mathrm{Dec}_{sk}(c_1 \oplus c_2) \approx \mathbf{p}_1 + \mathbf{p}_2, \qquad
\mathrm{Dec}_{sk}(c_1 \otimes c_2) \approx \mathbf{p}_1 \odot \mathbf{p}_2, \qquad
\mathrm{Dec}_{sk}(\mathrm{Rot}_k(c_1)) \approx \pi^k(\mathbf{p}_1),
```

where $`\oplus`$ and $`\otimes`$ are encrypted addition and element-wise multiplication, and
$`\mathrm{Rot}_k`$ rotates the vector cyclically by $`k`$ positions. The server has these three
operations and nothing else.

**Encoding.** Let $`N`$ be a power of two ($`N = 2^{16}`$ here) and $`R = \mathbb{Z}[X]/(X^N + 1)`$. A vector
$`\mathbf{z} \in \mathbb{C}^{N/2}`$ is encoded at a scale $`\Delta`$ as the integer polynomial
$`m = \lfloor p \rceil \in R`$, where $`p`$ takes the value $`\Delta z_j`$ at the root $`\omega_j`$ of
$`X^N + 1`$ and $`\Delta \bar{z}_j`$ at its conjugate. Decoding reads $`z_j \approx m(\omega_j)/\Delta`$.
The $`N/2`$ entries of $`\mathbf{z}`$ are the *slots*. Evaluation at $`\omega_j`$ is a ring homomorphism,
so sums and products of polynomials act on the slots element-wise, and encoding and decoding are an
inverse and a forward discrete Fourier transform.

**Levels.** A ciphertext of $`m`$ is a pair $`(b, a)`$ of polynomials modulo $`Q`$ with
$`b + a s \equiv m + e \pmod{Q}`$, for the secret key $`s`$ and a small noise $`e`$. The product of two
ciphertexts holds its message at scale $`\Delta^2`$, and *rescaling* divides it by $`\Delta`$, which
removes a factor $`\Delta`$ from the modulus. With $`Q_\ell = q_0 \Delta^\ell`$, a fresh ciphertext has
level $`L`$ and each multiplication spends one level, so a circuit deeper than $`L`$ needs a refresh.

**Bootstrapping.** Bootstrapping [5] (for CKKS, [6]) refreshes a ciphertext
whose levels are spent. At level 0, $`b + a s = m + q_0 t`$ for an integer polynomial $`t`$ that depends
on $`s`$. Bootstrapping raises the modulus, moves the coefficients into the slots with a homomorphic
DFT (CoeffToSlot), removes the multiple of $`q_0`$ with a polynomial approximation of a sine (EvalMod)

```math
m_i \approx \frac{q_0}{2\pi} \sin\left(\frac{2\pi\,(m_i + q_0 t_i)}{q_0}\right), \qquad |m_i| \ll q_0,
```

and moves the result back (SlotToCoeff). A bootstrap costs two to three orders of magnitude more than
any other operation, so the depth of the circuit and where its bootstraps go decide most of the
latency.

**Why approximations.** Only additions, multiplications and rotations run under encryption, so every
nonlinear operation is approximated. In GPT-2 these are the $`1/\sqrt{x}`$ of LayerNorm, the
exponential and the division of softmax, GELU and the argmax. Common choices are Goldschmidt
iterations for division, Newton–Raphson iterations for $`1/\sqrt{x}`$ [7], and Chebyshev [8] or Remez
polynomials. An iterative method refines a seed $`F_0(x)`$ into estimates
$`F_1(x), \ldots, F_n(x) \to f(x)`$, and for quadratically convergent methods

```math
\varepsilon_n := \sup_{x \in \mathcal{D}} |F_n(x) - f(x)| \le M\, r^{2^n}, \qquad M > 0,\ r \in (0, 1),
```

where $`\mathcal{D}`$ is the set of inputs whose seed lies in the convergence interval. The error falls
doubly exponentially in $`n`$, but each iteration costs a fixed number of multiplications, so the
depth grows linearly. Every approximation trades accuracy for levels, and so for bootstraps. This
trade is open in the speedrun: an entry can refit any approximation while the KL stays below
$`10^{-2}`$ in prefill and $`10^{-3}`$ in decode.

**Where the latency comes from.** Everything described above adds extra, inefficient computation on very large numbers. The main latency bottlenecks are:

- **Key switching.** Every rotation and every product of two ciphertexts ends with a key switch against an
  evaluation key, and the matrix products need many rotations.
- **Bootstrapping.** Each refresh costs two to three orders of magnitude more than any other operation, and the
  depth of the approximations calls for many of them.
- **Placement.** Where the bootstraps go, and the level each operation runs at, decide how many bootstraps a token
  needs and what each operation costs.
- **Data movement.** A ciphertext takes tens of megabytes, and an evaluation key more, so the GPU is bound by its
  memory bandwidth rather than its arithmetic [9].

**Reference hardware.** Scored attempts will be run on a server with:

- **GPU:** one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).
- **CPU:** 2 x Xeon 6960P.
- **RAM:** 256 GB.

The server runs pinned to the GPU's NUMA node, with its CPU time capped at that node's CPUs and its memory at
160 GB; the client's memory is capped at 64 GB.

**Renting the GPU.** You can rent an RTX PRO 6000 by the hour, for instance on [RunPod](https://www.runpod.io),
[Vast.ai](https://vast.ai) or [Lium](https://lium.io). We are not affiliated with these providers, and none of them
sponsors the speedrun.

# Authors

- **Authors and maintainers:** Alessandro Zirilli ([@alexzilligmm](https://github.com/alexzilligmm)) and Davide
  Marincione ([@Davide-F5-Marincione](https://github.com/Davide-F5-Marincione)).
- **Scientific supervisors:** Evgenios M. Kornaropoulos, Giuseppe Ateniese and Emanuele Rodolà.

# References

[1] Jungho Moon, Dongwoo Yoo, Xiaoqian Jiang, Miran Kim: THOR: Secure Transformer Inference with Homomorphic Encryption. CCS 2025: 3765-3779

[2] Matan Avitan, Moran Baruch, Nir Drucker, Itamar Zimerman, Yoav Goldberg: Efficient Decoding Methods for Language Models on Encrypted Data. IJCNLP-AACL (Findings) 2025: 1777-1794

[3] Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, et al.: modded-nanogpt: Speedrunning the NanoGPT baseline. https://github.com/KellerJordan/modded-nanogpt, 2024

[4] Jung Hee Cheon, Andrey Kim, Miran Kim, Yong Soo Song: Homomorphic Encryption for Arithmetic of Approximate Numbers. ASIACRYPT (1) 2017: 409-437

[5] Craig Gentry: A fully homomorphic encryption scheme. Stanford University, USA 2009

[6] Jung Hee Cheon, Kyoohyung Han, Andrey Kim, Miran Kim, Yongsoo Song: Bootstrapping for Approximate Homomorphic Encryption. EUROCRYPT (1) 2018: 360-384

[7] Hongyuan Qu, Guangwu Xu: Improvements of Homomorphic Secure Evaluation of Inverse Square Root. ICICS 2023: 110-127

[8] Lloyd N. Trefethen: Approximation Theory and Approximation Practice. SIAM 2012, ISBN 978-1-611-97239-9, pp. I-VII, 1-305

[9] Wonkyung Jung, Sangpyo Kim, Jung Ho Ahn, Jung Hee Cheon, Younho Lee: Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-centric Optimization with GPUs. IACR Trans. Cryptogr. Hardw. Embed. Syst. 2021(4): 114-148 (2021)

## License

In `speedrun/`, `harness/`, `runner/`, `prompts/` and `records/` are under [Apache-2.0](speedrun/LICENSE). The rest
of this branch, Perseus with the server files in `speedrun/` (`server.py`, `serve.sh`, `record1.py`,
`make_tail_plan.sh`), is under [BUSL-1.1](LICENSE): free for non-commercial use.

*We will always be grateful to Alessandro Panconesi, whose mentorship, teaching, inspirational words and work led
us here.*
