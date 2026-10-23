# Entering the Encrypted GPT-2 Speedrun: instructions for Claude

You are helping someone make an entry: a change to Perseus that runs encrypted GPT-2 small faster than the current
record, at the same security and accuracy. `speedrun/README.md` is the rulebook. When this file and the README
disagree, the README wins. Paths below are from the root of the Perseus checkout.

## Setup

The entrant's fork of Perseus (the code being changed; it also holds this harness, in `speedrun/`), and next to it a
build of the commit `speedrun/harness/perseus_core_client.py` pins (`PERSEUS`), as a worktree.

```bash
git clone https://github.com/<handle>/perseus && cd perseus     # a fork of gladia-research-group/perseus
git remote add upstream https://github.com/gladia-research-group/perseus
git worktree add ../perseus-client <the PERSEUS sha>
for d in . ../perseus-client; do (cd $d &&
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[hf,examples,dev]" &&
  .venv/bin/maturin develop --release &&                        # the Rust + CUDA runtime
  .venv/bin/perseus-export --model openai-community/gpt2 --out .cache/models/openai-community/gpt2); done
```

`../perseus-client` is the client of every run (the client holds the secret key, so the harness runs it only at its
pinned commit, never on the code you are changing) and, today, also the current record.

You need Linux, `bwrap`, a systemd user session, a GPU with 48 GB or more, and `HF_HOME` holding the
`openai-community/gpt2` weights.

## The loop

1. Time the current record first, on the GPU you will use:
   `CLIENT=../perseus-client bash speedrun/runner/entry.sh perseus-core <the record's checkout> decode 0` (and
   `prefill`); the record's checkout is `../perseus-client` while record 1 holds. Each run writes
   `speedrun/runs/<run>/run.json`. Time the entry the same way, with `.` as the checkout.
2. Make one change at a time: kernels in `crates/`, Python in `python/perseus/`, plans in `bootstrap_placements/`,
   approximations, CKKS parameters. The server is `speedrun/server.py`. It must keep speaking the harness protocol
   (`speedrun/harness/protocol.py`, summarized in the README's "Client and server"). Rebuild after each change.
3. Time the change through the harness, never with an in-process script. Alternate the record and the entry on the
   same idle GPU (record, entry, entry, record, ...). On a machine with several NUMA nodes, pin the server with
   `NUMA=NODE[:MEMS]`: as in the official runs, its CPU time is then capped at that node's CPUs.
4. Score: `python3 speedrun/harness/score.py --track decode --entry speedrun/runs/<entry>/run.json ... --record
   speedrun/runs/<record>/run.json ...`. It reports the KL test and whether the entry beats the record by the margin
   (Rule 5). Keep only changes that pass both.
5. Write each change and its effect into the ablation table as you go.

## Rules you must not break

- Do not change the weights, the prompts, or anything in `speedrun/harness/`, `speedrun/runner/`,
  `speedrun/prompts/`, the existing records, the rulebook, `.github/` or the licence. The harness owns the inputs,
  the KL and argmax checks and the clock. The server reads the weights from `$WEIGHTS_PATH`.
- 128-bit security for every key. Keep log QP within the HE standard's bound for the ring dimension (N = 2^16:
  1747). Do not change a secret's distribution or Hamming weight without saying so in Disclosures; the maintainers
  run the Lattice Estimator on it.
- The server computes on ciphertexts only. It never holds the secret key and never sees a token, an activation or
  a logit in the clear.
- Oblivious execution: no control flow or memory access may depend on the prompt.
- Every input-dependent step runs inside the clock. Setup (keys, weights, plans, warm-up on a public prompt) is
  untimed and must not depend on the prompt.
- Do not move work into the unscored decode positions 0-7, and do not tune anything to the dev prompts. The
  official runs use a hidden test set.
- Never report a number the harness did not print. Keep the `run.json` files.

## Submitting

1. Push every submodule repository the change pins.
2. Copy `speedrun/records/_template` to `speedrun/records/<YYYY-MM-DD>_<Name>/` and fill in:
   - `README.md`: what changed and why it is faster, the ablation table, the dev results, and Disclosures
     (new untimed work, changed approximations, plans, parameters or keys; "none" if nothing);
   - `submission.json`: the authors, a one-line summary, the `ckks` block (log N, log QP, the secret, every sparse
     key), `ai_assistance` (be accurate: say that you were used, and roughly which share of the changed lines you
     wrote), and the dev numbers next to the record's on the same GPU;
   - `dev_logs/`: the `run.json` files.
3. Only if the entry has its own key or ciphertext formats: add one file, `speedrun/harness/<name>_client.py`, whose
   class `Backend` generates the keys the server's key request asks for, encrypts the inputs and decrypts the outputs
   (see `speedrun/harness/perseus_core_client.py`), and that pins the code's last commit, `PERSEUS = "<sha>"`; the
   client file and the record folder go in a later commit, with nothing else. It runs as
   `bash speedrun/runner/entry.sh <name> ...`. The PR check requires it when the PR changes the client's code
   (`crates/pc-client/`, `python/perseus/client/`).
4. Before opening the PR, run the same check the PR gets:
   `python3 speedrun/harness/check_pr.py --base upstream/main --head HEAD`.
5. Open the PR against `gladia-research-group/perseus` `main`, with "I have read the Perseus CLA and agree to it for
   my contributions." in its description (`CLA.md`; only the entrant can agree to it). The maintainers time the commit the check names, on
   the reference GPU and the hidden prompts, and post the results.
