# Maintaining the speedrun

What the maintainers (`MAINTAINERS` in [`harness/check_pr.py`](harness/check_pr.py), and
[`.github/CODEOWNERS`](../.github/CODEOWNERS)) do when an entry's PR arrives. Commands run from the root of a clone of
upstream `main`, your harness: never run the harness from an entry's checkout.

The machine provides what this folder cannot:

- an isolated place to build and run entries' code: an account without your credentials, or a VM with the GPU passed
  through; `runner/build.sh` and `runner/official.sh` run there;
- the hidden test set, outside every checkout (its SHA-256 is [`prompts/test.sha256`](prompts/test.sha256));
- the reference GPU, idle, and its NUMA node;
- cargo (rustup), uv and CUDA 13.2, for the builds.

## Once

- Repository settings: protect `main`, require a code owner's review, allow the Speedrun PR check workflow.
- Build the commit the record's client pins (`PERSEUS` in [`harness/perseus_core_client.py`](harness/perseus_core_client.py)),
  with GPT-2's weights, and the current record if it is another commit:
  `EXPORT=1 bash speedrun/runner/build.sh https://github.com/gladia-research-group/perseus <commit> <client build>`.
  While record 1 holds, the client build is also the record's.

## When a PR arrives

1. **Read the check's report** on the PR. It names the commit to time, says whether the keys need the Lattice
   Estimator, and lists client-side changes. A new push runs it again: review and time only the commit it names.
2. **Review the diff** against the [rules](README.md#rules): computation on ciphertexts only, oblivious execution,
   every input-dependent step inside the clock, no work moved into decode positions 0-7, the weights read from
   `$WEIGHTS_PATH`, Disclosures that match the diff. A new client file is security code: it holds the secret key.
3. **Build the commit**, in the isolated place:
   `bash speedrun/runner/build.sh https://github.com/gladia-research-group/perseus <commit> <entry build>`.
4. **Time it**, for each track the entry claims:
   `PROMPTS=<hidden test set> CLIENT=<client build> NUMA=<the GPU's node> bash speedrun/runner/official.sh decode
   <GPU> <record build> <entry build>` (and `prefill`). An entry with its own client adds `CLIENT_FILE=<the client
   file of its PR>`. At record 1's speed a track takes about 1.7 h (Decode-8) or 1 h (Prefill-128): 6 pairs of 8
   prompts. Pass the same `OFFSET` to both tracks; change it between PRs to spread them over the hidden set.
5. **If the check flagged new keys**, run the Lattice Estimator on the declared sparse keys: each must reach 128 bits
   at the modulus where it is used.
6. **Post the results**: `gh pr comment <PR> --body-file <OUT>/report.md`, with the estimator's result and the review.
7. **Accept** when the quality test passes, the speed test says NEW RECORD, the review finds no broken rule and the
   keys hold: merge the PR with a merge commit, so the commit you timed stays in `main`'s history. Then, in a
   maintainer commit: the record's rows in the two tables of [`README.md`](README.md#records) and in
   [`records/progress.csv`](records/progress.csv), `python3 speedrun/records/plot.py`, and "Running the current
   record" if the record's commit or its client's pin changed (rebuild the record, step "Once").

Keep the `run.json` files of official runs; post only `report.md`.
