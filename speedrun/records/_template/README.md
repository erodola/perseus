# <Record name>

<One paragraph: what changed and why it is faster.>

## Changes against record #N

- <change>, in `python/perseus/<path>`, `crates/<path>` or a submodule, `<submodule>/<path>`

## Ablation

| Change | Track | Δ score on the dev set (same GPU) |
|---|---|---|
| <change> | <track> | <Δ> |

## Dev result

<Per track: GPU, score of this entry and of the current record on the same GPU, mean KL. Logs in `dev_logs/`.>

## Disclosures

<Everything a reviewer should read here rather than find in the diff. Examples: new untimed work,
changed approximations or plans, changed CKKS parameters or key distributions (the maintainer runs the
estimator), GPT-2-specific constants and why they would carry to other models. Write
"none" if there is nothing.>
