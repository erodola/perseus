"""Make the prompt sets from C4 (maintainers, once): the public dev set and the hidden test set.

    python runner/make_prompts.py --shard C4_ARROW --n 32 --seed 0 --out prompts/dev.json
    python runner/make_prompts.py --shard C4_ARROW --n 256 --out /private/test.json     # a secret seed

A prompt is the first 128 GPT-2 tokens of a C4 document (allenai/c4 en, train), the documents taken in a seeded
random order and kept when they have 128 tokens or more. Decode-8 uses a prompt's first 16 tokens. The dev and test
sets come from different shards, so they share no document; the approximations were calibrated on OpenWebText,
which neither uses. Prints the file's SHA-256, which the repository records for the hidden set (prompts/test.sha256).
Needs `datasets` and `transformers`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True, help="a C4 en train shard (.arrow) from the HuggingFace cache")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=None, help="default: a secret one from os.urandom")
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little")
    docs = Dataset.from_file(a.shard)
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    prompts = []
    for i in np.random.default_rng(seed).permutation(len(docs)):
        ids = tok.encode(docs[int(i)]["text"])
        if len(ids) >= a.length:
            prompts.append(ids[:a.length])
            if len(prompts) == a.n:
                break
    data = {"source": f"allenai/c4 en train, {os.path.basename(a.shard)}", "tokenizer": "openai-community/gpt2",
            "length": a.length, "seed": a.seed, "prompts": prompts}
    body = json.dumps(data).encode()
    with open(a.out, "wb") as f:
        f.write(body)
    print(f"{a.out}: {len(prompts)} prompts, sha256 {hashlib.sha256(body).hexdigest()}")


if __name__ == "__main__":
    main()
