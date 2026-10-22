"""BERT-base SST-2 planned on the new runtime: every block a compiled `Program` run in Rust
(`examples.bert.planned.BertPlannedPrefill`), the PCMM linears and the CCMM attention `Finish` ops served by the
session's matmul services; the 128-token text vs the float64 model.

Gates (the full gate is the paired `run_bert --text 128 --plan bootstrap_placements/bert_sst2_128`):
label agrees with float64, KL <= 1.2e-9 (2-class softmax), device free memory flat.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

PLAN = "bootstrap_placements/bert_sst2_128"


@pytest.mark.skipif(not os.environ.get("HF_HOME"), reason="HF_HOME with the BERT export / tokenizer needed")
def test_bert_planned_text128(tmp_path, monkeypatch):
    from perseus.examples.bert import run_bert
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "0")
    got = []
    rc = run_bert.main(["--text", "128", "--plan", PLAN, "--repeat", "2", "--work-dir", str(tmp_path / "w")],
                       results=got.append)
    assert rc == 0 and len(got) == 1
    r = got[0]
    assert r["agree"] == 1, r
    assert r["kl"] <= 1.2e-9, r["kl"]
    assert len(set(r["free_gb"])) == 1, r["free_gb"]
