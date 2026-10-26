"""Every default plan of the example drivers resolves to a plan dir with per-block placements."""
import os

from perseus._paths import resolve
from perseus.examples.defaults import PLANS, default_plan


def test_default_plans_exist():
    for model, table in PLANS.items():
        for t, plan in table.items():
            d = resolve(plan)
            assert os.path.isdir(d), (model, t, plan)
            assert any(f.startswith("block_") for f in os.listdir(d)), (model, t, plan)


def test_shape_lookup():
    assert default_plan("gpt2_prefill", 128).endswith("gpt2_prefill128_m16s0")
    assert default_plan("gpt2_prefill", 64) is None
    assert default_plan("gpt2_decode", 16).endswith("_kc16_cm")
    assert default_plan("gpt2_decode", 64).endswith("_m16i_fcf_cm")
    assert default_plan("nope") is None
