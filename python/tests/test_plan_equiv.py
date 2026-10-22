"""The baseline-gate contract (perseus.plan.equiv): runtime content
byte-identical, `summary` additive-only."""
import copy
import json

import pytest

from perseus._paths import ASSETS
from perseus.plan import equiv as _equiv

PLAN = ASSETS / "bootstrap_placements" / "gpt2_decode_python_n32" / "block_0_placement.json"


@pytest.fixture(scope="module")
def equiv():
    return _equiv.equivalent


@pytest.fixture
def plan(tmp_path):
    if not PLAN.exists():
        pytest.skip("shipping plan not present")
    ref = json.loads(PLAN.read_text(encoding="utf-8"))

    def write(mutate):
        d = copy.deepcopy(ref)
        mutate(d)
        p = tmp_path / "new.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        return str(p)
    return write


def test_identical_and_additive_summary_pass(equiv, plan):
    assert equiv(plan(lambda d: None), str(PLAN)) == (True, "ok")
    ok, why = equiv(plan(lambda d: d["summary"]["bts_quality"].__setitem__("extra_diagnostic", 3)), str(PLAN))
    assert ok, why


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: d["summary"]["bts_quality"].__setitem__("num_prescale", 99), "summary regressed at summary.bts_quality.num_prescale"),
    (lambda d: d["summary"]["bts_quality"].pop("cf_histogram"), "summary regressed at summary.bts_quality.cf_histogram: missing"),
    (lambda d: d.__setitem__("rescale_after", {}), "runtime content differs: rescale_after"),
    (lambda d: d["placements"].clear(), "runtime content differs: placements"),
    # nested summary dicts are walked recursively
    (lambda d: d["summary"]["final_named_levels"].__setitem__(next(iter(d["summary"]["final_named_levels"])), -1),
     "summary regressed at summary.final_named_levels."),
    (lambda d: d["summary"]["bts_quality"]["cf_histogram"].clear(), "summary regressed at summary.bts_quality.cf_histogram."),
    # leaves are type-strict: True == 1 in Python, not here
    (lambda d: d["summary"].__setitem__("num_placements", True), "summary regressed at summary.num_placements"),
])
def test_regressions_fail(equiv, plan, mutate, expect):
    ok, why = equiv(plan(mutate), str(PLAN))
    assert not ok
    assert expect in why
