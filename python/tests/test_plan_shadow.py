"""The planner's Shadow adapter (perseus.plan.placer.sim.PlannerShadow): coordinates through pc-shadow's
conversions, route landings from the recipe, the recipe knobs the sim reads, and `simulate` running its per-op rules
through `_pc.Shadow.step`."""
import pytest

from perseus import _pc
from perseus.plan.placer.ir import Graph
from perseus.plan.placer.sim import Budget, PlannerShadow, simulate

RECIPE = {"PLAN_SPARSE_BTS_OUT": "1:16,512:26", "PLAN_STC_FIRST_ENTRY": "46", "PLAN_STC_FIRST_FALLBACK": "34:1"}


def _sh(env=None, bl=28, unit=2):
    return PlannerShadow(bl, unit, _pc.PlanConfig.from_legacy_env(env if env is not None else RECIPE))


def test_coordinates_round_trip():
    sh = _sh()
    for c in (-12.0, 0.0, 6.0, 30.0):
        m = sh.meta(c, 2)
        assert (m.level, m.deg) == (28 + int(c), 2) and sh.consumed(m.level) == c
    with pytest.raises(_pc.LevelError):
        sh.meta(1.5, 1)                         # a fractional depth has no ciphertext meaning
    with pytest.raises(ValueError):
        PlannerShadow(27.5, 2)


def test_landings_are_the_recipe():
    sh = _sh()
    assert sh.landing(0) == (28, 1) and sh.landing(512) == (26, 1) and sh.landing(1) == (16, 1)
    assert sh.landing(256) is None              # no measured landing: the planner's "assume dense" case
    assert sh.shipped_landing() == (34, 1)      # BtsConfig.dense_shipped
    assert sh.dense_fallback() == (46.0, 6.0, 1)
    bare = _sh({})
    assert bare.routes == (0,) and bare.dense_fallback() is None and not bare.has_shipped
    assert _sh({"PLAN_STC_FIRST_ENTRY": "46"}).dense_fallback() is None   # entry without a fallback landing


def test_kv_landing_is_checked_against_the_shipped_order():
    assert _sh({**RECIPE, "PLAN_STC_FIRST_KV_LANDING": "32:2"}).kv_landing == (4.0, 2)
    with pytest.raises(ValueError, match="stc_first_kv_landing"):
        _sh({**RECIPE, "PLAN_STC_FIRST_KV_LANDING": "30:2"})
    with pytest.raises(ValueError, match="stc_first_fallback"):
        _sh({"PLAN_STC_FIRST_KV_LANDING": "32:2"})


def _node(op, inputs, output, ilv=None, olv=None, step="s"):
    return {"op_type": op, "inputs": inputs, "input_levels": ilv or [None] * len(inputs), "output": output,
            "output_level": olv, "step": step}


def test_simulate_rules_come_from_the_shadow():
    # x -> x*x (deg 2 at the effective level) -> + pt (AddPtRealizes: realized) -> rotate (pass-through) -> negate
    nodes = [_node("mult", ["x", "x"], "a"), _node("add", ["a", "pt_w"], "b"),
             _node("rotate", ["b", "rot(3)"], "c"), _node("negate", ["c"], "d")]
    g = Graph.from_nodes(nodes, level_unit=2)
    sh = _sh()
    sim = simulate(g, bootstrap_level=28.0, budget=Budget(L=22.0, unit=2), seed_consumed=lambda v: 2.0,
                   seed_deg=lambda v: 1, refreshed={}, shadow=sh)
    assert [(sim.consumed[v], sim.deg[v]) for v in "abcd"] == [(2.0, 2), (4.0, 1), (4.0, 1), (4.0, 2)]
    # the same node through the binding directly
    out = sh.shadow.step(sh.node(g.nodes[0]), [_pc.CtMeta(30, 1), _pc.CtMeta(30, 1)]).out
    assert (out.level, out.deg) == (30, 2)
    with pytest.raises(ValueError, match="does not match"):
        simulate(g, bootstrap_level=34.0, budget=Budget(L=16.0, unit=2), seed_consumed=lambda v: 0.0,
                 seed_deg=lambda v: 1, refreshed={}, shadow=sh)


def test_refresh_planner_default_shadow_is_its_route_table():
    from perseus.plan import btserr
    from perseus.plan.placer.refresh import RefreshPlanner
    g = Graph.from_nodes([_node("mult", ["x", "x"], "a")], level_unit=2)
    rp = RefreshPlanner(table=btserr.AccuracyTable(), policy=btserr.SitePolicy(), graph=g,
                        sparse_out_levels={512: 26}, bootstrap_level=28)
    assert rp.shadow.routes == (0, 512) and rp.shadow.landing(512) == (26, 1)
