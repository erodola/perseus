"""Client-generated keys == device-generated keys in function (pc-client, `Session.from_bundle`).

One client secret; session A generates every key on the device from it (`KeySource.generate(secret=)`), the client
generates the same request (`Session.key_manifest`) on the host into a bundle. Gates:
  * switch relation, max |e|: the bundle's keys on the host (client) and A's / B's device keys downloaded (host
    reference), all within the same bound;
  * rotate (every step, hoisted), conjugate, relinearized product: decrypted results of A and of B (a debug session on
    the bundle holding the client's secret) equal the numpy result and each other within the noise;
  * the client -> server -> client path: session C on the bundle holds no secret (decrypt refused); the client
    encrypts, C runs the op on the imported frame, the client decrypts the exported frame.

    tools/gpu.sh out/m12.keys.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_client_keys.py -q -s
"""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

N = 1 << 15
STEPS = [1, 2, 3, -1, -5, 64, 1024, 16384]
GROUPS = [[1, 2, 3]]
LEVELS = [0, 20, 40]
TOL = 1e-6


def rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-300))


@pytest.fixture(scope="module")
def setup():
    cfg = _pc.Config.preset("n32_gpt2_decode")
    secret = _pc.ClientSecret.generate(seed=5)
    client = _pc.Client(secret, seed=6)
    ks = _pc.KeySource.generate(STEPS, groups=GROUPS, layout="packed", seed=7, secret=secret.to_bytes())
    manifest = _pc.Session.key_manifest(cfg, ks)
    tmp = tempfile.mkdtemp(prefix="m12_keys_", dir=os.environ.get("PC_WORK_DIR"))
    path = os.path.join(tmp, "keys.pckb")
    t0 = time.time()
    stats = client.keygen(manifest, path, threads=8)
    print(f"\n[m12] client keygen {time.time() - t0:.2f} s: {dict(stats)}; bundle {_pc.bundle_info(path)}")
    a = _pc.Session(cfg, ks, pool=48, pt_slots=8).complete_setup()
    b = _pc.Session.from_bundle(cfg, ks, path, pool=48, pt_slots=8, debug_secret=secret.to_bytes()).complete_setup()
    c = _pc.Session.from_bundle(cfg, ks, path, pool=48, pt_slots=8).complete_setup()
    yield dict(cfg=cfg, client=client, path=path, a=a, b=b, c=c)
    for s in (a, b, c):
        s.close()
    os.remove(path)
    os.rmdir(tmp)


def test_relations(setup):
    host = setup["client"].check_bundle(setup["path"])
    na, ea = setup["a"].debug_key_relation()
    nb, eb = setup["b"].debug_key_relation()
    print(f"[m12] switch relation max |e|: client bundle {host['max_err']} ({host['keys']} keys, worst {host['worst']}), "
          f"device keygen {ea} ({na} keys), bundle on the device {eb} ({nb} keys)")
    assert host["keys"] == na == nb
    assert 0 < host["max_err"] <= 64 and 0 < ea <= 64 and 0 < eb <= 64


def test_server_holds_no_secret(setup):
    c = setup["c"]
    assert not c.has_secret and not c.fhe.has_secret_key
    x = np.arange(8) * 0.125
    ct = c.fhe.encrypt(x, 10)                                  # public-key encryption of a public value
    with pytest.raises(RuntimeError, match="no secret"):
        c.fhe.decrypt_slots(ct)
    with pytest.raises(RuntimeError):
        c.debug_key_relation()
    got = setup["client"].decrypt(_pc.export_ct(c, ct))[0][:8]
    assert rel(got.real, x) < 1e-8


def ops(fhe, ct, ct2):
    out = {f"rot{s}": fhe.rotate(ct, s) for s in STEPS}
    out["conj"] = fhe.conjugate(ct)
    out["mult"] = fhe.mult(ct, ct2)
    for s, r in zip(GROUPS[0], fhe.rotate_hoisted(ct, GROUPS[0])):
        out[f"hoist{s}"] = r
    return out


def want(x, y):
    w = {f"rot{s}": np.roll(x, -s) for s in STEPS}
    w["conj"] = np.conj(x)
    w["mult"] = x * y
    for s in GROUPS[0]:
        w[f"hoist{s}"] = np.roll(x, -s)
    return w


@pytest.mark.parametrize("level", LEVELS)
def test_function(setup, level):
    rng = np.random.default_rng(level)
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / 2
    y = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / 2
    w = want(x, y)
    res = {}
    for name in ("a", "b"):
        f = setup[name].fhe
        r = ops(f, f.encrypt(x, level), f.encrypt(y, level))
        res[name] = {k: np.asarray(f.decrypt_slots_complex(v)) for k, v in r.items()}
    # the client's own path: encrypt on the host, run on the secret-free session, decrypt on the host
    cl, sc = setup["client"], setup["c"]
    fx, fy = _pc.import_ct(sc, cl.encrypt(x, level)), _pc.import_ct(sc, cl.encrypt(y, level))
    r = ops(sc.fhe, fx, fy)
    res["c"] = {k: cl.decrypt(_pc.export_ct(sc, v))[0] for k, v in r.items()}
    worst = {}
    for k in w:
        e = {n: rel(res[n][k], w[k]) for n in res}
        d_ab = rel(res["a"][k], res["b"][k])
        worst[k] = (max(e.values()), d_ab)
        assert max(e.values()) < TOL, (k, e)
        assert d_ab < TOL, (k, d_ab)
    bits = {n: -np.log2(max(rel(res[n][k], w[k]) for k in w)) for n in res}
    print(f"[m12] level {level}: worst bits device keys {bits['a']:.1f}, bundle keys (debug) {bits['b']:.1f}, "
          f"client->server->client {bits['c']:.1f}; max |A - B| rel {max(v[1] for v in worst.values()):.2e}")
