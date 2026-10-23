"""Checks for harness/score.py: the t distribution, the quality gate and the record rule on synthetic runs.

    python3 -I harness/test_score.py -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score  # noqa: E402


def run(times, kls, argmax=True, track="decode", sha="dev", offset=0, harness="a" * 40):
    steps = 16 if track == "decode" else 1
    reqs = [{"warmup": False, "score": t, "kl_scored_mean": k, "argmax_ok": [argmax] * steps, "prompt": offset + i}
            for i, (t, k) in enumerate(zip(times, kls))]
    return {"run": "x", "void": None, "harness": harness, "result": {"prompts_sha256": sha, "requests": reqs}}


class ScoreTest(unittest.TestCase):
    def test_t_tail_matches_known_values(self):
        for t, df, want in ((3.365, 5, 0.01), (2.015, 5, 0.05), (2.326, 10**6, 0.01), (0.0, 7, 0.5)):
            self.assertAlmostEqual(score.t_sf(t, df), want, delta=2e-4)
        self.assertAlmostEqual(score.t_sf(-2.015, 5), 0.95, delta=2e-4)

    def test_quality_passes_under_the_bar_and_fails_over_it(self):
        under = [run([3.4] * 4, [4e-4, 5e-4, 6e-4, 4.5e-4]) for _ in range(2)]
        over = [run([3.4] * 4, [9e-4, 1.2e-3, 1.1e-3, 9.5e-4]) for _ in range(2)]
        self.assertTrue(score.quality("decode", under)["pass"])
        self.assertFalse(score.quality("decode", over)["pass"])
        self.assertTrue(score.quality("prefill", [run([22.0] * 4, [2.7e-3] * 3 + [2.8e-3], track="prefill")])["pass"])

    def test_quality_needs_the_argmax(self):
        self.assertFalse(score.quality("decode", [run([3.4] * 4, [4e-4] * 4, argmax=False)])["pass"])

    def test_record_rule(self):
        record = [run([3.99 + 0.01 * i], [4e-4]) for i in range(6)]
        faster = [run([3.38 + 0.01 * i], [4e-4]) for i in range(6)]
        same = [run([3.99 + 0.01 * ((i * 7) % 6)], [4e-4]) for i in range(6)]
        s = score.speed(faster, record)
        self.assertTrue(s["new_record"])
        self.assertAlmostEqual(s["speedup"], math.exp(sum(math.log((3.99 + 0.01 * i) / (3.38 + 0.01 * i))
                                                           for i in range(6)) / 6), places=9)
        self.assertFalse(score.speed(same, record)["new_record"])

    def test_a_pair_must_share_its_prompts(self):
        record = [run([3.99], [4e-4], offset=i) for i in range(2)]
        with self.assertRaises(ValueError):
            score.speed([run([3.38], [4e-4], offset=0), run([3.38], [4e-4], offset=0)], record)
        with self.assertRaises(ValueError):
            score.speed([run([3.38], [4e-4], offset=i, sha="test") for i in range(2)], record)

    def test_official_runs_are_on_the_hidden_set_from_one_harness(self):
        with open(score.TEST_SHA) as f:
            test = f.read().split()[0]
        score.official([run([3.4], [4e-4], sha=test) for _ in range(2)])
        for bad in ([run([3.4], [4e-4])], [run([3.4], [4e-4], sha=test, harness="a" * 40 + "+dirty")],
                    [run([3.4], [4e-4], sha=test), run([3.4], [4e-4], sha=test, harness="b" * 40)]):
            with self.assertRaises(ValueError):
                score.official(bad)

    def test_void_runs_are_refused(self):
        bad = run([3.4], [4e-4])
        bad["void"] = "server exited"
        with self.assertRaises(ValueError):
            score.quality("decode", [bad])


if __name__ == "__main__":
    unittest.main()
