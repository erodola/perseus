"""Checks for speedrun/runner/plan_diff.py on a throwaway git repository: a changed plan field, and a summary-only
difference.

    python3 -I speedrun/runner/test_plan_diff.py -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class PlanDiffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.git("init", "-q")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", "-C", self.tmp.name, "-c", "user.email=t@t", "-c", "user.name=t", *args],
                              check=True, capture_output=True, text=True).stdout.strip()

    def plan(self, name, cf, solve_s):
        os.makedirs(os.path.join(self.tmp.name, name), exist_ok=True)
        with open(os.path.join(self.tmp.name, name, "block_0_placement.json"), "w") as f:
            json.dump({"correction_factor": {"v_1": cf}, "summary": {"placer_meta": {"solve_s": solve_s}}}, f)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", name)
        return self.git("rev-parse", "HEAD")

    def run_diff(self, a, b):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "plan_diff.py"), a, "old", b, "new"],
                              cwd=self.tmp.name, capture_output=True, text=True)

    def test_a_changed_field_is_printed(self):
        a, b = self.plan("old", 10, 1.0), self.plan("new", 2, 1.5)
        p = self.run_diff(a, b)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn(".correction_factor.v_1: 10 -> 2", p.stdout)
        self.assertIn("1 plan fields differ, 1 summary fields", p.stdout)

    def test_a_summary_only_difference_passes(self):
        a, b = self.plan("old", 10, 1.0), self.plan("new", 10, 1.5)
        p = self.run_diff(a, b)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertIn("plan fields that differ: 0", p.stdout)


if __name__ == "__main__":
    unittest.main()
