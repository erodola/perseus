"""Checks for speedrun/harness/check_pr.py on throwaway git repositories: an entry, a plain PR, and each way a PR
fails.

    python3 -I speedrun/harness/test_check_pr.py -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_pr  # noqa: E402

GOOD = {"name": "Fast", "tracks": ["decode"], "authors": [{"name": "A", "github": "a"}], "summary": "s",
        "dev_result": {}, "ckks": {"log_n": 16, "log_qp": 1706, "secret": "uniform-ternary",
                                   "sparse_keys": [{"role": "encapsulation", "hamming_weight": 32}]}}
FOLDER = "speedrun/records/2026-11-01_Fast"


class CheckPrTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        for c in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            subprocess.run(["git", *c], check=True)
        for path in ("speedrun/harness/perseus_core_client.py", "speedrun/records/2026-10-22_PerseusCore/README.md",
                     "speedrun/server.py", "crates/pc-client/src/lib.rs", "python/perseus/model.py"):
            self.write(path, "x\n")
        self.write("speedrun/records/2026-10-22_PerseusCore/submission.json", json.dumps(GOOD))
        self.base = self.commit()

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def write(self, path, text):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def commit(self):
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-q", "-m", "c"], check=True)
        return check_pr.git("rev-parse", "HEAD").strip()

    def record(self, sub=GOOD, readme="# Fast\n\n## Disclosures\n\nnone\n", folder=FOLDER):
        self.write(f"{folder}/README.md", readme)
        self.write(f"{folder}/submission.json", json.dumps(sub))

    def run_check(self, author="", body="I have read the Perseus CLA and agree to it for my contributions."):
        self.commit()
        return check_pr.check(self.base, "HEAD", author, body)

    def test_an_entry_with_its_code_passes_and_names_its_commit(self):
        self.record()
        self.write("python/perseus/model.py", "faster\n")
        self.write("speedrun/server.py", "faster\n")
        errors, notes, entry = self.run_check()
        self.assertEqual((errors, entry), ([], True))
        self.assertIn(f"time commit `{check_pr.git('rev-parse', 'HEAD').strip()}`", notes[0])
        self.assertTrue(any("Lattice Estimator" in n for n in notes))
        self.assertTrue(any("of record `2026-10-22_PerseusCore`, no new estimate" in n for n in notes))

    def test_an_entry_must_accept_the_cla(self):
        self.record()
        self.assertTrue(any("CLA" in e for e in self.run_check(body="Faster GELU.")[0]))
        errors, notes, _ = check_pr.check(self.base, "HEAD")          # run locally: no description, a reminder
        self.assertEqual(errors, [])
        self.assertTrue(any("CLA" in n for n in notes))

    def test_new_keys_are_flagged_for_the_estimator(self):
        self.record({**GOOD, "ckks": {**GOOD["ckks"], "log_qp": 1600}})
        self.assertTrue(any(n.startswith("keys: new or changed") for n in self.run_check()[1]))

    def test_a_plain_pr_is_not_an_entry(self):
        self.write("python/perseus/model.py", "fix\n")
        self.assertEqual(self.run_check(), ([], [], False))

    def test_touching_the_harness_fails_in_any_pr(self):
        self.write("speedrun/harness/perseus_core_client.py", "evil\n")
        self.assertTrue(any("perseus_core_client.py" in e for e in self.run_check()[0]))
        self.record()
        self.write(".github/workflows/x.yml", "evil\n")
        self.write("README.md", "new rules\n")
        self.write("CLAUDE.md", "new rules\n")
        errors = self.run_check()[0]
        self.assertTrue(all(any(p in e for e in errors) for p in (".github", "`README.md`", "`CLAUDE.md`")))

    def test_a_maintainer_may_touch_the_harness(self):
        self.write("speedrun/runner/run.py", "better\n")
        errors, notes, _ = self.run_check(author=check_pr.MAINTAINERS[0])
        self.assertEqual(errors, [])
        self.assertTrue(any("maintainer" in n for n in notes))

    def test_adding_to_an_existing_record_fails(self):
        self.record()
        self.write("speedrun/records/2026-10-22_PerseusCore/evil.md", "x\n")
        self.assertTrue(any("2026-10-22_PerseusCore/evil.md" in e for e in self.run_check()[0]))

    def test_renaming_into_the_harness_fails(self):
        self.record()
        subprocess.run(["git", "mv", "python/perseus/model.py", "speedrun/harness/model.py"], check=True)
        self.assertTrue(any("speedrun/harness/model.py" in e for e in self.run_check()[0]))

    def test_two_record_folders_fail(self):
        self.record()
        self.record(folder="speedrun/records/2026-11-02_Faster")
        self.assertTrue(any("2 record folders" in e for e in self.run_check()[0]))
        errors, _, entry = check_pr.check(self.base, "HEAD", check_pr.MAINTAINERS[0])   # a maintainer adding records
        self.assertEqual((errors, entry), ([], False))

    def test_log_qp_over_the_bound_fails(self):
        self.record({**GOOD, "ckks": {**GOOD["ckks"], "log_qp": 1800}})
        self.assertTrue(any("1800 > 1747" in e for e in self.run_check()[0]))

    def test_missing_fields_and_disclosures_fail(self):
        self.record({k: v for k, v in GOOD.items() if k != "ckks"}, readme="# Fast\n")
        errors = self.run_check()[0]
        self.assertTrue(any("Disclosures" in e for e in errors))
        self.assertTrue(any("ckks" in e for e in errors))

    def test_changing_the_clients_code_needs_a_client(self):
        self.record()
        self.write("crates/pc-client/src/lib.rs", "new format\n")
        self.assertTrue(any("your own client" in e for e in self.run_check()[0]))

    def test_a_client_pinned_to_a_commit_of_the_pr_passes(self):
        self.write("crates/pc-client/src/lib.rs", "new format\n")
        code = self.commit()
        self.record()
        self.write("speedrun/harness/rust_client.py", f'PERSEUS = "{code}"\n')
        errors, notes, _ = self.run_check()
        self.assertEqual(errors, [])
        self.assertTrue(any("rust_client.py" in n for n in notes) and any("pc-client" in n for n in notes))
        self.assertIn(f"time commit `{code}`", notes[0])

    def test_a_client_pinned_outside_the_pr_fails(self):
        self.record()
        self.write("speedrun/harness/rust_client.py", f'PERSEUS = "{self.base}"\n')
        self.assertTrue(any("not a commit of this PR" in e for e in self.run_check()[0]))

    def test_a_client_pinned_before_the_last_client_change_fails(self):
        self.write("crates/pc-client/src/lib.rs", "new format\n")
        code = self.commit()
        self.write("crates/pc-client/src/lib.rs", "newer format\n")
        self.record()
        self.write("speedrun/harness/rust_client.py", f'PERSEUS = "{code}"\n')
        self.assertTrue(any("may only add the record folder" in e for e in self.run_check()[0]))

    def test_a_client_file_outside_an_entry_fails(self):
        self.write("speedrun/harness/rust_client.py", f'PERSEUS = "{self.base}"\n')
        self.assertTrue(any("rust_client.py" in e for e in self.run_check()[0]))

    def test_a_framing_change_is_only_noted(self):
        self.record()
        self.write("speedrun/record1.py", "pack\n")
        errors, notes, _ = self.run_check()
        self.assertEqual(errors, [])
        self.assertTrue(any("record1.py" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
