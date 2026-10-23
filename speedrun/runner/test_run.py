"""Checks for runner/run.py: whole sessions with the client and harness/dummy_server.py each sandboxed, and what the
runner refuses before any session: a server checkout linking outside the sandbox, a client checkout off its pin.

    python3 -I runner/test_run.py -v
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")
DUMMY = ["python3", "-I", os.path.join(HARNESS, "dummy_server.py")]
SYSCALLS = """
import ctypes, sys
f = ctypes.CDLL(None, use_errno=True).syscall
calls = {"io_uring_setup": (425, 1, 0), "keyctl": (250, 0, -3, 0), "userfaultfd": (323, 1)}
refused = [n for n, c in calls.items() if f(*c) == -1 and ctypes.get_errno() == 1]
print(" ".join(refused))
if sys.argv[1:] == ["x32"]:
    f(39 | 0x40000000)          # getpid through the x32 ABI: the filter kills the process
    print("x32 allowed")
"""


def wrapped(script):
    """A server command that runs `script` (sh) in the sandbox, then the dummy server."""
    return ["sh", "-c", f'{script} && exec "$0" "$@"', *DUMMY]


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = os.path.join(self.tmp.name, "runs")

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, track, server, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "run.py"), "--track", track,
                               "--runs", self.runs, "--mem", "2G", "--time", "120", "--ro", HARNESS,
                               *args, "--", *server], capture_output=True, text=True, timeout=60)

    def run_track(self, track, server, *args):
        p = self.start(track, server, *args)
        (run,) = os.listdir(self.runs)
        with open(os.path.join(self.runs, run, "run.json")) as f:
            return p.returncode, json.load(f)

    def checkout(self, link_to=None):
        """A server checkout, with a top-level symlink to `link_to`."""
        path = os.path.join(self.tmp.name, "entry")
        os.makedirs(path, exist_ok=True)
        if link_to:
            os.symlink(link_to, os.path.join(path, "data"))
        return path

    def assert_clean(self, rc, report, requests):
        self.assertEqual(rc, 0, report)
        self.assertIsNone(report["void"])
        self.assertEqual(len(report["result"]["requests"]), requests)
        self.assertTrue(all(r["outputs_ok"] for r in report["result"]["requests"]))
        self.assertEqual(os.listdir("/dev/shm").count(f"speedrun-{report['run']}"), 0)

    def test_decode_in_sandboxes(self):
        rc, report = self.run_track("decode", [*DUMMY, "--delay", "0.01"], "--requests", "2", "--warmup", "1")
        self.assert_clean(rc, report, 3)
        self.assertEqual([r["warmup"] for r in report["result"]["requests"]], [True, False, False])
        self.assertEqual(len(report["result"]["requests"][0]["times_s"]), 16)
        self.assertLess(report["server_exit_s"], 10)

    def test_prefill_in_sandboxes(self):
        rc, report = self.run_track("prefill", [*DUMMY, "--delay", "0.05"], "--requests", "3")
        self.assert_clean(rc, report, 3)
        self.assertGreaterEqual(min(r["score"] for r in report["result"]["requests"]), 0.05)

    def test_server_cannot_write_its_inbox(self):
        rc, report = self.run_track("decode", [*DUMMY, "--misbehave", "poke"])
        self.assert_clean(rc, report, 1)

    def test_lingering_server_voids_the_run(self):
        rc, report = self.run_track("prefill", [*DUMMY, "--misbehave", "linger"], "--bound", "1")
        self.assertEqual(rc, 1)
        self.assertIn("still running", report["void"])

    def test_a_link_out_of_the_server_checkout_is_refused(self):
        private = os.path.join(self.tmp.name, "private")
        os.makedirs(private)
        p = self.start("decode", DUMMY, "--ro", self.checkout(private))
        self.assertEqual(p.returncode, 2)
        self.assertIn("links to", p.stderr)
        self.assertFalse(os.path.exists(self.runs))

    def test_allow_lets_a_link_through(self):
        private = os.path.join(self.tmp.name, "private")
        os.makedirs(private)
        rc, report = self.run_track("decode", DUMMY, "--ro", self.checkout(private), "--allow", private)
        self.assert_clean(rc, report, 1)

    def test_the_server_reads_the_weights_at_weights_path(self):
        weights = os.path.join(self.tmp.name, "weights.bin.zip")
        with open(weights, "w") as f:
            f.write("w")
        rc, report = self.run_track("decode", wrapped('test "$(cat "$WEIGHTS_PATH")" = w'), "--weights", weights)
        self.assert_clean(rc, report, 1)

    def test_what_a_sandbox_leaves_is_capped(self):
        big = "head -c 3000000 /dev/zero"
        rc, report = self.run_track("prefill", wrapped(f"{big} > /out/big && {big}"), "--keep", "1000000")
        self.assert_clean(rc, report, 1)
        kept = [os.path.join(self.runs, report["run"], *f) for f in (["server.log"], ["server", "big"])]
        self.assertLessEqual(sum(os.path.getsize(f) for f in kept if os.path.exists(f)), 1000000)

    def test_the_client_checkout_must_be_at_its_pin(self):
        repo = self.checkout()
        for c in (["init", "-q"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty",
                                   "-m", "c"]):
            subprocess.run(["git", "-C", repo, *c], check=True)
        p = self.start("decode", DUMMY, "--perseus", repo, "--backend", "perseus-core", "--prompts", repo,
                       "--ref", repo)
        self.assertEqual(p.returncode, 2)
        self.assertIn("runs Perseus 89bc830017fe", p.stderr)

    def sandboxed(self, *args):
        return subprocess.run(["bash", os.path.join(HERE, "sandbox.sh"), "--gpu", "none", "--out",
                               os.path.join(self.tmp.name, "out"), *args], capture_output=True, text=True, timeout=60)

    def test_the_sandbox_filters_system_calls(self):
        p = self.sandboxed("--", "python3", "-c", SYSCALLS)
        self.assertEqual(p.stdout.split(), ["io_uring_setup", "keyctl", "userfaultfd"], p.stderr)
        p = self.sandboxed("--", "python3", "-c", SYSCALLS, "x32")
        self.assertNotIn("x32 allowed", p.stdout)
        self.assertNotEqual(p.returncode, 0)

    def test_cpus_sets_the_quota(self):
        unit = f"sbx-test-cpus-{os.getpid()}"
        with subprocess.Popen(["bash", os.path.join(HERE, "sandbox.sh"), "--gpu", "none", "--out",
                               os.path.join(self.tmp.name, "out"), "--cpus", "3", "--unit", unit, "--", "sleep", "3"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
            quota = ""
            for _ in range(40):
                quota = subprocess.run(["systemctl", "--user", "show", unit, "-p", "CPUQuotaPerSecUSec", "--value"],
                                       capture_output=True, text=True).stdout.strip()
                if quota not in ("", "infinity"):
                    break
                time.sleep(0.05)
            proc.wait()
        self.assertEqual(quota, "3s")

    def test_memory_peaks_are_sampled(self):
        rc, report = self.run_track("prefill", [*wrapped("python3 -c 'x = bytearray(300 << 20); x[::4096] = b\"1\" * "
                                                         "len(x[::4096])'"), "--delay", "1"])
        self.assert_clean(rc, report, 1)
        self.assertGreaterEqual(report["peak_mem_mib"]["server"], 300)

    def test_server_exiting_early_voids_the_run(self):
        rc, report = self.run_track("decode", ["python3", "-c", "raise SystemExit(5)"])
        self.assertEqual(rc, 1)
        self.assertIn("server exited (5) before the client quit", report["void"])


if __name__ == "__main__":
    unittest.main()
