"""Checks for harness/protocol.py: both tracks against harness/dummy_server.py, in a separate process.

    python3 -I harness/test_protocol.py -v
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wire  # noqa: E402
from protocol import Session  # noqa: E402


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inbox = os.path.join(self.tmp.name, "in")
        self.outbox = os.path.join(self.tmp.name, "out")
        os.mkdir(self.inbox)
        os.mkdir(self.outbox)
        self.server = None

    def tearDown(self):
        if self.server and self.server.poll() is None:
            self.server.kill()
            self.server.wait()
        self.tmp.cleanup()

    def start(self, track, *args, timeout=10.0):
        self.server = subprocess.Popen(
            [sys.executable, "-I", os.path.join(HERE, "dummy_server.py"), "--track", track,
             "--inbox", self.inbox, "--outbox", self.outbox, *args])
        session = Session(self.inbox, self.outbox, track, timeout=timeout)
        session.send_keys(lambda path, spec: os.makedirs(path))
        session.ready(timeout=10.0)
        return session

    def finish(self, session):
        session.quit()
        self.assertEqual(self.server.wait(timeout=5), 0)

    def test_prefill_answers_each_request(self):
        s = self.start("prefill", "--delay", "0.05")
        self.assertIn(b'"track": "prefill"', s.manifest)
        for i in range(3):
            prompt = os.urandom(1000 + i)
            r = s.request([prompt])
            self.assertEqual(r.outputs, [hashlib.sha256(prompt).digest()])
            self.assertGreaterEqual(r.score("prefill"), 0.05)
        self.finish(s)

    def test_decode_releases_one_token_at_a_time(self):
        s = self.start("decode", "--delay", "0.01")
        for _ in range(2):
            tokens = [os.urandom(64) for _ in range(16)]
            r = s.request(tokens)
            self.assertEqual(r.outputs, [hashlib.sha256(t).digest() for t in tokens])
            for j in range(15):
                self.assertGreaterEqual(r.starts[j + 1], r.ends[j])
        self.finish(s)  # the server exits 3 on an early release

    def test_decode_scores_positions_8_to_15_only(self):
        s = self.start("decode", "--delay", "0.02", "--fill-delay", "0.2")
        r = s.request([os.urandom(64) for _ in range(16)])
        self.assertGreaterEqual(min(r.times()[:8]), 0.2)
        self.assertGreaterEqual(r.score("decode"), 0.02)
        self.assertLess(r.score("decode"), 0.1)
        self.finish(s)

    def test_output_before_input_is_refused(self):
        s = self.start("decode", "--misbehave", "early")
        with self.assertRaisesRegex(wire.WireError, "output before its input"):
            s.request([os.urandom(64) for _ in range(16)])

    def test_missing_output_times_out(self):
        s = self.start("prefill", "--misbehave", "silent", timeout=0.3)
        with self.assertRaisesRegex(wire.WireError, "deadline"):
            s.request([os.urandom(64)])

    def test_request_before_ready_and_wrong_length_are_refused(self):
        s = Session(self.inbox, self.outbox, "decode")
        with self.assertRaises(RuntimeError):
            s.request([b""] * 16)
        s.manifest = b"{}"
        with self.assertRaises(ValueError):
            s.request([b""] * 15)


if __name__ == "__main__":
    unittest.main()
