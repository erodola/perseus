"""Checks for harness/wire.py: a round trip, and every way a hostile server could feed the client.

    python3 -I harness/test_wire.py -v
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402


class WireTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.secret = os.path.join(self.root, "secret.key")
        with open(self.secret, "wb") as f:
            f.write(b"the client's secret key")
        self.out = os.path.join(self.root, "out")
        os.mkdir(self.out)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        wire.write(self.out, "tok/1/logits.ct", b"\x00\x01payload")
        self.assertEqual(wire.read(self.out, "tok/1/logits.ct", cap=100), b"\x00\x01payload")

    def test_no_temporary_file_left(self):
        wire.write(self.out, "ack", b"")
        self.assertEqual(sorted(os.listdir(self.out)), ["ack"])

    def test_symlink_to_secret_is_refused(self):
        os.symlink(self.secret, os.path.join(self.out, "z.ct"))
        with self.assertRaises(wire.WireError):
            wire.read(self.out, "z.ct", cap=100)

    def test_symlinked_directory_is_refused(self):
        os.symlink(self.root, os.path.join(self.out, "tok"))
        with self.assertRaises(wire.WireError):
            wire.read(self.out, "tok/secret.key", cap=100)

    def test_fifo_is_refused_without_blocking(self):
        os.mkfifo(os.path.join(self.out, "logits.ct"))
        with self.assertRaises(wire.WireError):
            wire.read(self.out, "logits.ct", cap=100)

    def test_oversized_file_is_refused(self):
        wire.write(self.out, "big.ct", b"x" * 1000)
        with self.assertRaises(wire.WireError):
            wire.read(self.out, "big.ct", cap=999)

    def test_bad_frame_is_refused(self):
        with open(os.path.join(self.out, "short.ct"), "wb") as f:
            f.write(wire.HEADER.pack(50) + b"only ten b")
        with self.assertRaises(wire.WireError):
            wire.read(self.out, "short.ct", cap=100)

    def test_escaping_paths_are_refused(self):
        for rel in ("../secret.key", "/etc/passwd", "a//b", "./x"):
            with self.assertRaises(wire.WireError):
                wire.read(self.out, rel, cap=100)

    def test_wait_returns_when_written_and_times_out_otherwise(self):
        wire.write(self.out, "done", b"")
        self.assertLessEqual(wire.wait(self.out, "done", time.monotonic() + 1), time.monotonic())
        with self.assertRaises(wire.WireError):
            wire.wait(self.out, "never", time.monotonic() + 0.05)


if __name__ == "__main__":
    unittest.main()
