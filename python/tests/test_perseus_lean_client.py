"""speedrun/harness/perseus_lean_client.py's key-request policy, without Perseus: the requests of both chains pass; a
sparse key over more special primes or above the bottom, another chain, CCMM and a weak SPRU key are refused. Skipped
in a checkout without the client (it comes in the record's own commit).
"""
import copy
import importlib.util
import unittest
from pathlib import Path

import pytest

CLIENT = Path(__file__).resolve().parents[2] / "speedrun" / "harness" / "perseus_lean_client.py"
if not CLIENT.is_file():
    pytest.skip("no speedrun/harness/perseus_lean_client.py in this checkout", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("perseus_lean_client", CLIENT)
plc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plc)


def request(chain, nq, ndig, kp, kp_sparse):
    full = {"q_top": nq - 1, "ndig": ndig, "nq": nq, "kp": kp}
    bottom = {"q_top": 1, "ndig": 1, "nq": nq, "kp": kp_sparse}
    return {"chain": chain, "params_hash": "0" * 16, "contract": "c", "pcmm_t": [8], "ccmm": False, "public_key": True,
            "keys": [{"role": "relin", "shape": full}, {"role": "rot", "step": 1, "galois": 5, "shape": full},
                     {"role": "conj", "shape": full}, {"role": "enc_dense_to_sparse", "shape": bottom},
                     {"role": "enc_sparse_to_dense", "shape": full}, {"role": "spru_atob", "shape": bottom},
                     {"role": "auto", "galois": 131067, "shape": full}],
            "spru": {"h": 64, "n": 2, "cs_level": 0, "cs_nf": 1.0}}


N32, N32A4 = request("n32", 54, 6, 9, 9), request("n32a4", 51, 4, 13, 9)


class Policy(unittest.TestCase):
    def test_both_chains_pass(self):
        self.assertIsNone(plc.refuse(N32))
        self.assertIsNone(plc.refuse(N32A4))

    def test_sparse_key_over_all_of_p(self):
        for role in plc.SPARSE:
            r = copy.deepcopy(N32A4)
            k = next(k for k in r["keys"] if k["role"] == role)
            k["shape"]["kp"] = 13
            self.assertIn("special primes", plc.refuse(r))

    def test_sparse_key_above_the_bottom(self):
        r = copy.deepcopy(N32)
        r["keys"][3]["shape"]["q_top"] = 3
        self.assertIn("above the bottom", plc.refuse(r))

    def test_other_chain_ccmm_spru(self):
        self.assertIn("chain", plc.refuse({**N32A4, "chain": "n64"}))
        self.assertIn("CCMM", plc.refuse({**N32A4, "ccmm": True}))
        self.assertIn("blocks", plc.refuse({**N32A4, "spru": {"h": 2, "n": 2}}))

    def test_repeated_key(self):
        with self.assertRaises(ValueError):
            plc.parse(b'{"chain": "n32", "chain": "n32a4"}')

