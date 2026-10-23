"""Checks for harness/perseus_core_client.py's key-request policy, without Perseus: record 1's request passes, a weak
key is refused.

    python3 -I harness/test_perseus_core_client.py -v
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perseus_core_client as pcc  # noqa: E402

FULL, BOTTOM = {"q_top": 53, "ndig": 6, "nq": 54, "kp": 9}, {"q_top": 1, "ndig": 1, "nq": 54, "kp": 9}
RECORD = {"chain": "n32", "params_hash": "0" * 16, "contract": "c", "pcmm_t": [8], "ccmm": False, "public_key": True,
          "keys": [{"role": "relin", "shape": FULL}, {"role": "rot", "step": 1, "galois": 5, "shape": FULL},
                   {"role": "conj", "shape": FULL}, {"role": "enc_dense_to_sparse", "shape": BOTTOM},
                   {"role": "enc_sparse_to_dense", "shape": FULL}, {"role": "spru_atob", "shape": BOTTOM},
                   {"role": "auto", "galois": 131067, "shape": FULL}],
          "spru": {"h": 64, "n": 2, "cs_level": 0, "cs_nf": 1.0}}


def with_key(role, shape):
    return {**RECORD, "keys": [k if k["role"] != role else {**k, "shape": shape} for k in RECORD["keys"]]}


class KeyRequestTest(unittest.TestCase):
    def test_record_1s_request_passes(self):
        self.assertIsNone(pcc.refuse(pcc.parse(json.dumps(RECORD).encode())))
        self.assertIsNone(pcc.refuse({**RECORD, "spru": None}))

    def test_a_small_spru_block_key_is_refused(self):
        self.assertIn("2 blocks", pcc.refuse({**RECORD, "spru": {**RECORD["spru"], "h": 2}}))

    def test_a_sparse_secret_above_the_bottom_level_is_refused(self):
        for role in ("enc_dense_to_sparse", "spru_atob"):
            self.assertIn(role, pcc.refuse(with_key(role, FULL)))
            self.assertIn(role, pcc.refuse(with_key(role, {})))
            self.assertIn(role, pcc.refuse({**RECORD, "keys": [{"role": role}]}))
        self.assertIsNone(pcc.refuse(with_key("enc_sparse_to_dense", FULL)))   # its new secret is the dense one

    def test_undeclared_keys_and_chains_are_refused(self):
        self.assertIn("CCMM", pcc.refuse({**RECORD, "ccmm": True}))
        self.assertIn("n64", pcc.refuse({**RECORD, "chain": "n64"}))

    def test_a_repeated_json_key_is_refused(self):
        body = json.dumps(RECORD)[:-1] + ', "spru": {"h": 2, "n": 2, "cs_level": 0, "cs_nf": 1.0}}'
        with self.assertRaises(ValueError):
            pcc.parse(body.encode())


if __name__ == "__main__":
    unittest.main()
