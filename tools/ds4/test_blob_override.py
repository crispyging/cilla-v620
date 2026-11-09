#!/usr/bin/env python3
"""test_blob_override.py - the pack's --q8-dense hook takes per-tensor qtype from the manifest.

RED target: manifest entries for the MXFP4 expert re-encode carry qtype "MXFP4" and 17 bytes
per 32 values (not the Q8_0 nel//32*34 the hook hardcoded); legacy mixq8.json entries without
qtype must keep resolving as Q8_0; a blob tensor must split into per-expert entries at
base + e * (bytes // n_exp) exactly like the main loop does for shard tensors.
"""
import sys, unittest
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import pack_gguf as pg


class TestBlobOverride(unittest.TestCase):
    def test_q8_default_for_legacy_manifest(self):
        e = {"off": 0, "bytes": 34 * (4529848320 // 32)}
        tname, per, bpb = pg.resolve_override("blk.0.attn_q.weight", e, 4529848320)
        self.assertEqual((tname, per, bpb), ("Q8_0", 32, 34))

    def test_mxfp4_entry(self):
        nel = 384 * 2304 * 5120
        e = {"off": 256, "bytes": nel // 32 * 17, "qtype": "MXFP4"}
        tname, per, bpb = pg.resolve_override("blk.15.ffn_gate_exps.weight", e, nel)
        self.assertEqual((tname, per, bpb), ("MXFP4", 32, 17))

    def test_bytes_must_match_qtype(self):
        nel = 384 * 2304 * 5120
        with self.assertRaises(SystemExit):
            pg.resolve_override("blk.15.ffn_gate_exps.weight", {"off": 0, "bytes": nel // 32 * 34,
                                                                "qtype": "MXFP4"}, nel)

    def test_partial_blocks_rejected(self):
        with self.assertRaises(SystemExit):
            pg.resolve_override("x", {"off": 0, "bytes": 33 * 17, "qtype": "MXFP4"}, 1056 + 16)

    def test_unknown_qtype_rejected(self):
        with self.assertRaises(SystemExit):
            pg.resolve_override("x", {"off": 0, "bytes": 8, "qtype": "MXFP8-fake"}, 64)

    def test_expert_split_of_a_blob_tensor(self):
        # [n_exp][out][in] MXFP4 override -> per-expert entries at off + e*part, part from blob bytes
        n_exp, out, k = 384, 2304, 5120
        nel, base = n_exp * out * k, 256
        part = nel // 32 * 17 // n_exp
        rows = pg.expert_entries("layers.15.ffn.experts.w1", "MXFP4", 7, base, part * n_exp, [n_exp, out, k])
        self.assertEqual(len(rows), n_exp)
        self.assertEqual(rows[0][0], "layers.15.ffn.experts.0.w1.weight")
        for e, (name, dt, fi, off, nb, shape) in enumerate(rows):
            self.assertEqual((dt, fi, off, nb), ("MXFP4", 7, base + e * part, part))
            self.assertEqual(shape, [out, k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
