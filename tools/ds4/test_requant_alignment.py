"""RED tests: requant_dense_q8 must read general.alignment from each real
shard header and refuse any non-32 alignment, instead of silently assuming
32 (pack_gguf.py honours the key; a mismatch would silently requant garbage).

Run: python -m unittest discover -s tools/ds4 -p 'test_requant_alignment.py' -v
"""
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

T_UINT32, T_STRING, T_ARRAY, T_INT32 = 4, 8, 9, 5


def _s(b):
    return struct.pack('<Q', len(b)) + b


def _kv(key, t, payload):
    return _s(key.encode()) + struct.pack('<I', t) + payload


def synth_gguf(kvs):
    """Minimal GGUF header: magic, v3, 0 tensors, given KVs."""
    body = b''.join(kvs)
    return (b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', 0)
            + struct.pack('<Q', len(kvs)) + body)


class AlignmentGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _w(self, name, data):
        p = self.dir / name
        p.write_bytes(data)
        return p

    def test_default_32_when_key_absent(self):
        from requant_dense_q8 import read_gguf_alignment
        p = self._w('a.gguf', synth_gguf([
            _kv('general.architecture', T_STRING, _s(b'deepseek41')),
        ]))
        self.assertEqual(read_gguf_alignment(p), 32)

    def test_declared_alignment_is_returned(self):
        from requant_dense_q8 import read_gguf_alignment
        p = self._w('b.gguf', synth_gguf([
            _kv('general.architecture', T_STRING, _s(b'deepseek41')),
            _kv('general.alignment', T_UINT32, struct.pack('<I', 4096)),
        ]))
        self.assertEqual(read_gguf_alignment(p), 4096)

    def test_parser_walks_arrays_before_key(self):
        from requant_dense_q8 import read_gguf_alignment
        p = self._w('c.gguf', synth_gguf([
            _kv('x.roles', T_ARRAY,
                struct.pack('<IQ', T_STRING, 2) + _s(b'one') + _s(b'two')),
            _kv('x.nums', T_ARRAY, struct.pack('<IQ', T_INT32, 3)
                + struct.pack('<3i', 7, 8, 9)),
            _kv('general.alignment', T_UINT32, struct.pack('<I', 64)),
        ]))
        self.assertEqual(read_gguf_alignment(p), 64)

    def test_guard_accepts_32_and_default(self):
        from requant_dense_q8 import require_align32
        p = self._w('ok.gguf', synth_gguf([]))
        self.assertEqual(require_align32(p), 32)

    def test_guard_rejects_4096(self):
        from requant_dense_q8 import require_align32
        p = self._w('bad.gguf', synth_gguf([
            _kv('general.alignment', T_UINT32, struct.pack('<I', 4096)),
        ]))
        with self.assertRaises(SystemExit):
            require_align32(p)

    def test_real_shard1_default_32(self):
        import os
        source = os.environ.get('CILLA_TEST_GGUF')
        if not source:
            self.skipTest('set CILLA_TEST_GGUF to test a local alignment-32 shard')
        f = pathlib.Path(source)
        if not f.exists():
            self.skipTest('shard 1 not staged')
        from requant_dense_q8 import read_gguf_alignment
        self.assertEqual(read_gguf_alignment(f), 32)


if __name__ == '__main__':
    unittest.main(verbosity=2)
