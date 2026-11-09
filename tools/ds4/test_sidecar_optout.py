"""RED/GREEN: the pack's sidecar resolution must respect a 'none' opt-out.

Mix-256G ships neither a DSpark drafter nor a vision tower, but its model dir — like
Maya-S's — sits under weights/, and pack_gguf.py's defaults resolve <model>/../dspark
and <model>/../vision, which find Maya-S's sidecars and silently borrow weights the
artifact does not ship.  'none' on either flag must take neither, while the old default
behaviour (complete sidecar present -> taken) stays intact for Maya-S's rollback pack.
"""
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_gguf  # noqa: E402


class DrafterOptout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_parent = pathlib.Path(self.tmp.name) / "weights"
        (self.model_parent / "dspark").mkdir(parents=True)
        self.dj = self.model_parent / "dspark" / "dspark.json"
        self.db = self.model_parent / "dspark" / "dspark.bin"
        self.dj.write_text("{}")
        self.db.write_bytes(b"\0")

    def test_none_opts_out_even_if_sibling_present(self):
        self.assertIsNone(pack_gguf.drafter_json_path("none", self.model_parent))

    def test_default_still_picks_up_complete_sidecar(self):
        self.assertEqual(pack_gguf.drafter_json_path(None, self.model_parent), self.dj)

    def test_default_without_bin_is_none(self):
        self.db.unlink()
        self.assertIsNone(pack_gguf.drafter_json_path(None, self.model_parent))

    def test_explicit_missing_raises(self):
        with self.assertRaises(SystemExit):
            pack_gguf.drafter_json_path(str(self.model_parent / "nope.json"), self.model_parent)

    def test_explicit_present_returns(self):
        self.assertEqual(pack_gguf.drafter_json_path(str(self.dj), self.model_parent), self.dj)


class VisionOptout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_parent = pathlib.Path(self.tmp.name) / "weights"
        self.vdir = self.model_parent / "vision"
        self.vdir.mkdir(parents=True)
        (self.vdir / "vision.json").write_text("{}")
        (self.vdir / "vision.bin").write_bytes(b"\0")

    def test_none_opts_out_even_if_sibling_present(self):
        self.assertIsNone(pack_gguf.vision_dir_path("none", self.model_parent))

    def test_default_picks_complete_folder(self):
        self.assertEqual(pack_gguf.vision_dir_path(None, self.model_parent), self.vdir)

    def test_default_incomplete_folder_is_none(self):
        (self.vdir / "vision.bin").unlink()
        self.assertIsNone(pack_gguf.vision_dir_path(None, self.model_parent))


if __name__ == "__main__":
    unittest.main()
