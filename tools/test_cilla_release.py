"""CPU-only release-boundary tests; no model load or HIP runtime required."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIN = '3cf03257f219afbe7334045ff7c6a06ac68c627d'


class MmqBuildBoundary(unittest.TestCase):
    def setUp(self):
        if not shutil.which('cmake') or not shutil.which('git'):
            self.skipTest('cmake and git required')
        self.source = ROOT / 'third_party/llama.cpp/ggml/src/ggml-cuda'
        if not self.source.is_dir():
            self.skipTest('install pinned llama.cpp source first')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        start = cmake.index('    set(_cilla_mmq_glu OFF)', cmake.index('  elseif(STRATA_PREFILL_MMQ)'))
        end = cmake.index('    set(_strata_mmq_sources', start)
        self.block = cmake[start:end]

    def run_block(self, source=None, disabled_git=False):
        build = self.base / 'build'
        build.mkdir(exist_ok=True)
        script = self.base / 'patch.cmake'
        script.write_text(
            'cmake_minimum_required(VERSION 3.24)\n'
            f'set(CMAKE_CURRENT_SOURCE_DIR "{ROOT.as_posix()}")\n'
            f'set(CMAKE_BINARY_DIR "{build.as_posix()}")\n'
            f'set(_strata_mmq "{(source or self.source).as_posix()}")\n'
            + ('set(CMAKE_DISABLE_FIND_PACKAGE_Git TRUE)\n' if disabled_git else '')
            + self.block)
        return subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True)

    def test_pinned_source_applies_both_edits(self):
        original = (self.source / 'mmq.cuh').read_bytes()
        r = self.run_block()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = (self.base / 'build/strata-ggml-cuda-hip/mmq.cuh').read_text()
        self.assertIn('stride_channel_x < 0', out)
        self.assertIn('(x_, x_gate, offset_x, ', out)
        self.assertIn('ggml_cuda_mmq_fusion_supported', out)
        self.assertEqual((self.source / 'mmq.cuh').read_bytes(), original)

    def test_missing_required_seam_fails_closed(self):
        source = self.base / 'source'
        shutil.copytree(self.source, source)
        p = source / 'mmq.cuh'
        p.write_text(p.read_text().replace('const int offset_x = fastdiv(wt, sample_ratio)',
                                          'const int changed_offset_x = fastdiv(wt, sample_ratio)'))
        r = self.run_block(source)
        self.assertNotEqual(r.returncode, 0)

    def test_missing_git_fails_closed(self):
        r = self.run_block(disabled_git=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('HIP MMQ requires Git', r.stderr)


class ReleaseConfiguration(unittest.TestCase):
    def test_local_example_is_loopback_and_not_pinned(self):
        c = json.loads((ROOT / 'examples/cilla.local.example.json').read_text())
        self.assertEqual(c['host'], '127.0.0.1')
        self.assertNotIn('api_key', c)
        self.assertEqual(c['env']['DS4_ENGRAM_RAM'], '0')
        self.assertNotIn('DS4_SPLIT', c['env'])

    def test_requant_cli_requires_explicit_inputs(self):
        r = subprocess.run([sys.executable, str(ROOT / 'tools/ds4/requant_dense_q8.py')],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('--model', r.stderr)
        self.assertIn('--index', r.stderr)
        self.assertIn('--out', r.stderr)

    def test_requant_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            r = subprocess.run([sys.executable, str(ROOT / 'tools/ds4/requant_dense_q8.py'),
                                '--model', tmp, '--index', str(p / 'absent.json'), '--out', tmp],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('refusing to overwrite', r.stderr)


if __name__ == '__main__':
    unittest.main()
