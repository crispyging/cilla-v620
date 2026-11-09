"""Compile the production scratch helper AND extracted caller with a CPU fake allocator."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MmqScratchAllocation(unittest.TestCase):
    def setUp(self):
        self.cxx = shutil.which('g++')
        if not self.cxx:
            self.skipTest('g++ required for CPU allocation fault injection')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def compile_run(self, path):
        exe = self.base / 'test'
        r = subprocess.run([self.cxx, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-UNDEBUG',
                            '-I', str(ROOT / 'src/ds4'), str(path), '-o', str(exe)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_production_pair_helper_faults(self):
        self.compile_run(ROOT / 'tests/mmq_scratch_alloc_test.cpp')

    def test_actual_caller_faults_and_iota_byte_count(self):
        source = (ROOT / 'src/ds4/ds4_gguf.cu').read_text()
        start = source.index('struct MmqScratch {')
        struct = source[start:source.index('\n};', start) + 3]
        start = source.index('MmqScratch* mmq_scratch(')
        caller = source[start:source.index('\nbool mmq_ok(', start)]
        launch = 'cilla_iota<<<(rows + 255) / 256, 256, 0, s>>>(m.iota, rows);'
        self.assertEqual(caller.count(launch), 1)
        caller = caller.replace(launch, 'cilla_iota(m.iota, rows, s);')
        fixture = r'''
#include "mmq_scratch_alloc.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
using cudaStream_t = int;
constexpr int cudaSuccess = 0;
std::unordered_map<void*, size_t> live;
int calls = 0, fail_at = 0, iota_launches = 0;
int cudaGetDevice(int* dev) { *dev = 0; return cudaSuccess; }
template<class T> int cudaMalloc(T** out, size_t bytes) {
    ++calls;
    if (calls == fail_at) { *out = nullptr; return 1; }
    // Track requested device bytes without actually allocating/touching large GPU-sized buffers.
    void* p = std::malloc(1);
    assert(p);
    live[p] = bytes;
    *out = static_cast<T*>(p);
    return cudaSuccess;
}
int cudaFree(void* p) { assert(live.erase(p) == 1); std::free(p); return cudaSuccess; }
void cilla_iota(int* p, int rows, cudaStream_t) {
    assert(live.at(p) >= size_t(rows) * sizeof(int));
    ++iota_launches;
}
namespace mq {
struct Context {};
size_t q8_bytes(int rows, int cols) { return size_t(rows) * ((cols + 31) / 32) * 34; }
}
'''
        fixture += '\n' + struct + '\nstd::mutex g_mq_mu;\nstd::unordered_map<int, MmqScratch> g_mq;\n' + caller
        fixture += r'''
void reset() {
    for (auto& item : g_mq) {
        auto& m = item.second;
        for (void* p : {m.xq, static_cast<void*>(m.g), static_cast<void*>(m.u),
                        static_cast<void*>(m.bnd), static_cast<void*>(m.iota)})
            if (p) cudaFree(p);
        delete m.ctx;
    }
    g_mq.clear();
    assert(live.empty());
}
int main() {
    for (int failure = 1; failure <= 5; ++failure) {
        calls = 0; fail_at = failure;
        assert(mmq_scratch(1024, 5120, 2304, 16, 0) == nullptr);
        // An earlier complete group may remain owned; retry must succeed without leaking it.
        calls = 0; fail_at = 0;
        MmqScratch* m = mmq_scratch(1024, 5120, 2304, 16, 0);
        assert(m && m->rows_cap == 8192 && m->iota_cap == 8192);
        assert(m->gu_cap == size_t(8192) * 2304);
        assert(live.size() == 5 && live.at(m->iota) == size_t(8192) * sizeof(int));
        calls = 0;
        assert(mmq_scratch(8192, 5120, 2304, 16, 0) == m);
        assert(calls == 0);
        assert(mmq_scratch(8193, 5120, 2304, 16, 0) == nullptr);
        assert(mmq_scratch(1024, 5121, 2304, 16, 0) == nullptr);
        assert(mmq_scratch(1024, 5120, 2305, 16, 0) == nullptr);
        assert(mmq_scratch(0, 5120, 2304, 16, 0) == nullptr);
        assert(calls == 0 && live.size() == 5);
        reset();
    }
    assert(iota_launches == 5);
}
'''
        path = self.base / 'caller.cpp'
        path.write_text(fixture)
        self.compile_run(path)


if __name__ == '__main__':
    unittest.main()
