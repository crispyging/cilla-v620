// tests/ds4_iq4nl_test.cu - the Engram's IQ4_NL row decode (src/ds4/ds4_iq4nl.cuh) against ggml's OWN dequantize
// reference (GGML_TYPE_IQ4_NL's to_float), bit for bit: every nibble of both halves, a sweep of the scale's fp16 bit
// patterns (zeros, denormals, max normal, infinities, NaNs), and the pack's CONTIGUOUS records read with the engine's
// own addressing - a row of hd values at id * (hd / 32 * 18) bytes (ds4_model.cu's eg_issue), its codes inside the
// row's hd-byte stride at (e / 32) * kIq4nlBytes with the tail poisoned, proving the kernel reads none of it.
// The Mix-256G Engram table (hd = 256, IQ4_NL) is exactly this layout. CPU only: no GPU, no model, no disk.
// RED until src/ds4/ds4_iq4nl.cuh exists; the parity against ggml's compiled code is the golden, not a copy.
#include "../src/ds4/ds4_iq4nl.cuh"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using strata::ds4::iq4nl_value;
using strata::ds4::kIq4nlBytes;

namespace {

int g_fail = 0;
std::mt19937 rng(20261026);

void check(bool ok, const char* what) {
    std::printf("%-64s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// ggml's reference over k values of one packed block stream
std::vector<float> ggml_ref(const uint8_t* packed, int k) {
    std::vector<float> want(k);
    ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(packed, want.data(), (int64_t) k);
    return want;
}

bool same(float a, float b) {   // bit for bit, NaNs equal by bit pattern
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
}

// 1) a block's values: nibble order, the codebook and the scale's full fp16 range
void test_values() {
    const uint16_t sweeps[] = {0x0000, 0x8000, 0x0001, 0x03FF, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x3F80};
    const uint8_t code_sweeps[] = {0x00, 0x0F, 0xF0, 0xFF, 0x55, 0xAA, 0x81, 0x7E};
    int blocks = 0;
    for (const uint16_t d : sweeps)
        for (const uint8_t c : code_sweeps) {
            uint8_t b[kIq4nlBytes];
            b[0] = (uint8_t) d;
            b[1] = (uint8_t) (d >> 8);
            for (int j = 0; j < 16; ++j) b[2 + j] = (uint8_t) ((j & 1) ? ~c : c);
            const std::vector<float> want = ggml_ref(b, 32);
            for (int i = 0; i < 32; ++i)
                if (!same(iq4nl_value(b, i), want[i])) { ++g_fail; std::printf("value FAIL d=%04X c=%02X i=%d\n", d, c, i); return; }
            blocks += 1;
        }
    // random blocks: random finite-ish scale bits, random codes
    for (int n = 0; n < 4096; ++n) {
        uint8_t b[kIq4nlBytes];
        const uint16_t d = (uint16_t) rng();
        b[0] = (uint8_t) d;
        b[1] = (uint8_t) (d >> 8);
        for (int j = 0; j < 16; ++j) b[2 + j] = (uint8_t) rng();
        const std::vector<float> want = ggml_ref(b, 32);
        for (int i = 0; i < 32; ++i)
            if (!same(iq4nl_value(b, i), want[i])) { ++g_fail; std::printf("value FAIL n=%d i=%d\n", n, i); return; }
    }
    check(g_fail == 0, "IQ4_NL values = ggml's, bit for bit (sweep + 4096 random)");
}

// 2) the pack's records and the engine's addressing: the Mix table's hd, rows read at id * (hd/32*18)
void test_records(int hd) {
    const int rec = (hd / 32) * kIq4nlBytes;            // (eg_issue's r.n)
    check(kIq4nlBytes == 18, "an IQ4_NL block is 18 bytes");
    check(rec == 144 && hd == 256, "a Mix row (256 values) is a 144-byte record");
    check(rec <= hd, "the record fits the row's hd-byte stride (tail never read)");

    const int R = 17;
    std::vector<uint8_t> file((size_t) R * rec);
    for (auto& x : file) x = (uint8_t) rng();
    // row 0: structured - every code 0, then every code 0xF
    std::memset(file.data(), 0, 2);                        // row 0 codes all zero (scale at the front: keep d=1.0)
    file[0] = 0x00; file[1] = 0x3C;                        // (fp16 1.0)
    for (int j = 0; j < 16; ++j) file[2 + j] = 0x00;
    std::memcpy(file.data() + rec + 2 - 2, "\x00\x3c", 2); // row 1: scale 1.0 too
    for (int j = 0; j < 16; ++j) file[rec + 2 + j] = 0xFF; // all hi nibbles: 0xF

    bool ok = true;
    for (int id = 0; id < R && ok; ++id) {
        const std::vector<float> want = ggml_ref(file.data() + (size_t) id * rec, hd);
        for (int e = 0; e < hd; ++e)
            if (!same(iq4nl_value(file.data() + (size_t) id * rec + (e >> 5) * kIq4nlBytes, e & 31), want[e])) {
                std::printf("record FAIL id=%d e=%d\n", id, e);
                ok = false;
                break;
            }
    }
    check(ok, "pack records: a row at id*(hd/32*18), a value at (e/32)*18 = ggml's");

    // 3) the gathered batch's hd-stride rows: codes at the row's start, the tail POISONED - the engine must not read it
    std::vector<uint8_t> buf((size_t) R * hd, 0xCD);       // stride hd, like the window's row area
    for (int id = 0; id < R; ++id) std::memcpy(buf.data() + (size_t) id * hd, file.data() + (size_t) id * rec, (size_t) rec);
    ok = true;
    for (int id = 0; id < R && ok; ++id) {
        const std::vector<float> want = ggml_ref(file.data() + (size_t) id * rec, hd);
        const uint8_t* rows = buf.data() + (size_t) id * hd;
        for (int e = 0; e < hd; ++e)
            if (!same(iq4nl_value(rows + (e >> 5) * kIq4nlBytes, e & 31), want[e])) {
                std::printf("stride FAIL id=%d e=%d\n", id, e);
                ok = false;
                break;
            }
    }
    check(ok, "strided rows decode past the poisoned tail (only hd/32*18 bytes read)");
}

}  // namespace

int main() {
    static_assert(kIq4nlBytes == 18, "IQ4_NL block bytes");
    std::printf("ds4_iq4nl_test - the Engram IQ4_NL decode against ggml's own reference\n");
    test_values();
    test_records(256);   // (the Mix-256G table's real eg_hd)
    std::printf("%s\n", g_fail == 0 ? "ALL OK" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
