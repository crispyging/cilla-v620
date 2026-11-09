// src/ds4/ds4_iq4nl.cuh - one value of a ggml IQ4_NL block (32 values, 18 bytes: d F16 then 16 bytes of 4-bit
// indices, the low nibble the block's first 16 values), as ggml's dequantize_row_iq4_nl computes it: kvalues * d.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace strata::ds4 {

constexpr int kIq4nlBlock = 32;    // values a block
constexpr int kIq4nlBytes = 18;    // d[2] + qs[16]

// ggml's IQ4_NL codebook (int8), the exact values its kvalues_iq4nl holds
__device__ __constant__ int8_t kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                    1,   13,   25,  38,  53,  69,  89, 113};

// value i (0..31) of the block at b
__device__ __forceinline__ float iq4nl_value(const uint8_t* __restrict__ b, int i) {
    const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
    const uint8_t* qs = b + 2;                 // 16 bytes: byte j holds values j (low) and j + 16 (high)
    const int j = i & 15;
    const int nib = i < 16 ? (qs[j] & 0xF) : (qs[j] >> 4);
    return __fmul_rn((float) kIq4nlValues[nib], d);
}

}  // namespace strata::ds4
