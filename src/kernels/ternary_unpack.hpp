#pragma once

#include "kitteni/common.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace kitteni {

// Pack an array of trits (values in {-1, 0, +1}) and per-128-group float scales
// into KittenI's lane-interleaved 2-bit SIMD TernaryMatrix.
// In each 128-element group:
//   byte m (0..31) stores codes for elements m, m+32, m+64, m+96:
//   bits [1:0] = code(m), bits [3:2] = code(m+32), bits [5:4] = code(m+64), bits [7:6] = code(m+96)
//   where code = trit + 1 in {0, 1, 2}.
void pack_trits_to_ternary_matrix(
    const int8_t* trits,       // [out_features, in_features] in {-1, 0, +1}
    const float* scales,       // [out_features, in_features / 128]
    int64_t out_features,
    int64_t in_features,
    TernaryMatrix& out
);

// Unpack a TernaryMatrix back to exact float32 or bfloat16 values for parity verification
void unpack_ternary_matrix_to_f32(
    const TernaryMatrix& mat,
    float* dst_f32             // [out_features * in_features]
);

void unpack_ternary_matrix_to_bf16(
    const TernaryMatrix& mat,
    uint16_t* dst_bf16         // [out_features * in_features]
);

// Format 1: Unpack TL2 tensor (`tl2/*` in model-tl2-emb4.safetensors)
// Shape on disk: [rows_blocks, group_blocks, 928] uint8, where:
//   out_features = rows_blocks * 32
//   in_features  = group_blocks * 128
void load_tl2_to_ternary_matrix(
    const uint8_t* tl2_buf,
    int64_t rows_blocks,
    int64_t group_blocks,
    TernaryMatrix& out
);

// Format 2: Unpack 1.58-bit base-3 tensor (`*::codes` + `*::scale` in model-ternary.safetensors)
// 5 trits per byte (least-significant base-3 digit first: sum_{i=0..4} d_i * 3^i),
// where d_i in {0, 1, 2} corresponds to trit = d_i - 1 in {-1, 0, +1}.
void load_ternary158_to_ternary_matrix(
    const uint8_t* packed_base3,
    size_t packed_byte_len,
    const uint16_t* scales_bf16,
    DType scale_dtype,
    int64_t out_features,
    int64_t in_features,
    TernaryMatrix& out
);

// Format 3: Load GGUF TQ2_1 blocks (68 bytes per 256 weights: 64 bytes lane-interleaved 2-bit codes + 2x FP16 scales)
void load_tq2_1_to_ternary_matrix(
    const uint8_t* tq2_1_blocks,
    int64_t out_features,
    int64_t in_features,
    TernaryMatrix& out
);

} // namespace kitteni
