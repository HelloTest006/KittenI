#pragma once

#include "kitteni/common.hpp"
#include "core/thread_pool.hpp"
#include <vector>
#include <cstdint>

namespace kitteni {

// Quantized activation buffer for fast INT8 x Ternary SIMD GEMV/GEMM.
// Each 128-element group `g` of `x` is quantized to:
//   `q[g * 128 + i] = round(x[g * 128 + i] / scale_x[g]) + 128` in `uint8_t` [1..255]
//   `scale_x[g] = max_abs / 127.0f`
// Shifting activations by +128 into unsigned `uint8_t` allows AVX2 `_mm256_maddubs_epi16`
// (unsigned byte x signed byte) to multiply directly with signed trits `{-1, 0, +1}`!
// The +128 offset is canceled in O(1) per group using `TernaryMatrix::group_sums[g]`:
//   `dot(w_trits, x_int8) = dot(w_trits, x_u8) - 128 * sum(w_trits)`
struct QuantizedActivation {
    int64_t rows = 0;          // Number of tokens (1 for decode GEMV, seq_len for prefill GEMM)
    int64_t in_features = 0;
    std::vector<uint8_t> u8_data; // [rows, in_features] biased by +128
    std::vector<float> scales;    // [rows, in_features / 128]

    void quantize_from(const float* x, int64_t num_rows, int64_t in_dim);
};

// High-speed SIMD INT8-Activation x 2-bit-Ternary-Weight GEMV / GEMM:
//   `y[r, out_col] = sum_in (x[r, in] * W[out_col, in])`
// Uses `_mm256_maddubs_epi16` on AVX2 CPUs (128 trits processed in 4 vector instructions!).
void ternary_matmul_quantized(
    const TernaryMatrix& W,
    const QuantizedActivation& X_q,
    float* Y,                  // [X_q.rows, W.out_features]
    const ThreadPool* pool = nullptr
);

// Exact FP32-Activation x 2-bit-Ternary-Weight GEMV / GEMM (zero activation quantization loss):
// Uses AVX2 sign-mask / LUT FMA directly over FP32 activations.
void ternary_matmul_f32(
    const TernaryMatrix& W,
    const float* X,            // [num_rows, W.in_features]
    int64_t num_rows,
    float* Y,                  // [num_rows, W.out_features]
    const ThreadPool* pool = nullptr
);

} // namespace kitteni
