#pragma once

#include "kitteni/common.hpp"
#include "core/thread_pool.hpp"
#include <vector>
#include <cstdint>

namespace kitteni {

// BitNet per-token 8-bit activation quantizer (`activation_quant` in student_flow.py):
//   `scale = 127.0 / max(|x|, eps)`
//   `x_q = clamp(round(x * scale), -128, 127) / scale`
// When paired with a `BitWeightMatrix`, we also store the integer `u8 = q_int8 + 128`
// so AVX2 `_mm256_maddubs_epi16` executes integer dot products directly!
struct BitNetWeight {
    int64_t out_features = 0;   // For Linear: out_features; for Conv1d: out_channels
    int64_t in_features = 0;    // For Linear: in_features;  for Conv1d: in_channels * kernel_size
    int64_t in_channels = 0;    // For Conv1d: in_channels
    int64_t kernel_size = 1;    // For Conv1d: kernel_size
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;
    int64_t causal_pad_left = 0;
    int64_t causal_pad_right = 0;
    bool norm_input = false;
    bool quant_acts = true;

    // Integer codes in `[-128..127]` (INT4 codes are in `[-8..7]`, INT8 in `[-128..127]`, Ternary in `[-1..1]`)
    // Padded along `in_features` to a multiple of 32 for fast 32-wide AVX2 `_mm256_maddubs_epi16`.
    int64_t padded_in = 0;
    std::vector<int8_t> codes_i8;     // [out_features, padded_in]
    std::vector<int32_t> row_sums;    // [out_features] sum of codes_i8 along row (for +128 uint8 activation bias cancel)
    std::vector<float> inv_scales;    // [out_features] 1.0f / scale[o] (since Python divides `codes / scale`)
    std::vector<float> bias;          // [out_features] optional bias

    // Also allow initializing from a pre-dequantized float32 weight tensor (such as from TorchScript constants)
    // by recovering the exact integer codes and per-channel scale!
    void init_from_dequantized_f32(
        const float* weight_f32,
        const float* bias_f32,
        int64_t out_dim,
        int64_t in_ch,
        int64_t k_size = 1,
        int bits = 4,
        int64_t c_pad_left = 0,
        int64_t c_pad_right = 0
    );

    void init_from_packed_w4(
        const uint8_t* packed_u8,
        const float* scale_f32,
        const float* bias_f32,
        int64_t out_dim,
        int64_t in_ch,
        int64_t k_size = 1,
        int64_t c_pad_left = 0,
        int64_t c_pad_right = 0
    );
};

// Execute `BitLinear`:
// - Optionally applies `simple_rms_norm` if `W.norm_input` is true
// - Quantizes each row of `x` (`[num_rows, in_features]`) to 8-bit integers
// - Multiplies against `W.codes_i8` using AVX2 `_mm256_maddubs_epi16` and scales by `1 / (scale_x * scale_w)` + bias
void bitlinear_forward(
    const BitNetWeight& W,
    const float* x,            // [num_rows, W.in_features]
    int64_t num_rows,
    float* y,                  // [num_rows, W.out_features]
    const ThreadPool* pool = nullptr
);

// Execute `BitConv1d` on input `[C_in, T_in]` -> output `[C_out, T_out]`:
// - Applies optional causal padding `(causal_pad_left, causal_pad_right)`
// - Transposes to `[T_padded, C_in]`, applies per-frame 8-bit activation quantization over `C_in`
// - Gathers `kernel_size` patches and runs AVX2 INT8 x INT4/INT8 dot products!
void bitconv1d_forward(
    const BitNetWeight& W,
    const float* x,            // [W.in_channels, T_in]
    int64_t T_in,
    float* y,                  // [W.out_features, T_out]
    const ThreadPool* pool = nullptr
);

// Standard FP32 Conv1d on `[C_in, T_in]` -> `[C_out, T_out]` (with stride, padding, dilation)
void conv1d_f32(
    const float* x,            // [in_channels, T_in]
    const float* weight,       // [out_channels, in_channels, kernel_size]
    const float* bias,         // [out_channels] or nullptr
    int64_t in_channels,
    int64_t out_channels,
    int64_t T_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    float* y,                  // [out_channels, T_out]
    const ThreadPool* pool = nullptr
);

// Standard FP32 ConvTranspose1d on `[C_in, T_in]` -> `[C_out, T_out]`
// where `T_out = (T_in - 1) * stride - 2 * padding + kernel_size`
void conv_transpose1d_f32(
    const float* x,            // [in_channels, T_in]
    const float* weight,       // [in_channels, out_channels, kernel_size] (PyTorch ConvTranspose1d layout!)
    const float* bias,         // [out_channels] or nullptr
    int64_t in_channels,
    int64_t out_channels,
    int64_t T_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    float* y,                  // [out_channels, T_out]
    const ThreadPool* pool = nullptr
);

} // namespace kitteni
