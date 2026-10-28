#pragma once

#include <cstdint>
#include <cstddef>

namespace kitteni {

// RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
void rms_norm_f32(
    const float* x,
    const float* weight,
    int64_t dim,
    float eps,
    float* out
);

// LayerNorm (with weight and bias, used by spk_proj):
// out[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
void layer_norm_f32(
    const float* x,
    const float* weight,
    const float* bias,
    int64_t dim,
    float eps,
    float* out
);

// Qwen3 NeoX-style (non-traditional) Rotary Position Embedding (RoPE) in-place over
// `heads` vectors of length `head_dim` at sequence position `pos`.
// Pairs `(i, i + head_dim/2)` for `i in [0, head_dim/2)` with `freq = theta^(-2*i / head_dim)`.
void rope_neox_inplace(
    float* q_or_k,             // [num_heads, head_dim]
    int64_t num_heads,
    int64_t head_dim,
    int64_t pos,
    float rope_theta = 1000000.0f
);

// SwiGLU activation: out[i] = silu(gate[i]) * up[i], where silu(z) = z / (1 + exp(-z))
void swiglu_f32(
    const float* gate,
    const float* up,
    int64_t dim,
    float* out
);

// Dense Linear projection (used for spk_proj.0: 512 -> hidden_size):
// out[o] = bias[o] + sum_i (x[i] * weight[o * in_dim + i])
void dense_linear_f32(
    const float* x,
    const float* weight,
    const float* bias,
    int64_t in_dim,
    int64_t out_dim,
    float* out
);

} // namespace kitteni
