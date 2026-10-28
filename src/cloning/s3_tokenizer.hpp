#pragma once

#include "core/mmap_loader.hpp"
#include "core/thread_pool.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace kitteni {

// Native C++20 implementation of `S3TokenizerV2` (`speech_tokenizer_v2_25hz` from `s3gen_meanflow.safetensors`):
// - 16 kHz mono audio -> 128-bin Whisper-style log-Mel spectrogram (`n_fft=400, hop=160`)
// - `conv1`: Conv1d(128 -> 1280, k=3, stride=2, pad=1) + GELU
// - `conv2`: Conv1d(1280 -> 1280, k=3, stride=2, pad=1) + GELU -> permute `[T_tok, 1280]` (25 Hz)
// - 6 `ResidualAttentionBlock`s (`n_state=1280, n_head=20, head_dim=64`):
//     Each block has:
//       - `attn_ln` (LayerNorm(1280))
//       - `FSMNMultiHeadAttention` (RoPE theta=10000 on Q & K, depthwise Conv1d(1280, k=31, groups=1280) + residual on V)
//       - `mlp_ln` (LayerNorm(1280)) + MLP (Linear(1280->5120) + GELU + Linear(5120->1280))
// - `FSQCodebook`: `project_down` (Linear(1280 -> 8)) -> `round(tanh(h) * 0.9990000128746033) + 1` -> base-3 dot `[1, 3, 9, 27, 81, 243, 729, 2187]` -> 0..6560 token IDs!
class S3AudioTokenizer {
public:
    struct AttentionBlock {
        std::vector<float> attn_ln_w, attn_ln_b; // [1280]
        std::vector<float> q_w, q_b;             // [1280, 1280] and [1280]
        std::vector<float> k_w;                  // [1280, 1280] (no bias!)
        std::vector<float> v_w, v_b;             // [1280, 1280] and [1280]
        std::vector<float> out_w, out_b;         // [1280, 1280] and [1280]
        std::vector<float> fsmn_w;               // [1280, 31] depthwise conv

        std::vector<float> mlp_ln_w, mlp_ln_b;   // [1280]
        std::vector<float> mlp0_w, mlp0_b;       // [5120, 1280] and [5120]
        std::vector<float> mlp2_w, mlp2_b;       // [1280, 5120] and [1280]
    };

    S3AudioTokenizer() = default;

    // Load from a SafetensorsFile containing `tokenizer.*` keys
    void load(const SafetensorsFile& sf, const std::string& prefix = "tokenizer.");

    // Encode 16 kHz mono audio into 25 Hz S3 codec token IDs (`[0..6560]`)
    std::vector<int32_t> encode_16k(
        const float* wav_16k,
        int64_t num_samples,
        const ThreadPool* pool = nullptr
    ) const;

private:
    std::vector<float> mel_filters_; // [128, 201]
    std::vector<float> window_400_;  // [400]

    std::vector<float> conv1_w_, conv1_b_; // [1280, 128, 3] and [1280]
    std::vector<float> conv2_w_, conv2_b_; // [1280, 1280, 3] and [1280]

    AttentionBlock blocks_[6];

    std::vector<float> proj_down_w_; // [8, 1280]
    std::vector<float> proj_down_b_; // [8]
};

} // namespace kitteni
