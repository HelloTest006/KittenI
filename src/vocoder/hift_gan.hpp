#pragma once

#include "vocoder/s3_flow.hpp"
#include "core/thread_pool.hpp"
#include <vector>
#include <random>
#include <cstdint>

namespace kitteni {

// Native C++20 implementation of HiFTGenerator (Neural Source Filter + ISTFTNet vocoder at 24 kHz):
// - ConvRNNF0Predictor (5x Conv1d + ELU + Linear -> abs F0)
// - SourceModuleHnNSF (480x F0 upsample -> 9-harmonic SineGen + UV noise -> Linear + Tanh -> 16-point STFT)
// - 3-stage ConvTranspose1d upsampler (rates [8, 5, 3], channels 512 -> 256 -> 128 -> 64)
//   with Snake-activated source fusion (`source_downs` + `source_resblocks`) and multi-kernel `ResBlock`s (kernels [3, 7, 11])
// - Final Conv1d(64, 18, 7) -> exp(mag) & sin(phase) -> 16-point ISTFT (hop=4) -> 20ms cosine trim_fade
class HiFTGenerator {
public:
    struct SnakeResBlock {
        int64_t channels = 0;
        int64_t kernel_size = 3;
        int64_t dilations[3] = {1, 3, 5};

        // 3 dilated residual stages per ResBlock:
        // Each stage has:
        //   snake1 (alpha1, inv_beta1) -> conv1 (dilation d) -> snake2 (alpha2, inv_beta2) -> conv2 (dilation 1)
        std::vector<float> alpha1[3];
        std::vector<float> inv_beta1[3];
        std::vector<float> conv1_w[3];
        std::vector<float> conv1_b[3];

        std::vector<float> alpha2[3];
        std::vector<float> inv_beta2[3];
        std::vector<float> conv2_w[3];
        std::vector<float> conv2_b[3];
    };

    HiFTGenerator() = default;

    // Load folded weights from `decoder/constants/1115..1447` of TorchScript `decoder.pt`
    void load_from_torchscript_zip(const TorchScriptZipConstants& zip);

    // Synthesize 24 kHz waveform `[T_mel * 480]` from `[80, T_mel]` mel spectrogram
    std::vector<float> forward(
        const float* mel_80xT,
        int64_t T_mel,
        std::mt19937& rng,
        const ThreadPool* pool = nullptr,
        bool deterministic_source = false
    ) const;

private:
    // F0 predictor: 5x Conv1d(k=3, pad=1) + ELU + Linear(512, 1)
    std::vector<float> f0_conv_w_[5];
    std::vector<float> f0_conv_b_[5];
    std::vector<float> f0_linear_w_; // [512]
    float f0_linear_b_ = 0.0f;

    // SourceModuleHnNSF: Linear(9, 1) + Tanh, plus 16-point Hann window
    std::vector<float> source_linear_w_; // [9]
    float source_linear_b_ = 0.0f;
    std::vector<float> stft_window_16_;  // [16]

    // Pre-conv: Conv1d(80, 512, 7, pad=3)
    std::vector<float> conv_pre_w_;
    std::vector<float> conv_pre_b_;

    // 3 upsample stages:
    //   Stage 0: 512 -> 256, rate=8, k=16, pad=4; source_down Conv1d(18, 256, k=30, stride=15, pad=7); source_resblock k=7
    //   Stage 1: 256 -> 128, rate=5, k=11, pad=3; source_down Conv1d(18, 128, k=6,  stride=3,  pad=1); source_resblock k=7
    //   Stage 2: 128 -> 64,  rate=3, k=7,  pad=2 (+ reflect_pad(1, 0)); source_down Conv1d(18, 64, k=1, stride=1, pad=0); source_resblock k=11
    std::vector<float> ups_w_[3];
    std::vector<float> ups_b_[3];

    std::vector<float> source_downs_w_[3];
    std::vector<float> source_downs_b_[3];
    SnakeResBlock source_resblocks_[3];

    // 3 ResBlocks (kernels 3, 7, 11) per upsample stage = 9 ResBlocks total
    SnakeResBlock resblocks_[3][3];

    // Post-conv: Conv1d(64, 18, 7, pad=3) + 960-sample (40ms) trim_fade
    std::vector<float> conv_post_w_;
    std::vector<float> conv_post_b_;
    std::vector<float> trim_fade_960_;

    void run_snake_resblock(
        const SnakeResBlock& rb,
        float* x,                   // [rb.channels, T] modified in-place
        int64_t T,
        const ThreadPool* pool
    ) const;
};

} // namespace kitteni
