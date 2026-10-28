#pragma once

#include "core/mmap_loader.hpp"
#include "core/thread_pool.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace kitteni {

// Native C++20 implementation of pyannote's `XVectorSincNet` (`speaker/model.safetensors`):
// - `wav_norm1d` (InstanceNorm1d(1, affine=True))
// - `ParamSincFilterbank(80, 251, stride=10)` (40 cos + 40 sin band-pass sinc filters) -> abs -> MaxPool1d(3) -> InstanceNorm1d(80) -> LeakyReLU
// - `conv1d.1(80, 60, 5)` -> MaxPool1d(3) -> InstanceNorm1d(60) -> LeakyReLU
// - `conv1d.2(60, 60, 5)` -> MaxPool1d(3) -> InstanceNorm1d(60) -> LeakyReLU
// - 5 TDNN blocks:
//     channels = (512, 512, 512, 512, 1500)
//     kernels  = (5, 3, 3, 1, 1)
//     dilations= (1, 2, 3, 1, 1)
//     each: Conv1d -> LeakyReLU(0.01) -> BatchNorm1d
// - Statistics pooling: concat(mean(dim=-1), std(dim=-1, correction=1)) -> [3000]
// - Linear(3000, 512) -> L2 unit normalization -> [512] speaker embedding for `spk_proj`.
class SincNetSpeakerEncoder {
public:
    SincNetSpeakerEncoder() = default;

    // Load from `speaker/model.safetensors`
    void load(const std::string& safetensors_path);

    // Encode a 16 kHz mono waveform into a unit-norm 512-dim speaker embedding
    std::vector<float> encode_16k(
        const float* wav_16k,
        int64_t num_samples,
        const ThreadPool* pool = nullptr
    ) const;

private:
    float wav_norm_w_ = 1.0f;
    float wav_norm_b_ = 0.0f;

    // Pre-built [80, 251] sinc filters built once at load time from `low_hz_` and `band_hz_`
    std::vector<float> sinc_filters_; // [80 * 251]

    std::vector<float> sinc_conv_w_[2]; // [60, 80, 5] and [60, 60, 5]
    std::vector<float> sinc_conv_b_[2]; // [60] and [60]

    std::vector<float> sinc_inorm_w_[3]; // [80], [60], [60]
    std::vector<float> sinc_inorm_b_[3]; // [80], [60], [60]

    struct TdnnLayer {
        int64_t in_ch = 0;
        int64_t out_ch = 0;
        int64_t kernel_size = 1;
        int64_t dilation = 1;
        std::vector<float> conv_w;
        std::vector<float> conv_b;
        std::vector<float> bn_scale; // folded `gamma / sqrt(running_var + 1e-5)`
        std::vector<float> bn_shift; // folded `beta - running_mean * bn_scale`
    };
    TdnnLayer tdnns_[5];

    std::vector<float> emb_linear_w_; // [512, 3000]
    std::vector<float> emb_linear_b_; // [512]
};

} // namespace kitteni
