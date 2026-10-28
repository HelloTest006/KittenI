#pragma once

#include "core/mmap_loader.hpp"
#include "core/thread_pool.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace kitteni {

// Native C++20 implementation of CosyVoice/Chatterbox `CAMPPlus` speaker encoder (192-dim x-vector):
// - Input: 16 kHz waveform -> Kaldi 80-bin fbank with CMN (`[T, 80]`) -> permute to `[80, T]`
// - `head` (FCM 2D ResNet over `[C, F=80, T]`):
//     conv1(1->32, 3x3, pad=1) + bn1 + relu
//     layer1 (2x BasicResBlock, stride=(2,1) on block 0 -> F=40)
//     layer2 (2x BasicResBlock, stride=(2,1) on block 0 -> F=20)
//     conv2(32->32, 3x3, stride=(2,1), pad=1 -> F=10) + bn2 + relu -> reshape `[320, T]`
// - `xvector`:
//     tdnn: Conv1d(320->128, k=5, stride=2, pad=2) + BN + ReLU
//     block1 (12x CAMDenseTDNNLayer, growth=32, bn_ch=128, k=3, dil=1) -> 512 ch -> transit1(512->256)
//     block2 (24x CAMDenseTDNNLayer, growth=32, bn_ch=128, k=3, dil=2) -> 1024 ch -> transit2(1024->512)
//     block3 (16x CAMDenseTDNNLayer, growth=32, bn_ch=128, k=3, dil=2) -> 1024 ch -> transit3(1024->512)
//     out_nonlinear (BN + ReLU, 512 ch) -> StatsPool (mean + unbiased std -> 1024)
//     dense: Conv1d(1024->192, k=1, bias=False) + BatchNorm1d(192, affine=False) -> `[192]` x-vector.
class CampPlusEncoder {
public:
    struct FoldedBN {
        std::vector<float> scale; // gamma / sqrt(var + 1e-5)
        std::vector<float> shift; // beta - mean * scale
    };

    struct BasicResBlock2D {
        int stride_f = 1;
        std::vector<float> conv1_w; // [32, 32, 3, 3]
        FoldedBN bn1;
        std::vector<float> conv2_w; // [32, 32, 3, 3]
        FoldedBN bn2;
        bool has_shortcut = false;
        std::vector<float> shortcut_w; // [32, 32, 1, 1]
        FoldedBN shortcut_bn;
    };

    struct CamDenseLayer {
        int64_t in_ch = 0;
        int64_t out_ch = 32;
        int64_t bn_ch = 128;
        int64_t kernel_size = 3;
        int64_t dilation = 1;
        FoldedBN nonlinear1_bn;
        std::vector<float> linear1_w; // [128, in_ch, 1]
        FoldedBN nonlinear2_bn;
        std::vector<float> cam_linear_local_w; // [32, 128, 3]
        std::vector<float> cam_linear1_w;      // [64, 128, 1]
        std::vector<float> cam_linear1_b;      // [64]
        std::vector<float> cam_linear2_w;      // [32, 64, 1]
        std::vector<float> cam_linear2_b;      // [32]
    };

    struct TransitBlock {
        int64_t in_ch = 0;
        int64_t out_ch = 0;
        FoldedBN bn;
        std::vector<float> linear_w; // [out_ch, in_ch, 1]
    };

    CampPlusEncoder() = default;

    // Load from a SafetensorsFile containing `speaker_encoder.*` keys
    void load(const SafetensorsFile& sf, const std::string& prefix = "speaker_encoder.");

    // Compute 192-dim x-vector from 16 kHz mono audio
    std::vector<float> encode_16k(
        const float* wav_16k,
        int64_t num_samples,
        const ThreadPool* pool = nullptr
    ) const;

private:
    // FCM 2D head
    std::vector<float> head_conv1_w_; // [32, 1, 3, 3]
    FoldedBN head_bn1_;
    BasicResBlock2D head_layer1_[2];
    BasicResBlock2D head_layer2_[2];
    std::vector<float> head_conv2_w_; // [32, 32, 3, 3]
    FoldedBN head_bn2_;

    // TDNN initial layer: 320 -> 128, k=5, stride=2, pad=2
    std::vector<float> tdnn_w_; // [128, 320, 5]
    FoldedBN tdnn_bn_;

    std::vector<CamDenseLayer> block1_; // 12 layers
    TransitBlock transit1_;             // 512 -> 256
    std::vector<CamDenseLayer> block2_; // 24 layers
    TransitBlock transit2_;             // 1024 -> 512
    std::vector<CamDenseLayer> block3_; // 16 layers
    TransitBlock transit3_;             // 1024 -> 512

    FoldedBN out_nonlinear_bn_;         // 512
    std::vector<float> dense_w_;        // [192, 1024, 1]
    FoldedBN dense_bn_;                 // 192 (affine=False)
};

} // namespace kitteni
