#pragma once

#include "kitteni/common.hpp"
#include "core/mmap_loader.hpp"
#include "core/thread_pool.hpp"
#include "kernels/bitnet_ops.hpp"
#include <vector>
#include <string>
#include <random>
#include <cstdint>

namespace kitteni {

// Reads raw tensor blobs (`<prefix>/constants/<idx>`) from a TorchScript `.pt` (ZIP) archive
class TorchScriptZipConstants {
public:
    explicit TorchScriptZipConstants(const std::string& pt_path);

    bool has_constant(int idx) const;
    TensorView get_constant(int idx) const;
    std::vector<float> get_f32_vec(int idx) const;

private:
    MmapFile mmap_;
    std::map<int, TensorView> constants_;
};

// S3 Student Flow (`student_w4` / `student_w8`) architecture and weights:
// Maps S3 codec tokens + reference prompt tokens/mel/x-vector -> 80-bin mel spectrogram (`[80, T_mel]`)
// Uses native low-bit `BitLinear` and `BitConv1d` integer kernels.
class S3StudentFlow {
public:
    struct ConformerLayer {
        std::vector<float> norm_mha_w, norm_mha_b;
        BitNetWeight linear_q, linear_k, linear_v, linear_pos, linear_out;
        std::vector<float> pos_bias_u, pos_bias_v; // [8, 46]
        std::vector<float> norm_ff_w, norm_ff_b;
        BitNetWeight ff_w1, ff_w2;
    };

    struct TransformerBlock {
        std::vector<float> norm1_w, norm1_b;
        BitNetWeight to_q, to_k, to_v, to_out; // [352, 176] and [176, 352]
        std::vector<float> norm3_w, norm3_b;
        BitNetWeight ff_proj, ff_out;          // [704, 176] and [176, 704]
    };

    struct EstimatorBlock {
        // CausalResnetBlock1D
        BitNetWeight block1_conv;              // [176, in_ch, 3] causal_pad=2
        std::vector<float> block1_ln_w, block1_ln_b;
        std::vector<float> time_emb_proj;      // [176] pre-evaluated `mlp(Mish(t_emb))` for t=0, r=1
        BitNetWeight block2_conv;              // [176, 176, 3] causal_pad=2
        std::vector<float> block2_ln_w, block2_ln_b;
        BitNetWeight res_conv;                 // [176, in_ch, 1]

        // 4 BasicTransformerBlocks
        TransformerBlock transformers[4];

        // Final CausalConv1d(176, 176, 3) on down_blocks and up_blocks (unused on mid_blocks)
        BitNetWeight post_conv;
        bool has_post_conv = false;
    };

    S3StudentFlow() = default;

    // Load from `cpp/student_w4/decoder.pt` or `cpp/student_w8/decoder.pt`
    void load_from_torchscript_zip(const TorchScriptZipConstants& zip, int bits = 4);

    // Run 1-step MeanFlow Euler (`meanflow_forward` in `student_flow.py`):
    // Returns mel spectrogram `[80, T_speech_mel]` where `T_speech_mel = speech_tokens.size() * 2`.
    // If `fixed_noise` is nullptr, draws `z ~ N(0, 1)` from `rng`.
    std::vector<float> forward(
        const std::vector<int32_t>& speech_tokens, // [T_tok] (including 3 trailing 4299 silence tokens)
        const std::vector<int32_t>& prompt_tokens, // [T_prompt]
        const float* prompt_feat,                  // [T_prompt * 2, 80] row-major
        int64_t prompt_feat_frames,                // T_prompt * 2
        const float* spk_embedding_192,            // [192] CAMPPlus x-vector
        std::mt19937& rng,
        const ThreadPool* pool = nullptr,
        const float* fixed_noise = nullptr         // optional [80, 2 * (T_prompt + T_tok)] for exact parity tests
    ) const;

private:
    int64_t enc_dim_ = 368;
    int64_t enc_ff_dim_ = 1472;
    int64_t enc_heads_ = 8;
    int64_t enc_head_dim_ = 46;
    int64_t est_dim_ = 176;
    int64_t est_heads_ = 8;
    int64_t est_head_dim_ = 44;

    // CausalMaskedDiffWithXvec top-level weights
    std::vector<float> spk_embed_affine_w_; // [80, 192]
    std::vector<float> spk_embed_affine_b_; // [80]
    std::vector<float> input_embedding_;    // [6561, 368]

    // UpsampleConformerEncoder
    BitNetWeight embed_linear_;             // [368, 368]
    std::vector<float> embed_ln_w_, embed_ln_b_;
    std::vector<float> rel_pe_table_;       // [9999, 368] (CONSTANTS.c12)

    BitNetWeight pre_lookahead_conv1_;      // [368, 368, 4]
    BitNetWeight pre_lookahead_conv2_;      // [368, 368, 3]

    std::vector<ConformerLayer> encoders_;  // 6 blocks
    BitNetWeight up_layer_conv_;            // [368, 368, 5]
    BitNetWeight up_embed_linear_;          // [368, 368]
    std::vector<float> up_embed_ln_w_, up_embed_ln_b_;
    std::vector<ConformerLayer> up_encoders_; // 4 blocks
    std::vector<float> after_norm_w_, after_norm_b_;
    BitNetWeight encoder_proj_;             // [80, 368]

    // ConditionalDecoder UNet (1 down_block, 12 mid_blocks, 1 up_block)
    EstimatorBlock down_block_;
    std::vector<EstimatorBlock> mid_blocks_; // 12 blocks
    EstimatorBlock up_block_;

    BitNetWeight final_block_conv_;         // [176, 176, 3] causal_pad=2
    std::vector<float> final_block_ln_w_, final_block_ln_b_;
    std::vector<float> final_proj_w_;       // [80, 176, 1] (dense!)
    std::vector<float> final_proj_b_;       // [80]

    void run_conformer_stack(
        float* xs,                          // [T, 368] modified in-place
        int64_t T,
        const std::vector<ConformerLayer>& layers,
        const ThreadPool* pool
    ) const;

    void run_estimator_block(
        const EstimatorBlock& blk,
        const float* x_in,                  // [in_ch, T]
        int64_t in_ch,
        int64_t T,
        float* x_out,                       // [176, T]
        const ThreadPool* pool
    ) const;
};

} // namespace kitteni
