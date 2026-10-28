#include "cloning/s3_tokenizer.hpp"
#include "kernels/dsp_fft.hpp"
#include "kernels/bitnet_ops.hpp"
#include "kernels/nn_ops.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace kitteni {

namespace {

std::vector<float> get_f32_tok(const SafetensorsFile& sf, const std::string& key) {
    TensorView tv = sf.get_tensor(key);
    size_t n = tv.numel();
    std::vector<float> out(n);
    std::memcpy(out.data(), tv.data, n * sizeof(float));
    return out;
}

inline float gelu_exact_tok(float x) {
    return 0.5f * x * (1.0f + std::erff(x * 0.70710678118654752440f));
}

// Parallel dense matrix multiplication `Y[T, out_dim] = X[T, in_dim] @ W^T[out_dim, in_dim] + bias[out_dim]`
void dense_matmul_rows_f32(
    const float* X,
    const float* W,
    const float* bias,
    int64_t T,
    int64_t in_dim,
    int64_t out_dim,
    float* Y,
    const ThreadPool* pool
) {
    auto compute_cols = [&](int64_t o_begin, int64_t o_end, int /*tid*/) {
        for (int64_t o = o_begin; o < o_end; ++o) {
            const float* w_row = W + o * in_dim;
            const float b_val = bias ? bias[o] : 0.0f;
            for (int64_t t = 0; t < T; ++t) {
                const float* x_row = X + t * in_dim;
                float sum = b_val;
                for (int64_t i = 0; i < in_dim; ++i) {
                    sum += x_row[i] * w_row[i];
                }
                Y[t * out_dim + o] = sum;
            }
        }
    };

    if (pool && out_dim >= 32) {
        pool->parallel_for(out_dim, compute_cols);
    } else {
        compute_cols(0, out_dim, 0);
    }
}

} // namespace

void S3AudioTokenizer::load(const SafetensorsFile& sf, const std::string& prefix) {
    mel_filters_ = get_f32_tok(sf, prefix + "_mel_filters");
    window_400_  = get_f32_tok(sf, prefix + "window");

    conv1_w_ = get_f32_tok(sf, prefix + "encoder.conv1.weight");
    conv1_b_ = get_f32_tok(sf, prefix + "encoder.conv1.bias");
    conv2_w_ = get_f32_tok(sf, prefix + "encoder.conv2.weight");
    conv2_b_ = get_f32_tok(sf, prefix + "encoder.conv2.bias");

    for (int i = 0; i < 6; ++i) {
        std::string bp = prefix + "encoder.blocks." + std::to_string(i) + ".";
        AttentionBlock& b = blocks_[i];
        b.attn_ln_w = get_f32_tok(sf, bp + "attn_ln.weight");
        b.attn_ln_b = get_f32_tok(sf, bp + "attn_ln.bias");

        b.q_w   = get_f32_tok(sf, bp + "attn.query.weight");
        b.q_b   = get_f32_tok(sf, bp + "attn.query.bias");
        b.k_w   = get_f32_tok(sf, bp + "attn.key.weight");
        b.v_w   = get_f32_tok(sf, bp + "attn.value.weight");
        b.v_b   = get_f32_tok(sf, bp + "attn.value.bias");
        b.out_w = get_f32_tok(sf, bp + "attn.out.weight");
        b.out_b = get_f32_tok(sf, bp + "attn.out.bias");
        b.fsmn_w = get_f32_tok(sf, bp + "attn.fsmn_block.weight"); // [1280, 1, 31]

        b.mlp_ln_w = get_f32_tok(sf, bp + "mlp_ln.weight");
        b.mlp_ln_b = get_f32_tok(sf, bp + "mlp_ln.bias");
        b.mlp0_w   = get_f32_tok(sf, bp + "mlp.0.weight");
        b.mlp0_b   = get_f32_tok(sf, bp + "mlp.0.bias");
        b.mlp2_w   = get_f32_tok(sf, bp + "mlp.2.weight");
        b.mlp2_b   = get_f32_tok(sf, bp + "mlp.2.bias");
    }

    proj_down_w_ = get_f32_tok(sf, prefix + "quantizer._codebook.project_down.weight");
    proj_down_b_ = get_f32_tok(sf, prefix + "quantizer._codebook.project_down.bias");
}

std::vector<int32_t> S3AudioTokenizer::encode_16k(
    const float* wav_16k,
    int64_t num_samples,
    const ThreadPool* pool
) const {
    if (num_samples < 400) return {};

    // 1. Extract 128-bin log-Mel spectrogram `[128, T_mel]` at 100 Hz (`hop=160`)
    int64_t T_mel = 0;
    std::vector<float> mel_128xT = extract_s3tokenizer_mel128_16k(
        wav_16k, num_samples, mel_filters_.data(), window_400_.data(), T_mel
    );
    if (T_mel <= 0) return {};

    // Note: S3TokenizerV2 pads short audio (< 3000 mel frames = 30s) with zeros up to 3000 mel frames
    // inside `_quantize_mixed_batch`? Wait: let's check `S3TokenizerV2.quantize` in `model_v2.py`!
    // In `quantize`:
    //   `long_audio_mask = mel_len > max_frames` (where `max_frames = 3000`).
    //   `if long_audio_mask.any(): ... else: hidden, code_len = self.encoder(mel, mel_len); code = self.quantizer.encode(hidden)`!
    // So for audio <= 30 seconds (`T_mel <= 3000`), `S3TokenizerV2.quantize` does NOT pad to 3000 frames!
    // It runs `self.encoder(mel, mel_len)` directly on the exact length `T_mel`!

    // 2. `conv1`: Conv1d(128 -> 1280, k=3, stride=2, pad=1) + GELU
    const int64_t T1 = (T_mel + 2 * 1 - 3) / 2 + 1;
    std::vector<float> c1_out(static_cast<size_t>(1280 * T1));
    conv1d_f32(mel_128xT.data(), conv1_w_.data(), conv1_b_.data(), 128, 1280, T_mel, 3, 2, 1, 1, c1_out.data(), pool);
    for (float& v : c1_out) v = gelu_exact_tok(v);

    // 3. `conv2`: Conv1d(1280 -> 1280, k=3, stride=2, pad=1) + GELU
    const int64_t T2 = (T1 + 2 * 1 - 3) / 2 + 1;
    std::vector<float> c2_out(static_cast<size_t>(1280 * T2));
    conv1d_f32(c1_out.data(), conv2_w_.data(), conv2_b_.data(), 1280, 1280, T1, 3, 2, 1, 1, c2_out.data(), pool);
    for (float& v : c2_out) v = gelu_exact_tok(v);

    // Permute `[1280, T2] -> [T2, 1280]`
    const int64_t D = 1280;
    const int64_t H = 20;
    const int64_t Dh = 64;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(Dh)); // (64^-0.25)^2 = 1 / 8 = 0.125

    std::vector<float> x(static_cast<size_t>(T2 * D));
    for (int64_t c = 0; c < D; ++c) {
        for (int64_t t = 0; t < T2; ++t) {
            x[t * D + c] = c2_out[c * T2 + t];
        }
    }

    // 4. Run 6 `ResidualAttentionBlock`s
    std::vector<float> norm_x(static_cast<size_t>(T2 * D));
    std::vector<float> q_buf(static_cast<size_t>(T2 * D));
    std::vector<float> k_buf(static_cast<size_t>(T2 * D));
    std::vector<float> v_buf(static_cast<size_t>(T2 * D));
    std::vector<float> fsmn_mem(static_cast<size_t>(T2 * D));
    std::vector<float> attn_wv(static_cast<size_t>(T2 * D));
    std::vector<float> proj_out(static_cast<size_t>(T2 * D));
    std::vector<float> mlp_mid(static_cast<size_t>(T2 * 5120));

    for (int l = 0; l < 6; ++l) {
        const AttentionBlock& blk = blocks_[l];

        // a) LayerNorm(1280, eps=1e-5)
        for (int64_t t = 0; t < T2; ++t) {
            layer_norm_f32(x.data() + t * D, blk.attn_ln_w.data(), blk.attn_ln_b.data(), D, 1e-5f, norm_x.data() + t * D);
        }

        // b) Q, K, V projections
        dense_matmul_rows_f32(norm_x.data(), blk.q_w.data(), blk.q_b.data(), T2, D, D, q_buf.data(), pool);
        dense_matmul_rows_f32(norm_x.data(), blk.k_w.data(), nullptr,        T2, D, D, k_buf.data(), pool);
        dense_matmul_rows_f32(norm_x.data(), blk.v_w.data(), blk.v_b.data(), T2, D, D, v_buf.data(), pool);

        // c) `forward_fsmn` on unrotated `v_buf`:
        //    Depthwise 1D conv over time (kernel_size=31, left_pad=15, right_pad=15) + residual `v_buf`
        for (int64_t t = 0; t < T2; ++t) {
            float* dst = fsmn_mem.data() + t * D;
            const float* v_cur = v_buf.data() + t * D;
            std::memcpy(dst, v_cur, static_cast<size_t>(D) * sizeof(float));

            for (int k = 0; k < 31; ++k) {
                int64_t src_t = t + k - 15;
                if (src_t < 0 || src_t >= T2) continue;
                const float* v_src = v_buf.data() + src_t * D;
                for (int64_t c = 0; c < D; ++c) {
                    dst[c] += v_src[c] * blk.fsmn_w[c * 31 + k];
                }
            }
        }

        // d) Apply RoPE (`theta=10000.0`, `dim=64`) to Q and K:
        //    Notice `precompute_freqs_cis(64, end, theta=10000.0)` in `model_v2.py`!
        //    In `precompute_freqs_cis`:
        //      `freqs = 1.0 / (10000 ** (arange(0, 64, 2) / 64))` (32 frequencies)
        //      `freqs_cis = cat((freqs_cis, freqs_cis), dim=-1)` (length 64!)
        //    And in `apply_rotary_emb`:
        //      `cos` and `sin` have shape `[T, 64]` where `cos[t, i] == cos[t, i + 32]` for `i in [0..31]`!
        //      `half_l, half_r = x[..., :32], x[..., 32:]`
        //      `x_r = cat((-half_r, half_l), dim=-1)`
        //      `out = x * cos + x_r * sin`
        //    Which is identical to NeoX RoPE with `rope_theta = 10000.0f`!
        for (int64_t t = 0; t < T2; ++t) {
            rope_neox_inplace(q_buf.data() + t * D, H, Dh, t, 10000.0f);
            rope_neox_inplace(k_buf.data() + t * D, H, Dh, t, 10000.0f);
        }

        // e) Multi-head full attention over `T2` frames (20 heads x 64 dim)
        auto run_heads = [&](int64_t h_begin, int64_t h_end, int /*tid*/) {
            std::vector<float> scores(static_cast<size_t>(T2));
            for (int64_t h = h_begin; h < h_end; ++h) {
                for (int64_t t1 = 0; t1 < T2; ++t1) {
                    const float* q_vec = q_buf.data() + t1 * D + h * Dh;
                    float max_sc = -1e30f;
                    for (int64_t t2 = 0; t2 < T2; ++t2) {
                        const float* k_vec = k_buf.data() + t2 * D + h * Dh;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < Dh; ++d) dot += q_vec[d] * k_vec[d];
                        float s = dot * attn_scale;
                        scores[t2] = s;
                        if (s > max_sc) max_sc = s;
                    }
                    float sum_exp = 0.0f;
                    for (int64_t t2 = 0; t2 < T2; ++t2) {
                        scores[t2] = std::exp(scores[t2] - max_sc);
                        sum_exp += scores[t2];
                    }
                    const float inv_sum = 1.0f / sum_exp;

                    float* out_vec = attn_wv.data() + t1 * D + h * Dh;
                    for (int64_t d = 0; d < Dh; ++d) out_vec[d] = 0.0f;
                    for (int64_t t2 = 0; t2 < T2; ++t2) {
                        const float w = scores[t2] * inv_sum;
                        const float* v_vec = v_buf.data() + t2 * D + h * Dh;
                        for (int64_t d = 0; d < Dh; ++d) out_vec[d] += w * v_vec[d];
                    }
                }
            }
        };
        if (pool) {
            pool->parallel_for(H, run_heads);
        } else {
            run_heads(0, H, 0);
        }

        // f) `self.out(wv) + fsm_memory` + residual onto `x`
        dense_matmul_rows_f32(attn_wv.data(), blk.out_w.data(), blk.out_b.data(), T2, D, D, proj_out.data(), pool);
        for (int64_t i = 0; i < T2 * D; ++i) {
            x[i] += proj_out[i] + fsmn_mem[i];
        }

        // g) `mlp_ln` + MLP (`Linear(1280->5120) -> GELU -> Linear(5120->1280)`) + residual
        for (int64_t t = 0; t < T2; ++t) {
            layer_norm_f32(x.data() + t * D, blk.mlp_ln_w.data(), blk.mlp_ln_b.data(), D, 1e-5f, norm_x.data() + t * D);
        }
        dense_matmul_rows_f32(norm_x.data(), blk.mlp0_w.data(), blk.mlp0_b.data(), T2, D, 5120, mlp_mid.data(), pool);
        for (float& v : mlp_mid) v = gelu_exact_tok(v);
        dense_matmul_rows_f32(mlp_mid.data(), blk.mlp2_w.data(), blk.mlp2_b.data(), T2, 5120, D, proj_out.data(), pool);
        for (int64_t i = 0; i < T2 * D; ++i) {
            x[i] += proj_out[i];
        }
    }

    // 5. `FSQCodebook.encode`:
    //    `h = project_down(x)` (`[T2, 8]`)
    //    `h = round(tanh(h) * 0.9990000128746033) + 1` in `{0, 1, 2}`
    //    `ind = sum_{k=0..7} h[k] * 3^k` in `[0..6560]`
    static constexpr int32_t kPowers3[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
    std::vector<int32_t> tokens(static_cast<size_t>(T2));

    for (int64_t t = 0; t < T2; ++t) {
        const float* x_row = x.data() + t * D;
        int32_t code = 0;
        for (int k = 0; k < 8; ++k) {
            const float* w_row = proj_down_w_.data() + k * D;
            float sum = proj_down_b_[k];
            for (int64_t c = 0; c < D; ++c) {
                sum += x_row[c] * w_row[c];
            }
            float th = std::tanh(sum) * 0.9990000128746033f;
            int digit = static_cast<int>(std::nearbyintf(th)) + 1;
            digit = std::clamp(digit, 0, 2);
            code += digit * kPowers3[k];
        }
        tokens[t] = code;
    }

    return tokens;
}

} // namespace kitteni
