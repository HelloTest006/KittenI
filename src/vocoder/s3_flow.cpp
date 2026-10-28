#include "vocoder/s3_flow.hpp"
#include "kernels/nn_ops.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace kitteni {

TorchScriptZipConstants::TorchScriptZipConstants(const std::string& pt_path) : mmap_(pt_path) {
    const uint8_t* base = mmap_.data();
    const size_t total = mmap_.size();
    if (total < 22) {
        throw std::runtime_error("ZIP file too small: " + pt_path);
    }

    // Find End of Central Directory (EOCD) signature 0x06054b50 near the end of the file
    size_t eocd_pos = total;
    size_t min_search = (total > 65557) ? (total - 65557) : 0;
    for (size_t pos = total - 22; pos >= min_search; --pos) {
        uint32_t sig = 0;
        std::memcpy(&sig, base + pos, 4);
        if (sig == 0x06054B50U) {
            eocd_pos = pos;
            break;
        }
        if (pos == 0) break;
    }
    if (eocd_pos == total) {
        throw std::runtime_error("Cannot find ZIP End of Central Directory in: " + pt_path);
    }

    uint16_t num_entries16 = 0;
    uint32_t cd_size32 = 0, cd_offset32 = 0;
    std::memcpy(&num_entries16, base + eocd_pos + 10, 2);
    std::memcpy(&cd_size32, base + eocd_pos + 12, 4);
    std::memcpy(&cd_offset32, base + eocd_pos + 16, 4);

    uint64_t cd_offset = cd_offset32;
    uint64_t num_entries = num_entries16;

    // Check for ZIP64 End of Central Directory Locator (20 bytes before EOCD, sig 0x07064b50)
    if (eocd_pos >= 20) {
        uint32_t loc_sig = 0;
        std::memcpy(&loc_sig, base + eocd_pos - 20, 4);
        if (loc_sig == 0x07064B50U) {
            uint64_t zip64_eocd_off = 0;
            std::memcpy(&zip64_eocd_off, base + eocd_pos - 12, 8);
            if (zip64_eocd_off + 56 <= total) {
                uint32_t z64_sig = 0;
                std::memcpy(&z64_sig, base + zip64_eocd_off, 4);
                if (z64_sig == 0x06064B50U) {
                    std::memcpy(&num_entries, base + zip64_eocd_off + 32, 8);
                    std::memcpy(&cd_offset, base + zip64_eocd_off + 48, 8);
                }
            }
        }
    }

    size_t pos = static_cast<size_t>(cd_offset);
    for (uint64_t e = 0; e < num_entries && pos + 46 <= total; ++e) {
        uint32_t sig = 0;
        std::memcpy(&sig, base + pos, 4);
        if (sig != 0x02014B50U) break; // Central Directory File Header signature

        uint16_t method = 0, name_len = 0, extra_len = 0, comment_len = 0;
        uint32_t comp_size32 = 0, local_off32 = 0;
        std::memcpy(&method, base + pos + 10, 2);
        std::memcpy(&comp_size32, base + pos + 20, 4);
        std::memcpy(&name_len, base + pos + 28, 2);
        std::memcpy(&extra_len, base + pos + 30, 2);
        std::memcpy(&comment_len, base + pos + 32, 2);
        std::memcpy(&local_off32, base + pos + 42, 4);

        if (pos + 46 + name_len + extra_len + comment_len > total) break;

        std::string name(reinterpret_cast<const char*>(base + pos + 46), name_len);
        uint64_t comp_size = comp_size32;
        uint64_t local_off = local_off32;

        // Parse ZIP64 extra field in central directory if needed
        if (comp_size32 == 0xFFFFFFFFU || local_off32 == 0xFFFFFFFFU) {
            const uint8_t* ex = base + pos + 46 + name_len;
            size_t ex_pos = 0;
            while (ex_pos + 4 <= extra_len) {
                uint16_t tag = 0, sz = 0;
                std::memcpy(&tag, ex + ex_pos, 2);
                std::memcpy(&sz, ex + ex_pos + 2, 2);
                if (tag == 0x0001U && ex_pos + 4 + sz <= extra_len) {
                    size_t f_off = ex_pos + 4;
                    if (comp_size32 == 0xFFFFFFFFU && f_off + 16 <= ex_pos + 4 + sz) {
                        std::memcpy(&comp_size, ex + f_off + 8, 8);
                        f_off += 16;
                    }
                    if (local_off32 == 0xFFFFFFFFU && f_off + 8 <= ex_pos + 4 + sz) {
                        std::memcpy(&local_off, ex + f_off, 8);
                    }
                    break;
                }
                ex_pos += 4 + sz;
            }
        }

        const std::string prefix = "decoder/constants/";
        if (method == 0 && name.rfind(prefix, 0) == 0 && local_off + 30 <= total) {
            uint16_t loc_name_len = 0, loc_extra_len = 0;
            std::memcpy(&loc_name_len, base + local_off + 26, 2);
            std::memcpy(&loc_extra_len, base + local_off + 28, 2);
            size_t data_start = static_cast<size_t>(local_off) + 30 + loc_name_len + loc_extra_len;
            if (data_start + comp_size <= total) {
                int idx = std::stoi(name.substr(prefix.size()));
                TensorView tv;
                tv.data = base + data_start;
                tv.byte_size = static_cast<size_t>(comp_size);
                tv.dtype = DType::F32;
                tv.shape = {static_cast<int64_t>(comp_size / 4)};
                constants_[idx] = tv;
            }
        }

        pos += 46 + name_len + extra_len + comment_len;
    }
}

bool TorchScriptZipConstants::has_constant(int idx) const {
    return constants_.find(idx) != constants_.end();
}

TensorView TorchScriptZipConstants::get_constant(int idx) const {
    auto it = constants_.find(idx);
    if (it == constants_.end()) {
        throw std::runtime_error("Missing TorchScript constant c" + std::to_string(idx));
    }
    return it->second;
}

std::vector<float> TorchScriptZipConstants::get_f32_vec(int idx) const {
    TensorView tv = get_constant(idx);
    size_t n = tv.byte_size / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), tv.data, n * sizeof(float));
    return out;
}

namespace {

inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float gelu_exact(float x) {
    return 0.5f * x * (1.0f + std::erff(x * 0.70710678118654752440f));
}

inline float mish_scalar(float x) {
    // mish(x) = x * tanh(softplus(x))
    float sp = (x > 20.0f) ? x : std::log1pf(std::exp(x));
    return x * std::tanh(sp);
}

// Transpose `[rows, cols]` -> `[cols, rows]`
void transpose_2d(const float* src, int64_t rows, int64_t cols, float* dst) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

} // namespace

void S3StudentFlow::load_from_torchscript_zip(const TorchScriptZipConstants& zip, int bits) {
    spk_embed_affine_w_ = zip.get_f32_vec(1);
    spk_embed_affine_b_ = zip.get_f32_vec(2);
    input_embedding_    = zip.get_f32_vec(3);

    auto load_bit_linear = [&](int w_idx, int b_idx, int64_t out_d, int64_t in_d, BitNetWeight& bw) {
        TensorView w_tv = zip.get_constant(w_idx);
        const float* b_ptr = (b_idx >= 0) ? zip.get_constant(b_idx).ptr<float>() : nullptr;
        bw.init_from_dequantized_f32(w_tv.ptr<float>(), b_ptr, out_d, in_d, 1, bits, 0, 0);
    };

    auto load_bit_conv = [&](int w_idx, int b_idx, int64_t out_c, int64_t in_c, int64_t k_sz,
                             int64_t c_pad_l, int64_t c_pad_r, BitNetWeight& bw) {
        TensorView w_tv = zip.get_constant(w_idx);
        const float* b_ptr = (b_idx >= 0) ? zip.get_constant(b_idx).ptr<float>() : nullptr;
        bw.init_from_dequantized_f32(w_tv.ptr<float>(), b_ptr, out_c, in_c, k_sz, bits, c_pad_l, c_pad_r);
    };

    load_bit_linear(5, 6, 368, 368, embed_linear_);
    embed_ln_w_   = zip.get_f32_vec(7);
    embed_ln_b_   = zip.get_f32_vec(8);
    rel_pe_table_ = zip.get_f32_vec(12); // [1, 9999, 368]

    // PreLookaheadLayer pads right by 3 before conv1 (k=4) and pads left by 2 before conv2 (k=3)
    load_bit_conv(13, 14, 368, 368, 4, /*c_pad_l=*/0, /*c_pad_r=*/3, pre_lookahead_conv1_);
    load_bit_conv(15, 16, 368, 368, 3, /*c_pad_l=*/2, /*c_pad_r=*/0, pre_lookahead_conv2_);

    auto load_conformer = [&](int base_c, ConformerLayer& cl, bool first_encoder_block) {
        // Notice: in TorchScript trace, encoder.encoders.0 has extra constants c28, c29, c38
        // so its linear_out starts at base_c + 13 instead of base_c + 11!
        cl.norm_mha_w = zip.get_f32_vec(base_c + 0);
        cl.norm_mha_b = zip.get_f32_vec(base_c + 1);
        load_bit_linear(base_c + 2, base_c + 3, 368, 368, cl.linear_q);
        load_bit_linear(base_c + 4, base_c + 5, 368, 368, cl.linear_k);
        load_bit_linear(base_c + 6, base_c + 7, 368, 368, cl.linear_v);
        load_bit_linear(base_c + 8, -1,         368, 368, cl.linear_pos);
        cl.pos_bias_u = zip.get_f32_vec(base_c + 9);
        cl.pos_bias_v = zip.get_f32_vec(base_c + 10);

        int out_offset = first_encoder_block ? 13 : 11;
        load_bit_linear(base_c + out_offset + 0, base_c + out_offset + 1, 368, 368, cl.linear_out);
        cl.norm_ff_w = zip.get_f32_vec(base_c + out_offset + 2);
        cl.norm_ff_b = zip.get_f32_vec(base_c + out_offset + 3);
        load_bit_linear(base_c + out_offset + 4, base_c + out_offset + 5, 1472, 368, cl.ff_w1);
        load_bit_linear(base_c + out_offset + 6, base_c + out_offset + 7, 368, 1472, cl.ff_w2);
    };

    encoders_.resize(6);
    load_conformer(17, encoders_[0], true);
    for (int i = 1; i < 6; ++i) {
        load_conformer(39 + (i - 1) * 19, encoders_[i], false);
    }

    // up_layer_conv (stride=2 nearest upsample + pad(4, 0) + conv(368, 368, k=5))
    load_bit_conv(134, 135, 368, 368, 5, /*c_pad_l=*/4, /*c_pad_r=*/0, up_layer_conv_);
    load_bit_linear(136, 137, 368, 368, up_embed_linear_);
    up_embed_ln_w_ = zip.get_f32_vec(138);
    up_embed_ln_b_ = zip.get_f32_vec(139);

    up_encoders_.resize(4);
    for (int i = 0; i < 4; ++i) {
        load_conformer(140 + i * 19, up_encoders_[i], false);
    }

    after_norm_w_ = zip.get_f32_vec(216);
    after_norm_b_ = zip.get_f32_vec(217);
    load_bit_linear(218, 219, 80, 368, encoder_proj_);

    // Helper to load an EstimatorBlock (down_blocks.0, mid_blocks.0..11, up_blocks.0)
    auto load_estimator_block = [&](int& cursor, int64_t in_ch, bool has_extra_231_234, bool has_post, EstimatorBlock& eb) {
        load_bit_conv(cursor + 0, cursor + 1, 176, in_ch, 3, /*c_pad_l=*/2, 0, eb.block1_conv);
        eb.block1_ln_w   = zip.get_f32_vec(cursor + 2);
        eb.block1_ln_b   = zip.get_f32_vec(cursor + 3);
        eb.time_emb_proj = zip.get_f32_vec(cursor + 4); // [176, 1]
        load_bit_conv(cursor + 5, cursor + 6, 176, 176, 3, /*c_pad_l=*/2, 0, eb.block2_conv);
        eb.block2_ln_w   = zip.get_f32_vec(cursor + 7);
        eb.block2_ln_b   = zip.get_f32_vec(cursor + 8);
        load_bit_conv(cursor + 9, cursor + 10, 176, in_ch, 1, 0, 0, eb.res_conv);
        cursor += 11;

        for (int tb = 0; tb < 4; ++tb) {
            if (tb == 0 && has_extra_231_234) {
                cursor += 1; // skip c231 (-1e10 mask constant)
            }
            TransformerBlock& t = eb.transformers[tb];
            t.norm1_w = zip.get_f32_vec(cursor + 0);
            t.norm1_b = zip.get_f32_vec(cursor + 1);
            cursor += 2;
            if (tb == 0 && has_extra_231_234) {
                cursor += 1; // skip c234 (8 heads constant)
            }
            load_bit_linear(cursor + 0, -1,         352, 176, t.to_q);
            load_bit_linear(cursor + 1, -1,         352, 176, t.to_k);
            load_bit_linear(cursor + 2, -1,         352, 176, t.to_v);
            load_bit_linear(cursor + 3, cursor + 4, 176, 352, t.to_out);
            t.norm3_w = zip.get_f32_vec(cursor + 5);
            t.norm3_b = zip.get_f32_vec(cursor + 6);
            load_bit_linear(cursor + 7, cursor + 8, 704, 176, t.ff_proj);
            load_bit_linear(cursor + 9, cursor + 10, 176, 704, t.ff_out);
            cursor += 11;
        }

        eb.has_post_conv = has_post;
        if (has_post) {
            load_bit_conv(cursor + 0, cursor + 1, 176, 176, 3, /*c_pad_l=*/2, 0, eb.post_conv);
            cursor += 2;
        }
    };

    int cursor = 220;
    load_estimator_block(cursor, 320, /*has_extra_231_234=*/true, /*has_post=*/true, down_block_);
    mid_blocks_.resize(12);
    for (int m = 0; m < 12; ++m) {
        load_estimator_block(cursor, 176, /*has_extra_231_234=*/false, /*has_post=*/false, mid_blocks_[m]);
    }
    load_estimator_block(cursor, 352, /*has_extra_231_234=*/false, /*has_post=*/true, up_block_);

    if (cursor != 1108) {
        throw std::runtime_error("Estimator constant cursor mismatch: expected 1108, got " + std::to_string(cursor));
    }

    load_bit_conv(1108, 1109, 176, 176, 3, /*c_pad_l=*/2, 0, final_block_conv_);
    final_block_ln_w_ = zip.get_f32_vec(1110);
    final_block_ln_b_ = zip.get_f32_vec(1111);
    final_proj_w_     = zip.get_f32_vec(1112); // [80, 176, 1]
    final_proj_b_     = zip.get_f32_vec(1113); // [80]
}

void S3StudentFlow::run_conformer_stack(
    float* xs,
    int64_t T,
    const std::vector<ConformerLayer>& layers,
    const ThreadPool* pool
) const {
    const int64_t D = enc_dim_;       // 368
    const int64_t F = enc_ff_dim_;    // 1472
    const int64_t H = enc_heads_;     // 8
    const int64_t Dh = enc_head_dim_; // 46
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    // Extract relative positional embedding slice `pos_emb` of shape `[2*T - 1, D]`
    // from `rel_pe_table_` (`[9999, 368]`, center at index 4999)
    const int64_t P_len = 2 * T - 1;
    const int64_t pe_start = 4999 - T + 1;
    const float* pos_emb = rel_pe_table_.data() + pe_start * D;

    std::vector<float> norm_x(static_cast<size_t>(T * D));
    std::vector<float> q_buf(static_cast<size_t>(T * D));
    std::vector<float> k_buf(static_cast<size_t>(T * D));
    std::vector<float> v_buf(static_cast<size_t>(T * D));
    std::vector<float> p_buf(static_cast<size_t>(P_len * D));
    std::vector<float> attn_out(static_cast<size_t>(T * D));
    std::vector<float> proj_out(static_cast<size_t>(T * D));
    std::vector<float> ff_mid(static_cast<size_t>(T * F));

    for (const auto& layer : layers) {
        // 1. LayerNorm before MHA (eps=1e-12)
        for (int64_t t = 0; t < T; ++t) {
            layer_norm_f32(xs + t * D, layer.norm_mha_w.data(), layer.norm_mha_b.data(), D, 1e-12f, norm_x.data() + t * D);
        }

        // 2. BitLinear Q, K, V, and relative pos projection P
        bitlinear_forward(layer.linear_q, norm_x.data(), T, q_buf.data(), pool);
        bitlinear_forward(layer.linear_k, norm_x.data(), T, k_buf.data(), pool);
        bitlinear_forward(layer.linear_v, norm_x.data(), T, v_buf.data(), pool);
        bitlinear_forward(layer.linear_pos, pos_emb, P_len, p_buf.data(), pool);

        // 3. RelPositionMultiHeadedAttention over H=8 heads (parallelized across heads)
        auto run_rel_heads = [&](int64_t h_begin, int64_t h_end, int /*tid*/) {
            std::vector<float> scores(static_cast<size_t>(T));
            for (int64_t h = h_begin; h < h_end; ++h) {
                const float* bias_u = layer.pos_bias_u.data() + h * Dh;
                const float* bias_v = layer.pos_bias_v.data() + h * Dh;

                for (int64_t t1 = 0; t1 < T; ++t1) {
                    const float* q_t1 = q_buf.data() + t1 * D + h * Dh;
                    float qu[46], qv[46];
                    for (int64_t d = 0; d < Dh; ++d) {
                        qu[d] = q_t1[d] + bias_u[d];
                        qv[d] = q_t1[d] + bias_v[d];
                    }

                    float max_sc = -1e30f;
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        const float* k_t2 = k_buf.data() + t2 * D + h * Dh;
                        const int64_t p_idx = (T - 1) - t1 + t2;
                        const float* p_vec = p_buf.data() + p_idx * D + h * Dh;

                        float ac = 0.0f, bd = 0.0f;
                        for (int64_t d = 0; d < Dh; ++d) {
                            ac += qu[d] * k_t2[d];
                            bd += qv[d] * p_vec[d];
                        }
                        float s = (ac + bd) * scale;
                        scores[t2] = s;
                        if (s > max_sc) max_sc = s;
                    }

                    float sum_exp = 0.0f;
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        scores[t2] = std::exp(scores[t2] - max_sc);
                        sum_exp += scores[t2];
                    }
                    const float inv_sum = 1.0f / sum_exp;

                    float* out_vec = attn_out.data() + t1 * D + h * Dh;
                    for (int64_t d = 0; d < Dh; ++d) out_vec[d] = 0.0f;
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        const float w = scores[t2] * inv_sum;
                        const float* v_t2 = v_buf.data() + t2 * D + h * Dh;
                        for (int64_t d = 0; d < Dh; ++d) {
                            out_vec[d] += w * v_t2[d];
                        }
                    }
                }
            }
        };
        if (pool) {
            pool->parallel_for(H, run_rel_heads);
        } else {
            run_rel_heads(0, H, 0);
        }

        // 4. linear_out + residual
        bitlinear_forward(layer.linear_out, attn_out.data(), T, proj_out.data(), pool);
        for (int64_t i = 0; i < T * D; ++i) xs[i] += proj_out[i];

        // 5. LayerNorm before FFN (eps=1e-12) + PositionwiseFeedForward (w_1 -> SiLU -> w_2) + residual
        for (int64_t t = 0; t < T; ++t) {
            layer_norm_f32(xs + t * D, layer.norm_ff_w.data(), layer.norm_ff_b.data(), D, 1e-12f, norm_x.data() + t * D);
        }
        bitlinear_forward(layer.ff_w1, norm_x.data(), T, ff_mid.data(), pool);
        for (int64_t i = 0; i < T * F; ++i) {
            ff_mid[i] = silu_scalar(ff_mid[i]);
        }
        bitlinear_forward(layer.ff_w2, ff_mid.data(), T, proj_out.data(), pool);
        for (int64_t i = 0; i < T * D; ++i) xs[i] += proj_out[i];
    }
}

void S3StudentFlow::run_estimator_block(
    const EstimatorBlock& blk,
    const float* x_in,
    int64_t in_ch,
    int64_t T,
    float* x_out,
    const ThreadPool* pool
) const {
    const int64_t C = est_dim_;       // 176
    const int64_t H = est_heads_;     // 8
    const int64_t Dh = est_head_dim_; // 44
    const int64_t inner_dim = H * Dh; // 352
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    std::vector<float> h1(static_cast<size_t>(C * T));
    std::vector<float> h1_tr(static_cast<size_t>(T * C));
    std::vector<float> h2(static_cast<size_t>(C * T));
    std::vector<float> res(static_cast<size_t>(C * T));

    // 1. CausalBlock1D 1: CausalConv1d(in_ch, 176, 3) -> Transpose -> LayerNorm(176, eps=1e-5) -> Transpose -> Mish
    bitconv1d_forward(blk.block1_conv, x_in, T, h1.data(), pool);
    transpose_2d(h1.data(), C, T, h1_tr.data());
    for (int64_t t = 0; t < T; ++t) {
        layer_norm_f32(h1_tr.data() + t * C, blk.block1_ln_w.data(), blk.block1_ln_b.data(), C, 1e-5f, h1_tr.data() + t * C);
    }
    transpose_2d(h1_tr.data(), T, C, h1.data());
    for (int64_t c = 0; c < C; ++c) {
        const float t_bias = blk.time_emb_proj[c];
        float* row = h1.data() + c * T;
        for (int64_t t = 0; t < T; ++t) {
            row[t] = mish_scalar(row[t]) + t_bias;
        }
    }

    // 2. CausalBlock1D 2: CausalConv1d(176, 176, 3) -> Transpose -> LayerNorm -> Transpose -> Mish
    bitconv1d_forward(blk.block2_conv, h1.data(), T, h2.data(), pool);
    transpose_2d(h2.data(), C, T, h1_tr.data());
    for (int64_t t = 0; t < T; ++t) {
        layer_norm_f32(h1_tr.data() + t * C, blk.block2_ln_w.data(), blk.block2_ln_b.data(), C, 1e-5f, h1_tr.data() + t * C);
    }
    transpose_2d(h1_tr.data(), T, C, h2.data());

    // 3. Residual 1x1 conv + sum
    bitconv1d_forward(blk.res_conv, x_in, T, res.data(), pool);
    for (int64_t i = 0; i < C * T; ++i) {
        h2[i] = mish_scalar(h2[i]) + res[i];
    }

    // 4. Transpose `[C, T] -> [T, C]` and run 4 BasicTransformerBlocks
    std::vector<float> cur_t(static_cast<size_t>(T * C));
    transpose_2d(h2.data(), C, T, cur_t.data());

    std::vector<float> norm_t(static_cast<size_t>(T * C));
    std::vector<float> q_buf(static_cast<size_t>(T * inner_dim));
    std::vector<float> k_buf(static_cast<size_t>(T * inner_dim));
    std::vector<float> v_buf(static_cast<size_t>(T * inner_dim));
    std::vector<float> attn_out(static_cast<size_t>(T * inner_dim));
    std::vector<float> proj_out(static_cast<size_t>(T * C));
    std::vector<float> ff_mid(static_cast<size_t>(T * 704));

    for (int tb = 0; tb < 4; ++tb) {
        const TransformerBlock& tr = blk.transformers[tb];
        for (int64_t t = 0; t < T; ++t) {
            layer_norm_f32(cur_t.data() + t * C, tr.norm1_w.data(), tr.norm1_b.data(), C, 1e-5f, norm_t.data() + t * C);
        }

        bitlinear_forward(tr.to_q, norm_t.data(), T, q_buf.data(), pool);
        bitlinear_forward(tr.to_k, norm_t.data(), T, k_buf.data(), pool);
        bitlinear_forward(tr.to_v, norm_t.data(), T, v_buf.data(), pool);

        // Multi-head self-attention (8 heads x 44 dim = 352, parallelized across heads)
        (void)in_ch;
        auto run_est_heads = [&](int64_t h_begin, int64_t h_end, int /*tid*/) {
            std::vector<float> scores(static_cast<size_t>(T));
            for (int64_t h = h_begin; h < h_end; ++h) {
                for (int64_t t1 = 0; t1 < T; ++t1) {
                    const float* q_vec = q_buf.data() + t1 * inner_dim + h * Dh;
                    float max_sc = -1e30f;
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        const float* k_vec = k_buf.data() + t2 * inner_dim + h * Dh;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < Dh; ++d) dot += q_vec[d] * k_vec[d];
                        float s = dot * attn_scale;
                        scores[t2] = s;
                        if (s > max_sc) max_sc = s;
                    }
                    float sum_exp = 0.0f;
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        scores[t2] = std::exp(scores[t2] - max_sc);
                        sum_exp += scores[t2];
                    }
                    const float inv_sum = 1.0f / sum_exp;
                    float* out_vec = attn_out.data() + t1 * inner_dim + h * Dh;
                    for (int64_t d = 0; d < Dh; ++d) out_vec[d] = 0.0f;
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        const float w = scores[t2] * inv_sum;
                        const float* v_vec = v_buf.data() + t2 * inner_dim + h * Dh;
                        for (int64_t d = 0; d < Dh; ++d) out_vec[d] += w * v_vec[d];
                    }
                }
            }
        };
        if (pool) {
            pool->parallel_for(H, run_est_heads);
        } else {
            run_est_heads(0, H, 0);
        }

        bitlinear_forward(tr.to_out, attn_out.data(), T, proj_out.data(), pool);
        for (int64_t i = 0; i < T * C; ++i) cur_t[i] += proj_out[i];

        for (int64_t t = 0; t < T; ++t) {
            layer_norm_f32(cur_t.data() + t * C, tr.norm3_w.data(), tr.norm3_b.data(), C, 1e-5f, norm_t.data() + t * C);
        }
        bitlinear_forward(tr.ff_proj, norm_t.data(), T, ff_mid.data(), pool);
        for (int64_t i = 0; i < T * 704; ++i) {
            ff_mid[i] = gelu_exact(ff_mid[i]);
        }
        bitlinear_forward(tr.ff_out, ff_mid.data(), T, proj_out.data(), pool);
        for (int64_t i = 0; i < T * C; ++i) cur_t[i] += proj_out[i];
    }

    // 5. Transpose back `[T, C] -> [C, T]` and apply optional `post_conv`
    if (blk.has_post_conv) {
        transpose_2d(cur_t.data(), T, C, h2.data());
        bitconv1d_forward(blk.post_conv, h2.data(), T, x_out, pool);
    } else {
        transpose_2d(cur_t.data(), T, C, x_out);
    }
}

std::vector<float> S3StudentFlow::forward(
    const std::vector<int32_t>& speech_tokens,
    const std::vector<int32_t>& prompt_tokens,
    const float* prompt_feat,
    int64_t prompt_feat_frames,
    const float* spk_embedding_192,
    std::mt19937& rng,
    const ThreadPool* pool,
    const float* fixed_noise
) const {
    const int64_t T_prompt = static_cast<int64_t>(prompt_tokens.size());
    const int64_t T_speech = static_cast<int64_t>(speech_tokens.size());
    const int64_t T_tok = T_prompt + T_speech;
    const int64_t T_mel = T_tok * 2;
    const int64_t D = enc_dim_; // 368

    // 1. L2-normalize 192-d CAMPPlus speaker embedding and project to 80-d (`spk_embed_affine_layer`)
    float norm_sq = 0.0f;
    for (int i = 0; i < 192; ++i) norm_sq += spk_embedding_192[i] * spk_embedding_192[i];
    const float inv_norm = 1.0f / std::max(std::sqrt(norm_sq), 1e-12f);
    float spk_normed[192];
    for (int i = 0; i < 192; ++i) spk_normed[i] = spk_embedding_192[i] * inv_norm;

    float spk_80[80];
    dense_linear_f32(spk_normed, spk_embed_affine_w_.data(), spk_embed_affine_b_.data(), 192, 80, spk_80);

    // 2. Concatenate prompt_tokens + speech_tokens and lookup `input_embedding_` ([6561, 368])
    std::vector<float> xs(static_cast<size_t>(T_tok * D));
    for (int64_t t = 0; t < T_tok; ++t) {
        int32_t tid = (t < T_prompt) ? prompt_tokens[t] : speech_tokens[t - T_prompt];
        tid = std::clamp<int32_t>(tid, 0, 6560);
        std::memcpy(xs.data() + t * D, input_embedding_.data() + tid * D, static_cast<size_t>(D) * sizeof(float));
    }

    // 3. `encoder.embed`: BitLinear(368, 368) -> LayerNorm(368, eps=1e-5) -> * sqrt(368)
    const float xscale = std::sqrt(static_cast<float>(D));
    std::vector<float> tmp_tok(static_cast<size_t>(T_tok * D));
    bitlinear_forward(embed_linear_, xs.data(), T_tok, tmp_tok.data(), pool);
    for (int64_t t = 0; t < T_tok; ++t) {
        layer_norm_f32(tmp_tok.data() + t * D, embed_ln_w_.data(), embed_ln_b_.data(), D, 1e-5f, xs.data() + t * D);
        for (int64_t c = 0; c < D; ++c) xs[t * D + c] *= xscale;
    }

    // 4. `pre_lookahead_layer`:
    //    Transpose [T_tok, 368] -> [368, T_tok], pad(0, 3) + conv1 -> leaky_relu(0.01) -> pad(2, 0) + conv2 -> + residual
    std::vector<float> xs_ch(static_cast<size_t>(D * T_tok));
    std::vector<float> conv_mid(static_cast<size_t>(D * T_tok));
    std::vector<float> conv_out(static_cast<size_t>(D * T_tok));
    transpose_2d(xs.data(), T_tok, D, xs_ch.data());
    bitconv1d_forward(pre_lookahead_conv1_, xs_ch.data(), T_tok, conv_mid.data(), pool);
    for (float& v : conv_mid) v = (v >= 0.0f) ? v : (0.01f * v);
    bitconv1d_forward(pre_lookahead_conv2_, conv_mid.data(), T_tok, conv_out.data(), pool);
    for (int64_t c = 0; c < D; ++c) {
        for (int64_t t = 0; t < T_tok; ++t) {
            xs[t * D + c] += conv_out[c * T_tok + t];
        }
    }

    // 5. Run 6 Conformer blocks at token rate `T_tok` (25 Hz)
    run_conformer_stack(xs.data(), T_tok, encoders_, pool);

    // 6. `up_layer`: nearest-neighbor 2x upsample `[368, T_tok] -> [368, T_mel]` + pad(4, 0) + conv(368, 368, 5)
    transpose_2d(xs.data(), T_tok, D, xs_ch.data());
    std::vector<float> up_ch_in(static_cast<size_t>(D * T_mel));
    for (int64_t c = 0; c < D; ++c) {
        for (int64_t t = 0; t < T_tok; ++t) {
            float v = xs_ch[c * T_tok + t];
            up_ch_in[c * T_mel + 2 * t + 0] = v;
            up_ch_in[c * T_mel + 2 * t + 1] = v;
        }
    }
    std::vector<float> up_ch_out(static_cast<size_t>(D * T_mel));
    bitconv1d_forward(up_layer_conv_, up_ch_in.data(), T_mel, up_ch_out.data(), pool);

    std::vector<float> xs_mel(static_cast<size_t>(T_mel * D));
    transpose_2d(up_ch_out.data(), D, T_mel, xs_mel.data());

    // 7. `up_embed`: BitLinear(368, 368) -> LayerNorm(368, eps=1e-5) -> * sqrt(368)
    std::vector<float> tmp_mel(static_cast<size_t>(T_mel * D));
    bitlinear_forward(up_embed_linear_, xs_mel.data(), T_mel, tmp_mel.data(), pool);
    for (int64_t t = 0; t < T_mel; ++t) {
        layer_norm_f32(tmp_mel.data() + t * D, up_embed_ln_w_.data(), up_embed_ln_b_.data(), D, 1e-5f, xs_mel.data() + t * D);
        for (int64_t c = 0; c < D; ++c) xs_mel[t * D + c] *= xscale;
    }

    // 8. Run 4 `up_encoders_` Conformer blocks at mel rate `T_mel` (50 Hz) + `after_norm` + `encoder_proj` -> `mu` [80, T_mel]
    run_conformer_stack(xs_mel.data(), T_mel, up_encoders_, pool);
    for (int64_t t = 0; t < T_mel; ++t) {
        layer_norm_f32(xs_mel.data() + t * D, after_norm_w_.data(), after_norm_b_.data(), D, 1e-5f, tmp_mel.data() + t * D);
    }
    std::vector<float> h_proj(static_cast<size_t>(T_mel * 80));
    bitlinear_forward(encoder_proj_, tmp_mel.data(), T_mel, h_proj.data(), pool);

    std::vector<float> mu(static_cast<size_t>(80 * T_mel));
    transpose_2d(h_proj.data(), T_mel, 80, mu.data());

    // 9. Assemble 320-channel ConditionalDecoder input `x_320` (`[320, T_mel]`):
    //    channels 0..79:   `z` (initial Gaussian noise `[80, T_mel]`)
    //    channels 80..159: `mu` (`[80, T_mel]`)
    //    channels 160..239:`spks` (`spk_80` repeated across `T_mel`)
    //    channels 240..319:`cond` (`prompt_feat` transposed into `[80, 0..prompt_feat_frames-1]`, 0 elsewhere)
    std::vector<float> z(static_cast<size_t>(80 * T_mel));
    if (fixed_noise) {
        std::memcpy(z.data(), fixed_noise, z.size() * sizeof(float));
    } else {
        std::normal_distribution<float> ndist(0.0f, 1.0f);
        for (float& v : z) v = ndist(rng);
    }

    std::vector<float> x_320(static_cast<size_t>(320 * T_mel), 0.0f);
    std::memcpy(x_320.data() + 0 * 80 * T_mel, z.data(), 80 * T_mel * sizeof(float));
    std::memcpy(x_320.data() + 1 * 80 * T_mel, mu.data(), 80 * T_mel * sizeof(float));
    for (int c = 0; c < 80; ++c) {
        float sv = spk_80[c];
        float* dst_row = x_320.data() + (160 + c) * T_mel;
        for (int64_t t = 0; t < T_mel; ++t) dst_row[t] = sv;
    }
    const int64_t cond_frames = std::min(prompt_feat_frames, T_mel);
    for (int64_t t = 0; t < cond_frames; ++t) {
        for (int c = 0; c < 80; ++c) {
            x_320[(240 + c) * T_mel + t] = prompt_feat[t * 80 + c];
        }
    }

    // 10. Run UNet Estimator:
    //     down_block_ (320 -> 176, saving skip connection *before* down_block_.post_conv!)
    //     Wait: let's check lines 1185-1197 of TorchScript trace!
    //     In ConditionalDecoder.forward:
    //       `hiddens.append(x)` happens BEFORE `x = downsample(x * mask_down)`!
    //     Let's make sure we save `skip_down` before `down_block_.post_conv`!
    std::vector<float> skip_down(static_cast<size_t>(176 * T_mel));
    std::vector<float> x_cur(static_cast<size_t>(176 * T_mel));
    {
        EstimatorBlock down_no_post = down_block_;
        down_no_post.has_post_conv = false;
        run_estimator_block(down_no_post, x_320.data(), 320, T_mel, skip_down.data(), pool);
        bitconv1d_forward(down_block_.post_conv, skip_down.data(), T_mel, x_cur.data(), pool);
    }

    std::vector<float> x_next(static_cast<size_t>(176 * T_mel));
    for (int m = 0; m < 12; ++m) {
        run_estimator_block(mid_blocks_[m], x_cur.data(), 176, T_mel, x_next.data(), pool);
        x_cur.swap(x_next);
    }

    // Concatenate `[x_cur, skip_down]` -> `[352, T_mel]` and run `up_block_`
    std::vector<float> up_in(static_cast<size_t>(352 * T_mel));
    std::memcpy(up_in.data(), x_cur.data(), 176 * T_mel * sizeof(float));
    std::memcpy(up_in.data() + 176 * T_mel, skip_down.data(), 176 * T_mel * sizeof(float));
    run_estimator_block(up_block_, up_in.data(), 352, T_mel, x_cur.data(), pool);

    // 11. `final_block` (CausalConv1d -> LN -> Mish) + `final_proj` (dense Conv1d(176, 80, 1))
    bitconv1d_forward(final_block_conv_, x_cur.data(), T_mel, x_next.data(), pool);
    std::vector<float> fb_tr(static_cast<size_t>(T_mel * 176));
    transpose_2d(x_next.data(), 176, T_mel, fb_tr.data());
    for (int64_t t = 0; t < T_mel; ++t) {
        layer_norm_f32(fb_tr.data() + t * 176, final_block_ln_w_.data(), final_block_ln_b_.data(), 176, 1e-5f, fb_tr.data() + t * 176);
    }
    transpose_2d(fb_tr.data(), T_mel, 176, x_next.data());
    for (float& v : x_next) v = mish_scalar(v);

    std::vector<float> dxdt(static_cast<size_t>(80 * T_mel));
    conv1d_f32(x_next.data(), final_proj_w_.data(), final_proj_b_.data(), 176, 80, T_mel, 1, 1, 0, 1, dxdt.data(), pool);

    // 12. 1-step Euler update `mel_full = z + dxdt`, then slice off `prompt_feat_frames`:
    //     Output `mel` has shape `[80, T_speech_mel]` where `T_speech_mel = T_mel - prompt_feat_frames`.
    const int64_t T_speech_mel = T_mel - prompt_feat_frames;
    std::vector<float> out_mel(static_cast<size_t>(80 * T_speech_mel));
    for (int c = 0; c < 80; ++c) {
        for (int64_t t = 0; t < T_speech_mel; ++t) {
            int64_t src_t = prompt_feat_frames + t;
            out_mel[c * T_speech_mel + t] = z[c * T_mel + src_t] + dxdt[c * T_mel + src_t];
        }
    }
    return out_mel;
}

} // namespace kitteni
