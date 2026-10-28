#include "lm/qwen3_ternary.hpp"
#include "kernels/nn_ops.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace kitteni {

void Qwen3KVCache::init(const Qwen3Config& cfg, int64_t max_tokens) {
    num_layers = cfg.num_hidden_layers;
    num_kv_heads = cfg.num_key_value_heads;
    head_dim = cfg.head_dim;
    max_seq_len = max_tokens;
    cur_seq_len = 0;

    const size_t layer_elems = static_cast<size_t>(num_kv_heads * max_seq_len * head_dim);
    keys.resize(static_cast<size_t>(num_layers));
    values.resize(static_cast<size_t>(num_layers));
    for (int64_t l = 0; l < num_layers; ++l) {
        keys[l].assign(layer_elems, 0.0f);
        values[l].assign(layer_elems, 0.0f);
    }
}

namespace {

inline float dot_f32(const float* a, const float* b, int64_t n) {
    float sum = 0.0f;
    int64_t i = 0;
#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 1));
    sum = _mm_cvtss_f32(s128);
#endif
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline void axpy_f32(float alpha, const float* x, float* y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
    __m256 valpha = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(valpha, vx, vy));
    }
#endif
    for (; i < n; ++i) y[i] += alpha * x[i];
}

} // namespace

Qwen3TernaryEngine::Qwen3TernaryEngine(
    const KittenTernaryLMWeights& weights,
    int num_threads,
    bool use_int8_act
) : weights_(weights), pool_(num_threads), use_int8_act_(use_int8_act) {}

void Qwen3TernaryEngine::forward_hidden(
    float* hidden,
    int64_t seq_len,
    Qwen3KVCache& kv_cache
) {
    const Qwen3Config& cfg = weights_.config;
    const int64_t D = cfg.hidden_size;
    const int64_t I = cfg.intermediate_size;
    const int64_t H_q = cfg.num_attention_heads;
    const int64_t H_kv = cfg.num_key_value_heads;
    const int64_t Dh = cfg.head_dim;
    const int64_t q_dim = H_q * Dh;
    const int64_t kv_dim = H_kv * Dh;
    const int64_t gqa_ratio = H_q / H_kv;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const int64_t start_pos = kv_cache.cur_seq_len;

    if (start_pos + seq_len > kv_cache.max_seq_len) {
        throw std::runtime_error("Qwen3KVCache capacity exceeded");
    }

    const int64_t qkv_dim = q_dim + 2 * kv_dim;
    norm_buf_.resize(static_cast<size_t>(seq_len * D));
    q_buf_.resize(static_cast<size_t>(seq_len * qkv_dim));
    attn_out_buf_.resize(static_cast<size_t>(seq_len * q_dim));
    proj_out_buf_.resize(static_cast<size_t>(seq_len * D));
    gate_buf_.resize(static_cast<size_t>(seq_len * 2 * I));
    up_buf_.resize(static_cast<size_t>(seq_len * I));

    auto run_linear = [&](const TernaryMatrix& W, const float* x_in, float* y_out) {
        if (use_int8_act_) {
            ternary_matmul_quantized(W, quant_act_, y_out, &pool_);
        } else {
            ternary_matmul_f32(W, x_in, seq_len, y_out, &pool_);
        }
    };

    const int64_t num_layers = static_cast<int64_t>(weights_.layers.size());
    for (int64_t l = 0; l < num_layers; ++l) {
        const Qwen3LayerWeights& lw = weights_.layers[l];

        // 1. Input RMSNorm
        for (int64_t s = 0; s < seq_len; ++s) {
            rms_norm_f32(
                hidden + s * D,
                lw.input_layernorm.data(),
                D,
                cfg.rms_norm_eps,
                norm_buf_.data() + s * D
            );
        }

        // 2. Single fused QKV projection (`qkv_proj`: [q_dim + 2*kv_dim, D])!
        if (use_int8_act_) {
            quant_act_.quantize_from(norm_buf_.data(), seq_len, D);
        }
        run_linear(lw.qkv_proj, norm_buf_.data(), q_buf_.data());

        // 3. Per-head Q-RMSNorm & K-RMSNorm + RoPE + store K, V into cache
        float* layer_k_cache = kv_cache.keys[l].data();
        float* layer_v_cache = kv_cache.values[l].data();
        const int64_t max_seq = kv_cache.max_seq_len;

        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t pos = start_pos + s;
            float* qkv_s = q_buf_.data() + s * qkv_dim;
            float* qs = qkv_s;
            float* ks = qkv_s + q_dim;
            const float* vs = qkv_s + q_dim + kv_dim;

            for (int64_t h = 0; h < H_q; ++h) {
                rms_norm_f32(qs + h * Dh, lw.q_norm.data(), Dh, cfg.rms_norm_eps, qs + h * Dh);
            }
            for (int64_t h = 0; h < H_kv; ++h) {
                rms_norm_f32(ks + h * Dh, lw.k_norm.data(), Dh, cfg.rms_norm_eps, ks + h * Dh);
            }

            rope_neox_inplace(qs, H_q, Dh, pos, cfg.rope_theta);
            rope_neox_inplace(ks, H_kv, Dh, pos, cfg.rope_theta);

            // Copy K and V into `[H_kv, max_seq, Dh]` cache
            for (int64_t h = 0; h < H_kv; ++h) {
                std::memcpy(layer_k_cache + (h * max_seq + pos) * Dh, ks + h * Dh, static_cast<size_t>(Dh) * sizeof(float));
                std::memcpy(layer_v_cache + (h * max_seq + pos) * Dh, vs + h * Dh, static_cast<size_t>(Dh) * sizeof(float));
            }
        }

        // 4. Causal Grouped-Query Attention (parallelized across query heads)
        pool_.parallel_for(H_q, [&](int64_t h_begin, int64_t h_end, int /*tid*/) {
            std::vector<float> scores(static_cast<size_t>(start_pos + seq_len));
            for (int64_t h = h_begin; h < h_end; ++h) {
                const int64_t kv_h = h / gqa_ratio;
                const float* k_head_cache = layer_k_cache + kv_h * max_seq * Dh;
                const float* v_head_cache = layer_v_cache + kv_h * max_seq * Dh;

                for (int64_t s = 0; s < seq_len; ++s) {
                    const int64_t causal_len = start_pos + s + 1;
                    const float* q_vec = q_buf_.data() + s * qkv_dim + h * Dh;
                    float* out_vec = attn_out_buf_.data() + s * q_dim + h * Dh;

                    float max_score = -std::numeric_limits<float>::infinity();
                    for (int64_t t = 0; t < causal_len; ++t) {
                        float sc = dot_f32(q_vec, k_head_cache + t * Dh, Dh) * attn_scale;
                        scores[t] = sc;
                        if (sc > max_score) max_score = sc;
                    }

                    float sum_exp = 0.0f;
                    for (int64_t t = 0; t < causal_len; ++t) {
                        float e = std::exp(scores[t] - max_score);
                        scores[t] = e;
                        sum_exp += e;
                    }
                    const float inv_sum = 1.0f / sum_exp;

                    std::memset(out_vec, 0, static_cast<size_t>(Dh) * sizeof(float));
                    for (int64_t t = 0; t < causal_len; ++t) {
                        axpy_f32(scores[t] * inv_sum, v_head_cache + t * Dh, out_vec, Dh);
                    }
                }
            }
        });

        // 5. Output projection `o_proj` + residual connection
        if (use_int8_act_) {
            quant_act_.quantize_from(attn_out_buf_.data(), seq_len, q_dim);
        }
        run_linear(lw.o_proj, attn_out_buf_.data(), proj_out_buf_.data());

        for (int64_t i = 0; i < seq_len * D; ++i) {
            hidden[i] += proj_out_buf_[i];
        }

        // 6. Post-attention RMSNorm
        for (int64_t s = 0; s < seq_len; ++s) {
            rms_norm_f32(
                hidden + s * D,
                lw.post_attention_layernorm.data(),
                D,
                cfg.rms_norm_eps,
                norm_buf_.data() + s * D
            );
        }

        // 7. Single fused `gate_up_proj` ([2*I, D]) + SwiGLU + `down_proj`!
        if (use_int8_act_) {
            quant_act_.quantize_from(norm_buf_.data(), seq_len, D);
        }
        run_linear(lw.gate_up_proj, norm_buf_.data(), gate_buf_.data());

        for (int64_t s = 0; s < seq_len; ++s) {
            const float* gs = gate_buf_.data() + s * (2 * I);
            const float* us = gs + I;
            swiglu_f32(gs, us, I, up_buf_.data() + s * I);
        }
        if (use_int8_act_) {
            quant_act_.quantize_from(up_buf_.data(), seq_len, I);
        }
        run_linear(lw.down_proj, up_buf_.data(), proj_out_buf_.data());

        for (int64_t i = 0; i < seq_len * D; ++i) {
            hidden[i] += proj_out_buf_[i];
        }
    }

    kv_cache.cur_seq_len += seq_len;
}

void Qwen3TernaryEngine::compute_output_logits(
    const float* last_hidden,
    std::vector<float>& out_logits,
    const TokenMap* tm,
    bool speech_decode_only
) {
    const Qwen3Config& cfg = weights_.config;
    const int64_t D = cfg.hidden_size;
    const int64_t V = cfg.vocab_size;

    std::vector<float> normed(static_cast<size_t>(D));
    rms_norm_f32(last_hidden, weights_.final_norm.data(), D, cfg.rms_norm_eps, normed.data());

    const float neg_inf = -std::numeric_limits<float>::infinity();
    out_logits.assign(static_cast<size_t>(V), neg_inf);

    // During speech token generation, the LM only emits tokens in `[speech_start_id .. audio_id_end)`
    // (6567 tokens, all in Tier 0 of `emb4`!), which is 24x smaller than the 158,241 BPE vocabulary!
    int64_t v_start = 0;
    int64_t v_end = V;
    if (speech_decode_only && tm != nullptr) {
        v_start = std::min({tm->speech_start_id, tm->speech_end_id, tm->stop_id, tm->audio_id_base});
        v_end = std::max({tm->speech_end_id + 1, tm->stop_id + 1, tm->audio_id_end()});
    }

    if (weights_.weights_variant == "emb4") {
        std::vector<float> slice_logits(static_cast<size_t>(v_end - v_start), neg_inf);
        weights_.emb4_table.compute_tied_logits(normed.data(), slice_logits.data(), v_start, v_end, &pool_);
        std::memcpy(out_logits.data() + v_start, slice_logits.data(), slice_logits.size() * sizeof(float));
    } else {
        const uint16_t* emb_bf16 = weights_.bf16_embed_tokens.ptr<uint16_t>();
        pool_.parallel_for(v_end - v_start, [&](int64_t b, int64_t e, int /*tid*/) {
            for (int64_t idx = b; idx < e; ++idx) {
                int64_t vid = v_start + idx;
                const uint16_t* row = emb_bf16 + vid * D;
                float acc = 0.0f;
                for (int64_t c = 0; c < D; ++c) {
                    acc += bf16_to_fp32(row[c]) * normed[c];
                }
                out_logits[vid] = acc;
            }
        });
    }
}

void Qwen3TernaryEngine::prefill(
    const std::vector<int32_t>& prompt_ids,
    const float* speaker_embedding_512,
    Qwen3KVCache& kv_cache,
    std::vector<float>& out_logits,
    const TokenMap* tm,
    bool speech_decode_only
) {
    if (prompt_ids.empty()) {
        throw std::invalid_argument("prompt_ids cannot be empty in prefill");
    }
    const int64_t seq_len = static_cast<int64_t>(prompt_ids.size());
    const int64_t D = weights_.config.hidden_size;

    hidden_buf_.resize(static_cast<size_t>(seq_len * D));
    const bool inject_spk = (kv_cache.cur_seq_len == 0 && speaker_embedding_512 != nullptr);

    if (inject_spk) {
        weights_.project_speaker(speaker_embedding_512, hidden_buf_.data());
        // Match Python's `module.spk_proj(embedding.to(embeds.dtype))` BF16 rounding at prefill pos 0
        for (int64_t c = 0; c < D; ++c) {
            hidden_buf_[c] = bf16_to_fp32(fp32_to_bf16(hidden_buf_[c]));
        }
    }

    for (int64_t s = (inject_spk ? 1 : 0); s < seq_len; ++s) {
        weights_.embed_token(prompt_ids[s], hidden_buf_.data() + s * D);
    }

    forward_hidden(hidden_buf_.data(), seq_len, kv_cache);
    compute_output_logits(hidden_buf_.data() + (seq_len - 1) * D, out_logits, tm, speech_decode_only);
}

void Qwen3TernaryEngine::decode_step(
    int32_t token_id,
    Qwen3KVCache& kv_cache,
    std::vector<float>& out_logits,
    const TokenMap* tm,
    bool speech_decode_only
) {
    const int64_t D = weights_.config.hidden_size;
    hidden_buf_.resize(static_cast<size_t>(D));
    weights_.embed_token(token_id, hidden_buf_.data());

    forward_hidden(hidden_buf_.data(), 1, kv_cache);
    compute_output_logits(hidden_buf_.data(), out_logits, tm, speech_decode_only);
}

std::vector<int32_t> Qwen3TernaryEngine::generate_audio_tokens(
    const std::vector<int32_t>& prompt_ids,
    const float* speaker_embedding_512,
    const TokenMap& tm,
    const SamplingSettings& sampling,
    int32_t max_new_tokens,
    std::mt19937& rng,
    GenerationStats* out_stats
) {
    Qwen3KVCache kv_cache;
    kv_cache.init(weights_.config, static_cast<int64_t>(prompt_ids.size()) + max_new_tokens + 16);

    std::vector<float> logits;
    std::vector<int32_t> history = prompt_ids;
    std::vector<int32_t> audio_tokens;
    audio_tokens.reserve(static_cast<size_t>(max_new_tokens));

    auto t0 = std::chrono::steady_clock::now();
    prefill(prompt_ids, speaker_embedding_512, kv_cache, logits, &tm, /*speech_decode_only=*/true);
    auto t1 = std::chrono::steady_clock::now();

    bool terminated = false;
    int64_t gen_steps = 0;

    for (int32_t step = 0; step < max_new_tokens; ++step) {
        apply_logits_processors(logits, history, prompt_ids.size(), tm, sampling);
        int32_t next_token = sample_token(logits, rng, sampling.greedy);
        history.push_back(next_token);
        gen_steps++;

        if (next_token == tm.speech_end_id || next_token == tm.stop_id) {
            terminated = true;
            break;
        }
        if (next_token >= tm.audio_id_base && next_token < tm.audio_id_end()) {
            audio_tokens.push_back(next_token - tm.audio_id_base);
        }
        if (step + 1 < max_new_tokens) {
            decode_step(next_token, kv_cache, logits, &tm, /*speech_decode_only=*/true);
        }
    }
    auto t2 = std::chrono::steady_clock::now();

    if (out_stats) {
        out_stats->prompt_tokens = static_cast<int64_t>(prompt_ids.size());
        out_stats->generated_tokens = gen_steps;
        out_stats->prefill_seconds = std::chrono::duration<double>(t1 - t0).count();
        out_stats->decode_seconds = std::chrono::duration<double>(t2 - t1).count();
        out_stats->terminated = terminated;
    }

    return audio_tokens;
}

} // namespace kitteni
