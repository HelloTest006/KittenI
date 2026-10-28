#pragma once

#include "lm/model_loader.hpp"
#include "lm/sampler.hpp"
#include "core/thread_pool.hpp"
#include "kernels/ternary_gemv.hpp"
#include <vector>
#include <cstdint>
#include <memory>

namespace kitteni {

struct Qwen3KVCache {
    int64_t num_layers = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    int64_t cur_seq_len = 0;

    // Each layer stores K and V in layout `[num_kv_heads, max_seq_len, head_dim]`
    // so attention over past positions `t in [0..cur_seq_len-1]` for a given KV head is contiguous!
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;

    void init(const Qwen3Config& cfg, int64_t max_tokens = 2048);
    void reset() { cur_seq_len = 0; }
};

struct GenerationStats {
    int64_t prompt_tokens = 0;
    int64_t generated_tokens = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    bool terminated = false;
};

class Qwen3TernaryEngine {
public:
    explicit Qwen3TernaryEngine(const KittenTernaryLMWeights& weights, int num_threads = 0, bool use_int8_act = true);

    const KittenTernaryLMWeights& weights() const { return weights_; }
    ThreadPool& thread_pool() { return pool_; }

    // Prefill a sequence of token IDs starting at `kv_cache.cur_seq_len`.
    // When `kv_cache.cur_seq_len == 0` and `speaker_embedding_512 != nullptr`,
    // position 0's input embedding is replaced by `spk_proj(speaker_embedding_512)`.
    // Computes and writes the final position's logits into `out_logits[0..vocab_size-1]`.
    // If `speech_decode_only == true` and `tm != nullptr`, only computes logits for
    // `[speech_start_id .. audio_id_end)` (6567 tokens instead of 158,241!).
    void prefill(
        const std::vector<int32_t>& prompt_ids,
        const float* speaker_embedding_512,
        Qwen3KVCache& kv_cache,
        std::vector<float>& out_logits,
        const TokenMap* tm = nullptr,
        bool speech_decode_only = false
    );

    // Run a single autoregressive decode step for `token_id` at `kv_cache.cur_seq_len`.
    void decode_step(
        int32_t token_id,
        Qwen3KVCache& kv_cache,
        std::vector<float>& out_logits,
        const TokenMap* tm = nullptr,
        bool speech_decode_only = false
    );

    // High-level chunk audio token generator (`_generate_tokens` in KittenTTS 2):
    // Returns 0-based S3 codec token IDs (`[0..6560]`).
    std::vector<int32_t> generate_audio_tokens(
        const std::vector<int32_t>& prompt_ids,
        const float* speaker_embedding_512,
        const TokenMap& tm,
        const SamplingSettings& sampling,
        int32_t max_new_tokens,
        std::mt19937& rng,
        GenerationStats* out_stats = nullptr
    );

private:
    const KittenTernaryLMWeights& weights_;
    ThreadPool pool_;
    bool use_int8_act_ = true;

    // Reusable scratch buffers to avoid allocations during decode steps
    std::vector<float> hidden_buf_;
    std::vector<float> norm_buf_;
    std::vector<float> q_buf_;
    std::vector<float> k_buf_;
    std::vector<float> v_buf_;
    std::vector<float> attn_out_buf_;
    std::vector<float> proj_out_buf_;
    std::vector<float> gate_buf_;
    std::vector<float> up_buf_;
    QuantizedActivation quant_act_;

    void forward_hidden(
        float* hidden,         // [seq_len, hidden_size] modified in-place
        int64_t seq_len,
        Qwen3KVCache& kv_cache
    );

    void compute_output_logits(
        const float* last_hidden, // [hidden_size]
        std::vector<float>& out_logits,
        const TokenMap* tm,
        bool speech_decode_only
    );
};

} // namespace kitteni
