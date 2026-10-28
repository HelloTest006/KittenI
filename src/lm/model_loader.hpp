#pragma once

#include "kitteni/common.hpp"
#include "core/mmap_loader.hpp"
#include "core/hub_cache.hpp"
#include "kernels/fwht_emb4.hpp"
#include "kernels/ternary_unpack.hpp"
#include <vector>
#include <string>
#include <memory>

namespace kitteni {

struct Qwen3Config {
    int64_t hidden_size = 2048;
    int64_t intermediate_size = 6144;
    int64_t num_hidden_layers = 28;
    int64_t num_attention_heads = 16;
    int64_t num_key_value_heads = 8;
    int64_t head_dim = 128;
    int64_t vocab_size = 158241;
    int64_t spk_embedding_dim = 512;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    bool tie_word_embeddings = true;
};

struct Qwen3LayerWeights {
    std::vector<float> input_layernorm;          // [hidden_size]
    std::vector<float> post_attention_layernorm; // [hidden_size]
    std::vector<float> q_norm;                   // [head_dim]
    std::vector<float> k_norm;                   // [head_dim]

    TernaryMatrix q_proj;                        // [num_attention_heads * head_dim, hidden_size]
    TernaryMatrix k_proj;                        // [num_key_value_heads * head_dim, hidden_size]
    TernaryMatrix v_proj;                        // [num_key_value_heads * head_dim, hidden_size]
    TernaryMatrix qkv_proj;                      // Fused [q_dim + 2 * kv_dim, hidden_size] for single-dispatch GEMV
    TernaryMatrix o_proj;                        // [hidden_size, num_attention_heads * head_dim]

    TernaryMatrix gate_proj;                     // [intermediate_size, hidden_size]
    TernaryMatrix up_proj;                       // [intermediate_size, hidden_size]
    TernaryMatrix gate_up_proj;                  // Fused [2 * intermediate_size, hidden_size] for single-dispatch GEMV
    TernaryMatrix down_proj;                     // [hidden_size, intermediate_size]
};

struct SpeakerProjWeights {
    std::vector<float> linear_weight; // [hidden_size, spk_embedding_dim]
    std::vector<float> linear_bias;   // [hidden_size]
    std::vector<float> norm_weight;   // [hidden_size]
    std::vector<float> norm_bias;     // [hidden_size]
};

class KittenTernaryLMWeights {
public:
    Qwen3Config config;
    std::string weights_variant; // "emb4" or "packed"

    // Holds the memory-mapped safetensors file alive so `emb4` codes can be accessed zero-copy
    std::unique_ptr<SafetensorsFile> mapped_file;

    // For `emb4`: compressed tiered 4-bit Hadamard embedding table
    TieredEmb4Table emb4_table;

    // For `packed`: BF16 embedding table [vocab_size, hidden_size]
    TensorView bf16_embed_tokens;

    std::vector<Qwen3LayerWeights> layers;
    std::vector<float> final_norm; // [hidden_size]
    SpeakerProjWeights spk_proj;

    // Load model weights from a ResolvedRepo (`emb4` by default for fast prototyping)
    // Optional `max_layers` parameter allows partial layer loading for fast unit tests.
    void load(const ResolvedRepo& repo, const std::string& variant = "emb4", int64_t max_layers = -1, int num_threads = 0);

    // Lookup token embedding into `out_f32[0..hidden_size-1]`
    void embed_token(int64_t token_id, float* out_f32) const;

    // Project a 512-d speaker embedding through `spk_proj` (Linear + LayerNorm) into `out_f32[0..hidden_size-1]`
    void project_speaker(const float* spk_512, float* out_f32) const;

    // Total memory footprint of packed ternary matrices in bytes
    size_t ternary_body_bytes() const;
};

} // namespace kitteni
