#include "lm/model_loader.hpp"
#include "kernels/nn_ops.hpp"
#include "core/thread_pool.hpp"
#include <stdexcept>
#include <cstring>

namespace kitteni {

namespace {

// Concatenate 2 or 3 TernaryMatrices sharing the same `in_features` along `out_features`
// and release the source matrices' memory so total RAM is unchanged.
void concat_and_move_ternary_matrices(std::vector<TernaryMatrix*> srcs, TernaryMatrix& dst) {
    if (srcs.empty()) return;
    int64_t in_feat = srcs[0]->in_features;
    int64_t total_out = 0;
    size_t total_packed = 0;
    size_t total_scales = 0;
    for (auto* m : srcs) {
        total_out += m->out_features;
        total_packed += m->packed_2bit.size();
        total_scales += m->scales.size();
    }
    dst.out_features = total_out;
    dst.in_features = in_feat;
    dst.packed_2bit.resize(total_packed);
    dst.scales.resize(total_scales);
    dst.group_sums.resize(total_scales);

    size_t p_off = 0, s_off = 0;
    for (auto* m : srcs) {
        std::memcpy(dst.packed_2bit.data() + p_off, m->packed_2bit.data(), m->packed_2bit.size());
        std::memcpy(dst.scales.data() + s_off, m->scales.data(), m->scales.size() * sizeof(float));
        std::memcpy(dst.group_sums.data() + s_off, m->group_sums.data(), m->group_sums.size() * sizeof(int16_t));
        p_off += m->packed_2bit.size();
        s_off += m->scales.size();
        m->packed_2bit.clear();
        m->packed_2bit.shrink_to_fit();
        m->scales.clear();
        m->scales.shrink_to_fit();
        m->group_sums.clear();
        m->group_sums.shrink_to_fit();
    }
}

std::vector<float> load_1d_or_2d_f32(const SafetensorsFile& sf, const std::string& name) {
    TensorView tv = sf.get_tensor(name);
    size_t n = tv.numel();
    std::vector<float> out(n);
    if (tv.dtype == DType::F32) {
        std::memcpy(out.data(), tv.data, n * sizeof(float));
    } else if (tv.dtype == DType::BF16) {
        const uint16_t* p = tv.ptr<uint16_t>();
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(p[i]);
    } else if (tv.dtype == DType::F16) {
        const uint16_t* p = tv.ptr<uint16_t>();
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(p[i]);
    } else {
        throw std::runtime_error("Unsupported dense tensor dtype for: " + name);
    }
    return out;
}

} // namespace

void KittenTernaryLMWeights::load(const ResolvedRepo& repo, const std::string& variant, int64_t max_layers, int num_threads) {
    weights_variant = variant;
    std::string lm_dir = repo.config.value_string("lm_dir", "lm");
    auto lm_cfg_path = ensure_repo_file(repo, lm_dir + "/config.json", true);
    JsonValue lm_cfg = parse_json_file(lm_cfg_path.string());

    config.hidden_size = lm_cfg.value_int("hidden_size", 2048);
    config.intermediate_size = lm_cfg.value_int("intermediate_size", 6144);
    config.num_hidden_layers = lm_cfg.value_int("num_hidden_layers", 28);
    config.num_attention_heads = lm_cfg.value_int("num_attention_heads", 16);
    config.num_key_value_heads = lm_cfg.value_int("num_key_value_heads", 8);
    config.head_dim = lm_cfg.value_int("head_dim", 128);
    config.vocab_size = lm_cfg.value_int("vocab_size", 158241);
    config.spk_embedding_dim = repo.config.value_int("speaker_embedding_dim", 512);
    config.rms_norm_eps = static_cast<float>(lm_cfg.value_float("rms_norm_eps", 1e-6));
    config.rope_theta = static_cast<float>(lm_cfg.value_float("rope_theta", 1000000.0));
    config.tie_word_embeddings = lm_cfg.value_bool("tie_word_embeddings", true);

    std::string sf_rel_path;
    if (variant == "emb4") {
        sf_rel_path = repo.config.value_string("lm_emb4", "lm/model-tl2-emb4.safetensors");
    } else if (variant == "packed") {
        sf_rel_path = repo.config.value_string("lm_packed", "lm/model-ternary.safetensors");
    } else {
        throw std::runtime_error("Unsupported weight variant: " + variant);
    }

    auto sf_full_path = ensure_repo_file(repo, sf_rel_path, true);
    mapped_file = std::make_unique<SafetensorsFile>(sf_full_path.string());
    const SafetensorsFile& sf = *mapped_file;

    int64_t layers_to_load = (max_layers >= 0 && max_layers < config.num_hidden_layers)
                                 ? max_layers
                                 : config.num_hidden_layers;
    layers.resize(static_cast<size_t>(layers_to_load));
    ThreadPool loader_pool(num_threads);

    if (variant == "emb4") {
        std::string schema = sf.get_metadata("schema");
        if (schema != "kitten_ternary_tl2_emb4_v1") {
            throw std::runtime_error("Unexpected emb4 schema: " + schema);
        }

        // 1. Load compressed Hadamard 4-bit tiered embedding table
        emb4_table.load(sf);

        // 2. Load dense norms and speaker projection (`fp/*`)
        final_norm = load_1d_or_2d_f32(sf, "fp/model.norm.weight");
        spk_proj.linear_weight = load_1d_or_2d_f32(sf, "fp/spk_proj.0.weight");
        spk_proj.linear_bias   = load_1d_or_2d_f32(sf, "fp/spk_proj.0.bias");
        spk_proj.norm_weight   = load_1d_or_2d_f32(sf, "fp/spk_proj.1.weight");
        spk_proj.norm_bias     = load_1d_or_2d_f32(sf, "fp/spk_proj.1.bias");

        auto load_tl2_mat = [&](const std::string& name, TernaryMatrix& mat) {
            TensorView tv = sf.get_tensor("tl2/" + name);
            if (tv.shape.size() != 3 || tv.shape[2] != 928) {
                throw std::runtime_error("Invalid TL2 tensor shape for: " + name);
            }
            load_tl2_to_ternary_matrix(tv.data, tv.shape[0], tv.shape[1], mat);
        };

        loader_pool.parallel_for(layers_to_load, [&](int64_t b, int64_t e, int /*tid*/) {
            for (int64_t i = b; i < e; ++i) {
                std::string p = "model.layers." + std::to_string(i) + ".";
                Qwen3LayerWeights& lw = layers[i];
                lw.input_layernorm          = load_1d_or_2d_f32(sf, "fp/" + p + "input_layernorm.weight");
                lw.post_attention_layernorm = load_1d_or_2d_f32(sf, "fp/" + p + "post_attention_layernorm.weight");
                lw.q_norm                   = load_1d_or_2d_f32(sf, "fp/" + p + "self_attn.q_norm.weight");
                lw.k_norm                   = load_1d_or_2d_f32(sf, "fp/" + p + "self_attn.k_norm.weight");

                load_tl2_mat(p + "self_attn.q_proj.weight", lw.q_proj);
                load_tl2_mat(p + "self_attn.k_proj.weight", lw.k_proj);
                load_tl2_mat(p + "self_attn.v_proj.weight", lw.v_proj);
                concat_and_move_ternary_matrices({&lw.q_proj, &lw.k_proj, &lw.v_proj}, lw.qkv_proj);

                load_tl2_mat(p + "self_attn.o_proj.weight", lw.o_proj);
                load_tl2_mat(p + "mlp.gate_proj.weight",    lw.gate_proj);
                load_tl2_mat(p + "mlp.up_proj.weight",      lw.up_proj);
                concat_and_move_ternary_matrices({&lw.gate_proj, &lw.up_proj}, lw.gate_up_proj);

                load_tl2_mat(p + "mlp.down_proj.weight",    lw.down_proj);
            }
        });
    } else if (variant == "packed") {
        std::string fmt = sf.get_metadata("format");
        if (fmt != "kittenml-ternary-1.58") {
            throw std::runtime_error("Unexpected packed ternary format: " + fmt);
        }
        JsonValue packed_meta = parse_json(sf.get_metadata("packed", "{}"));

        bf16_embed_tokens = sf.get_tensor("model.embed_tokens.weight");
        final_norm = load_1d_or_2d_f32(sf, "model.norm.weight");
        spk_proj.linear_weight = load_1d_or_2d_f32(sf, "spk_proj.0.weight");
        spk_proj.linear_bias   = load_1d_or_2d_f32(sf, "spk_proj.0.bias");
        spk_proj.norm_weight   = load_1d_or_2d_f32(sf, "spk_proj.1.weight");
        spk_proj.norm_bias     = load_1d_or_2d_f32(sf, "spk_proj.1.bias");

        auto load_158_mat = [&](const std::string& name, TernaryMatrix& mat) {
            const JsonValue& spec = packed_meta.at(name);
            const auto& sh = spec.at("shape").as_array();
            int64_t rows = sh[0].as_int();
            int64_t cols = sh[1].as_int();
            TensorView codes_tv = sf.get_tensor(name + "::codes");
            TensorView scale_tv = sf.get_tensor(name + "::scale");
            load_ternary158_to_ternary_matrix(
                codes_tv.data,
                codes_tv.byte_size,
                scale_tv.ptr<uint16_t>(),
                scale_tv.dtype,
                rows,
                cols,
                mat
            );
        };

        loader_pool.parallel_for(layers_to_load, [&](int64_t b, int64_t e, int /*tid*/) {
            for (int64_t i = b; i < e; ++i) {
                std::string p = "model.layers." + std::to_string(i) + ".";
                Qwen3LayerWeights& lw = layers[i];
                lw.input_layernorm          = load_1d_or_2d_f32(sf, p + "input_layernorm.weight");
                lw.post_attention_layernorm = load_1d_or_2d_f32(sf, p + "post_attention_layernorm.weight");
                lw.q_norm                   = load_1d_or_2d_f32(sf, p + "self_attn.q_norm.weight");
                lw.k_norm                   = load_1d_or_2d_f32(sf, p + "self_attn.k_norm.weight");

                load_158_mat(p + "self_attn.q_proj.weight", lw.q_proj);
                load_158_mat(p + "self_attn.k_proj.weight", lw.k_proj);
                load_158_mat(p + "self_attn.v_proj.weight", lw.v_proj);
                concat_and_move_ternary_matrices({&lw.q_proj, &lw.k_proj, &lw.v_proj}, lw.qkv_proj);

                load_158_mat(p + "self_attn.o_proj.weight", lw.o_proj);
                load_158_mat(p + "mlp.gate_proj.weight",    lw.gate_proj);
                load_158_mat(p + "mlp.up_proj.weight",      lw.up_proj);
                concat_and_move_ternary_matrices({&lw.gate_proj, &lw.up_proj}, lw.gate_up_proj);

                load_158_mat(p + "mlp.down_proj.weight",    lw.down_proj);
            }
        });
    }
}

void KittenTernaryLMWeights::embed_token(int64_t token_id, float* out_f32) const {
    if (weights_variant == "emb4") {
        emb4_table.lookup_row(token_id, out_f32, true);
    } else {
        const uint16_t* row = bf16_embed_tokens.ptr<uint16_t>() + token_id * config.hidden_size;
        for (int64_t i = 0; i < config.hidden_size; ++i) {
            out_f32[i] = bf16_to_fp32(row[i]);
        }
    }
}

void KittenTernaryLMWeights::project_speaker(const float* spk_512, float* out_f32) const {
    std::vector<float> tmp(static_cast<size_t>(config.hidden_size));
    dense_linear_f32(
        spk_512,
        spk_proj.linear_weight.data(),
        spk_proj.linear_bias.data(),
        config.spk_embedding_dim,
        config.hidden_size,
        tmp.data()
    );
    layer_norm_f32(
        tmp.data(),
        spk_proj.norm_weight.data(),
        spk_proj.norm_bias.data(),
        config.hidden_size,
        1e-5f,
        out_f32
    );
}

size_t KittenTernaryLMWeights::ternary_body_bytes() const {
    size_t total = 0;
    for (const auto& lw : layers) {
        total += lw.qkv_proj.memory_bytes();
        total += lw.o_proj.memory_bytes();
        total += lw.gate_up_proj.memory_bytes();
        total += lw.down_proj.memory_bytes();
    }
    return total;
}

} // namespace kitteni
