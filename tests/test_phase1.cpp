#include "kitteni/common.hpp"
#include "core/json.hpp"
#include "core/mmap_loader.hpp"
#include "core/hub_cache.hpp"
#include "kernels/cpu_features.hpp"
#include "kernels/ternary_unpack.hpp"
#include "kernels/fwht_emb4.hpp"
#include "kernels/ternary_gemv.hpp"
#include "kernels/nn_ops.hpp"
#include "lm/model_loader.hpp"

#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>
#include <cstring>
#include <filesystem>

using namespace kitteni;

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

static void test_fp16_bf16_conversions() {
    std::cout << "[Test 1] Testing FP16 and BF16 conversions...\n";
    const float test_vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.25f, 3.14159f, 1e-4f, 65504.0f};
    for (float v : test_vals) {
        uint16_t bf = fp32_to_bf16(v);
        float v_bf = bf16_to_fp32(bf);
        if (v != 0.0f) {
            float rel_err = std::fabs(v - v_bf) / std::fabs(v);
            REQUIRE(rel_err < 1e-2f);
        }
        uint16_t hf = fp32_to_fp16(v);
        float v_hf = fp16_to_fp32(hf);
        if (v != 0.0f) {
            float rel_err = std::fabs(v - v_hf) / std::fabs(v);
            REQUIRE(rel_err < 2e-3f);
        }
    }
}

static void test_fwht_orthogonality() {
    std::cout << "[Test 2] Testing Fast Walsh-Hadamard Transform (FWHT) involution & accuracy...\n";
    const int64_t dim = 2048;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> orig(dim);
    for (int64_t i = 0; i < dim; ++i) orig[i] = dist(rng);

    std::vector<float> buf = orig;
    // H is symmetric and orthogonal: FWHT(FWHT(x)) == x
    fwht_inplace(buf.data(), dim);
    fwht_inplace(buf.data(), dim);

    float max_diff = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        max_diff = std::max(max_diff, std::fabs(orig[i] - buf[i]));
    }
    std::cout << "  FWHT 2048 round-trip max error: " << max_diff << "\n";
    REQUIRE(max_diff < 1e-5f);
}

static void test_ternary_pack_and_tl2_unpack() {
    std::cout << "[Test 3] Testing TL2 & 1.58-bit ternary packing and exact reconstruction...\n";
    const int64_t rows_blocks = 2; // 64 rows
    const int64_t group_blocks = 2; // 256 cols
    const int64_t out_features = rows_blocks * 32;
    const int64_t in_features = group_blocks * 128;

    std::mt19937 rng(1337);
    std::vector<int8_t> trits(out_features * in_features);
    for (auto& t : trits) {
        int v = (rng() % 3) - 1;
        t = static_cast<int8_t>(v);
    }

    std::vector<float> scales(out_features * group_blocks);
    for (size_t i = 0; i < scales.size(); ++i) {
        // Use exact FP16/BF16 representable scale (e.g. multiples of 1/64)
        scales[i] = static_cast<float>((i % 31) + 1) * 0.03125f;
    }

    // 1. Test direct pack -> unpack
    TernaryMatrix mat;
    pack_trits_to_ternary_matrix(trits.data(), scales.data(), out_features, in_features, mat);

    std::vector<float> unpacked(out_features * in_features);
    unpack_ternary_matrix_to_f32(mat, unpacked.data());

    for (int64_t r = 0; r < out_features; ++r) {
        for (int64_t c = 0; c < in_features; ++c) {
            float expected = static_cast<float>(trits[r * in_features + c]) * scales[r * group_blocks + (c / 128)];
            float actual = unpacked[r * in_features + c];
            REQUIRE(std::fabs(expected - actual) < 1e-7f);
        }
    }

    // 2. Test base-3 1.58-bit encoding -> load_ternary158_to_ternary_matrix
    const size_t numel = trits.size();
    const size_t packed_len = (numel + 4) / 5;
    std::vector<uint8_t> base3_bytes(packed_len, 0);
    const int powers[5] = {1, 3, 9, 27, 81};
    for (size_t i = 0; i < numel; ++i) {
        int digit = static_cast<int>(trits[i]) + 1; // {0, 1, 2}
        base3_bytes[i / 5] = static_cast<uint8_t>(base3_bytes[i / 5] + digit * powers[i % 5]);
    }
    std::vector<uint16_t> scales_bf16(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) scales_bf16[i] = fp32_to_bf16(scales[i]);

    TernaryMatrix mat158;
    load_ternary158_to_ternary_matrix(
        base3_bytes.data(), base3_bytes.size(),
        scales_bf16.data(), DType::BF16,
        out_features, in_features, mat158
    );

    std::vector<float> unpacked158(out_features * in_features);
    unpack_ternary_matrix_to_f32(mat158, unpacked158.data());
    for (size_t i = 0; i < numel; ++i) {
        REQUIRE(std::fabs(unpacked[i] - unpacked158[i]) < 1e-7f);
    }

    // 3. Test TL2 synthetic block encoding -> load_tl2_to_ternary_matrix
    std::vector<uint8_t> tl2_buf(rows_blocks * group_blocks * 928, 0);
    for (int64_t rb = 0; rb < rows_blocks; ++rb) {
        for (int64_t gb = 0; gb < group_blocks; ++gb) {
            uint8_t* blk = tl2_buf.data() + (rb * group_blocks + gb) * 928;
            // Write 32 FP16 scales
            for (int r = 0; r < 32; ++r) {
                uint16_t h = fp32_to_fp16(scales[(rb * 32 + r) * group_blocks + gb]);
                std::memcpy(blk + r * 2, &h, 2);
            }
            size_t cursor = 64;
            for (int sub = 0; sub < 4; ++sub) {
                for (int t_idx = 0; t_idx < 10; ++t_idx) {
                    uint8_t* seg_row = blk + cursor + t_idx * 20;
                    uint32_t sign_bits = 0;
                    for (int pair = 0; pair < 16; ++pair) {
                        int r_even = pair * 2;
                        int r_odd = pair * 2 + 1;
                        int col = sub * 32 + t_idx * 3;

                        auto encode_triple = [&](int r_local, bool& is_neg) -> uint8_t {
                            int64_t gr = rb * 32 + r_local;
                            int d0 = trits[(gr * group_blocks + gb) * 128 + col + 0] + 1;
                            int d1 = trits[(gr * group_blocks + gb) * 128 + col + 1] + 1;
                            int d2 = trits[(gr * group_blocks + gb) * 128 + col + 2] + 1;
                            int n = d0 * 9 + d1 * 3 + d2; // 0..26
                            int bal = n - 13;             // -13..+13
                            is_neg = (bal < 0);
                            return static_cast<uint8_t>(std::abs(bal));
                        };

                        bool neg_e = false, neg_o = false;
                        uint8_t mag_e = encode_triple(r_even, neg_e);
                        uint8_t mag_o = encode_triple(r_odd, neg_o);
                        seg_row[pair] = static_cast<uint8_t>((mag_e & 0x0FU) | ((mag_o & 0x0FU) << 4));
                        if (neg_e) sign_bits |= (1U << r_even);
                        if (neg_o) sign_bits |= (1U << r_odd);
                    }
                    std::memcpy(seg_row + 16, &sign_bits, 4);
                }
                cursor += 200;
                uint8_t* tail_bytes = blk + cursor;
                cursor += 16;
                int col_tail = sub * 32 + 30;
                for (int pair = 0; pair < 16; ++pair) {
                    int r_even = pair * 2;
                    int r_odd = pair * 2 + 1;
                    int64_t gr_e = rb * 32 + r_even;
                    int64_t gr_o = rb * 32 + r_odd;
                    uint8_t te0 = static_cast<uint8_t>(trits[(gr_e * group_blocks + gb) * 128 + col_tail + 0] + 1);
                    uint8_t te1 = static_cast<uint8_t>(trits[(gr_e * group_blocks + gb) * 128 + col_tail + 1] + 1);
                    uint8_t to0 = static_cast<uint8_t>(trits[(gr_o * group_blocks + gb) * 128 + col_tail + 0] + 1);
                    uint8_t to1 = static_cast<uint8_t>(trits[(gr_o * group_blocks + gb) * 128 + col_tail + 1] + 1);
                    uint8_t tail_e = static_cast<uint8_t>(te0 | (te1 << 2));
                    uint8_t tail_o = static_cast<uint8_t>(to0 | (to1 << 2));
                    tail_bytes[pair] = static_cast<uint8_t>(tail_e | (tail_o << 4));
                }
            }
        }
    }

    TernaryMatrix mat_tl2;
    load_tl2_to_ternary_matrix(tl2_buf.data(), rows_blocks, group_blocks, mat_tl2);
    std::vector<float> unpacked_tl2(out_features * in_features);
    unpack_ternary_matrix_to_f32(mat_tl2, unpacked_tl2.data());

    for (size_t i = 0; i < numel; ++i) {
        REQUIRE(std::fabs(unpacked[i] - unpacked_tl2[i]) < 1e-7f);
    }
    std::cout << "  TL2 & 1.58-bit exact reconstruction passed (0 bit errors across " << numel << " trits).\n";
}

static void test_simd_ternary_gemv_accuracy() {
    std::cout << "[Test 4] Testing AVX2 Ternary GEMV (FP32 & INT8-Quantized) accuracy...\n";
    const int64_t out_features = 128;
    const int64_t in_features = 512;
    const int64_t groups_per_row = in_features / 128;

    std::mt19937 rng(99);
    std::normal_distribution<float> x_dist(0.0f, 1.0f);

    std::vector<int8_t> trits(out_features * in_features);
    for (auto& t : trits) t = static_cast<int8_t>((rng() % 3) - 1);

    std::vector<float> scales(out_features * groups_per_row);
    for (auto& s : scales) s = 0.05f + 0.02f * std::fabs(x_dist(rng));

    TernaryMatrix W;
    pack_trits_to_ternary_matrix(trits.data(), scales.data(), out_features, in_features, W);

    std::vector<float> W_dense(out_features * in_features);
    unpack_ternary_matrix_to_f32(W, W_dense.data());

    std::vector<float> x(in_features);
    for (auto& v : x) v = x_dist(rng);

    // Reference dense GEMV
    std::vector<float> y_ref(out_features, 0.0f);
    for (int64_t o = 0; o < out_features; ++o) {
        float sum = 0.0f;
        for (int64_t i = 0; i < in_features; ++i) {
            sum += W_dense[o * in_features + i] * x[i];
        }
        y_ref[o] = sum;
    }

    // 1. AVX2 FP32 Ternary GEMV
    std::vector<float> y_f32(out_features, 0.0f);
    ternary_matmul_f32(W, x.data(), 1, y_f32.data());

    float max_f32_err = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
        max_f32_err = std::max(max_f32_err, std::fabs(y_ref[o] - y_f32[o]));
    }
    std::cout << "  AVX2 FP32 Ternary GEMV max error vs reference: " << max_f32_err << "\n";
    REQUIRE(max_f32_err < 1e-4f);

    // 2. AVX2 INT8-Activation x Ternary GEMV (`_mm256_maddubs_epi16`)
    QuantizedActivation xq;
    xq.quantize_from(x.data(), 1, in_features);
    std::vector<float> y_q8(out_features, 0.0f);
    ternary_matmul_quantized(W, xq, y_q8.data());

    // Compute cosine similarity between y_ref and y_q8
    double dot = 0.0, nr = 0.0, nq = 0.0;
    for (int64_t o = 0; o < out_features; ++o) {
        dot += static_cast<double>(y_ref[o]) * y_q8[o];
        nr  += static_cast<double>(y_ref[o]) * y_ref[o];
        nq  += static_cast<double>(y_q8[o]) * y_q8[o];
    }
    double cosine = dot / (std::sqrt(nr) * std::sqrt(nq));
    std::cout << "  AVX2 INT8 x Ternary GEMV cosine similarity vs FP32: " << cosine << "\n";
    REQUIRE(cosine > 0.9999);
}

static void test_local_hf_cache_assets() {
    std::cout << "[Test 5] Testing local HuggingFace cache resolution & NPZ voice loading...\n";
    // Check if KittenML/kitten-tts-2 is in local HF cache without downloading weights yet
    ResolvedRepo repo = resolve_repository("KittenML/kitten-tts-2", "", "", false);
    REQUIRE(repo.config.value_string("type") == "KITTEN2");
    REQUIRE(repo.config.value_int("sample_rate") == 24000);
    REQUIRE(repo.config.value_string("lm_emb4") == "lm/model-tl2-emb4.safetensors");

    // Check loading a built-in voice .npz file from the cached snapshot
    auto voices_json_path = repo.root_dir / "voices" / "voices.json";
    if (std::filesystem::exists(voices_json_path)) {
        JsonValue voices = parse_json_file(voices_json_path.string());
        REQUIRE(voices.contains("Bruno"));
        std::string bruno_npz_rel = "voices/" + voices.at("Bruno").value_string("artifacts");
        auto bruno_npz_path = repo.root_dir / bruno_npz_rel;
        if (std::filesystem::exists(bruno_npz_path)) {
            NpzArchive npz(bruno_npz_path.string());
            REQUIRE(npz.has_array("embedding"));
            REQUIRE(npz.has_array("reference_tokens"));
            TensorView emb = npz.get_array("embedding");
            REQUIRE(emb.numel() == 512);
            std::cout << "  Loaded Bruno voice NPZ: embedding=" << emb.numel()
                      << " dims, reference_tokens=" << npz.get_array("reference_tokens").numel() << "\n";
        }
    }

    // Verify real emb4 checkpoint loading, FWHT embedding lookup, and dual-space tied logits
    auto emb4_path = repo.root_dir / "lm" / "model-tl2-emb4.safetensors";
    if (std::filesystem::exists(emb4_path)) {
        std::cout << "[Test 6] Verifying real KittenTTS 2 emb4 checkpoint (TL2 + FWHT emb4 + tied logits)...\n";
        KittenTernaryLMWeights lm;
        lm.load(repo, "emb4", 1); // Load layer 0 + full emb4 table + spk_proj
        REQUIRE(lm.layers.size() == 1);
        REQUIRE(lm.emb4_table.vocab_size() == 158241);
        REQUIRE(lm.emb4_table.dim() == 2048);

        // Verify FWHT emb4 single-row lookup and compare dual-space tied logits vs explicit dot products
        std::vector<float> hidden(2048);
        lm.embed_token(151673, hidden.data()); // start_id
        rms_norm_f32(hidden.data(), lm.final_norm.data(), 2048, lm.config.rms_norm_eps, hidden.data());

        // Compute tied logits for audio tokens [151675..151775) via fast compressed FWHT dual-space kernel
        const int64_t v_start = 151675;
        const int64_t v_end = 151775;
        std::vector<float> fast_logits(v_end - v_start);
        lm.emb4_table.compute_tied_logits(hidden.data(), fast_logits.data(), v_start, v_end);

        // Compare against explicit row-by-row unrounded FWHT dequantization + dot product
        std::vector<float> row_vec(2048);
        float max_logit_err = 0.0f;
        for (int64_t vid = v_start; vid < v_end; ++vid) {
            lm.emb4_table.lookup_row(vid, row_vec.data(), /*round_via_bf16=*/false);
            float ref_dot = 0.0f;
            for (int64_t c = 0; c < 2048; ++c) ref_dot += row_vec[c] * hidden[c];
            max_logit_err = std::max(max_logit_err, std::fabs(ref_dot - fast_logits[vid - v_start]));
        }
        std::cout << "  Dual-space FWHT tied-logits max error vs row-dequant dot: " << max_logit_err << "\n";
        REQUIRE(max_logit_err < 1e-4f);
    }
}

int main() {
    std::cout << "=== KittenI Phase 1 Unit Tests ===\n";
    std::cout << "Detected CPU: " << cpu_capabilities_summary() << "\n";
    test_fp16_bf16_conversions();
    test_fwht_orthogonality();
    test_ternary_pack_and_tl2_unpack();
    test_simd_ternary_gemv_accuracy();
    test_local_hf_cache_assets();
    std::cout << "=== All Phase 1 Unit Tests PASSED ===\n";
    return 0;
}
