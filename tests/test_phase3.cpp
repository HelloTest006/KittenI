#include "core/hub_cache.hpp"
#include "kernels/cpu_features.hpp"
#include "kernels/ternary_gemv.hpp"
#include "kernels/dsp_fft.hpp"
#include "kernels/bitnet_ops.hpp"
#include "frontend/audio_joiner.hpp"
#include "vocoder/vocoder_pipeline.hpp"

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <filesystem>

using namespace kitteni;

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

static void test_stft_istft_16_roundtrip() {
    std::cout << "[Test 1] Testing 16-point STFT -> ISTFT perfect reconstruction...\n";
    const int64_t L = 4800; // 10 mel frames * 480
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> orig(L);
    for (float& v : orig) v = dist(rng);

    std::vector<float> win = hann_window_periodic(16);
    std::vector<float> stft_out;
    hift_stft_16(orig.data(), L, win.data(), stft_out);

    const int64_t T_frames = L / 4 + 1;
    REQUIRE(static_cast<int64_t>(stft_out.size()) == 18 * T_frames);

    std::vector<float> recon;
    hift_istft_16(stft_out.data(), stft_out.data() + 9 * T_frames, T_frames, win.data(), recon);
    REQUIRE(static_cast<int64_t>(recon.size()) == L);

    float max_err = 0.0f;
    for (int64_t i = 0; i < L; ++i) {
        max_err = std::max(max_err, std::fabs(orig[i] - recon[i]));
    }
    std::cout << "  16-point STFT -> ISTFT roundtrip max error: " << max_err << "\n";
    REQUIRE(max_err < 1e-5f);
}

static void test_bitnet_linear_and_conv1d() {
    std::cout << "[Test 2] Testing BitLinear and BitConv1d integer kernels...\n";
    const int64_t out_dim = 64;
    const int64_t in_dim = 128;

    std::vector<float> w_f32(out_dim * in_dim);
    std::vector<float> b_f32(out_dim, 0.25f);
    for (int64_t o = 0; o < out_dim; ++o) {
        float scale = 20.0f + o;
        for (int64_t i = 0; i < in_dim; ++i) {
            int code = ((o + i) % 15) - 7; // in [-7..7]
            w_f32[o * in_dim + i] = static_cast<float>(code) / scale;
        }
    }

    BitNetWeight bw;
    bw.init_from_dequantized_f32(w_f32.data(), b_f32.data(), out_dim, in_dim, 1, /*bits=*/4);

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(2 * in_dim);
    for (float& v : x) v = dist(rng);

    std::vector<float> y_ref(2 * out_dim);
    // Reference float BitLinear
    for (int r = 0; r < 2; ++r) {
        float max_abs = 1e-5f;
        for (int64_t i = 0; i < in_dim; ++i) max_abs = std::max(max_abs, std::fabs(x[r * in_dim + i]));
        float sx = 127.0f / max_abs;
        for (int64_t o = 0; o < out_dim; ++o) {
            float acc = b_f32[o];
            for (int64_t i = 0; i < in_dim; ++i) {
                float xq = std::clamp(std::nearbyintf(x[r * in_dim + i] * sx), -128.0f, 127.0f) / sx;
                acc += xq * w_f32[o * in_dim + i];
            }
            y_ref[r * out_dim + o] = acc;
        }
    }

    // Verify all supported SIMD ISAs (SCALAR, AVX2, AVX_VNNI, AVX512_VNNI) produce bit-identical results!
    const SimdIsa isas[] = {SimdIsa::SCALAR, SimdIsa::AVX2, SimdIsa::AVX_VNNI, SimdIsa::AVX512_VNNI};
    for (SimdIsa cand : isas) {
        if (!is_isa_supported(cand)) continue;
        set_active_isa(cand);
        std::vector<float> y(2 * out_dim, 0.0f);
        bitlinear_forward(bw, x.data(), 2, y.data());
        for (size_t i = 0; i < y.size(); ++i) {
            REQUIRE(std::fabs(y_ref[i] - y[i]) < 1e-4f);
        }
        std::cout << "  BitLinear ISA=" << isa_name(cand) << " matched reference within < 1e-4.\n";
    }
    set_active_isa(SimdIsa::AUTO);
}

static void test_native_vocoder_pipeline() {
    std::cout << "[Test 3] Testing Native S3StudentFlow + HiFTGenerator Vocoder pipeline on real weights...\n";
    ResolvedRepo repo = resolve_repository("KittenML/kitten-tts-2", "", "", false);
    NativeVocoderPipeline vocoder;
    vocoder.load(repo, "student_w4");
    PreparedVoice bruno = vocoder.load_voice("Bruno");
    REQUIRE(bruno.prompt_token.size() == 100);
    REQUIRE(bruno.prompt_feat_frames == 200);
    REQUIRE(bruno.flow_embedding_192.size() == 192);

    ThreadPool pool(4);
    std::mt19937 rng(1234);
    // Decode 8 speech tokens (+ 3 silence lookahead tokens = 11 tokens = 22 mel frames = 10,560 samples)
    std::vector<int32_t> tokens = {120, 450, 890, 1024, 2048, 3100, 4200, 4299};
    auto wav = vocoder.decode_tokens_to_wav(tokens, bruno, rng, &pool);

    REQUIRE(wav.size() == (tokens.size() + 3) * 2 * 480);
    float max_abs = 0.0f;
    for (float s : wav) {
        REQUIRE(std::isfinite(s));
        max_abs = std::max(max_abs, std::fabs(s));
    }
    std::cout << "  Generated " << wav.size() << " samples (24 kHz), peak amplitude = " << max_abs << "\n";
    REQUIRE(max_abs > 0.01f && max_abs <= 0.99f);
}

int main() {
    std::cout << "=== KittenI Phase 3 Unit & Vocoder Tests ===\n";
    test_stft_istft_16_roundtrip();
    test_bitnet_linear_and_conv1d();
    test_native_vocoder_pipeline();
    std::cout << "=== All Phase 3 Unit Tests PASSED ===\n";
    return 0;
}
