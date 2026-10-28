#include "kitteni/kitteni.hpp"
#include "kitteni/kitteni.h"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

using namespace kitteni;

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

static void test_cpp_engine_and_pipelined_streaming() {
    std::cout << "[Test 1] Testing C++20 kitteni::Engine & multi-chunk pipelined producer-consumer streaming...\n";
    EngineOptions eo;
    eo.model_or_path = "KittenML/kitten-tts-2";
    eo.weights = "emb4";
    eo.decoder = "student_w4";
    eo.threads = 4;
    eo.max_layers = 2; // 2 layers for fast unit test
    eo.allow_download = false;

    Engine eng(eo);
    REQUIRE(eng.sample_rate() == 24000);
    auto voices = eng.available_voices();
    REQUIRE(voices.size() >= 47);

    // Multi-chunk text (chunk_chars=80, chunk_min_chars=30 -> 2 chunks)
    std::string text =
        "The old lighthouse stood on the rocky cliff above the sea. "
        "Every night its bright beam guided ships safely through the storm.";

    GenerateOptions go;
    go.voice = "Bruno";
    go.max_new_tokens = 8; // 8 tokens per chunk for fast test
    go.use_reference_prompt = 0;
    go.chunk_chars = 65;
    go.chunk_min_chars = 30;
    go.seed = 42;

    size_t callbacks_received = 0;
    size_t total_stream_samples = 0;
    SynthesisReport rep;

    eng.generate_stream(
        text,
        [&](size_t idx, size_t total, const float* pcm, size_t n) -> bool {
            std::cout << "  Stream callback chunk " << (idx + 1) << "/" << total << ": " << n << " samples\n";
            REQUIRE(total == 2);
            REQUIRE(n > 0);
            for (size_t i = 0; i < n; ++i) REQUIRE(std::isfinite(pcm[i]));
            callbacks_received++;
            total_stream_samples += n;
            return true;
        },
        go,
        &rep
    );

    REQUIRE(callbacks_received == 2);
    REQUIRE(rep.num_chunks == 2);
    REQUIRE(rep.total_samples == total_stream_samples);
    std::cout << "  Pipelined stream finished: " << rep.num_chunks << " chunks, "
              << rep.total_samples << " samples, wall=" << rep.wall_seconds << "s\n";
}

static void test_c_abi_library() {
    std::cout << "[Test 2] Testing C ABI (`kitteni.h`: create, voice query, generate, generate_to_file, destroy)...\n";
    kitteni_engine_options_t eo;
    kitteni_default_engine_options(&eo);
    eo.threads = 4;
    eo.max_layers = 1;
    eo.allow_download = 0;

    kitteni_engine_t* handle = kitteni_create(&eo);
    REQUIRE(handle != nullptr);
    REQUIRE(kitteni_sample_rate(handle) == 24000);
    REQUIRE(kitteni_num_voices(handle) >= 47);
    REQUIRE(std::string(kitteni_voice_name(handle, 0)).size() > 0);

    kitteni_generate_options_t go;
    kitteni_default_generate_options(&go);
    go.voice = "Bruno";
    go.max_new_tokens = 6;
    go.use_reference_prompt = 0;
    go.seed = 123;

    float* pcm = nullptr;
    size_t n_samples = 0;
    kitteni_report_t rep{};

    int rc = kitteni_generate(handle, "Hello from the C API.", &go, &pcm, &n_samples, &rep);
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(n_samples > 0);
    REQUIRE(rep.total_samples == n_samples);
    kitteni_free_audio(pcm);

    std::string wav_out = "build/test_c_api.wav";
    rc = kitteni_generate_to_file(handle, "Hello file.", wav_out.c_str(), &go, &rep);
    REQUIRE(rc == 0);
    REQUIRE(std::filesystem::exists(wav_out));

    kitteni_destroy(handle);
    std::cout << "  C ABI verified (" << n_samples << " samples generated and written to " << wav_out << ").\n";
}

int main() {
    std::cout << "=== KittenI Phase 5 C++ Engine, Streaming & C ABI Tests ===\n";
    test_cpp_engine_and_pipelined_streaming();
    test_c_abi_library();
    std::cout << "=== All Phase 5 Unit Tests PASSED ===\n";
    return 0;
}
