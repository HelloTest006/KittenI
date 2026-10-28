#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace kitteni {

struct EngineOptions {
    std::string model_or_path = "KittenML/kitten-tts-2";
    std::string weights = "emb4";       // "emb4" | "packed"
    std::string decoder = "student_w4"; // "student_w4" | "student_w8"
    std::string cache_dir = "";
    std::string isa = "auto";           // "auto" | "scalar" | "avx2" | "avx_vnni" | "avx512_vnni"
    int threads = 0;                    // 0 = auto hardware_concurrency
    int vocoder_threads = 0;            // 0 = auto (used by pipelined background vocoder worker)
    int max_layers = -1;                // -1 = all layers (e.g. 28 for 1.7B, fewer for 80M/15M/7M or fast testing)
    bool allow_download = true;
};

struct GenerateOptions {
    std::string voice = "";             // Built-in voice name (default from config.json, e.g. "Bruno")
    std::string reference_wav = "";     // Optional path to 5-30s WAV for zero-shot runtime cloning
    std::string reference_text = "";    // Optional transcript for reference_wav
    std::string preset = "";            // "stable" | "expressive"
    int max_new_tokens = 1000;
    bool normalize = true;
    int use_reference_prompt = -1;      // -1 = use model config default, 0 = false, 1 = true
    float temperature = -1.0f;          // < 0 = use preset default
    int top_k = -1;                     // < 0 = use preset default
    float top_p = -1.0f;                // < 0 = use preset default
    float min_p = -1.0f;                // < 0 = use preset default
    float repetition_penalty = 1.1f;
    int repetition_window = -1;         // < 0 = use config default (50)
    float token_run_penalty = 1.3f;
    int token_run_grace = 10;
    int chunk_chars = -1;               // < 0 = use config default (380)
    int chunk_min_chars = -1;           // < 0 = use config default (130)
    float chunk_gap_s = 0.16f;
    int seed = 1234;
    bool greedy = false;
};

struct SynthesisReport {
    std::string normalized_text;
    size_t num_chunks = 0;
    size_t total_generated_tokens = 0;
    size_t total_audio_tokens = 0;
    size_t total_samples = 0;
    double lm_seconds = 0.0;
    double vocoder_seconds = 0.0;
    double wall_seconds = 0.0;
    double audio_seconds = 0.0;
    double rtf = 0.0;
};

// Callback invoked as each chunk is synthesized in `generate_stream`:
//   `chunk_index`: 0-based index of chunk (`0 .. total_chunks - 1`)
//   `total_chunks`: total number of chunks
//   `pcm_f32`: pointer to 24 kHz mono float32 audio samples for this chunk
//   `num_samples`: number of float32 samples in `pcm_f32`
// Return `true` to continue streaming, or `false` to abort early.
using StreamChunkCallback = std::function<bool(
    size_t chunk_index,
    size_t total_chunks,
    const float* pcm_f32,
    size_t num_samples
)>;

class Engine {
public:
    explicit Engine(const EngineOptions& opts = EngineOptions{});
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    int sample_rate() const;
    std::vector<std::string> available_voices() const;
    std::vector<std::string> available_decoders() const;
    std::vector<std::string> available_weights() const;

    std::string normalize_text(const std::string& text, bool normalize = true) const;

    // Synthesize full text into a joined 24 kHz float32 waveform.
    // Uses dual-stage pipelined execution when `chunks.size() > 1` (LM generates chunk N+1 while Vocoder decodes chunk N).
    std::vector<float> generate(
        const std::string& text,
        const GenerateOptions& opts = GenerateOptions{},
        SynthesisReport* out_report = nullptr
    );

    // Stream each chunk's 24 kHz float32 audio to `callback` as soon as its vocoder pass finishes,
    // while the Speech LM concurrently generates tokens for the next chunk in the background!
    void generate_stream(
        const std::string& text,
        StreamChunkCallback callback,
        const GenerateOptions& opts = GenerateOptions{},
        SynthesisReport* out_report = nullptr
    );

    // Synthesize and write directly to a 24 kHz IEEE Float32 WAV file.
    std::string generate_to_file(
        const std::string& text,
        const std::string& output_wav_path,
        const GenerateOptions& opts = GenerateOptions{},
        SynthesisReport* out_report = nullptr
    );

    // Pre-extract and save a cloned voice profile from a 5-30s WAV file to a JSON profile file.
    void clone_voice_to_file(
        const std::string& reference_wav_path,
        const std::string& reference_text,
        const std::string& output_profile_json_path
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kitteni
