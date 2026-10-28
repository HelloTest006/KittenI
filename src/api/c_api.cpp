#include "kitteni/kitteni.h"
#include "kitteni/kitteni.hpp"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct kitteni_engine {
    kitteni::Engine impl;
    std::vector<std::string> cached_voices;
    explicit kitteni_engine(const kitteni::EngineOptions& opts)
        : impl(opts), cached_voices(impl.available_voices()) {}
};

namespace {

thread_local std::string g_last_error;

kitteni::GenerateOptions convert_gen_opts(const kitteni_generate_options_t* c_opts) {
    kitteni::GenerateOptions o;
    if (!c_opts) return o;
    if (c_opts->voice) o.voice = c_opts->voice;
    if (c_opts->reference_wav) o.reference_wav = c_opts->reference_wav;
    if (c_opts->reference_text) o.reference_text = c_opts->reference_text;
    if (c_opts->preset) o.preset = c_opts->preset;
    o.max_new_tokens       = c_opts->max_new_tokens > 0 ? c_opts->max_new_tokens : 1000;
    o.normalize            = (c_opts->normalize != 0);
    o.use_reference_prompt = c_opts->use_reference_prompt;
    o.temperature          = c_opts->temperature;
    o.top_k                = c_opts->top_k;
    o.top_p                = c_opts->top_p;
    o.min_p                = c_opts->min_p;
    o.repetition_penalty   = c_opts->repetition_penalty > 0.0f ? c_opts->repetition_penalty : 1.1f;
    o.repetition_window    = c_opts->repetition_window;
    o.token_run_penalty    = c_opts->token_run_penalty >= 0.0f ? c_opts->token_run_penalty : 1.3f;
    o.token_run_grace      = c_opts->token_run_grace > 0 ? c_opts->token_run_grace : 10;
    o.chunk_chars          = c_opts->chunk_chars;
    o.chunk_min_chars      = c_opts->chunk_min_chars;
    o.chunk_gap_s          = c_opts->chunk_gap_s >= 0.0f ? c_opts->chunk_gap_s : 0.16f;
    o.seed                 = c_opts->seed;
    o.greedy               = (c_opts->greedy != 0);
    return o;
}

void copy_report(const kitteni::SynthesisReport& src, kitteni_report_t* dst) {
    if (!dst) return;
    dst->num_chunks             = src.num_chunks;
    dst->total_generated_tokens = src.total_generated_tokens;
    dst->total_audio_tokens     = src.total_audio_tokens;
    dst->total_samples          = src.total_samples;
    dst->lm_seconds             = src.lm_seconds;
    dst->vocoder_seconds        = src.vocoder_seconds;
    dst->wall_seconds           = src.wall_seconds;
    dst->audio_seconds          = src.audio_seconds;
    dst->rtf                    = src.rtf;
}

} // namespace

extern "C" {

KITTENI_API void kitteni_default_engine_options(kitteni_engine_options_t* opts) {
    if (!opts) return;
    opts->model_or_path   = "KittenML/kitten-tts-2";
    opts->weights         = "emb4";
    opts->decoder         = "student_w4";
    opts->cache_dir       = "";
    opts->isa             = "auto";
    opts->threads         = 0;
    opts->vocoder_threads = 0;
    opts->max_layers      = -1;
    opts->allow_download  = 1;
}

KITTENI_API void kitteni_default_generate_options(kitteni_generate_options_t* opts) {
    if (!opts) return;
    opts->voice                = nullptr;
    opts->reference_wav        = nullptr;
    opts->reference_text       = nullptr;
    opts->preset               = nullptr;
    opts->max_new_tokens       = 1000;
    opts->normalize            = 1;
    opts->use_reference_prompt = -1;
    opts->temperature          = -1.0f;
    opts->top_k                = -1;
    opts->top_p                = -1.0f;
    opts->min_p                = -1.0f;
    opts->repetition_penalty   = 1.1f;
    opts->repetition_window    = -1;
    opts->token_run_penalty    = 1.3f;
    opts->token_run_grace      = 10;
    opts->chunk_chars          = -1;
    opts->chunk_min_chars      = -1;
    opts->chunk_gap_s          = 0.16f;
    opts->seed                 = 1234;
    opts->greedy               = 0;
}

KITTENI_API const char* kitteni_last_error(void) {
    return g_last_error.c_str();
}

KITTENI_API kitteni_engine_t* kitteni_create(const kitteni_engine_options_t* c_opts) {
    try {
        kitteni::EngineOptions o;
        if (c_opts) {
            if (c_opts->model_or_path && *c_opts->model_or_path) o.model_or_path = c_opts->model_or_path;
            if (c_opts->weights && *c_opts->weights) o.weights = c_opts->weights;
            if (c_opts->decoder && *c_opts->decoder) o.decoder = c_opts->decoder;
            if (c_opts->cache_dir && *c_opts->cache_dir) o.cache_dir = c_opts->cache_dir;
            if (c_opts->isa && *c_opts->isa) o.isa = c_opts->isa;
            o.threads         = c_opts->threads;
            o.vocoder_threads = c_opts->vocoder_threads;
            o.max_layers      = c_opts->max_layers;
            o.allow_download  = (c_opts->allow_download != 0);
        }
        return new kitteni_engine(o);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

KITTENI_API void kitteni_destroy(kitteni_engine_t* engine) {
    delete engine;
}

KITTENI_API int32_t kitteni_sample_rate(const kitteni_engine_t* engine) {
    return engine ? engine->impl.sample_rate() : 24000;
}

KITTENI_API size_t kitteni_num_voices(const kitteni_engine_t* engine) {
    return engine ? engine->cached_voices.size() : 0;
}

KITTENI_API const char* kitteni_voice_name(const kitteni_engine_t* engine, size_t index) {
    if (!engine || index >= engine->cached_voices.size()) return nullptr;
    return engine->cached_voices[index].c_str();
}

KITTENI_API int kitteni_generate(
    kitteni_engine_t* engine,
    const char* text,
    const kitteni_generate_options_t* opts,
    float** out_pcm,
    size_t* out_num_samples,
    kitteni_report_t* out_report
) {
    if (!engine || !text || !out_pcm || !out_num_samples) {
        g_last_error = "Invalid null argument passed to kitteni_generate";
        return -1;
    }
    try {
        kitteni::SynthesisReport rep;
        std::vector<float> wav = engine->impl.generate(text, convert_gen_opts(opts), &rep);
        copy_report(rep, out_report);

        float* buf = static_cast<float*>(std::malloc(wav.size() * sizeof(float)));
        if (!buf && !wav.empty()) {
            g_last_error = "Out of memory allocating audio buffer";
            return -2;
        }
        if (!wav.empty()) {
            std::memcpy(buf, wav.data(), wav.size() * sizeof(float));
        }
        *out_pcm = buf;
        *out_num_samples = wav.size();
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -3;
    }
}

KITTENI_API int kitteni_generate_stream(
    kitteni_engine_t* engine,
    const char* text,
    const kitteni_generate_options_t* opts,
    kitteni_stream_callback_t callback,
    void* user_data,
    kitteni_report_t* out_report
) {
    if (!engine || !text || !callback) {
        g_last_error = "Invalid null argument passed to kitteni_generate_stream";
        return -1;
    }
    try {
        kitteni::SynthesisReport rep;
        engine->impl.generate_stream(
            text,
            [&](size_t idx, size_t total, const float* pcm, size_t n) -> bool {
                return callback(idx, total, pcm, n, user_data) != 0;
            },
            convert_gen_opts(opts),
            &rep
        );
        copy_report(rep, out_report);
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -3;
    }
}

KITTENI_API int kitteni_generate_to_file(
    kitteni_engine_t* engine,
    const char* text,
    const char* output_wav_path,
    const kitteni_generate_options_t* opts,
    kitteni_report_t* out_report
) {
    if (!engine || !text || !output_wav_path) {
        g_last_error = "Invalid null argument passed to kitteni_generate_to_file";
        return -1;
    }
    try {
        kitteni::SynthesisReport rep;
        engine->impl.generate_to_file(text, output_wav_path, convert_gen_opts(opts), &rep);
        copy_report(rep, out_report);
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -3;
    }
}

KITTENI_API void kitteni_free_audio(float* pcm) {
    std::free(pcm);
}

} // extern "C"
