#ifndef KITTENI_H
#define KITTENI_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#  if defined(KITTENI_SHARED_BUILD)
#    define KITTENI_API __declspec(dllexport)
#  elif defined(KITTENI_SHARED_USE)
#    define KITTENI_API __declspec(dllimport)
#  else
#    define KITTENI_API
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define KITTENI_API __attribute__((visibility("default")))
#  else
#    define KITTENI_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kitteni_engine kitteni_engine_t;

typedef struct kitteni_engine_options {
    const char* model_or_path; /* Default: "KittenML/kitten-tts-2" */
    const char* weights;       /* "emb4" | "packed" (Default: "emb4") */
    const char* decoder;       /* "student_w4" | "student_w8" (Default: "student_w4") */
    const char* cache_dir;     /* NULL or "" for default HF cache */
    const char* isa;           /* "auto" | "scalar" | "avx2" | "avx_vnni" | "avx512_vnni" */
    int32_t threads;           /* 0 = auto */
    int32_t vocoder_threads;   /* 0 = auto */
    int32_t max_layers;        /* -1 = all layers */
    int32_t allow_download;    /* 1 = true, 0 = offline */
} kitteni_engine_options_t;

typedef struct kitteni_generate_options {
    const char* voice;           /* Built-in voice name or NULL for default ("Bruno") */
    const char* reference_wav;   /* Path to 5-30s WAV for runtime voice cloning, or NULL */
    const char* reference_text;  /* Transcript of reference_wav, or NULL */
    const char* preset;          /* "stable" | "expressive", or NULL */
    int32_t max_new_tokens;      /* Default: 1000 */
    int32_t normalize;           /* 1 = normalize text, 0 = raw */
    int32_t use_reference_prompt;/* -1 = default, 0 = off, 1 = on */
    float temperature;           /* < 0 = preset default */
    int32_t top_k;               /* < 0 = preset default */
    float top_p;                 /* < 0 = preset default */
    float min_p;                 /* < 0 = preset default */
    float repetition_penalty;    /* Default: 1.1 */
    int32_t repetition_window;   /* < 0 = config default (50) */
    float token_run_penalty;     /* Default: 1.3 */
    int32_t token_run_grace;     /* Default: 10 */
    int32_t chunk_chars;         /* < 0 = config default (380) */
    int32_t chunk_min_chars;     /* < 0 = config default (130) */
    float chunk_gap_s;           /* Default: 0.16 */
    int32_t seed;                /* Default: 1234 */
    int32_t greedy;              /* 0 = sample, 1 = greedy argmax */
} kitteni_generate_options_t;

typedef struct kitteni_report {
    size_t num_chunks;
    size_t total_generated_tokens;
    size_t total_audio_tokens;
    size_t total_samples;
    double lm_seconds;
    double vocoder_seconds;
    double wall_seconds;
    double audio_seconds;
    double rtf;
} kitteni_report_t;

/* Callback for chunk streaming. Return 1 to continue, 0 to stop early. */
typedef int (*kitteni_stream_callback_t)(
    size_t chunk_index,
    size_t total_chunks,
    const float* pcm_f32,
    size_t num_samples,
    void* user_data
);

/* Populate default options structs */
KITTENI_API void kitteni_default_engine_options(kitteni_engine_options_t* opts);
KITTENI_API void kitteni_default_generate_options(kitteni_generate_options_t* opts);

/* Return the last error message on the calling thread */
KITTENI_API const char* kitteni_last_error(void);

/* Create and destroy a KittenI engine instance */
KITTENI_API kitteni_engine_t* kitteni_create(const kitteni_engine_options_t* opts);
KITTENI_API void kitteni_destroy(kitteni_engine_t* engine);

/* Query engine metadata */
KITTENI_API int32_t kitteni_sample_rate(const kitteni_engine_t* engine);
KITTENI_API size_t kitteni_num_voices(const kitteni_engine_t* engine);
KITTENI_API const char* kitteni_voice_name(const kitteni_engine_t* engine, size_t index);

/* Generate speech into a newly allocated float32 buffer (`*out_pcm`).
   Caller must free `*out_pcm` using `kitteni_free_audio(*out_pcm)`.
   Returns 0 on success, non-zero on error. */
KITTENI_API int kitteni_generate(
    kitteni_engine_t* engine,
    const char* text,
    const kitteni_generate_options_t* opts,
    float** out_pcm,
    size_t* out_num_samples,
    kitteni_report_t* out_report
);

/* Stream speech chunk-by-chunk using pipelined LM + Vocoder execution.
   Returns 0 on success, non-zero on error. */
KITTENI_API int kitteni_generate_stream(
    kitteni_engine_t* engine,
    const char* text,
    const kitteni_generate_options_t* opts,
    kitteni_stream_callback_t callback,
    void* user_data,
    kitteni_report_t* out_report
);

/* Generate speech and write directly to a 24 kHz float32 WAV file.
   Returns 0 on success, non-zero on error. */
KITTENI_API int kitteni_generate_to_file(
    kitteni_engine_t* engine,
    const char* text,
    const char* output_wav_path,
    const kitteni_generate_options_t* opts,
    kitteni_report_t* out_report
);

/* Free audio buffer allocated by `kitteni_generate`. */
KITTENI_API void kitteni_free_audio(float* pcm);

#ifdef __cplusplus
}
#endif

#endif /* KITTENI_H */
