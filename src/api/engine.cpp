#include "kitteni/kitteni.hpp"
#include "core/hub_cache.hpp"
#include "core/mmap_loader.hpp"
#include "core/thread_pool.hpp"
#include "kernels/cpu_features.hpp"
#include "frontend/bpe_tokenizer.hpp"
#include "frontend/text_normalizer.hpp"
#include "frontend/audio_joiner.hpp"
#include "lm/model_loader.hpp"
#include "lm/sampler.hpp"
#include "lm/qwen3_ternary.hpp"
#include "vocoder/vocoder_pipeline.hpp"
#include "cloning/voice_cloner.hpp"

#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace kitteni {

struct Engine::Impl {
    EngineOptions eng_opts;
    ResolvedRepo repo;
    TokenMap token_map;
    Qwen3BPETokenizer tokenizer;
    KittenTernaryLMWeights lm_weights;
    std::unique_ptr<Qwen3TernaryEngine> lm_engine;
    NativeVocoderPipeline vocoder;
    ThreadPool vocoder_pool;

    std::vector<std::string> voice_names;
    mutable std::map<std::string, ClonedVoiceProfile> clone_cache;
    mutable std::unique_ptr<VoiceCloner> cloner;

    explicit Impl(const EngineOptions& opts)
        : eng_opts(opts),
          vocoder_pool(opts.vocoder_threads > 0 ? opts.vocoder_threads : opts.threads) {
        set_active_isa(parse_isa_name(opts.isa));
        repo = resolve_repository(opts.model_or_path, opts.weights, opts.cache_dir, opts.allow_download);
        token_map = TokenMap::from_config(repo.config);

        std::string lm_dir = repo.config.value_string("lm_dir", "lm");
        ensure_repo_file(repo, lm_dir + "/vocab.json", opts.allow_download);
        ensure_repo_file(repo, lm_dir + "/merges.txt", opts.allow_download);
        ensure_repo_file(repo, lm_dir + "/added_tokens.json", opts.allow_download);
        tokenizer.load((repo.root_dir / lm_dir).string());

        lm_weights.load(repo, opts.weights, opts.max_layers, opts.threads);
        lm_engine = std::make_unique<Qwen3TernaryEngine>(lm_weights, opts.threads, /*use_int8_act=*/true);

        vocoder.load(repo, opts.decoder);

        std::string voices_rel = repo.config.value_string("voices", "voices/voices.json");
        auto v_path = ensure_repo_file(repo, voices_rel, opts.allow_download);
        JsonValue v_json = parse_json_file(v_path.string());
        for (const auto& [k, _] : v_json.as_object()) {
            voice_names.push_back(k);
        }
    }

    ClonedVoiceProfile resolve_speaker(const GenerateOptions& opts) const {
        if (!opts.reference_wav.empty()) {
            std::string cache_key = opts.reference_wav + "|" + opts.reference_text;
            auto it = clone_cache.find(cache_key);
            if (it != clone_cache.end()) {
                return it->second;
            }
            if (!cloner) {
                cloner = std::make_unique<VoiceCloner>();
                cloner->load(repo, /*ensure_s3_cloner=*/true, eng_opts.allow_download);
            }
            ClonedVoiceProfile prof = cloner->clone_from_wav(
                opts.reference_wav,
                opts.reference_text,
                &lm_weights,
                /*for_student_flow=*/true,
                &vocoder_pool
            );
            clone_cache.emplace(cache_key, prof);
            return prof;
        }

        std::string vname = opts.voice.empty()
                                ? repo.config.value_string("default_voice", "Bruno")
                                : opts.voice;

        std::string voices_rel = repo.config.value_string("voices", "voices/voices.json");
        JsonValue voices = parse_json_file((repo.root_dir / voices_rel).string());
        if (!voices.contains(vname)) {
            throw std::runtime_error("Unknown built-in voice: " + vname);
        }
        const JsonValue& spec = voices.at(vname);

        ClonedVoiceProfile prof;
        prof.vocoder_voice = vocoder.load_voice(vname);

        std::string npz_rel = "voices/" + spec.value_string("artifacts");
        auto npz_path = ensure_repo_file(repo, npz_rel, eng_opts.allow_download);
        NpzArchive npz(npz_path.string());
        TensorView emb_tv = npz.get_array("embedding");
        prof.embedding_512.resize(emb_tv.numel());
        if (emb_tv.dtype == DType::F32) {
            std::memcpy(prof.embedding_512.data(), emb_tv.data, emb_tv.numel() * sizeof(float));
        } else if (emb_tv.dtype == DType::F16) {
            const uint16_t* p = emb_tv.ptr<uint16_t>();
            for (size_t i = 0; i < emb_tv.numel(); ++i) prof.embedding_512[i] = fp16_to_fp32(p[i]);
        }
        return prof;
    }

    SamplingSettings resolve_sampling(const GenerateOptions& opts) const {
        SamplingSettings s;
        std::string preset = opts.preset.empty()
                                 ? repo.config.value_string("default_preset", "stable")
                                 : opts.preset;
        if (repo.config.contains("decode_presets") && repo.config.at("decode_presets").contains(preset)) {
            const JsonValue& p = repo.config.at("decode_presets").at(preset);
            s.temperature = static_cast<float>(p.value_float("temperature", 0.8));
            s.top_p       = static_cast<float>(p.value_float("top_p", 0.8));
            s.top_k       = static_cast<int32_t>(p.value_int("top_k", 50));
            s.min_p       = static_cast<float>(p.value_float("min_p", 0.0));
        }
        if (opts.temperature > 0.0f) s.temperature = opts.temperature;
        if (opts.top_k >= 0) s.top_k = opts.top_k;
        if (opts.top_p > 0.0f && opts.top_p <= 1.0f) s.top_p = opts.top_p;
        if (opts.min_p >= 0.0f && opts.min_p <= 1.0f) s.min_p = opts.min_p;

        s.repetition_penalty = opts.repetition_penalty;
        int rep_win = opts.repetition_window;
        if (rep_win < 0 && repo.config.contains("generation")) {
            rep_win = static_cast<int>(repo.config.at("generation").value_int("repetition_window", 50));
        }
        s.repetition_window = (rep_win >= 0) ? rep_win : 50;
        s.token_run_penalty = opts.token_run_penalty;
        s.token_run_grace   = opts.token_run_grace;
        s.greedy            = opts.greedy;
        return s;
    }
};

Engine::Engine(const EngineOptions& opts) : impl_(std::make_unique<Impl>(opts)) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

int Engine::sample_rate() const {
    return static_cast<int>(impl_->repo.config.value_int("sample_rate", 24000));
}

std::vector<std::string> Engine::available_voices() const {
    return impl_->voice_names;
}

std::vector<std::string> Engine::available_decoders() const {
    return {"default", "student_w4", "student_w8"};
}

std::vector<std::string> Engine::available_weights() const {
    return {"emb4", "packed", "full"};
}

std::string Engine::normalize_text(const std::string& text, bool normalize) const {
    return kitteni::normalize_text(text, normalize);
}

void Engine::clone_voice_to_file(
    const std::string& reference_wav_path,
    const std::string& reference_text,
    const std::string& output_profile_json_path
) {
    if (!impl_->cloner) {
        impl_->cloner = std::make_unique<VoiceCloner>();
        impl_->cloner->load(impl_->repo, /*ensure_s3_cloner=*/true, impl_->eng_opts.allow_download);
    }
    ClonedVoiceProfile prof = impl_->cloner->clone_from_wav(
        reference_wav_path, reference_text, &impl_->lm_weights, /*for_student_flow=*/true, &impl_->vocoder_pool
    );
    VoiceCloner::save_profile_json(output_profile_json_path, prof);
}

void Engine::generate_stream(
    const std::string& text,
    StreamChunkCallback callback,
    const GenerateOptions& opts,
    SynthesisReport* out_report
) {
    auto wall_t0 = std::chrono::steady_clock::now();

    std::string spoken = kitteni::normalize_text(text, opts.normalize);
    if (spoken.empty()) {
        throw std::runtime_error("No text to speak after normalization");
    }

    int max_c = opts.chunk_chars > 0 ? opts.chunk_chars : 380;
    int min_c = opts.chunk_min_chars > 0 ? opts.chunk_min_chars : 130;
    std::vector<std::string> chunks = split_for_synthesis(spoken, max_c, min_c);
    if (chunks.empty()) {
        throw std::runtime_error("No chunks produced for synthesis");
    }

    ClonedVoiceProfile spk = impl_->resolve_speaker(opts);
    SamplingSettings sampling = impl_->resolve_sampling(opts);

    bool use_ref = true;
    if (opts.use_reference_prompt >= 0) {
        use_ref = (opts.use_reference_prompt != 0);
    } else if (impl_->repo.config.contains("generation")) {
        use_ref = impl_->repo.config.at("generation").value_bool("use_reference_prompt", true);
    }
    if (spk.vocoder_voice.transcript.empty() || spk.vocoder_voice.reference_tokens.empty()) {
        use_ref = false;
    }

    std::vector<int32_t> ref_prefix;
    if (use_ref) {
        auto ref_text_ids = impl_->tokenizer.encode(spk.vocoder_voice.transcript);
        ref_prefix = build_reference_prefix(impl_->token_map, ref_text_ids, spk.vocoder_voice.reference_tokens);
    }

    std::vector<int32_t> emotion_ids;
    if (has_expression_tag(text) && impl_->repo.config.contains("generation")) {
        std::string emo_ctrl = impl_->repo.config.at("generation").value_string("emotion_control");
        if (!emo_ctrl.empty()) {
            emotion_ids = impl_->tokenizer.encode(emo_ctrl);
        }
    }

    SynthesisReport report;
    report.normalized_text = spoken;
    report.num_chunks = chunks.size();

    // Fast path for single-chunk input (no producer-consumer queue overhead needed)
    if (chunks.size() == 1) {
        std::mt19937 lm_rng(static_cast<uint32_t>(opts.seed));
        std::mt19937 voc_rng(static_cast<uint32_t>(opts.seed));

        auto target_ids = impl_->tokenizer.encode(chunks[0]);
        auto prompt_ids = build_generation_prompt(
            impl_->token_map, target_ids,
            use_ref ? &ref_prefix : nullptr,
            emotion_ids.empty() ? nullptr : &emotion_ids
        );

        GenerationStats stats;
        auto audio_ids = impl_->lm_engine->generate_audio_tokens(
            prompt_ids, spk.embedding_512.data(), impl_->token_map, sampling,
            opts.max_new_tokens, lm_rng, &stats
        );
        report.lm_seconds = stats.prefill_seconds + stats.decode_seconds;
        report.total_generated_tokens = static_cast<size_t>(stats.generated_tokens);
        report.total_audio_tokens = audio_ids.size();

        if (!audio_ids.empty()) {
            auto vt0 = std::chrono::steady_clock::now();
            auto wav = impl_->vocoder.decode_tokens_to_wav(
                audio_ids, spk.vocoder_voice, voc_rng, &impl_->vocoder_pool
            );
            auto vt1 = std::chrono::steady_clock::now();
            report.vocoder_seconds = std::chrono::duration<double>(vt1 - vt0).count();
            report.total_samples = wav.size();
            if (callback && !wav.empty()) {
                callback(0, 1, wav.data(), wav.size());
            }
        }

        auto wall_t1 = std::chrono::steady_clock::now();
        report.wall_seconds = std::chrono::duration<double>(wall_t1 - wall_t0).count();
        report.audio_seconds = static_cast<double>(report.total_samples) / sample_rate();
        report.rtf = (report.audio_seconds > 0.0) ? (report.wall_seconds / report.audio_seconds) : 0.0;
        if (out_report) *out_report = report;
        return;
    }

    // Multi-chunk Pipelined Producer-Consumer Execution:
    // Producer (main thread): runs `Qwen3TernaryEngine` for chunk `c_idx` and pushes `audio_ids` to queue.
    // Consumer (vocoder thread): pops `audio_ids` for chunk `c_idx`, runs `NativeVocoderPipeline`, and invokes `callback`.
    // While the consumer vocodes chunk `N`, the producer is already generating tokens for chunk `N + 1`!
    struct QueuedChunk {
        size_t chunk_index = 0;
        std::vector<int32_t> audio_ids;
        bool is_sentinel = false;
    };

    std::queue<QueuedChunk> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> aborted{false};
    std::exception_ptr consumer_exc = nullptr;

    double voc_seconds_acc = 0.0;
    size_t total_samples_acc = 0;

    std::thread consumer_thread([&]() {
        try {
            std::mt19937 voc_rng(static_cast<uint32_t>(opts.seed));
            while (true) {
                QueuedChunk item;
                {
                    std::unique_lock<std::mutex> lk(mtx);
                    cv.wait(lk, [&]() { return !q.empty() || aborted.load(std::memory_order_acquire); });
                    if (aborted.load(std::memory_order_acquire) && q.empty()) break;
                    item = std::move(q.front());
                    q.pop();
                }
                if (item.is_sentinel) break;
                if (aborted.load(std::memory_order_acquire) || item.audio_ids.empty()) continue;

                auto vt0 = std::chrono::steady_clock::now();
                auto wav = impl_->vocoder.decode_tokens_to_wav(
                    item.audio_ids, spk.vocoder_voice, voc_rng, &impl_->vocoder_pool
                );
                auto vt1 = std::chrono::steady_clock::now();
                voc_seconds_acc += std::chrono::duration<double>(vt1 - vt0).count();
                total_samples_acc += wav.size();

                if (callback && !wav.empty()) {
                    bool keep_going = callback(item.chunk_index, chunks.size(), wav.data(), wav.size());
                    if (!keep_going) {
                        aborted.store(true, std::memory_order_release);
                        break;
                    }
                }
            }
        } catch (...) {
            consumer_exc = std::current_exception();
            aborted.store(true, std::memory_order_release);
        }
    });

    std::mt19937 lm_rng(static_cast<uint32_t>(opts.seed));
    try {
        for (size_t c_idx = 0; c_idx < chunks.size(); ++c_idx) {
            if (aborted.load(std::memory_order_acquire)) break;

            auto target_ids = impl_->tokenizer.encode(chunks[c_idx]);
            auto prompt_ids = build_generation_prompt(
                impl_->token_map, target_ids,
                use_ref ? &ref_prefix : nullptr,
                emotion_ids.empty() ? nullptr : &emotion_ids
            );

            int32_t expected = static_cast<int32_t>(utf8_codepoint_len(chunks[c_idx]) / 20.0 * 25.0 * 1.8);
            int32_t budget = std::min<int32_t>(opts.max_new_tokens, std::max<int32_t>(200, expected));

            GenerationStats stats;
            auto audio_ids = impl_->lm_engine->generate_audio_tokens(
                prompt_ids, spk.embedding_512.data(), impl_->token_map, sampling,
                budget, lm_rng, &stats
            );

            report.lm_seconds += (stats.prefill_seconds + stats.decode_seconds);
            report.total_generated_tokens += static_cast<size_t>(stats.generated_tokens);
            report.total_audio_tokens += audio_ids.size();

            {
                std::lock_guard<std::mutex> lk(mtx);
                q.push(QueuedChunk{c_idx, std::move(audio_ids), false});
            }
            cv.notify_one();
        }
    } catch (...) {
        aborted.store(true, std::memory_order_release);
        cv.notify_all();
        if (consumer_thread.joinable()) consumer_thread.join();
        throw;
    }

    {
        std::lock_guard<std::mutex> lk(mtx);
        q.push(QueuedChunk{0, {}, true});
    }
    cv.notify_one();
    if (consumer_thread.joinable()) consumer_thread.join();

    if (consumer_exc) {
        std::rethrow_exception(consumer_exc);
    }

    auto wall_t1 = std::chrono::steady_clock::now();
    report.vocoder_seconds = voc_seconds_acc;
    report.total_samples = total_samples_acc;
    report.wall_seconds = std::chrono::duration<double>(wall_t1 - wall_t0).count();
    report.audio_seconds = static_cast<double>(report.total_samples) / sample_rate();
    report.rtf = (report.audio_seconds > 0.0) ? (report.wall_seconds / report.audio_seconds) : 0.0;
    if (out_report) *out_report = report;
}

std::vector<float> Engine::generate(
    const std::string& text,
    const GenerateOptions& opts,
    SynthesisReport* out_report
) {
    std::vector<std::vector<float>> chunk_wavs;
    SynthesisReport rep;
    generate_stream(
        text,
        [&](size_t idx, size_t total, const float* pcm, size_t n) {
            if (chunk_wavs.size() < total) chunk_wavs.resize(total);
            chunk_wavs[idx].assign(pcm, pcm + n);
            return true;
        },
        opts,
        &rep
    );
    std::vector<float> joined = join_audio_chunks(chunk_wavs, sample_rate(), opts.chunk_gap_s);
    rep.total_samples = joined.size();
    rep.audio_seconds = static_cast<double>(joined.size()) / sample_rate();
    rep.rtf = (rep.audio_seconds > 0.0) ? (rep.wall_seconds / rep.audio_seconds) : 0.0;
    if (out_report) *out_report = rep;
    return joined;
}

std::string Engine::generate_to_file(
    const std::string& text,
    const std::string& output_wav_path,
    const GenerateOptions& opts,
    SynthesisReport* out_report
) {
    std::vector<float> wav = generate(text, opts, out_report);
    write_wav_f32(output_wav_path, wav, sample_rate());
    return output_wav_path;
}

} // namespace kitteni
