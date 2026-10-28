#include "core/hub_cache.hpp"
#include "core/mmap_loader.hpp"
#include "kernels/cpu_features.hpp"
#include "kernels/ternary_gemv.hpp"
#include "frontend/bpe_tokenizer.hpp"
#include "frontend/text_normalizer.hpp"
#include "frontend/audio_joiner.hpp"
#include "lm/model_loader.hpp"
#include "lm/sampler.hpp"
#include "lm/qwen3_ternary.hpp"
#include "vocoder/vocoder_pipeline.hpp"
#include "cloning/voice_cloner.hpp"
#include "kitteni/kitteni.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace kitteni;

namespace {

struct VoiceConditioning {
    std::string transcript;
    std::vector<float> embedding_512;
    std::vector<int32_t> reference_tokens;
};

VoiceConditioning load_builtin_voice_npz(const ResolvedRepo& repo, const std::string& voice_name) {
    std::string voices_rel = repo.config.value_string("voices", "voices/voices.json");
    auto voices_json_path = ensure_repo_file(repo, voices_rel, true);
    JsonValue voices = parse_json_file(voices_json_path.string());
    if (!voices.contains(voice_name)) {
        throw std::runtime_error("Unknown built-in voice: " + voice_name);
    }
    const JsonValue& spec = voices.at(voice_name);
    VoiceConditioning vc;
    vc.transcript = spec.value_string("transcript");

    std::string npz_rel = "voices/" + spec.value_string("artifacts");
    auto npz_path = ensure_repo_file(repo, npz_rel, true);
    NpzArchive npz(npz_path.string());

    TensorView emb_tv = npz.get_array("embedding");
    vc.embedding_512.resize(emb_tv.numel());
    if (emb_tv.dtype == DType::F32) {
        std::memcpy(vc.embedding_512.data(), emb_tv.data, emb_tv.numel() * sizeof(float));
    } else if (emb_tv.dtype == DType::F16) {
        const uint16_t* p = emb_tv.ptr<uint16_t>();
        for (size_t i = 0; i < emb_tv.numel(); ++i) vc.embedding_512[i] = fp16_to_fp32(p[i]);
    } else {
        throw std::runtime_error("Unsupported voice embedding dtype in NPZ");
    }

    TensorView tok_tv = npz.get_array("reference_tokens");
    vc.reference_tokens.resize(tok_tv.numel());
    if (tok_tv.dtype == DType::I64) {
        const int64_t* p = tok_tv.ptr<int64_t>();
        for (size_t i = 0; i < tok_tv.numel(); ++i) vc.reference_tokens[i] = static_cast<int32_t>(p[i]);
    } else if (tok_tv.dtype == DType::I32) {
        const int32_t* p = tok_tv.ptr<int32_t>();
        for (size_t i = 0; i < tok_tv.numel(); ++i) vc.reference_tokens[i] = p[i];
    } else {
        throw std::runtime_error("Unsupported reference_tokens dtype in NPZ");
    }
    return vc;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string repo_arg = "KittenML/kitten-tts-2";
        std::string weights_arg = "emb4";
        std::string decoder_arg = "student_w4";
        std::string isa_arg = "auto";
        std::string text_arg;
        std::string voice_arg;
        std::string reference_wav_arg;
        std::string reference_text_arg;
        std::string save_voice_path;
        std::string load_voice_path;
        std::string preset_arg;
        std::string output_wav_path = "output.wav";
        std::string tokens_out_path;
        std::string decode_tokens_path;
        int threads = 0;
        int max_layers = 28;
        int max_new_tokens = 1000;
        uint32_t seed = 1234;
        bool normalize = true;
        bool use_reference = true;
        bool greedy = false;
        bool tokens_only = false;
        bool stream_mode = false;
        bool benchmark = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--repo" && i + 1 < argc) repo_arg = argv[++i];
            else if (arg == "--weights" && i + 1 < argc) weights_arg = argv[++i];
            else if (arg == "--decoder" && i + 1 < argc) decoder_arg = argv[++i];
            else if (arg == "--isa" && i + 1 < argc) isa_arg = argv[++i];
            else if (arg == "--text" && i + 1 < argc) text_arg = argv[++i];
            else if (arg == "--voice" && i + 1 < argc) voice_arg = argv[++i];
            else if (arg == "--reference" && i + 1 < argc) reference_wav_arg = argv[++i];
            else if (arg == "--reference-text" && i + 1 < argc) reference_text_arg = argv[++i];
            else if (arg == "--save-voice" && i + 1 < argc) save_voice_path = argv[++i];
            else if (arg == "--load-voice" && i + 1 < argc) load_voice_path = argv[++i];
            else if (arg == "--preset" && i + 1 < argc) preset_arg = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output_wav_path = argv[++i];
            else if (arg == "--save-tokens" && i + 1 < argc) tokens_out_path = argv[++i];
            else if (arg == "--decode-tokens" && i + 1 < argc) decode_tokens_path = argv[++i];
            else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if (arg == "--max-layers" && i + 1 < argc) max_layers = std::stoi(argv[++i]);
            else if (arg == "--max-tokens" && i + 1 < argc) max_new_tokens = std::stoi(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--no-normalize") normalize = false;
            else if (arg == "--no-reference") use_reference = false;
            else if (arg == "--greedy") greedy = true;
            else if (arg == "--tokens-only") tokens_only = true;
            else if (arg == "--stream") stream_mode = true;
            else if (arg == "--bench") benchmark = true;
            else if (arg == "--help" || arg == "-h") {
                std::cout << "KittenI CLI (Pure C++20 Zero-LibTorch Ternary TTS & Voice Cloning Runtime)\n"
                          << "Usage: kitteni [options]\n"
                          << "  --text TEXT              Input text to synthesize into 24 kHz WAV\n"
                          << "  --output FILE            Output WAV file path (default: output.wav)\n"
                          << "  --voice NAME             Built-in voice name (default: Bruno)\n"
                          << "  --reference WAV          Clone speaker directly from a 5-30s WAV file at runtime\n"
                          << "  --reference-text TEXT    Transcript of the reference WAV (for in-context prompt)\n"
                          << "  --save-voice FILE        Save cloned voice profile (.json) for instant reuse\n"
                          << "  --load-voice FILE        Load a previously saved cloned voice profile (.json)\n"
                          << "  --preset NAME            Decode preset: stable | expressive (default: stable)\n"
                          << "  --weights VARIANT        LM weight variant: emb4 | packed (default: emb4)\n"
                          << "  --decoder NAME           Vocoder flow: student_w4 | student_w8 (default: student_w4)\n"
                          << "  --isa ISA                SIMD ISA: auto | scalar | avx2 | avx_vnni | avx512_vnni\n"
                          << "  --max-tokens N           Max generated audio tokens per chunk (default: 1000)\n"
                          << "  --threads N              CPU thread count (default: auto)\n"
                          << "  --seed N                 RNG seed (default: 1234)\n"
                          << "  --no-normalize           Disable English text normalization\n"
                          << "  --no-reference           Skip in-context audio reference prefix (faster prefill)\n"
                          << "  --greedy                 Use greedy argmax decoding\n"
                          << "  --stream                 Use pipelined producer-consumer chunk streaming\n"
                          << "  --tokens-only            Skip waveform decoding (only run Speech LM)\n"
                          << "  --save-tokens FILE       Write generated 0-based S3 codec tokens as JSON\n"
                          << "  --decode-tokens FILE     Decode an existing JSON array of 0-based S3 codec tokens to WAV\n"
                          << "  --bench                  Run multi-ISA kernel benchmark (scalar vs avx2 vs avx_vnni vs avx512_vnni)\n";
                return 0;
            }
        }

        set_active_isa(parse_isa_name(isa_arg));
        std::cout << "[KittenI] CPU: " << cpu_capabilities_summary() << "\n";
        bool need_lm = decode_tokens_path.empty() && (!text_arg.empty() || benchmark || (!reference_wav_arg.empty() && !save_voice_path.empty()));
        ResolvedRepo repo = resolve_repository(repo_arg, need_lm ? weights_arg : "", "", true);
        std::cout << "[KittenI] Resolved repository at: " << repo.root_dir.string() << "\n";
        std::string voice = voice_arg.empty() ? repo.config.value_string("default_voice", "Bruno") : voice_arg;

        // Mode 1: Direct `--decode-tokens FILE` (vocoder-only synthesis from saved codec tokens)
        if (!decode_tokens_path.empty()) {
            JsonValue tok_json = parse_json_file(decode_tokens_path);
            std::vector<int32_t> audio_ids;
            for (const auto& item : tok_json.as_array()) {
                audio_ids.push_back(static_cast<int32_t>(item.as_int()));
            }
            ThreadPool pool(threads);
            NativeVocoderPipeline vocoder;
            auto t0 = std::chrono::steady_clock::now();
            vocoder.load(repo, decoder_arg);
            PreparedVoice pv;
            if (!load_voice_path.empty()) {
                pv = VoiceCloner::load_profile_json(load_voice_path).vocoder_voice;
            } else if (!reference_wav_arg.empty()) {
                VoiceCloner cloner;
                cloner.load(repo, /*ensure_s3_cloner=*/true, /*allow_download=*/true);
                auto prof = cloner.clone_from_wav(reference_wav_arg, reference_text_arg, nullptr, true, &pool);
                if (!save_voice_path.empty()) VoiceCloner::save_profile_json(save_voice_path, prof);
                pv = std::move(prof.vocoder_voice);
            } else {
                pv = vocoder.load_voice(voice);
            }
            auto t1 = std::chrono::steady_clock::now();

            std::mt19937 rng(seed);
            auto wav = vocoder.decode_tokens_to_wav(audio_ids, pv, rng, &pool);
            auto t2 = std::chrono::steady_clock::now();

            write_wav_f32(output_wav_path, wav, 24000);
            double load_s = std::chrono::duration<double>(t1 - t0).count();
            double dec_s  = std::chrono::duration<double>(t2 - t1).count();
            double dur_s  = wav.size() / 24000.0;
            std::cout << "[KittenI] Vocoder (" << decoder_arg << ") loaded in " << std::fixed << std::setprecision(3) << load_s
                      << "s, decoded " << audio_ids.size() << " tokens -> " << std::setprecision(2) << dur_s
                      << "s audio in " << std::setprecision(3) << dec_s << "s (vocoder RTF=" << (dec_s / dur_s) << ")\n";
            std::cout << "[KittenI] Saved 24 kHz WAV to: " << output_wav_path << "\n";
            return 0;
        }

        // Mode 2: Load Ternary Qwen3 LM
        auto t0 = std::chrono::steady_clock::now();
        KittenTernaryLMWeights lm;
        lm.load(repo, weights_arg, max_layers, threads);
        auto t1 = std::chrono::steady_clock::now();
        double load_s = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "[KittenI] Loaded " << lm.layers.size() << "/" << lm.config.num_hidden_layers
                  << " ternary Qwen3 layers (" << weights_arg << ") in "
                  << std::fixed << std::setprecision(3) << load_s << " s ("
                  << (lm.ternary_body_bytes() / (1024.0 * 1024.0)) << " MiB packed ternary RAM)\n";

        if (benchmark && !lm.layers.empty()) {
            ThreadPool pool(threads);
            std::vector<float> x(static_cast<size_t>(lm.config.hidden_size), 0.1f);
            lm.embed_token(151673, x.data());
            QuantizedActivation xq;
            std::vector<float> y_gate_up(static_cast<size_t>(lm.config.intermediate_size * 2));
            const int iters = 200;

            SimdIsa saved_isa = get_active_isa();
            const SimdIsa test_isas[] = {SimdIsa::SCALAR, SimdIsa::AVX2, SimdIsa::AVX_VNNI, SimdIsa::AVX512_VNNI};
            std::cout << "[KittenI] Benchmarking 12288x2048 Fused Gate+Up Ternary GEMV (" << pool.num_threads() << " threads):\n";
            for (SimdIsa cand : test_isas) {
                if (!is_isa_supported(cand)) continue;
                set_active_isa(cand);
                xq.quantize_from(x.data(), 1, lm.config.hidden_size);
                ternary_matmul_quantized(lm.layers[0].gate_up_proj, xq, y_gate_up.data(), &pool);

                auto b0 = std::chrono::steady_clock::now();
                for (int it = 0; it < iters; ++it) {
                    xq.quantize_from(x.data(), 1, lm.config.hidden_size);
                    ternary_matmul_quantized(lm.layers[0].gate_up_proj, xq, y_gate_up.data(), &pool);
                }
                auto b1 = std::chrono::steady_clock::now();
                double us_per_gemv = std::chrono::duration<double, std::micro>(b1 - b0).count() / iters;
                double gops = (2.0 * lm.config.hidden_size * (lm.config.intermediate_size * 2)) / (us_per_gemv * 1e3);
                std::cout << "  ISA=" << std::left << std::setw(12) << isa_name(cand)
                          << ": " << std::fixed << std::setprecision(1) << std::setw(8) << us_per_gemv << " us/call ("
                          << std::setprecision(2) << gops << " effective GOPS)\n";
            }

            const int64_t bl_out = 1472, bl_in = 368, bl_rows = 256;
            std::vector<float> bl_w(bl_out * bl_in, 0.05f), bl_b(bl_out, 0.0f), bl_x(bl_rows * bl_in, 0.2f), bl_y(bl_rows * bl_out);
            BitNetWeight bw;
            bw.init_from_dequantized_f32(bl_w.data(), bl_b.data(), bl_out, bl_in, 1, 4);
            std::cout << "[KittenI] Benchmarking 256x1472x368 L2-Resident BitLinear GEMM (" << pool.num_threads() << " threads):\n";
            for (SimdIsa cand : test_isas) {
                if (!is_isa_supported(cand)) continue;
                set_active_isa(cand);
                bitlinear_forward(bw, bl_x.data(), bl_rows, bl_y.data(), &pool);
                auto b0 = std::chrono::steady_clock::now();
                for (int it = 0; it < 50; ++it) {
                    bitlinear_forward(bw, bl_x.data(), bl_rows, bl_y.data(), &pool);
                }
                auto b1 = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(b1 - b0).count() / 50.0;
                double gops = (2.0 * bl_rows * bl_out * bl_in) / (us * 1e3);
                std::cout << "  ISA=" << std::left << std::setw(12) << isa_name(cand)
                          << ": " << std::fixed << std::setprecision(1) << std::setw(8) << us << " us/call ("
                          << std::setprecision(2) << gops << " effective GOPS)\n";
            }
            set_active_isa(saved_isa);
        }

        // Handle standalone `--reference WAV --save-voice FILE` even if `--text` is omitted
        if (!reference_wav_arg.empty() && text_arg.empty() && !save_voice_path.empty()) {
            ThreadPool pool(threads);
            VoiceCloner cloner;
            cloner.load(repo, /*ensure_s3_cloner=*/true, /*allow_download=*/true);
            auto prof = cloner.clone_from_wav(reference_wav_arg, reference_text_arg, &lm, true, &pool);
            VoiceCloner::save_profile_json(save_voice_path, prof);
            std::cout << "[KittenI] Cloned voice from '" << reference_wav_arg << "' and saved profile to: "
                      << save_voice_path << "\n";
            return 0;
        }

        if (!text_arg.empty() && stream_mode && !tokens_only) {
            EngineOptions eo;
            eo.model_or_path = repo_arg;
            eo.weights = weights_arg;
            eo.decoder = decoder_arg;
            eo.isa = isa_arg;
            eo.threads = threads;
            eo.max_layers = max_layers;

            Engine eng(eo);
            GenerateOptions go;
            go.voice = voice;
            go.reference_wav = reference_wav_arg;
            go.reference_text = reference_text_arg;
            go.preset = preset_arg;
            go.max_new_tokens = max_new_tokens;
            go.normalize = normalize;
            go.use_reference_prompt = use_reference ? 1 : 0;
            go.seed = static_cast<int>(seed);
            go.greedy = greedy;

            std::vector<std::vector<float>> streamed_chunks;
            auto st0 = std::chrono::steady_clock::now();
            SynthesisReport rep;
            eng.generate_stream(
                text_arg,
                [&](size_t idx, size_t total, const float* pcm, size_t n) -> bool {
                    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - st0).count();
                    std::cout << "[KittenI Stream] Yielded chunk " << (idx + 1) << "/" << total
                              << " (" << std::fixed << std::setprecision(2) << (n / 24000.0)
                              << "s audio) at t=" << std::setprecision(3) << elapsed << "s\n";
                    if (streamed_chunks.size() < total) streamed_chunks.resize(total);
                    streamed_chunks[idx].assign(pcm, pcm + n);
                    return true;
                },
                go,
                &rep
            );
            auto joined = join_audio_chunks(streamed_chunks, 24000, go.chunk_gap_s);
            write_wav_f32(output_wav_path, joined, 24000);
            std::cout << "[KittenI Stream] Complete: " << std::fixed << std::setprecision(2) << rep.audio_seconds
                      << "s audio in " << rep.wall_seconds << "s wall clock (LM=" << rep.lm_seconds
                      << "s, Vocoder=" << rep.vocoder_seconds << "s, RTF=" << rep.rtf << ") -> " << output_wav_path << "\n";
            return 0;
        }

        if (!text_arg.empty()) {
            std::string lm_dir = repo.config.value_string("lm_dir", "lm");
            ensure_repo_file(repo, lm_dir + "/vocab.json", true);
            ensure_repo_file(repo, lm_dir + "/merges.txt", true);
            ensure_repo_file(repo, lm_dir + "/added_tokens.json", true);

            Qwen3BPETokenizer tokenizer;
            tokenizer.load((repo.root_dir / lm_dir).string());

            TokenMap tm = TokenMap::from_config(repo.config);
            std::string preset = preset_arg.empty() ? repo.config.value_string("default_preset", "stable") : preset_arg;

            Qwen3TernaryEngine engine(lm, threads, /*use_int8_act=*/true);
            NativeVocoderPipeline vocoder;
            if (!tokens_only) {
                vocoder.load(repo, decoder_arg);
            }

            VoiceConditioning vc;
            PreparedVoice pv;

            if (!load_voice_path.empty()) {
                ClonedVoiceProfile prof = VoiceCloner::load_profile_json(load_voice_path);
                vc.transcript = prof.vocoder_voice.transcript;
                vc.embedding_512 = prof.embedding_512;
                vc.reference_tokens = prof.vocoder_voice.reference_tokens;
                pv = std::move(prof.vocoder_voice);
                std::cout << "[KittenI] Loaded cloned voice profile from: " << load_voice_path << "\n";
            } else if (!reference_wav_arg.empty()) {
                auto ct0 = std::chrono::steady_clock::now();
                VoiceCloner cloner;
                bool need_s3 = (!tokens_only || use_reference || !save_voice_path.empty());
                cloner.load(repo, /*ensure_s3_cloner=*/need_s3, /*allow_download=*/true);
                ClonedVoiceProfile prof = cloner.clone_from_wav(
                    reference_wav_arg, reference_text_arg, &lm, /*for_student_flow=*/true, &engine.thread_pool()
                );
                auto ct1 = std::chrono::steady_clock::now();
                std::cout << "[KittenI] Cloned reference WAV '" << reference_wav_arg << "' in "
                          << std::fixed << std::setprecision(3) << std::chrono::duration<double>(ct1 - ct0).count()
                          << "s (reference_tokens=" << prof.vocoder_voice.reference_tokens.size()
                          << ", prompt_tokens=" << prof.vocoder_voice.prompt_token.size() << ")\n";
                if (!save_voice_path.empty()) {
                    VoiceCloner::save_profile_json(save_voice_path, prof);
                    std::cout << "[KittenI] Saved cloned voice profile to: " << save_voice_path << "\n";
                }
                vc.transcript = prof.vocoder_voice.transcript;
                vc.embedding_512 = prof.embedding_512;
                vc.reference_tokens = prof.vocoder_voice.reference_tokens;
                pv = std::move(prof.vocoder_voice);
                if (vc.transcript.empty()) {
                    use_reference = false;
                }
            } else {
                vc = load_builtin_voice_npz(repo, voice);
                if (!tokens_only) {
                    pv = vocoder.load_voice(voice);
                }
            }

            SamplingSettings sampling;
            if (repo.config.contains("decode_presets") && repo.config.at("decode_presets").contains(preset)) {
                const JsonValue& p = repo.config.at("decode_presets").at(preset);
                sampling.temperature = static_cast<float>(p.value_float("temperature", 0.8));
                sampling.top_p       = static_cast<float>(p.value_float("top_p", 0.8));
                sampling.top_k       = static_cast<int32_t>(p.value_int("top_k", 50));
                sampling.min_p       = static_cast<float>(p.value_float("min_p", 0.0));
            }
            if (repo.config.contains("generation")) {
                sampling.repetition_window = static_cast<int32_t>(
                    repo.config.at("generation").value_int("repetition_window", 50));
            }
            sampling.greedy = greedy;

            std::string spoken = normalize_text(text_arg, normalize);
            auto chunks = split_for_synthesis(spoken);

            std::cout << "[KittenI] Normalized (" << chunks.size() << " chunk(s)): \"" << spoken << "\"\n";

            std::vector<int32_t> ref_text_ids = vc.transcript.empty() ? std::vector<int32_t>{} : tokenizer.encode(vc.transcript);
            std::vector<int32_t> ref_prefix = (use_reference && !vc.reference_tokens.empty())
                                                  ? build_reference_prefix(tm, ref_text_ids, vc.reference_tokens)
                                                  : std::vector<int32_t>{};

            std::vector<int32_t> emotion_ids;
            if (has_expression_tag(text_arg) && repo.config.contains("generation")) {
                std::string emo_ctrl = repo.config.at("generation").value_string("emotion_control");
                if (!emo_ctrl.empty()) {
                    emotion_ids = tokenizer.encode(emo_ctrl);
                }
            }

            std::mt19937 rng(seed);
            std::vector<int32_t> all_audio_tokens;
            std::vector<std::vector<float>> chunk_wavs;
            double total_lm_s = 0.0;
            double total_voc_s = 0.0;

            for (size_t c_idx = 0; c_idx < chunks.size(); ++c_idx) {
                std::vector<int32_t> target_ids = tokenizer.encode(chunks[c_idx]);
                std::vector<int32_t> prompt_ids = build_generation_prompt(
                    tm,
                    target_ids,
                    (use_reference && !ref_prefix.empty()) ? &ref_prefix : nullptr,
                    emotion_ids.empty() ? nullptr : &emotion_ids
                );

                int32_t budget = max_new_tokens;
                if (chunks.size() > 1) {
                    int32_t expected = static_cast<int32_t>(utf8_codepoint_len(chunks[c_idx]) / 20.0 * 25.0 * 1.8);
                    budget = std::max<int32_t>(200, std::min<int32_t>(max_new_tokens, expected));
                }

                GenerationStats stats;
                auto audio_ids = engine.generate_audio_tokens(
                    prompt_ids,
                    vc.embedding_512.data(),
                    tm,
                    sampling,
                    budget,
                    rng,
                    &stats
                );
                total_lm_s += (stats.prefill_seconds + stats.decode_seconds);

                double voc_s = 0.0;
                if (!tokens_only && !audio_ids.empty()) {
                    auto vt0 = std::chrono::steady_clock::now();
                    auto wav = vocoder.decode_tokens_to_wav(audio_ids, pv, rng, &engine.thread_pool());
                    auto vt1 = std::chrono::steady_clock::now();
                    voc_s = std::chrono::duration<double>(vt1 - vt0).count();
                    total_voc_s += voc_s;
                    chunk_wavs.push_back(std::move(wav));
                }

                double decode_tok_s = stats.decode_seconds > 0.0 ? ((stats.generated_tokens - 1) / stats.decode_seconds) : 0.0;
                double audio_sec = audio_ids.size() / 25.0;
                std::cout << "[KittenI] Chunk " << (c_idx + 1) << "/" << chunks.size()
                          << ": prompt=" << stats.prompt_tokens
                          << " generated=" << stats.generated_tokens
                          << " audio_tokens=" << audio_ids.size() << " (" << std::setprecision(2) << audio_sec << "s)"
                          << " prefill=" << std::setprecision(3) << stats.prefill_seconds << "s"
                          << " decode=" << stats.decode_seconds << "s (" << std::setprecision(1) << decode_tok_s << " tok/s)";
                if (!tokens_only) {
                    std::cout << " vocoder=" << std::setprecision(3) << voc_s << "s";
                }
                std::cout << "\n";

                all_audio_tokens.insert(all_audio_tokens.end(), audio_ids.begin(), audio_ids.end());
            }

            if (!tokens_out_path.empty()) {
                std::ofstream out(tokens_out_path);
                out << "[";
                for (size_t i = 0; i < all_audio_tokens.size(); ++i) {
                    if (i > 0) out << ", ";
                    out << all_audio_tokens[i];
                }
                out << "]\n";
                std::cout << "[KittenI] Wrote " << all_audio_tokens.size() << " codec tokens to " << tokens_out_path << "\n";
            }

            if (!tokens_only && !chunk_wavs.empty()) {
                auto final_audio = join_audio_chunks(chunk_wavs, 24000, 0.16f);
                write_wav_f32(output_wav_path, final_audio, 24000);
                double dur_s = final_audio.size() / 24000.0;
                std::cout << "[KittenI] Wrote " << std::fixed << std::setprecision(2) << dur_s
                          << "s 24 kHz audio to " << output_wav_path
                          << " (LM=" << std::setprecision(2) << total_lm_s << "s, Vocoder=" << total_voc_s << "s)\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[KittenI Error] " << e.what() << "\n";
        return 1;
    }
}
