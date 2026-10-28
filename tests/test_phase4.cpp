#include "core/hub_cache.hpp"
#include "core/mmap_loader.hpp"
#include "kernels/dsp_fft.hpp"
#include "cloning/sincnet_xvector.hpp"
#include "cloning/campplus.hpp"
#include "cloning/s3_tokenizer.hpp"
#include "cloning/voice_cloner.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>

using namespace kitteni;

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

static double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static void test_wav_reader_and_sincnet_xvector_parity(const ResolvedRepo& repo) {
    std::cout << "[Test 1] Verifying WAV Reader, Resampler, and XVectorSincNet 512-d speaker embedding parity...\n";
    auto spk_path = repo.root_dir / "speaker" / "model.safetensors";
    SincNetSpeakerEncoder sincnet;
    sincnet.load(spk_path.string());

    auto wav_path = repo.root_dir / "voices" / "kitten_v08_voice_3_male.wav";
    auto npz_path = repo.root_dir / "voices" / "kitten_v08_voice_3_male.npz";

    AudioBuffer wav = read_wav_file(wav_path.string());
    std::cout << "  Loaded Bruno reference WAV: " << wav.samples.size()
              << " samples at " << wav.sample_rate << " Hz ("
              << (static_cast<double>(wav.samples.size()) / wav.sample_rate) << "s)\n";
    REQUIRE(wav.sample_rate > 0);
    REQUIRE(wav.samples.size() > 24000);

    std::vector<float> wav_16k = resample_audio(
        wav.samples.data(), static_cast<int64_t>(wav.samples.size()), wav.sample_rate, 16000
    );

    ThreadPool pool(4);
    std::vector<float> emb_512 = sincnet.encode_16k(wav_16k.data(), static_cast<int64_t>(wav_16k.size()), &pool);
    REQUIRE(emb_512.size() == 512);

    // Compare against ground-truth precomputed embedding in `kitten_v08_voice_3_male.npz`
    NpzArchive npz(npz_path.string());
    TensorView ref_tv = npz.get_array("embedding");
    std::vector<float> ref_512(512);
    if (ref_tv.dtype == DType::F32) {
        std::memcpy(ref_512.data(), ref_tv.data, 512 * sizeof(float));
    } else if (ref_tv.dtype == DType::F16) {
        const uint16_t* p = ref_tv.ptr<uint16_t>();
        for (int i = 0; i < 512; ++i) ref_512[i] = fp16_to_fp32(p[i]);
    }

    double cos_sim = cosine_similarity(emb_512, ref_512);
    std::cout << "  XVectorSincNet 512-d cosine similarity vs precomputed Bruno NPZ: " << cos_sim << "\n";
    REQUIRE(cos_sim > 0.99);
}

static void test_mel80_24k_parity(const ResolvedRepo& repo) {
    std::cout << "[Test 2] Verifying 24 kHz 80-bin Mel Spectrogram (`extract_mel80_24k`) parity vs Bruno `prompt_feat`...\n";
    auto wav_path = repo.root_dir / "voices" / "kitten_v08_voice_3_male.wav";
    AudioBuffer wav = read_wav_file(wav_path.string());
    std::vector<float> wav_24k = resample_audio(
        wav.samples.data(), static_cast<int64_t>(wav.samples.size()), wav.sample_rate, 24000
    );
    // Slice to 4.0s (96000 samples) as student_w4 does (`STUDENT_REF_SECONDS = 4.0`)
    if (wav_24k.size() > 96000) wav_24k.resize(96000);

    int64_t mel_frames = 0;
    std::vector<float> mel = extract_mel80_24k(wav_24k.data(), static_cast<int64_t>(wav_24k.size()), mel_frames);
    REQUIRE(mel_frames == 200);

    // Compare against Bruno's `prompt_feat` in `cpp/student_w4/voices.json`
    NativeVocoderPipeline vocoder;
    vocoder.load(repo, "student_w4");
    PreparedVoice bruno = vocoder.load_voice("Bruno");
    REQUIRE(bruno.prompt_feat_frames == 200);
    REQUIRE(bruno.prompt_feat.size() == mel.size());

    float max_diff = 0.0f;
    double sum_abs = 0.0;
    for (size_t i = 0; i < mel.size(); ++i) {
        float d = std::fabs(mel[i] - bruno.prompt_feat[i]);
        max_diff = std::max(max_diff, d);
        sum_abs += d;
    }
    double mean_abs = sum_abs / static_cast<double>(mel.size());
    std::cout << "  24 kHz Mel80 [200 x 80] mean abs error vs Bruno prompt_feat: "
              << mean_abs << " (max diff: " << max_diff << ")\n";
    REQUIRE(mean_abs < 1e-4);
}

static void test_kaldi_fbank80() {
    std::cout << "[Test 3] Testing 16 kHz Kaldi 80-bin fbank + CMN feature extractor...\n";
    std::vector<float> wav_16k(16000 * 2, 0.0f);
    for (size_t i = 0; i < wav_16k.size(); ++i) {
        wav_16k[i] = 0.2f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 16000.0f);
    }
    int64_t T_frames = 0;
    std::vector<float> fbank = extract_kaldi_fbank80_16k(wav_16k.data(), static_cast<int64_t>(wav_16k.size()), T_frames);
    REQUIRE(T_frames == (32000 - 400) / 160 + 1);
    REQUIRE(static_cast<int64_t>(fbank.size()) == T_frames * 80);

    // Verify zero mean along each column (CMN)
    for (int m = 0; m < 80; ++m) {
        double col_sum = 0.0;
        for (int64_t t = 0; t < T_frames; ++t) col_sum += fbank[t * 80 + m];
        REQUIRE(std::fabs(col_sum / T_frames) < 1e-4);
    }
    std::cout << "  Kaldi fbank80 extracted " << T_frames << " frames with verified zero-mean CMN.\n";
}

static void test_full_runtime_wav_cloner_parity(const ResolvedRepo& repo) {
    std::cout << "[Test 4] Verifying full runtime WAV VoiceCloner (XVectorSincNet + CAMPPlus + S3AudioTokenizer)...\n";
    ThreadPool pool(4);
    VoiceCloner cloner;
    cloner.load(repo, /*ensure_s3_cloner=*/true, /*allow_download=*/false);
    REQUIRE(cloner.has_s3_cloner());

    auto wav_path = repo.root_dir / "voices" / "kitten_v08_voice_3_male.wav";
    std::string transcript = "When the Western music overwhelms the Chinese music, it evokes the long history of conflict between Asia and the Western nations that sought to exploit and dominate it.";

    ClonedVoiceProfile prof = cloner.clone_from_wav(wav_path.string(), transcript, nullptr, true, &pool);

    // Compare against ground-truth Bruno in `cpp/student_w4/voices.json`
    NativeVocoderPipeline vocoder;
    vocoder.load(repo, "student_w4");
    PreparedVoice bruno = vocoder.load_voice("Bruno");

    // 1. Compare 192-d CAMPPlus x-vector (`flow_embedding_192`)
    REQUIRE(prof.vocoder_voice.flow_embedding_192.size() == 192);
    double camp_cos = cosine_similarity(prof.vocoder_voice.flow_embedding_192, bruno.flow_embedding_192);
    std::cout << "  CAMPPlus 192-d x-vector cosine similarity vs Bruno voices.json: " << camp_cos << "\n";
    REQUIRE(camp_cos > 0.99);

    // 2. Compare S3AudioTokenizer 25 Hz tokens (`prompt_token` and `reference_tokens`)
    REQUIRE(prof.vocoder_voice.prompt_token.size() == bruno.prompt_token.size());
    int exact_prompt_matches = 0;
    for (size_t i = 0; i < bruno.prompt_token.size(); ++i) {
        if (prof.vocoder_voice.prompt_token[i] == bruno.prompt_token[i]) exact_prompt_matches++;
    }
    double prompt_match_rate = static_cast<double>(exact_prompt_matches) / static_cast<double>(bruno.prompt_token.size());
    std::cout << "  S3AudioTokenizer prompt_token exact match rate: " << exact_prompt_matches << "/"
              << bruno.prompt_token.size() << " (" << (prompt_match_rate * 100.0) << "%)\n";
    REQUIRE(prompt_match_rate >= 0.90);

    // 3. Test round-trip save & load of cloned voice profile + vocoder synthesis with cloned voice
    VoiceCloner::save_profile_json("build/test_cloned_profile.json", prof);
    ClonedVoiceProfile loaded_prof = VoiceCloner::load_profile_json("build/test_cloned_profile.json");
    REQUIRE(loaded_prof.vocoder_voice.prompt_token == prof.vocoder_voice.prompt_token);

    std::mt19937 rng(42);
    std::vector<int32_t> test_tokens = {120, 450, 890, 1024, 2048, 3100};
    auto wav = vocoder.decode_tokens_to_wav(test_tokens, loaded_prof.vocoder_voice, rng, &pool);
    REQUIRE(wav.size() == (test_tokens.size() + 3) * 2 * 480);
    std::cout << "  Synthesized " << wav.size() << " samples from runtime-cloned voice profile.\n";
}

int main() {
    std::cout << "=== KittenI Phase 4 Voice Cloning Unit & Parity Tests ===\n";
    ResolvedRepo repo = resolve_repository("KittenML/kitten-tts-2", "", "", false);
    test_wav_reader_and_sincnet_xvector_parity(repo);
    test_mel80_24k_parity(repo);
    test_kaldi_fbank80();
    test_full_runtime_wav_cloner_parity(repo);
    std::cout << "=== All Phase 4 Unit Tests PASSED ===\n";
    return 0;
}
