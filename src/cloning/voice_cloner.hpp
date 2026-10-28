#pragma once

#include "core/hub_cache.hpp"
#include "core/thread_pool.hpp"
#include "lm/model_loader.hpp"
#include "vocoder/vocoder_pipeline.hpp"
#include "cloning/sincnet_xvector.hpp"
#include "cloning/campplus.hpp"
#include "cloning/s3_tokenizer.hpp"
#include <string>
#include <vector>
#include <memory>

namespace kitteni {

struct ClonedVoiceProfile {
    PreparedVoice vocoder_voice;
    std::vector<float> embedding_512; // 512-d unit-norm SincNet speaker embedding for LM `spk_proj`
};

// Runtime zero-shot WAV voice cloner:
// 1. Always loads `speaker/model.safetensors` (`SincNetSpeakerEncoder` -> 512-d embedding) from `KittenML/kitten-tts-2`.
// 2. If `s3gen_meanflow.safetensors` (or a slice containing `speaker_encoder.*` and `tokenizer.*`) is available,
//    loads `CampPlusEncoder` (192-d x-vector) and `S3AudioTokenizer` (25 Hz prompt tokens).
//    Also supports downloading ONLY the `speaker_encoder.*` + `tokenizer.*` byte range of `s3gen_meanflow.safetensors`
//    via HTTP Range request (`speaker/s3_cloner.safetensors`) so users don't have to download the unused 500 MB dense flow!
class VoiceCloner {
public:
    VoiceCloner() = default;

    // Initialize cloner models from `repo`.
    // When `ensure_s3_cloner` is true, fetches/loads `CampPlusEncoder` and `S3AudioTokenizer`.
    void load(const ResolvedRepo& repo, bool ensure_s3_cloner = true, bool allow_download = true);

    bool has_s3_cloner() const { return has_s3_cloner_; }
    const SincNetSpeakerEncoder& sincnet() const { return sincnet_; }
    const CampPlusEncoder& campplus() const { return campplus_; }
    const S3AudioTokenizer& s3_tokenizer() const { return s3_tokenizer_; }

    // Clone a voice at runtime from any 5-30s WAV file:
    // - Computes 512-d `SincNetSpeakerEncoder` embedding (and projects via `lm` if provided)
    // - Computes `reference_tokens` (full clip at 25 Hz), `prompt_token` (4.0s truncated for student flow),
    //   `prompt_feat` (24 kHz Mel80), and 192-d `CampPlusEncoder` x-vector.
    ClonedVoiceProfile clone_from_wav(
        const std::string& wav_path,
        const std::string& reference_text,
        const KittenTernaryLMWeights* lm = nullptr,
        bool for_student_flow = true,
        const ThreadPool* pool = nullptr
    ) const;

    // Save/load a `ClonedVoiceProfile` to/from a JSON file for instant reuse
    static void save_profile_json(const std::string& json_path, const ClonedVoiceProfile& profile);
    static ClonedVoiceProfile load_profile_json(const std::string& json_path);

private:
    SincNetSpeakerEncoder sincnet_;
    CampPlusEncoder campplus_;
    S3AudioTokenizer s3_tokenizer_;
    bool has_s3_cloner_ = false;
};

} // namespace kitteni
