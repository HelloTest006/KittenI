#pragma once

#include "core/hub_cache.hpp"
#include "core/thread_pool.hpp"
#include "vocoder/s3_flow.hpp"
#include "vocoder/hift_gan.hpp"
#include <vector>
#include <string>
#include <random>
#include <cstdint>

namespace kitteni {

struct PreparedVoice {
    std::string transcript;
    std::vector<int32_t> reference_tokens; // For LM in-context reference prefix
    std::vector<float> lm_speaker_2048;    // Pre-projected `spk_proj(embedding_512)` [2048]
    std::vector<int32_t> prompt_token;     // S3 flow reference prompt tokens [T_prompt]
    std::vector<float> prompt_feat;        // S3 flow reference mel [T_prompt * 2, 80]
    int64_t prompt_feat_frames = 0;
    std::vector<float> flow_embedding_192; // S3 flow CAMPPlus 192-d x-vector
};

class NativeVocoderPipeline {
public:
    NativeVocoderPipeline() = default;

    // Load student_w4 or student_w8 flow + HiFTGenerator vocoder from `repo`
    void load(const ResolvedRepo& repo, const std::string& decoder_name = "student_w4");

    // Load a built-in voice from `cpp/<decoder>/voices.json`
    PreparedVoice load_voice(const std::string& voice_name) const;

    // Decode 0-based S3 speech tokens `[0..6560]` into a 24 kHz mono float32 waveform.
    // Automatically appends 3 trailing silence tokens (`4299`) as required by s3gen lookahead.
    std::vector<float> decode_tokens_to_wav(
        const std::vector<int32_t>& audio_ids,
        const PreparedVoice& voice,
        std::mt19937& rng,
        const ThreadPool* pool = nullptr,
        const float* fixed_flow_noise = nullptr,
        bool deterministic_hift_source = false
    ) const;

    const S3StudentFlow& flow() const { return flow_; }
    const HiFTGenerator& hift() const { return hift_; }

private:
    std::string decoder_name_;
    std::string voices_json_path_;
    mutable JsonValue voices_cache_;
    S3StudentFlow flow_;
    HiFTGenerator hift_;
};

} // namespace kitteni
