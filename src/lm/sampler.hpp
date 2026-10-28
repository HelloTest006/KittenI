#pragma once

#include "core/json.hpp"
#include <cstdint>
#include <vector>
#include <random>

namespace kitteni {

struct TokenMap {
    int32_t audio_id_base = 151675;
    int32_t num_audio_tokens = 6561;
    int32_t speech_start_id = 151669;
    int32_t speech_end_id = 151670;
    int32_t text_start_id = 151671;
    int32_t text_end_id = 151672;
    int32_t start_id = 151673;
    int32_t stop_id = 151674;
    int32_t final_seg_id = 158236;
    int32_t reference_speech_start_id = 158237;
    int32_t reference_speech_end_id = 158238;
    int32_t reference_text_start_id = 158239;
    int32_t reference_text_end_id = 158240;

    int32_t audio_id_end() const { return audio_id_base + num_audio_tokens; }
    bool uses_reference_role_tokens() const { return reference_text_start_id != 0; }

    static TokenMap from_config(const JsonValue& root_config);
};

// Assemble in-context reference prefix (`prompt.build_reference_prefix`)
std::vector<int32_t> build_reference_prefix(
    const TokenMap& tm,
    const std::vector<int32_t>& reference_text_ids,
    const std::vector<int32_t>& reference_audio_tokens // 0-based S3 codec tokens
);

// Assemble full generation prompt for one chunk (`prompt.build_generation_prompt`)
std::vector<int32_t> build_generation_prompt(
    const TokenMap& tm,
    const std::vector<int32_t>& target_text_ids,
    const std::vector<int32_t>* reference_prefix = nullptr,
    const std::vector<int32_t>* emotion_ids = nullptr
);

struct SamplingSettings {
    float temperature = 0.8f;
    int32_t top_k = 50;
    float top_p = 0.8f;
    float min_p = 0.0f;
    float repetition_penalty = 1.1f;
    int32_t repetition_window = 50;
    float token_run_penalty = 1.3f;
    int32_t token_run_grace = 10;
    bool greedy = false;
};

// Apply KittenTTS 2 decode-time logits processors in exact Python order:
// 1. TokenRunPenalty (escalating penalty on trailing run >= grace of any audio token)
// 2. RepetitionPenaltyExempt (penalize tokens in window, exempting speech_end_id and stop_id;
//    note: silence token IS penalized as in Python logits.py!)
// 3. TemperatureLogitsWarper
// 4. TopKLogitsWarper
// 5. TopPLogitsWarper
// 6. MinPLogitsWarper
void apply_logits_processors(
    std::vector<float>& scores,
    const std::vector<int32_t>& history,
    size_t prompt_length,
    const TokenMap& tm,
    const SamplingSettings& settings
);

// Sample next token ID from processed logits
int32_t sample_token(
    const std::vector<float>& scores,
    std::mt19937& rng,
    bool greedy = false
);

} // namespace kitteni
