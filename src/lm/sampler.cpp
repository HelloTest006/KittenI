#include "lm/sampler.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kitteni {

TokenMap TokenMap::from_config(const JsonValue& root_config) {
    TokenMap tm;
    if (!root_config.contains("token_map")) return tm;
    const JsonValue& j = root_config.at("token_map");
    tm.audio_id_base             = static_cast<int32_t>(j.value_int("audio_id_base", 151675));
    tm.num_audio_tokens          = static_cast<int32_t>(j.value_int("num_audio_tokens", 6561));
    tm.speech_start_id           = static_cast<int32_t>(j.value_int("speech_start_id", 151669));
    tm.speech_end_id             = static_cast<int32_t>(j.value_int("speech_end_id", 151670));
    tm.text_start_id             = static_cast<int32_t>(j.value_int("text_start_id", 151671));
    tm.text_end_id               = static_cast<int32_t>(j.value_int("text_end_id", 151672));
    tm.start_id                  = static_cast<int32_t>(j.value_int("start_id", 151673));
    tm.stop_id                   = static_cast<int32_t>(j.value_int("stop_id", 151674));
    tm.final_seg_id              = static_cast<int32_t>(j.value_int("final_seg_id", 0));
    tm.reference_speech_start_id = static_cast<int32_t>(j.value_int("reference_speech_start_id", 0));
    tm.reference_speech_end_id   = static_cast<int32_t>(j.value_int("reference_speech_end_id", 0));
    tm.reference_text_start_id   = static_cast<int32_t>(j.value_int("reference_text_start_id", 0));
    tm.reference_text_end_id     = static_cast<int32_t>(j.value_int("reference_text_end_id", 0));
    return tm;
}

std::vector<int32_t> build_reference_prefix(
    const TokenMap& tm,
    const std::vector<int32_t>& reference_text_ids,
    const std::vector<int32_t>& reference_audio_tokens
) {
    std::vector<int32_t> out;
    out.reserve(4 + reference_text_ids.size() + reference_audio_tokens.size());
    out.push_back(tm.start_id);

    if (tm.uses_reference_role_tokens()) {
        out.push_back(tm.reference_text_start_id);
        out.insert(out.end(), reference_text_ids.begin(), reference_text_ids.end());
        out.push_back(tm.reference_text_end_id);
        out.push_back(tm.reference_speech_start_id);
        for (int32_t t : reference_audio_tokens) {
            out.push_back(tm.audio_id_base + t);
        }
        out.push_back(tm.reference_speech_end_id);
    } else {
        out.push_back(tm.text_start_id);
        out.insert(out.end(), reference_text_ids.begin(), reference_text_ids.end());
        out.push_back(tm.speech_start_id);
        for (int32_t t : reference_audio_tokens) {
            out.push_back(tm.audio_id_base + t);
        }
        out.push_back(tm.speech_end_id);
    }
    return out;
}

std::vector<int32_t> build_generation_prompt(
    const TokenMap& tm,
    const std::vector<int32_t>& target_text_ids,
    const std::vector<int32_t>* reference_prefix,
    const std::vector<int32_t>* emotion_ids
) {
    std::vector<int32_t> out;
    if (reference_prefix && !reference_prefix->empty()) {
        out.insert(out.end(), reference_prefix->begin(), reference_prefix->end());
    } else {
        out.push_back(tm.start_id);
    }
    if (emotion_ids && !emotion_ids->empty()) {
        out.insert(out.end(), emotion_ids->begin(), emotion_ids->end());
    }
    out.push_back(tm.text_start_id);
    out.insert(out.end(), target_text_ids.begin(), target_text_ids.end());
    if (tm.final_seg_id != 0) {
        out.push_back(tm.final_seg_id);
    }
    out.push_back(tm.speech_start_id);
    return out;
}

void apply_logits_processors(
    std::vector<float>& scores,
    const std::vector<int32_t>& history,
    size_t prompt_length,
    const TokenMap& tm,
    const SamplingSettings& settings
) {
    const int32_t lo = tm.audio_id_base;
    const int32_t hi = tm.audio_id_end();

    // 1. TokenRunPenalty
    if (!history.empty() && settings.token_run_penalty > 0.0f) {
        int32_t last = history.back();
        if (last >= lo && last < hi) {
            int32_t run = 0;
            for (auto it = history.rbegin(); it != history.rend() && *it == last; ++it) {
                ++run;
            }
            if (run >= settings.token_run_grace && static_cast<size_t>(last) < scores.size()) {
                scores[last] -= static_cast<float>(run - settings.token_run_grace + 1) * settings.token_run_penalty;
            }
        }
    }

    // 2. RepetitionPenaltyExempt (exempts ONLY speech_end_id and stop_id, matching Python logits.py!)
    if (settings.repetition_penalty != 1.0f && !history.empty()) {
        size_t start = 0;
        if (settings.repetition_window > 0) {
            size_t win = static_cast<size_t>(settings.repetition_window);
            size_t win_start = history.size() > win ? (history.size() - win) : 0;
            start = std::max(prompt_length, win_start);
        }
        std::vector<uint8_t> seen(scores.size(), 0);
        for (size_t i = start; i < history.size(); ++i) {
            int32_t id = history[i];
            if (id < 0 || static_cast<size_t>(id) >= scores.size()) continue;
            if (id == tm.speech_end_id || id == tm.stop_id) continue;
            if (seen[id]) continue;
            seen[id] = 1;
            float s = scores[id];
            scores[id] = (s < 0.0f) ? (s * settings.repetition_penalty) : (s / settings.repetition_penalty);
        }
    }

    // 3. TemperatureLogitsWarper
    if (settings.temperature != 1.0f && settings.temperature > 0.0f) {
        const float inv_temp = 1.0f / settings.temperature;
        for (float& s : scores) s *= inv_temp;
    }

    const float neg_inf = -std::numeric_limits<float>::infinity();

    // 4. TopKLogitsWarper
    if (settings.top_k > 0 && static_cast<size_t>(settings.top_k) < scores.size()) {
        std::vector<float> finite_vals;
        finite_vals.reserve(8192);
        for (float s : scores) {
            if (std::isfinite(s)) finite_vals.push_back(s);
        }
        if (finite_vals.size() > static_cast<size_t>(settings.top_k)) {
            std::nth_element(
                finite_vals.begin(),
                finite_vals.begin() + (settings.top_k - 1),
                finite_vals.end(),
                std::greater<float>()
            );
            float cutoff = finite_vals[settings.top_k - 1];
            for (float& s : scores) {
                if (s < cutoff) s = neg_inf;
            }
        }
    }

    // 5. TopPLogitsWarper
    if (settings.top_p > 0.0f && settings.top_p < 1.0f) {
        std::vector<size_t> finite_indices;
        finite_indices.reserve(256);
        for (size_t i = 0; i < scores.size(); ++i) {
            if (std::isfinite(scores[i])) finite_indices.push_back(i);
        }
        if (finite_indices.size() > 1) {
            // Sort ascending as in transformers TopPLogitsWarper
            std::sort(finite_indices.begin(), finite_indices.end(), [&](size_t a, size_t b) {
                return scores[a] < scores[b];
            });
            float peak = scores[finite_indices.back()];
            double sum_exp = 0.0;
            for (size_t idx : finite_indices) {
                sum_exp += std::exp(static_cast<double>(scores[idx] - peak));
            }
            double cum_prob = 0.0;
            const double threshold = 1.0 - static_cast<double>(settings.top_p);
            for (size_t i = 0; i + 1 < finite_indices.size(); ++i) {
                size_t idx = finite_indices[i];
                cum_prob += std::exp(static_cast<double>(scores[idx] - peak)) / sum_exp;
                if (cum_prob <= threshold) {
                    scores[idx] = neg_inf;
                }
            }
        }
    }

    // 6. MinPLogitsWarper
    if (settings.min_p > 0.0f && settings.min_p <= 1.0f) {
        float max_logit = neg_inf;
        for (float s : scores) {
            if (s > max_logit) max_logit = s;
        }
        if (std::isfinite(max_logit)) {
            float cutoff = max_logit + std::log(settings.min_p);
            for (float& s : scores) {
                if (s < cutoff) s = neg_inf;
            }
        }
    }
}

int32_t sample_token(
    const std::vector<float>& scores,
    std::mt19937& rng,
    bool greedy
) {
    if (greedy) {
        auto it = std::max_element(scores.begin(), scores.end());
        return static_cast<int32_t>(std::distance(scores.begin(), it));
    }

    float peak = -std::numeric_limits<float>::infinity();
    for (float s : scores) {
        if (std::isfinite(s) && s > peak) peak = s;
    }
    if (!std::isfinite(peak)) {
        throw std::runtime_error("No finite logits available for sampling");
    }

    std::vector<int32_t> candidate_ids;
    std::vector<double> weights;
    candidate_ids.reserve(128);
    weights.reserve(128);

    for (size_t i = 0; i < scores.size(); ++i) {
        if (std::isfinite(scores[i])) {
            candidate_ids.push_back(static_cast<int32_t>(i));
            weights.push_back(std::exp(static_cast<double>(scores[i] - peak)));
        }
    }
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return candidate_ids[dist(rng)];
}

} // namespace kitteni
