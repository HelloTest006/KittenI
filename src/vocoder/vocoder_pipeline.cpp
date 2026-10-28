#include "vocoder/vocoder_pipeline.hpp"
#include <stdexcept>
#include <algorithm>

namespace kitteni {

void NativeVocoderPipeline::load(const ResolvedRepo& repo, const std::string& decoder_name) {
    decoder_name_ = (decoder_name == "default") ? "student_w4" : decoder_name;
    int bits = (decoder_name_ == "student_w8") ? 8 : 4;

    std::string pt_rel = "cpp/" + decoder_name_ + "/decoder.pt";
    std::string voices_rel = "cpp/" + decoder_name_ + "/voices.json";

    auto pt_path = ensure_repo_file(repo, pt_rel, true);
    auto v_path  = ensure_repo_file(repo, voices_rel, true);
    voices_json_path_ = v_path.string();

    TorchScriptZipConstants zip(pt_path.string());
    flow_.load_from_torchscript_zip(zip, bits);
    hift_.load_from_torchscript_zip(zip);
}

PreparedVoice NativeVocoderPipeline::load_voice(const std::string& voice_name) const {
    if (voices_cache_.is_null()) {
        voices_cache_ = parse_json_file(voices_json_path_);
    }
    if (!voices_cache_.contains(voice_name)) {
        throw std::runtime_error("Unknown voice in voices.json: " + voice_name);
    }
    const JsonValue& v = voices_cache_.at(voice_name);
    PreparedVoice pv;
    pv.transcript = v.value_string("transcript");

    for (const auto& t : v.at("reference_tokens").as_array()) {
        pv.reference_tokens.push_back(static_cast<int32_t>(t.as_int()));
    }
    for (const auto& s : v.at("speaker").as_array()) {
        pv.lm_speaker_2048.push_back(static_cast<float>(s.as_float()));
    }
    for (const auto& pt : v.at("prompt_token")[0].as_array()) {
        pv.prompt_token.push_back(static_cast<int32_t>(pt.as_int()));
    }

    const auto& pf_rows = v.at("prompt_feat")[0].as_array();
    pv.prompt_feat_frames = static_cast<int64_t>(pf_rows.size());
    pv.prompt_feat.reserve(static_cast<size_t>(pv.prompt_feat_frames * 80));
    for (const auto& row : pf_rows) {
        for (const auto& val : row.as_array()) {
            pv.prompt_feat.push_back(static_cast<float>(val.as_float()));
        }
    }

    for (const auto& e : v.at("embedding")[0].as_array()) {
        pv.flow_embedding_192.push_back(static_cast<float>(e.as_float()));
    }
    return pv;
}

std::vector<float> NativeVocoderPipeline::decode_tokens_to_wav(
    const std::vector<int32_t>& audio_ids,
    const PreparedVoice& voice,
    std::mt19937& rng,
    const ThreadPool* pool,
    const float* fixed_flow_noise,
    bool deterministic_hift_source
) const {
    std::vector<int32_t> valid_tokens;
    valid_tokens.reserve(audio_ids.size() + 3);
    for (int32_t id : audio_ids) {
        if (id >= 0 && id < 6561) valid_tokens.push_back(id);
    }
    if (valid_tokens.empty()) return {};

    // Append 3 silence tokens (4299) for s3gen lookahead
    valid_tokens.insert(valid_tokens.end(), 3, 4299);

    std::vector<float> mel = flow_.forward(
        valid_tokens,
        voice.prompt_token,
        voice.prompt_feat.data(),
        voice.prompt_feat_frames,
        voice.flow_embedding_192.data(),
        rng,
        pool,
        fixed_flow_noise
    );

    const int64_t T_speech_mel = static_cast<int64_t>(valid_tokens.size()) * 2;
    return hift_.forward(mel.data(), T_speech_mel, rng, pool, deterministic_hift_source);
}

} // namespace kitteni
