#include "cloning/voice_cloner.hpp"
#include "kernels/dsp_fft.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace kitteni {

namespace {

// Build a compact `.safetensors` file containing ONLY `speaker_encoder.*` (F32) and `tokenizer.*` (F32)
// from `ResembleAI/chatterbox-turbo/s3gen_meanflow.safetensors` using HTTP byte-range requests via curl!
// Why? In `s3gen_meanflow.safetensors` (1.06 GB), `speaker_encoder.*` F32 weights and `tokenizer.*` F32 weights
// occupy one contiguous 522 MB slice `[541872204 .. 1064557868]` at the end of the file!
// Fetching just the header + that slice saves ~542 MB of download and disk space.
fs::path ensure_s3_cloner_safetensors(const ResolvedRepo& repo, bool allow_download) {
    // 1. Check if full `s3gen_meanflow.safetensors` or compact `speaker/s3_cloner.safetensors` already exists
    fs::path compact_path = repo.root_dir / "speaker" / "s3_cloner.safetensors";
    if (fs::exists(compact_path) && fs::file_size(compact_path) > 1000000) {
        return compact_path;
    }

    fs::path full_in_repo = repo.root_dir / "speaker" / "s3gen_meanflow.safetensors";
    if (fs::exists(full_in_repo)) return full_in_repo;

    if (!allow_download) {
        return {};
    }

    fs::create_directories(compact_path.parent_path());
    const std::string url = "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/s3gen_meanflow.safetensors";
    fs::path hdr_tmp = compact_path;
    hdr_tmp += ".hdr";

    // Fetch first 330 KB to read the safetensors JSON header
    std::string cmd_hdr = "curl.exe -s -L --fail -r 0-330000 -o \"" + hdr_tmp.string() + "\" \"" + url + "\"";
#ifndef _WIN32
    cmd_hdr = "curl -s -L --fail -r 0-330000 -o \"" + hdr_tmp.string() + "\" \"" + url + "\"";
#endif
    if (std::system(cmd_hdr.c_str()) != 0 || !fs::exists(hdr_tmp)) {
        throw std::runtime_error("Failed to fetch s3gen_meanflow.safetensors header from HuggingFace");
    }

    std::string header_str;
    uint64_t orig_header_len = 0;
    {
        std::ifstream hf(hdr_tmp, std::ios::binary);
        hf.read(reinterpret_cast<char*>(&orig_header_len), 8);
        header_str.resize(static_cast<size_t>(orig_header_len));
        hf.read(header_str.data(), static_cast<std::streamsize>(orig_header_len));
    }
    fs::remove(hdr_tmp);

    JsonValue orig_meta = parse_json(header_str);
    const size_t orig_data_start = 8 + static_cast<size_t>(orig_header_len);

    // Find min and max byte offset across all F32 `speaker_encoder.*` and `tokenizer.*` tensors
    uint64_t min_off = UINT64_MAX;
    uint64_t max_off = 0;
    struct Entry {
        std::string name;
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t begin;
        uint64_t end;
    };
    std::vector<Entry> kept;

    for (const auto& [k, v] : orig_meta.as_object()) {
        if (k == "__metadata__") continue;
        if (k.rfind("speaker_encoder.", 0) != 0 && k.rfind("tokenizer.", 0) != 0) continue;
        std::string dt = v.at("dtype").as_string();
        if (dt != "F32") continue;

        const auto& offs = v.at("data_offsets").as_array();
        uint64_t b = static_cast<uint64_t>(offs[0].as_int());
        uint64_t e = static_cast<uint64_t>(offs[1].as_int());
        min_off = std::min(min_off, b);
        max_off = std::max(max_off, e);

        Entry ent;
        ent.name = k;
        ent.dtype = dt;
        for (const auto& d : v.at("shape").as_array()) ent.shape.push_back(d.as_int());
        ent.begin = b;
        ent.end = e;
        kept.push_back(std::move(ent));
    }

    // Build new compact safetensors JSON header with shifted offsets `begin - min_off`
    std::ostringstream js;
    js << "{";
    for (size_t i = 0; i < kept.size(); ++i) {
        if (i > 0) js << ",";
        const auto& e = kept[i];
        js << "\"" << e.name << "\":{\"dtype\":\"" << e.dtype << "\",\"shape\":[";
        for (size_t d = 0; d < e.shape.size(); ++d) {
            if (d > 0) js << ",";
            js << e.shape[d];
        }
        js << "],\"data_offsets\":[" << (e.begin - min_off) << "," << (e.end - min_off) << "]}";
    }
    js << "}";
    std::string new_header = js.str();
    while (new_header.size() % 8 != 0) new_header.push_back(' ');

    // Download the contiguous byte range `[orig_data_start + min_off .. orig_data_start + max_off - 1]`
    fs::path body_tmp = compact_path;
    body_tmp += ".body";
    uint64_t range_start = orig_data_start + min_off;
    uint64_t range_end   = orig_data_start + max_off - 1;

    std::cerr << "[KittenI] Downloading S3 Voice Cloner weights (CAMPPlus + S3TokenizerV2, "
              << ((range_end - range_start + 1) / (1024 * 1024)) << " MiB range) ...\n";

    std::string cmd_body = "curl.exe -L --fail --retry 3 -r " + std::to_string(range_start) + "-" +
                           std::to_string(range_end) + " -o \"" + body_tmp.string() + "\" \"" + url + "\"";
#ifndef _WIN32
    cmd_body = "curl -L --fail --retry 3 -r " + std::to_string(range_start) + "-" +
               std::to_string(range_end) + " -o \"" + body_tmp.string() + "\" \"" + url + "\"";
#endif
    if (std::system(cmd_body.c_str()) != 0 || !fs::exists(body_tmp)) {
        throw std::runtime_error("Failed to download S3 Voice Cloner weight range from HuggingFace");
    }

    // Assemble `compact_path`: 8-byte header length + `new_header` + `body_tmp`
    fs::path out_tmp = compact_path;
    out_tmp += ".partial";
    {
        std::ofstream out(out_tmp, std::ios::binary);
        uint64_t new_hdr_len = static_cast<uint64_t>(new_header.size());
        out.write(reinterpret_cast<const char*>(&new_hdr_len), 8);
        out.write(new_header.data(), static_cast<std::streamsize>(new_header.size()));

        std::ifstream body_in(body_tmp, std::ios::binary);
        out << body_in.rdbuf();
    }
    fs::remove(body_tmp);
    if (fs::exists(compact_path)) fs::remove(compact_path);
    fs::rename(out_tmp, compact_path);
    return compact_path;
}

} // namespace

void VoiceCloner::load(const ResolvedRepo& repo, bool ensure_s3_cloner, bool allow_download) {
    std::string spk_rel = repo.config.value_string("speaker_embedding", "speaker/model.safetensors");
    auto spk_path = ensure_repo_file(repo, spk_rel, allow_download);
    sincnet_.load(spk_path.string());

    if (ensure_s3_cloner) {
        fs::path s3_path = ensure_s3_cloner_safetensors(repo, allow_download);
        if (!s3_path.empty() && fs::exists(s3_path)) {
            SafetensorsFile sf(s3_path.string());
            campplus_.load(sf, "speaker_encoder.");
            s3_tokenizer_.load(sf, "tokenizer.");
            has_s3_cloner_ = true;
        }
    }
}

ClonedVoiceProfile VoiceCloner::clone_from_wav(
    const std::string& wav_path,
    const std::string& reference_text,
    const KittenTernaryLMWeights* lm,
    bool for_student_flow,
    const ThreadPool* pool
) const {
    AudioBuffer raw = read_wav_file(wav_path);
    if (raw.samples.empty()) {
        throw std::runtime_error("Reference WAV is empty: " + wav_path);
    }

    // 1. Resample to 16 kHz for `SincNetSpeakerEncoder`, `CampPlusEncoder`, and `S3AudioTokenizer`
    std::vector<float> wav_16k = resample_audio(raw.samples.data(), static_cast<int64_t>(raw.samples.size()),
                                                raw.sample_rate, 16000);

    ClonedVoiceProfile prof;
    prof.vocoder_voice.transcript = reference_text;

    // 2. Compute 512-d unit-norm SincNet speaker embedding for LM `spk_proj`
    prof.embedding_512 = sincnet_.encode_16k(wav_16k.data(), static_cast<int64_t>(wav_16k.size()), pool);
    if (lm) {
        prof.vocoder_voice.lm_speaker_2048.resize(static_cast<size_t>(lm->config.hidden_size));
        lm->project_speaker(prof.embedding_512.data(), prof.vocoder_voice.lm_speaker_2048.data());
    }

    if (!has_s3_cloner_) {
        return prof;
    }

    // 3. Encode full reference clip at 16 kHz into 25 Hz `reference_tokens` for LM in-context prompt
    prof.vocoder_voice.reference_tokens = s3_tokenizer_.encode_16k(
        wav_16k.data(), static_cast<int64_t>(wav_16k.size()), pool
    );

    // 4. Prepare S3 decoder reference audio at 24 kHz (`_load_decoder_reference` in `vocoder.py`):
    //    - Tile short clips to at least 6.0 seconds (`min_seconds = 6.0`, never silence-padded!)
    //    - Slice to `4.0` seconds (`STUDENT_REF_SECONDS`) for student flow or `10.0` seconds for default flow
    std::vector<float> wav_24k = resample_audio(raw.samples.data(), static_cast<int64_t>(raw.samples.size()),
                                                raw.sample_rate, 24000);
    const size_t min_samples_24k = 6 * 24000;
    if (wav_24k.size() < min_samples_24k) {
        std::vector<float> tiled;
        tiled.reserve(min_samples_24k + wav_24k.size());
        while (tiled.size() < min_samples_24k) {
            tiled.insert(tiled.end(), wav_24k.begin(), wav_24k.end());
        }
        wav_24k.swap(tiled);
    }
    const size_t max_cond_samples_24k = static_cast<size_t>((for_student_flow ? 4.0 : 10.0) * 24000);
    if (wav_24k.size() > max_cond_samples_24k) {
        wav_24k.resize(max_cond_samples_24k);
    }

    // 5. Extract 24 kHz Mel80 features (`prompt_feat`)
    int64_t mel_frames = 0;
    prof.vocoder_voice.prompt_feat = extract_mel80_24k(
        wav_24k.data(), static_cast<int64_t>(wav_24k.size()), mel_frames
    );

    // 6. Resample `wav_24k` conditioning slice to 16 kHz for `CampPlusEncoder` (192-d x-vector) and `prompt_token`
    std::vector<float> cond_16k = resample_audio(wav_24k.data(), static_cast<int64_t>(wav_24k.size()), 24000, 16000);
    prof.vocoder_voice.flow_embedding_192 = campplus_.encode_16k(
        cond_16k.data(), static_cast<int64_t>(cond_16k.size()), pool
    );

    std::vector<int32_t> cond_tokens = s3_tokenizer_.encode_16k(
        cond_16k.data(), static_cast<int64_t>(cond_16k.size()), pool
    );

    // 7. Align `prompt_token` and `prompt_feat` (`mel_len == 2 * token_len`, max 112 tokens for student flow)
    int64_t max_prompt_tokens = for_student_flow ? 112 : static_cast<int64_t>(cond_tokens.size());
    int64_t n_tok = std::min({static_cast<int64_t>(cond_tokens.size()), mel_frames / 2, max_prompt_tokens});

    prof.vocoder_voice.prompt_token.assign(cond_tokens.begin(), cond_tokens.begin() + n_tok);
    prof.vocoder_voice.prompt_feat_frames = n_tok * 2;
    prof.vocoder_voice.prompt_feat.resize(static_cast<size_t>(n_tok * 2 * 80));

    return prof;
}

void VoiceCloner::save_profile_json(const std::string& json_path, const ClonedVoiceProfile& p) {
    std::ofstream f(json_path);
    if (!f) throw std::runtime_error("Cannot write voice profile JSON: " + json_path);

    f << std::setprecision(8);
    f << "{\n";
    f << "  \"transcript\": \"" << p.vocoder_voice.transcript << "\",\n";

    auto write_int_arr = [&](const char* key, const std::vector<int32_t>& arr, bool nested) {
        f << "  \"" << key << "\": " << (nested ? "[[" : "[");
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) f << ", ";
            f << arr[i];
        }
        f << (nested ? "]],\n" : "],\n");
    };

    auto write_flt_arr = [&](const char* key, const std::vector<float>& arr, bool nested, bool last) {
        f << "  \"" << key << "\": " << (nested ? "[[" : "[");
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) f << ", ";
            f << arr[i];
        }
        f << (nested ? "]]" : "]") << (last ? "\n" : ",\n");
    };

    write_int_arr("reference_tokens", p.vocoder_voice.reference_tokens, false);
    write_flt_arr("embedding_512", p.embedding_512, false, false);
    write_flt_arr("speaker", p.vocoder_voice.lm_speaker_2048, false, false);
    write_int_arr("prompt_token", p.vocoder_voice.prompt_token, true);

    f << "  \"prompt_feat\": [[";
    for (int64_t t = 0; t < p.vocoder_voice.prompt_feat_frames; ++t) {
        if (t > 0) f << ", ";
        f << "[";
        for (int c = 0; c < 80; ++c) {
            if (c > 0) f << ", ";
            f << p.vocoder_voice.prompt_feat[t * 80 + c];
        }
        f << "]";
    }
    f << "]],\n";

    write_flt_arr("embedding", p.vocoder_voice.flow_embedding_192, true, true);
    f << "}\n";
}

ClonedVoiceProfile VoiceCloner::load_profile_json(const std::string& json_path) {
    JsonValue v = parse_json_file(json_path);
    ClonedVoiceProfile p;
    p.vocoder_voice.transcript = v.value_string("transcript");

    for (const auto& x : v.at("reference_tokens").as_array()) {
        p.vocoder_voice.reference_tokens.push_back(static_cast<int32_t>(x.as_int()));
    }
    if (v.contains("embedding_512")) {
        for (const auto& x : v.at("embedding_512").as_array()) {
            p.embedding_512.push_back(static_cast<float>(x.as_float()));
        }
    }
    if (v.contains("speaker")) {
        for (const auto& x : v.at("speaker").as_array()) {
            p.vocoder_voice.lm_speaker_2048.push_back(static_cast<float>(x.as_float()));
        }
    }
    for (const auto& x : v.at("prompt_token")[0].as_array()) {
        p.vocoder_voice.prompt_token.push_back(static_cast<int32_t>(x.as_int()));
    }
    const auto& pf_rows = v.at("prompt_feat")[0].as_array();
    p.vocoder_voice.prompt_feat_frames = static_cast<int64_t>(pf_rows.size());
    for (const auto& row : pf_rows) {
        for (const auto& val : row.as_array()) {
            p.vocoder_voice.prompt_feat.push_back(static_cast<float>(val.as_float()));
        }
    }
    for (const auto& x : v.at("embedding")[0].as_array()) {
        p.vocoder_voice.flow_embedding_192.push_back(static_cast<float>(x.as_float()));
    }
    return p;
}

} // namespace kitteni
