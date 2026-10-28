#include "frontend/audio_joiner.hpp"
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace kitteni {

std::vector<float> trim_silence(
    const std::vector<float>& wav,
    int sample_rate,
    float thresh_db,
    float keep_s
) {
    if (static_cast<int>(wav.size()) < static_cast<int>(sample_rate * 0.05f)) {
        return wav;
    }
    const size_t hop = std::max<size_t>(1, static_cast<size_t>(sample_rate * 0.01f)); // 240 samples at 24kHz
    const size_t n_frames = wav.size() / hop;
    const float thresh = std::pow(10.0f, thresh_db / 20.0f);

    size_t first_loud = n_frames;
    size_t last_loud = 0;
    bool found = false;

    for (size_t f = 0; f < n_frames; ++f) {
        float sum_sq = 0.0f;
        const float* frame = wav.data() + f * hop;
        for (size_t j = 0; j < hop; ++j) {
            sum_sq += frame[j] * frame[j];
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(hop) + 1e-12f);
        if (rms > thresh) {
            if (!found) {
                first_loud = f;
                found = true;
            }
            last_loud = f;
        }
    }

    if (!found) return wav;

    const size_t keep = static_cast<size_t>(sample_rate * keep_s); // 480 samples at 24kHz
    const size_t start = (first_loud * hop > keep) ? (first_loud * hop - keep) : 0;
    const size_t end = std::min<size_t>(wav.size(), (last_loud + 1) * hop + keep);
    return std::vector<float>(wav.begin() + static_cast<ptrdiff_t>(start),
                              wav.begin() + static_cast<ptrdiff_t>(end));
}

std::vector<float> join_audio_chunks(
    const std::vector<std::vector<float>>& chunks,
    int sample_rate,
    float gap_s
) {
    std::vector<const std::vector<float>*> non_empty;
    for (const auto& c : chunks) {
        if (!c.empty()) non_empty.push_back(&c);
    }
    if (non_empty.empty()) return {};
    if (non_empty.size() == 1) return *non_empty.front();

    const size_t gap_samples = static_cast<size_t>(sample_rate * gap_s);
    const size_t fade = std::max<size_t>(1, static_cast<size_t>(sample_rate * 0.008f)); // 192 samples at 24kHz

    std::vector<float> out;
    for (size_t i = 0; i < non_empty.size(); ++i) {
        std::vector<float> part = trim_silence(*non_empty[i], sample_rate);
        if (part.size() > 2 * fade) {
            for (size_t j = 0; j < fade; ++j) {
                float alpha = static_cast<float>(j) / static_cast<float>(fade - 1);
                part[j] *= alpha;
                part[part.size() - fade + j] *= (1.0f - alpha);
            }
        }
        out.insert(out.end(), part.begin(), part.end());
        if (i + 1 < non_empty.size() && gap_samples > 0) {
            out.insert(out.end(), gap_samples, 0.0f);
        }
    }
    return out;
}

void write_wav_f32(
    const std::string& path,
    const std::vector<float>& audio,
    int sample_rate
) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open WAV output file: " + path);
    }
    auto write_u16 = [&](uint16_t v) {
        f.put(static_cast<char>(v & 0xFFU));
        f.put(static_cast<char>((v >> 8) & 0xFFU));
    };
    auto write_u32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            f.put(static_cast<char>((v >> (i * 8)) & 0xFFU));
        }
    };

    const uint32_t data_bytes = static_cast<uint32_t>(audio.size() * sizeof(float));
    f.write("RIFF", 4);
    write_u32(36U + data_bytes);
    f.write("WAVEfmt ", 8);
    write_u32(16U);                                  // fmt chunk size
    write_u16(3U);                                   // format 3 = IEEE float
    write_u16(1U);                                   // 1 channel (mono)
    write_u32(static_cast<uint32_t>(sample_rate));   // sample rate
    write_u32(static_cast<uint32_t>(sample_rate * 4)); // byte rate
    write_u16(4U);                                   // block align
    write_u16(32U);                                  // bits per sample
    f.write("data", 4);
    write_u32(data_bytes);
    f.write(reinterpret_cast<const char*>(audio.data()), data_bytes);
    if (!f) {
        throw std::runtime_error("Error writing WAV data to: " + path);
    }
}

} // namespace kitteni
