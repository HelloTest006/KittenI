#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace kitteni {

// Strip lead-in and trail-out silence below `thresh_db` (default -45.0 dB), keeping `keep_s` (0.02s) padding.
std::vector<float> trim_silence(
    const std::vector<float>& wav,
    int sample_rate = 24000,
    float thresh_db = -45.0f,
    float keep_s = 0.02f
);

// Concatenate synthesized chunks seamlessly:
// - Trims each chunk's edge silence
// - Applies 8ms linear fade-in and fade-out at edges
// - Inserts `gap_s` (default 0.16s) pause between chunks
std::vector<float> join_audio_chunks(
    const std::vector<std::vector<float>>& chunks,
    int sample_rate = 24000,
    float gap_s = 0.16f
);

// Write mono IEEE float32 WAV file at `sample_rate` (default 24000 Hz)
void write_wav_f32(
    const std::string& path,
    const std::vector<float>& audio,
    int sample_rate = 24000
);

} // namespace kitteni
