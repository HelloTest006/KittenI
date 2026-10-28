#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

namespace kitteni {

struct AudioBuffer {
    std::vector<float> samples; // Mono float32 samples in [-1, 1]
    int sample_rate = 0;
};

// Read a WAV file (PCM 16-bit, PCM 24-bit, PCM 32-bit, or IEEE Float32; mono or multi-channel averaged to mono)
AudioBuffer read_wav_file(const std::string& path);

// Windowed sinc polyphase resampler (matching torchaudio / librosa Kaiser-windowed sinc resampling)
std::vector<float> resample_audio(
    const float* samples,
    int64_t num_samples,
    int src_rate,
    int dst_rate
);

// Periodic Hann window of length `win_length`: `w[n] = 0.5 * (1 - cos(2 * pi * n / win_length))`
std::vector<float> hann_window_periodic(int win_length);

// 1D reflect padding: pads `left_pad` samples on left and `right_pad` samples on right.
std::vector<float> pad_reflect_1d(const float* x, int64_t len, int64_t left_pad, int64_t right_pad);

// HiFTNet fast 16-point STFT (n_fft=16, hop_len=4, win_len=16, reflect-padded by 8 on both ends):
// Input `s` has length `L`. Output `out_stft` has shape `[18, T_frames]` where `T_frames = L / 4 + 1`:
//   channels `0..8`: real part of bins `0..8`
//   channels `9..17`: imaginary part of bins `0..8`
void hift_stft_16(
    const float* s,
    int64_t L,
    const float* window_16,
    std::vector<float>& out_stft // [18 * (L / 4 + 1)]
);

// HiFTNet fast 16-point ISTFT (n_fft=16, hop_len=4, win_len=16, center=True):
// Input `real_9xT` and `imag_9xT` have shape `[9, T_frames]`.
// Output `out_wav` has length `(T_frames - 1) * 4`.
void hift_istft_16(
    const float* real_9xT,
    const float* imag_9xT,
    int64_t T_frames,
    const float* window_16,
    std::vector<float>& out_wav
);

// Build Slaney-normalized triangular Mel filterbank `[n_mels, n_fft / 2 + 1]`
// (matching `librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)`)
std::vector<float> build_librosa_mel_filterbank(
    int sample_rate,
    int n_fft,
    int n_mels,
    float fmin = 0.0f,
    float fmax = -1.0f
);

// Compute 24 kHz 80-bin log-Mel spectrogram for S3 reference conditioning
// (`n_fft=1920, hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False` with reflect pad 720):
// Returns row-major `[T_frames, 80]` where `T_frames = num_samples / 480`.
std::vector<float> extract_mel80_24k(
    const float* wav_24k,
    int64_t num_samples,
    int64_t& out_frames
);

// Compute 16 kHz 128-bin Whisper-style log-Mel spectrogram for S3TokenizerV2
// (`n_fft=400, hop_size=160, center=True` reflect pad 200, drops last frame `stft[..., :-1]`):
// Uses `mel_filters_128x201` (`tokenizer._mel_filters`) and `window_400` (`tokenizer.window`).
// Output has shape `[128, T_frames]` where `T_frames = num_samples / 160`.
std::vector<float> extract_s3tokenizer_mel128_16k(
    const float* wav_16k,
    int64_t num_samples,
    const float* mel_filters_128x201,
    const float* window_400,
    int64_t& out_frames
);

// Compute 16 kHz 80-bin Kaldi fbank features with per-bin mean subtraction for CAMPPlus speaker encoder:
// (`frame_length=25ms (400), frame_shift=10ms (160), n_fft=512, preemphasis=0.97, Povey window, HTK mel 20..8000Hz`)
// Returns row-major `[T_frames, 80]` where `T_frames = (num_samples - 400) / 160 + 1`.
std::vector<float> extract_kaldi_fbank80_16k(
    const float* wav_16k,
    int64_t num_samples,
    int64_t& out_frames
);

} // namespace kitteni
