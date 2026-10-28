#include "kernels/dsp_fft.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <complex>

namespace kitteni {

namespace {
constexpr double kPi = 3.14159265358979323846;

struct Dft16Tables {
    float cos_tab[9][16];
    float sin_tab[9][16];
    Dft16Tables() {
        for (int k = 0; k < 9; ++k) {
            for (int n = 0; n < 16; ++n) {
                double angle = 2.0 * kPi * k * n / 16.0;
                cos_tab[k][n] = static_cast<float>(std::cos(angle));
                sin_tab[k][n] = static_cast<float>(-std::sin(angle));
            }
        }
    }
};

const Dft16Tables kDft16{};

// In-place radix-2 Cooley-Tukey complex FFT (for power-of-2 N, e.g. N=512)
void fft_radix2_inplace(std::complex<float>* a, int n) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * kPi / len;
        std::complex<float> wlen(static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang)));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<float> u = a[i + j];
                std::complex<float> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// General real-input DFT for composite N (e.g. N=400 = 16*25 or N=1920 = 128*15)
// Precomputes twiddle matrix `[N/2 + 1, N]` once per `N` for fast SIMD dot products!
struct RealDftPlan {
    int n_fft = 0;
    int n_bins = 0;
    std::vector<float> cos_mat; // [n_bins * n_fft]
    std::vector<float> sin_mat; // [n_bins * n_fft]

    explicit RealDftPlan(int n) : n_fft(n), n_bins(n / 2 + 1) {
        cos_mat.resize(static_cast<size_t>(n_bins * n_fft));
        sin_mat.resize(static_cast<size_t>(n_bins * n_fft));
        for (int k = 0; k < n_bins; ++k) {
            for (int t = 0; t < n_fft; ++t) {
                double ang = 2.0 * kPi * k * t / n_fft;
                cos_mat[k * n_fft + t] = static_cast<float>(std::cos(ang));
                sin_mat[k * n_fft + t] = static_cast<float>(-std::sin(ang));
            }
        }
    }

    void compute_power_or_mag(const float* frame, float* out_bins, bool power_spectrum) const {
        for (int k = 0; k < n_bins; ++k) {
            const float* c_row = cos_mat.data() + k * n_fft;
            const float* s_row = sin_mat.data() + k * n_fft;
            float re = 0.0f, im = 0.0f;
            for (int t = 0; t < n_fft; ++t) {
                float x = frame[t];
                re += x * c_row[t];
                im += x * s_row[t];
            }
            float p = re * re + im * im;
            out_bins[k] = power_spectrum ? p : std::sqrt(p + 1e-9f);
        }
    }
};

const RealDftPlan& get_dft_plan_400() {
    static const RealDftPlan plan(400);
    return plan;
}

const RealDftPlan& get_dft_plan_1920() {
    static const RealDftPlan plan(1920);
    return plan;
}

// Librosa Slaney mel scale conversions
double hz_to_mel_slaney(double freq) {
    const double f_min = 0.0;
    const double f_sp = 200.0 / 3.0;
    double mels = (freq - f_min) / f_sp;
    const double min_log_hz = 1000.0;
    const double min_log_mel = (min_log_hz - f_min) / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (freq >= min_log_hz) {
        mels = min_log_mel + std::log(freq / min_log_hz) / logstep;
    }
    return mels;
}

double mel_to_hz_slaney(double mels) {
    const double f_min = 0.0;
    const double f_sp = 200.0 / 3.0;
    double freqs = f_min + f_sp * mels;
    const double min_log_hz = 1000.0;
    const double min_log_mel = (min_log_hz - f_min) / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (mels >= min_log_mel) {
        freqs = min_log_hz * std::exp(logstep * (mels - min_log_mel));
    }
    return freqs;
}

} // namespace

AudioBuffer read_wav_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }
    char riff[4], wave[4];
    uint32_t file_size = 0;
    f.read(riff, 4);
    f.read(reinterpret_cast<char*>(&file_size), 4);
    f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        throw std::runtime_error("Invalid RIFF/WAVE header in: " + path);
    }

    uint16_t audio_format = 0, num_channels = 0, block_align = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, byte_rate = 0;
    std::vector<uint8_t> raw_data;

    while (f) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (!f.read(chunk_id, 4)) break;
        if (!f.read(reinterpret_cast<char*>(&chunk_size), 4)) break;

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            f.read(reinterpret_cast<char*>(&audio_format), 2);
            f.read(reinterpret_cast<char*>(&num_channels), 2);
            f.read(reinterpret_cast<char*>(&sample_rate), 4);
            f.read(reinterpret_cast<char*>(&byte_rate), 4);
            f.read(reinterpret_cast<char*>(&block_align), 2);
            f.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (chunk_size > 16) {
                // Check for WAVE_FORMAT_EXTENSIBLE (0xFFFE)
                if (audio_format == 0xFFFEU && chunk_size >= 40) {
                    uint16_t cb_size = 0, valid_bits = 0;
                    uint32_t channel_mask = 0;
                    uint16_t sub_format = 0;
                    f.read(reinterpret_cast<char*>(&cb_size), 2);
                    f.read(reinterpret_cast<char*>(&valid_bits), 2);
                    f.read(reinterpret_cast<char*>(&channel_mask), 4);
                    f.read(reinterpret_cast<char*>(&sub_format), 2);
                    audio_format = sub_format;
                    f.seekg(chunk_size - 26, std::ios::cur);
                } else {
                    f.seekg(chunk_size - 16, std::ios::cur);
                }
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            raw_data.resize(chunk_size);
            f.read(reinterpret_cast<char*>(raw_data.data()), chunk_size);
            break;
        } else {
            f.seekg(chunk_size + (chunk_size & 1U), std::ios::cur);
        }
    }

    if (raw_data.empty() || num_channels == 0 || sample_rate == 0) {
        throw std::runtime_error("WAV file contains no audio data: " + path);
    }

    const size_t bytes_per_sample = bits_per_sample / 8;
    const size_t frame_bytes = bytes_per_sample * num_channels;
    const size_t num_frames = raw_data.size() / frame_bytes;

    AudioBuffer out;
    out.sample_rate = static_cast<int>(sample_rate);
    out.samples.resize(num_frames);

    const uint8_t* ptr = raw_data.data();
    const float inv_ch = 1.0f / static_cast<float>(num_channels);

    for (size_t i = 0; i < num_frames; ++i) {
        float sum = 0.0f;
        for (uint16_t c = 0; c < num_channels; ++c) {
            const uint8_t* sp = ptr + (i * num_channels + c) * bytes_per_sample;
            if (audio_format == 3 && bits_per_sample == 32) {
                float v = 0.0f;
                std::memcpy(&v, sp, 4);
                sum += v;
            } else if (audio_format == 1 && bits_per_sample == 16) {
                int16_t v = 0;
                std::memcpy(&v, sp, 2);
                sum += static_cast<float>(v) * (1.0f / 32768.0f);
            } else if (audio_format == 1 && bits_per_sample == 24) {
                int32_t v = (static_cast<int32_t>(sp[0]) << 8) |
                            (static_cast<int32_t>(sp[1]) << 16) |
                            (static_cast<int32_t>(sp[2]) << 24);
                sum += static_cast<float>(v >> 8) * (1.0f / 8388608.0f);
            } else if (audio_format == 1 && bits_per_sample == 32) {
                int32_t v = 0;
                std::memcpy(&v, sp, 4);
                sum += static_cast<float>(v) * (1.0f / 2147483648.0f);
            } else {
                throw std::runtime_error("Unsupported WAV sample format in: " + path);
            }
        }
        out.samples[i] = sum * inv_ch;
    }

    return out;
}

// Modified Bessel function of order 0 (`I_0(x)`) for Kaiser window
double bessel_i0(double x) {
    double sum = 1.0;
    double y = (x * x) * 0.25;
    double term = 1.0;
    for (int k = 1; k <= 30; ++k) {
        term *= y / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        if (term < sum * 1e-15) break;
    }
    return sum;
}

std::vector<float> resample_audio(
    const float* samples,
    int64_t num_samples,
    int src_rate,
    int dst_rate
) {
    if (src_rate == dst_rate || num_samples <= 0) {
        return std::vector<float>(samples, samples + num_samples);
    }
    // Match `torchaudio.functional.resample` exact polyphase Kaiser-windowed sinc filter:
    //   orig_freq = src_rate / gcd, new_freq = dst_rate / gcd
    //   lowpass_filter_width = 6, rolloff = 0.99, beta = 14.769656459379492 (6 * pi * 0.78355...)
    int a = src_rate, b = dst_rate;
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    const int gcd = a;
    const int orig_freq = src_rate / gcd;
    const int new_freq  = dst_rate / gcd;

    const int lowpass_filter_width = 6;
    const double rolloff = 0.99;
    const double beta = 14.769656459379492; // 0.1102 * (100 - 8.7) for 100dB stopband, or torchaudio default 14.769656459379492
    const double base_freq = std::min(orig_freq, new_freq) * rolloff;
    const int width = static_cast<int>(std::ceil(lowpass_filter_width * orig_freq / base_freq));

    // Build polyphase kernel bank `kernels[p, idx]` for phase `p in [0..new_freq-1]` and `idx in [-width..width+orig_freq]`
    const int kernel_len = 2 * width + orig_freq;
    std::vector<float> kernels(static_cast<size_t>(new_freq * kernel_len), 0.0f);
    const double inv_i0_beta = 1.0 / bessel_i0(beta);
    const double scale = base_freq / orig_freq;

    for (int p = 0; p < new_freq; ++p) {
        float* k_row = kernels.data() + p * kernel_len;
        double t_offset = -static_cast<double>(p) / static_cast<double>(new_freq);
        for (int k = 0; k < kernel_len; ++k) {
            int j = k - width; // j in [-width .. width + orig_freq - 1]
            double t = (static_cast<double>(j) / static_cast<double>(orig_freq) + t_offset) * base_freq;
            double clamped_t = std::clamp(t, -static_cast<double>(lowpass_filter_width), static_cast<double>(lowpass_filter_width));
            double win_arg = clamped_t / lowpass_filter_width;
            double window = (std::fabs(t) < lowpass_filter_width)
                                ? (bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - win_arg * win_arg))) * inv_i0_beta)
                                : 0.0;
            double sinc = (std::fabs(t) < 1e-9) ? 1.0 : (std::sin(kPi * t) / (kPi * t));
            k_row[k] = static_cast<float>(sinc * window * scale);
        }
    }

    const int64_t target_len = (num_samples * new_freq + orig_freq - 1) / orig_freq;
    std::vector<float> out(static_cast<size_t>(target_len), 0.0f);

    for (int64_t out_idx = 0; out_idx < target_len; ++out_idx) {
        const int64_t block = out_idx / new_freq;
        const int p = static_cast<int>(out_idx % new_freq);
        const int64_t in_base = block * orig_freq - width;
        const float* k_row = kernels.data() + p * kernel_len;

        float acc = 0.0f;
        const int k_start = static_cast<int>(std::max<int64_t>(0, -in_base));
        const int k_end   = static_cast<int>(std::min<int64_t>(kernel_len, num_samples - in_base));
        for (int k = k_start; k < k_end; ++k) {
            acc += samples[in_base + k] * k_row[k];
        }
        out[out_idx] = acc;
    }
    return out;
}

std::vector<float> hann_window_periodic(int win_length) {
    std::vector<float> w(static_cast<size_t>(win_length));
    for (int n = 0; n < win_length; ++n) {
        w[n] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * n / win_length)));
    }
    return w;
}

std::vector<float> pad_reflect_1d(const float* x, int64_t len, int64_t left_pad, int64_t right_pad) {
    if (len <= 1) {
        throw std::invalid_argument("pad_reflect_1d requires len > 1");
    }
    std::vector<float> out(static_cast<size_t>(left_pad + len + right_pad));
    for (int64_t i = 0; i < left_pad; ++i) {
        int64_t src_idx = left_pad - i;
        src_idx = std::clamp<int64_t>(src_idx, 0, len - 1);
        out[i] = x[src_idx];
    }
    std::memcpy(out.data() + left_pad, x, static_cast<size_t>(len) * sizeof(float));
    for (int64_t i = 0; i < right_pad; ++i) {
        int64_t src_idx = len - 2 - i;
        src_idx = std::clamp<int64_t>(src_idx, 0, len - 1);
        out[left_pad + len + i] = x[src_idx];
    }
    return out;
}

void hift_stft_16(
    const float* s,
    int64_t L,
    const float* window_16,
    std::vector<float>& out_stft
) {
    std::vector<float> padded = pad_reflect_1d(s, L, 8, 8);
    const int64_t T_frames = (static_cast<int64_t>(padded.size()) - 16) / 4 + 1;
    out_stft.assign(static_cast<size_t>(18 * T_frames), 0.0f);

    float win_frame[16];
    for (int64_t t = 0; t < T_frames; ++t) {
        const float* src = padded.data() + t * 4;
        for (int n = 0; n < 16; ++n) {
            win_frame[n] = src[n] * window_16[n];
        }
        for (int k = 0; k < 9; ++k) {
            float re = 0.0f;
            float im = 0.0f;
            for (int n = 0; n < 16; ++n) {
                re += win_frame[n] * kDft16.cos_tab[k][n];
                im += win_frame[n] * kDft16.sin_tab[k][n];
            }
            out_stft[k * T_frames + t]       = re;
            out_stft[(k + 9) * T_frames + t] = im;
        }
    }
}

void hift_istft_16(
    const float* real_9xT,
    const float* imag_9xT,
    int64_t T_frames,
    const float* window_16,
    std::vector<float>& out_wav
) {
    if (T_frames <= 1) {
        out_wav.clear();
        return;
    }
    const int64_t full_len = (T_frames - 1) * 4 + 16;
    const int64_t out_len = (T_frames - 1) * 4;

    std::vector<float> ola_num(static_cast<size_t>(full_len), 0.0f);
    std::vector<float> ola_den(static_cast<size_t>(full_len), 0.0f);

    float win_sq[16];
    for (int n = 0; n < 16; ++n) {
        win_sq[n] = window_16[n] * window_16[n];
    }

    for (int64_t t = 0; t < T_frames; ++t) {
        float re_bins[9];
        float im_bins[9];
        for (int k = 0; k < 9; ++k) {
            re_bins[k] = real_9xT[k * T_frames + t];
            im_bins[k] = imag_9xT[k * T_frames + t];
        }

        const int64_t base = t * 4;
        for (int n = 0; n < 16; ++n) {
            float sum = re_bins[0] + ((n & 1) ? -re_bins[8] : re_bins[8]);
            for (int k = 1; k <= 7; ++k) {
                sum += 2.0f * (re_bins[k] * kDft16.cos_tab[k][n] + im_bins[k] * kDft16.sin_tab[k][n]);
            }
            float sample = (sum * (1.0f / 16.0f)) * window_16[n];
            ola_num[base + n] += sample;
            ola_den[base + n] += win_sq[n];
        }
    }

    out_wav.resize(static_cast<size_t>(out_len));
    for (int64_t i = 0; i < out_len; ++i) {
        float d = ola_den[i + 8];
        out_wav[i] = (d > 1e-11f) ? (ola_num[i + 8] / d) : 0.0f;
    }
}

std::vector<float> build_librosa_mel_filterbank(
    int sample_rate,
    int n_fft,
    int n_mels,
    float fmin,
    float fmax
) {
    if (fmax <= 0.0f) {
        fmax = static_cast<float>(sample_rate) / 2.0f;
    }
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> weights(static_cast<size_t>(n_mels * n_bins), 0.0f);

    std::vector<double> fftfreqs(static_cast<size_t>(n_bins));
    for (int i = 0; i < n_bins; ++i) {
        fftfreqs[i] = static_cast<double>(i) * sample_rate / n_fft;
    }

    const double min_mel = hz_to_mel_slaney(fmin);
    const double max_mel = hz_to_mel_slaney(fmax);
    std::vector<double> mel_f(static_cast<size_t>(n_mels + 2));
    for (int i = 0; i < n_mels + 2; ++i) {
        double m = min_mel + (max_mel - min_mel) * i / (n_mels + 1);
        mel_f[i] = mel_to_hz_slaney(m);
    }

    for (int m = 0; m < n_mels; ++m) {
        double f_left = mel_f[m];
        double f_center = mel_f[m + 1];
        double f_right = mel_f[m + 2];
        double enorm = 2.0 / (f_right - f_left);

        for (int k = 0; k < n_bins; ++k) {
            double lower = (fftfreqs[k] - f_left) / (f_center - f_left);
            double upper = (f_right - fftfreqs[k]) / (f_right - f_center);
            double val = std::max(0.0, std::min(lower, upper));
            weights[m * n_bins + k] = static_cast<float>(val * enorm);
        }
    }
    return weights;
}

std::vector<float> extract_mel80_24k(
    const float* wav_24k,
    int64_t num_samples,
    int64_t& out_frames
) {
    // Match `mel_spectrogram` in s3gen.py:
    // n_fft=1920, num_mels=80, sr=24000, hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False
    // Pre-padded with reflect pad `(1920 - 480)/2 = 720` on both ends!
    static const std::vector<float> mel_basis = build_librosa_mel_filterbank(24000, 1920, 80, 0.0f, 8000.0f);
    static const std::vector<float> hann_1920 = hann_window_periodic(1920);
    const RealDftPlan& plan = get_dft_plan_1920();

    std::vector<float> padded = pad_reflect_1d(wav_24k, num_samples, 720, 720);
    out_frames = (static_cast<int64_t>(padded.size()) - 1920) / 480 + 1;
    std::vector<float> out_mel(static_cast<size_t>(out_frames * 80));

    std::vector<float> win_frame(1920);
    std::vector<float> mag_bins(961);

    for (int64_t t = 0; t < out_frames; ++t) {
        const float* src = padded.data() + t * 480;
        for (int i = 0; i < 1920; ++i) {
            win_frame[i] = src[i] * hann_1920[i];
        }
        plan.compute_power_or_mag(win_frame.data(), mag_bins.data(), /*power_spectrum=*/false);

        for (int m = 0; m < 80; ++m) {
            const float* f_row = mel_basis.data() + m * 961;
            float sum = 0.0f;
            for (int k = 0; k < 961; ++k) {
                sum += f_row[k] * mag_bins[k];
            }
            out_mel[t * 80 + m] = std::log(std::max(sum, 1e-5f));
        }
    }
    return out_mel;
}

std::vector<float> extract_s3tokenizer_mel128_16k(
    const float* wav_16k,
    int64_t num_samples,
    const float* mel_filters_128x201,
    const float* window_400,
    int64_t& out_frames
) {
    // Match `S3Tokenizer.log_mel_spectrogram` in s3gen.py:
    // n_fft=400, hop=160, center=True (reflect pad 200 on both ends), drops last frame `stft[..., :-1]`
    const RealDftPlan& plan = get_dft_plan_400();
    std::vector<float> padded = pad_reflect_1d(wav_16k, num_samples, 200, 200);
    const int64_t total_stft_frames = (static_cast<int64_t>(padded.size()) - 400) / 160 + 1;
    out_frames = total_stft_frames - 1; // stft[..., :-1]

    std::vector<float> log_spec(static_cast<size_t>(128 * out_frames));
    std::vector<float> win_frame(400);
    std::vector<float> pow_bins(201);

    float global_max = -1e30f;
    for (int64_t t = 0; t < out_frames; ++t) {
        const float* src = padded.data() + t * 160;
        for (int i = 0; i < 400; ++i) {
            win_frame[i] = src[i] * window_400[i];
        }
        plan.compute_power_or_mag(win_frame.data(), pow_bins.data(), /*power_spectrum=*/true);

        for (int m = 0; m < 128; ++m) {
            const float* f_row = mel_filters_128x201 + m * 201;
            float sum = 0.0f;
            for (int k = 0; k < 201; ++k) {
                sum += f_row[k] * pow_bins[k];
            }
            float ls = std::log10(std::max(sum, 1e-10f));
            log_spec[m * out_frames + t] = ls;
            if (ls > global_max) global_max = ls;
        }
    }

    const float floor_val = global_max - 8.0f;
    for (float& v : log_spec) {
        v = (std::max(v, floor_val) + 4.0f) * 0.25f;
    }
    return log_spec;
}

std::vector<float> extract_kaldi_fbank80_16k(
    const float* wav_16k,
    int64_t num_samples,
    int64_t& out_frames
) {
    // Match `torchaudio.compliance.kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)` + CMN:
    // frame_length=400 (25ms), frame_shift=160 (10ms), padded_window_size=512,
    // remove_dc_offset=True, preemphasis=0.97, window_type="povey", low_freq=20.0, high_freq=8000.0 (HTK mel)
    if (num_samples < 400) {
        out_frames = 0;
        return {};
    }
    out_frames = (num_samples - 400) / 160 + 1;

    // Precompute Povey window (`hann(400)^0.85`)
    static const std::vector<float> povey_win = []() {
        std::vector<float> w(400);
        for (int n = 0; n < 400; ++n) {
            double a = 2.0 * kPi * n / 399.0;
            w[n] = static_cast<float>(std::pow(0.5 - 0.5 * std::cos(a), 0.85));
        }
        return w;
    }();

    // Precompute 80-bin HTK Mel filterbank over 256 FFT bins (`n_fft=512`, `low_freq=20`, `high_freq=8000`)
    static const std::vector<float> kaldi_mel_bins = []() {
        auto hz2mel = [](double f) { return 1127.0 * std::log(1.0 + f / 700.0); };
        const double low_mel = hz2mel(20.0);
        const double high_mel = hz2mel(8000.0);
        std::vector<double> mel_pts(82);
        for (int i = 0; i < 82; ++i) {
            mel_pts[i] = low_mel + (high_mel - low_mel) * i / 81.0;
        }
        std::vector<float> fb(80 * 257, 0.0f);
        for (int m = 0; m < 80; ++m) {
            double left = mel_pts[m];
            double center = mel_pts[m + 1];
            double right = mel_pts[m + 2];
            for (int k = 0; k <= 256; ++k) {
                double freq = static_cast<double>(k) * 16000.0 / 512.0;
                double mel = hz2mel(freq);
                if (mel > left && mel < right) {
                    double w = (mel <= center) ? ((mel - left) / (center - left))
                                               : ((right - mel) / (right - center));
                    fb[m * 257 + k] = static_cast<float>(w);
                }
            }
        }
        return fb;
    }();

    std::vector<float> feats(static_cast<size_t>(out_frames * 80));
    std::vector<std::complex<float>> fft_buf(512);
    float frame_buf[400];

    for (int64_t t = 0; t < out_frames; ++t) {
        const float* src = wav_16k + t * 160;
        // 1. Remove DC offset
        float mean = 0.0f;
        for (int i = 0; i < 400; ++i) mean += src[i];
        mean /= 400.0f;
        for (int i = 0; i < 400; ++i) frame_buf[i] = src[i] - mean;

        // 2. Pre-emphasis (0.97) + Povey window
        std::fill(fft_buf.begin(), fft_buf.end(), std::complex<float>(0.0f, 0.0f));
        for (int i = 399; i >= 1; --i) {
            float pe = frame_buf[i] - 0.97f * frame_buf[i - 1];
            fft_buf[i] = std::complex<float>(pe * povey_win[i], 0.0f);
        }
        fft_buf[0] = std::complex<float>((frame_buf[0] - 0.97f * frame_buf[0]) * povey_win[0], 0.0f);

        // 3. 512-point radix-2 FFT -> power spectrum
        fft_radix2_inplace(fft_buf.data(), 512);
        float power[257];
        for (int k = 0; k <= 256; ++k) {
            power[k] = std::norm(fft_buf[k]);
        }

        // 4. Mel filterbank dot product + log
        float* dst_row = feats.data() + t * 80;
        for (int m = 0; m < 80; ++m) {
            const float* fb_row = kaldi_mel_bins.data() + m * 257;
            float energy = 0.0f;
            for (int k = 0; k <= 256; ++k) {
                energy += fb_row[k] * power[k];
            }
            dst_row[m] = std::log(std::max(energy, 1.1920929e-7f));
        }
    }

    // 5. Cepstral Mean Normalization (`feature = feature - feature.mean(dim=0, keepdim=True)`)
    for (int m = 0; m < 80; ++m) {
        double sum = 0.0;
        for (int64_t t = 0; t < out_frames; ++t) {
            sum += feats[t * 80 + m];
        }
        float col_mean = static_cast<float>(sum / static_cast<double>(out_frames));
        for (int64_t t = 0; t < out_frames; ++t) {
            feats[t * 80 + m] -= col_mean;
        }
    }

    return feats;
}

} // namespace kitteni
