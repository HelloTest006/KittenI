#include "cloning/sincnet_xvector.hpp"
#include "kernels/bitnet_ops.hpp"
#include "kernels/nn_ops.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace kitteni {

namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<float> load_f32_tensor(const SafetensorsFile& sf, const std::string& name) {
    TensorView tv = sf.get_tensor(name);
    size_t n = tv.numel();
    std::vector<float> out(n);
    std::memcpy(out.data(), tv.data, n * sizeof(float));
    return out;
}

// InstanceNorm1d(C, eps=1e-5, affine=True) over `[C, T]` (unbiased=False in PyTorch InstanceNorm1d!)
void instance_norm1d_inplace(
    float* x,
    int64_t C,
    int64_t T,
    const float* weight,
    const float* bias,
    float eps = 1e-5f
) {
    for (int64_t c = 0; c < C; ++c) {
        float* row = x + c * T;
        double sum = 0.0;
        for (int64_t t = 0; t < T; ++t) sum += row[t];
        float mean = static_cast<float>(sum / static_cast<double>(T));

        double sq_sum = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            double d = static_cast<double>(row[t]) - mean;
            sq_sum += d * d;
        }
        float inv_std = 1.0f / std::sqrt(static_cast<float>(sq_sum / static_cast<double>(T)) + eps);
        const float w = weight[c];
        const float b = bias[c];
        for (int64_t t = 0; t < T; ++t) {
            row[t] = (row[t] - mean) * inv_std * w + b;
        }
    }
}

// MaxPool1d(kernel_size=3, stride=3) over `[C, T_in]` -> `[C, T_in / 3]`
std::vector<float> max_pool1d_k3_s3(const float* x, int64_t C, int64_t T_in, int64_t& T_out) {
    T_out = T_in / 3;
    std::vector<float> out(static_cast<size_t>(C * T_out));
    for (int64_t c = 0; c < C; ++c) {
        const float* src = x + c * T_in;
        float* dst = out.data() + c * T_out;
        for (int64_t t = 0; t < T_out; ++t) {
            float v0 = src[t * 3 + 0];
            float v1 = src[t * 3 + 1];
            float v2 = src[t * 3 + 2];
            dst[t] = std::max({v0, v1, v2});
        }
    }
    return out;
}

} // namespace

void SincNetSpeakerEncoder::load(const std::string& safetensors_path) {
    SafetensorsFile sf(safetensors_path);

    wav_norm_w_ = load_f32_tensor(sf, "sincnet.wav_norm1d.weight").at(0);
    wav_norm_b_ = load_f32_tensor(sf, "sincnet.wav_norm1d.bias").at(0);

    // Build [80, 251] parameterized sinc filterbank from `low_hz_` [40, 1] and `band_hz_` [40, 1]
    std::vector<float> low_hz  = load_f32_tensor(sf, "sincnet.conv1d.0.filterbank.low_hz_");
    std::vector<float> band_hz = load_f32_tensor(sf, "sincnet.conv1d.0.filterbank.band_hz_");

    constexpr int n_filters = 80;
    constexpr int half_filters = 40;
    constexpr int kernel_size = 251;
    constexpr int half_k = 125;
    constexpr float sample_rate = 16000.0f;
    constexpr float min_low_hz = 50.0f;
    constexpr float min_band_hz = 50.0f;

    float window[half_k];
    float n_vec[half_k];
    for (int i = 0; i < half_k; ++i) {
        // Hamming window of length 251: `0.54 - 0.46 * cos(2 * pi * i / 250)`
        window[i] = static_cast<float>(0.54 - 0.46 * std::cos(2.0 * kPi * i / (kernel_size - 1)));
        float t = static_cast<float>(i - half_k) / sample_rate; // -125.. -1 / 16000
        n_vec[i] = static_cast<float>(2.0 * kPi) * t;
    }

    sinc_filters_.resize(static_cast<size_t>(n_filters * kernel_size));
    for (int f = 0; f < half_filters; ++f) {
        float low = min_low_hz + std::fabs(low_hz[f]);
        float high = std::clamp(low + min_band_hz + std::fabs(band_hz[f]), min_low_hz, sample_rate * 0.5f);
        float band = high - low;

        float* cos_filt = sinc_filters_.data() + f * kernel_size;
        float* sin_filt = sinc_filters_.data() + (f + half_filters) * kernel_size;

        for (int i = 0; i < half_k; ++i) {
            float ft_low  = low * n_vec[i];
            float ft_high = high * n_vec[i];
            float half_n  = n_vec[i] * 0.5f;

            float left_cos = ((std::sin(ft_high) - std::sin(ft_low)) / half_n) * window[i] / (2.0f * band);
            float left_sin = ((std::cos(ft_low) - std::cos(ft_high)) / half_n) * window[i] / (2.0f * band);

            cos_filt[i] = left_cos;
            cos_filt[kernel_size - 1 - i] = left_cos;

            sin_filt[i] = left_sin;
            sin_filt[kernel_size - 1 - i] = -left_sin;
        }
        cos_filt[half_k] = 1.0f; // (2 * band) / (2 * band)
        sin_filt[half_k] = 0.0f;
    }

    for (int i = 0; i < 2; ++i) {
        std::string idx = std::to_string(i + 1);
        sinc_conv_w_[i] = load_f32_tensor(sf, "sincnet.conv1d." + idx + ".weight");
        sinc_conv_b_[i] = load_f32_tensor(sf, "sincnet.conv1d." + idx + ".bias");
    }
    for (int i = 0; i < 3; ++i) {
        std::string idx = std::to_string(i);
        sinc_inorm_w_[i] = load_f32_tensor(sf, "sincnet.norm1d." + idx + ".weight");
        sinc_inorm_b_[i] = load_f32_tensor(sf, "sincnet.norm1d." + idx + ".bias");
    }

    const int64_t in_chs[5]  = {60, 512, 512, 512, 512};
    const int64_t out_chs[5] = {512, 512, 512, 512, 1500};
    const int64_t k_sizes[5] = {5, 3, 3, 1, 1};
    const int64_t dils[5]    = {1, 2, 3, 1, 1};
    const int conv_idx[5]    = {0, 3, 6, 9, 12};
    const int bn_idx[5]      = {2, 5, 8, 11, 14};

    for (int i = 0; i < 5; ++i) {
        TdnnLayer& td = tdnns_[i];
        td.in_ch = in_chs[i];
        td.out_ch = out_chs[i];
        td.kernel_size = k_sizes[i];
        td.dilation = dils[i];

        std::string cp = "tdnns." + std::to_string(conv_idx[i]) + ".";
        std::string bp = "tdnns." + std::to_string(bn_idx[i]) + ".";

        td.conv_w = load_f32_tensor(sf, cp + "weight");
        td.conv_b = load_f32_tensor(sf, cp + "bias");

        auto gamma = load_f32_tensor(sf, bp + "weight");
        auto beta  = load_f32_tensor(sf, bp + "bias");
        auto mean  = load_f32_tensor(sf, bp + "running_mean");
        auto var   = load_f32_tensor(sf, bp + "running_var");

        td.bn_scale.resize(static_cast<size_t>(td.out_ch));
        td.bn_shift.resize(static_cast<size_t>(td.out_ch));
        for (int64_t c = 0; c < td.out_ch; ++c) {
            float sc = gamma[c] / std::sqrt(var[c] + 1e-5f);
            td.bn_scale[c] = sc;
            td.bn_shift[c] = beta[c] - mean[c] * sc;
        }
    }

    emb_linear_w_ = load_f32_tensor(sf, "embedding.weight");
    emb_linear_b_ = load_f32_tensor(sf, "embedding.bias");
}

std::vector<float> SincNetSpeakerEncoder::encode_16k(
    const float* wav_16k,
    int64_t num_samples,
    const ThreadPool* pool
) const {
    if (num_samples < 1000) {
        throw std::invalid_argument("Reference audio too short for SincNetSpeakerEncoder");
    }

    // 1. `wav_norm1d`: InstanceNorm1d(1, affine=True)
    std::vector<float> cur(wav_16k, wav_16k + num_samples);
    instance_norm1d_inplace(cur.data(), 1, num_samples, &wav_norm_w_, &wav_norm_b_, 1e-5f);
    int64_t T = num_samples;

    // 2. SincNet stage 0: Conv1d(1 -> 80, k=251, stride=10) -> abs -> MaxPool1d(3) -> InstanceNorm1d(80) -> LeakyReLU(0.01)
    int64_t T_conv0 = (T - 251) / 10 + 1;
    std::vector<float> conv_out(static_cast<size_t>(80 * T_conv0));
    conv1d_f32(cur.data(), sinc_filters_.data(), nullptr, 1, 80, T, 251, 10, 0, 1, conv_out.data(), pool);
    for (float& v : conv_out) v = std::fabs(v);

    int64_t T_pool = 0;
    cur = max_pool1d_k3_s3(conv_out.data(), 80, T_conv0, T_pool);
    instance_norm1d_inplace(cur.data(), 80, T_pool, sinc_inorm_w_[0].data(), sinc_inorm_b_[0].data(), 1e-5f);
    for (float& v : cur) v = (v >= 0.0f) ? v : (0.01f * v);
    T = T_pool;

    // 3. SincNet stages 1 and 2: Conv1d(in, 60, k=5, stride=1) -> MaxPool1d(3) -> InstanceNorm1d(60) -> LeakyReLU(0.01)
    const int64_t stage_in[2] = {80, 60};
    for (int st = 0; st < 2; ++st) {
        int64_t T_c = T - 5 + 1;
        conv_out.resize(static_cast<size_t>(60 * T_c));
        conv1d_f32(cur.data(), sinc_conv_w_[st].data(), sinc_conv_b_[st].data(),
                   stage_in[st], 60, T, 5, 1, 0, 1, conv_out.data(), pool);
        cur = max_pool1d_k3_s3(conv_out.data(), 60, T_c, T_pool);
        instance_norm1d_inplace(cur.data(), 60, T_pool, sinc_inorm_w_[st + 1].data(), sinc_inorm_b_[st + 1].data(), 1e-5f);
        for (float& v : cur) v = (v >= 0.0f) ? v : (0.01f * v);
        T = T_pool;
    }

    // 4. Five TDNN blocks: Conv1d(in, out, k, dilation=d) -> LeakyReLU(0.01) -> BatchNorm1d
    for (int i = 0; i < 5; ++i) {
        const TdnnLayer& td = tdnns_[i];
        int64_t T_next = T - td.dilation * (td.kernel_size - 1);
        std::vector<float> next_buf(static_cast<size_t>(td.out_ch * T_next));
        conv1d_f32(cur.data(), td.conv_w.data(), td.conv_b.data(),
                   td.in_ch, td.out_ch, T, td.kernel_size, 1, 0, td.dilation, next_buf.data(), pool);

        for (int64_t c = 0; c < td.out_ch; ++c) {
            const float sc = td.bn_scale[c];
            const float sh = td.bn_shift[c];
            float* row = next_buf.data() + c * T_next;
            for (int64_t t = 0; t < T_next; ++t) {
                float v = row[t];
                float lrelu = (v >= 0.0f) ? v : (0.01f * v);
                row[t] = lrelu * sc + sh;
            }
        }
        cur.swap(next_buf);
        T = T_next;
    }

    // 5. Statistics pooling over `T` for each of the 1500 channels:
    //    `pooled[0..1499] = mean`, `pooled[1500..2999] = std(correction=1)`
    std::vector<float> pooled(3000);
    for (int64_t c = 0; c < 1500; ++c) {
        const float* row = cur.data() + c * T;
        double sum = 0.0;
        for (int64_t t = 0; t < T; ++t) sum += row[t];
        double mean = sum / static_cast<double>(T);

        double sq_sum = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            double d = static_cast<double>(row[t]) - mean;
            sq_sum += d * d;
        }
        double std_unbiased = (T > 1) ? std::sqrt(sq_sum / static_cast<double>(T - 1)) : 0.0;
        pooled[c]        = static_cast<float>(mean);
        pooled[c + 1500] = static_cast<float>(std_unbiased);
    }

    // 6. Linear(3000 -> 512) + L2 unit normalization
    std::vector<float> emb(512);
    dense_linear_f32(pooled.data(), emb_linear_w_.data(), emb_linear_b_.data(), 3000, 512, emb.data());

    double norm_sq = 0.0;
    for (float v : emb) norm_sq += static_cast<double>(v) * v;
    float norm = static_cast<float>(std::sqrt(norm_sq));
    if (norm > 0.0f) {
        const float inv = 1.0f / norm;
        for (float& v : emb) v *= inv;
    }
    return emb;
}

} // namespace kitteni
