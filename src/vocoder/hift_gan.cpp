#include "vocoder/hift_gan.hpp"
#include "kernels/dsp_fft.hpp"
#include "kernels/nn_ops.hpp"
#include "kernels/cpu_features.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace kitteni {

namespace {
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kPi    = 3.14159265358979323846f;

inline float elu_scalar(float x) {
    return (x >= 0.0f) ? x : (std::exp(x) - 1.0f);
}

inline float leaky_relu_scalar(float x, float slope) {
    return (x >= 0.0f) ? x : (slope * x);
}

#if defined(__AVX2__)
// Fast degree-5 minimax polynomial sin(x) over [-pi, pi] in AVX2 (~1e-6 accuracy, 8x faster than scalar std::sin!)
inline __m256 fast_sin_avx2(__m256 x) {
    const __m256 inv_two_pi = _mm256_set1_ps(0.15915494309189533577f);
    const __m256 two_pi     = _mm256_set1_ps(6.28318530717958647692f);
    const __m256 half_pi    = _mm256_set1_ps(1.57079632679489661923f);
    const __m256 pi         = _mm256_set1_ps(3.14159265358979323846f);
    const __m256 sign_mask  = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    // Range reduce x to [-pi, pi]
    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, inv_two_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(k, two_pi, x);

    // Fold [-pi, pi] into [-pi/2, pi/2]: if |x| > pi/2, x = copysign(pi, x) - x
    __m256 abs_x = _mm256_and_ps(x, sign_mask);
    __m256 gt = _mm256_cmp_ps(abs_x, half_pi, _CMP_GT_OS);
    __m256 sign_x = _mm256_andnot_ps(sign_mask, x);
    __m256 folded = _mm256_sub_ps(_mm256_or_ps(pi, sign_x), x);
    x = _mm256_blendv_ps(x, folded, gt);

    __m256 x2 = _mm256_mul_ps(x, x);
    const __m256 c7 = _mm256_set1_ps(-0.0001984126984f);
    const __m256 c5 = _mm256_set1_ps(0.0083333333333f);
    const __m256 c3 = _mm256_set1_ps(-0.1666666666667f);
    __m256 p = _mm256_fmadd_ps(c7, x2, c5);
    p = _mm256_fmadd_ps(p, x2, c3);
    p = _mm256_mul_ps(p, x2);
    return _mm256_fmadd_ps(p, x, x);
}
#endif

// Snake activation: `x + inv_beta[c] * sin(alpha[c] * x)^2` (parallelized + AVX2 fast_sin)
void snake_1d_inplace(
    float* x,
    const float* alpha,
    const float* inv_beta,
    int64_t channels,
    int64_t T,
    const ThreadPool* pool
) {
    const SimdIsa isa = get_active_isa();
    auto compute_ch = [&](int64_t c_begin, int64_t c_end, int /*tid*/) {
        for (int64_t c = c_begin; c < c_end; ++c) {
            const float a = alpha[c];
            const float ib = inv_beta[c];
            float* row = x + c * T;
            int64_t t = 0;
#if defined(__AVX2__)
            if (isa != SimdIsa::SCALAR) {
                __m256 va = _mm256_set1_ps(a);
                __m256 vib = _mm256_set1_ps(ib);
                for (; t + 8 <= T; t += 8) {
                    __m256 vx = _mm256_loadu_ps(row + t);
                    __m256 vs = fast_sin_avx2(_mm256_mul_ps(vx, va));
                    __m256 vs2 = _mm256_mul_ps(vs, vs);
                    _mm256_storeu_ps(row + t, _mm256_fmadd_ps(vib, vs2, vx));
                }
            }
#endif
            for (; t < T; ++t) {
                float s = std::sin(row[t] * a);
                row[t] += ib * (s * s);
            }
        }
    };

    if (pool && channels >= 16) {
        pool->parallel_for(channels, compute_ch);
    } else {
        compute_ch(0, channels, 0);
    }
}

} // namespace

void HiFTGenerator::load_from_torchscript_zip(const TorchScriptZipConstants& zip) {
    // 1. ConvRNNF0Predictor: constants 1115..1126
    for (int i = 0; i < 5; ++i) {
        f0_conv_w_[i] = zip.get_f32_vec(1115 + i * 2);
        f0_conv_b_[i] = zip.get_f32_vec(1116 + i * 2);
    }
    f0_linear_w_ = zip.get_f32_vec(1125);
    f0_linear_b_ = zip.get_f32_vec(1126).at(0);

    // 2. SourceModuleHnNSF: constants 1140, 1141, 1142
    source_linear_w_ = zip.get_f32_vec(1140);
    source_linear_b_ = zip.get_f32_vec(1141).at(0);
    stft_window_16_  = zip.get_f32_vec(1142);

    // 3. conv_pre: constants 1143, 1144
    conv_pre_w_ = zip.get_f32_vec(1143);
    conv_pre_b_ = zip.get_f32_vec(1144);

    // Helper to load a SnakeResBlock (24 consecutive constants)
    auto load_resblock = [&](int& cursor, int64_t ch, int64_t k_sz, SnakeResBlock& rb) {
        rb.channels = ch;
        rb.kernel_size = k_sz;
        rb.dilations[0] = 1;
        rb.dilations[1] = 3;
        rb.dilations[2] = 5;

        for (int s = 0; s < 3; ++s) {
            rb.alpha1[s]    = zip.get_f32_vec(cursor + 0);
            rb.inv_beta1[s] = zip.get_f32_vec(cursor + 1);
            rb.conv1_w[s]   = zip.get_f32_vec(cursor + 2);
            rb.conv1_b[s]   = zip.get_f32_vec(cursor + 3);
            rb.alpha2[s]    = zip.get_f32_vec(cursor + 4);
            rb.inv_beta2[s] = zip.get_f32_vec(cursor + 5);
            rb.conv2_w[s]   = zip.get_f32_vec(cursor + 6);
            rb.conv2_b[s]   = zip.get_f32_vec(cursor + 7);
            cursor += 8;
        }
    };

    const int64_t stage_channels[3] = {256, 128, 64};
    const int64_t source_rb_kernels[3] = {7, 7, 11};
    const int64_t rb_kernels[3] = {3, 7, 11};

    int cursor = 1145;
    for (int st = 0; st < 3; ++st) {
        ups_w_[st] = zip.get_f32_vec(cursor + 0);
        ups_b_[st] = zip.get_f32_vec(cursor + 1);
        source_downs_w_[st] = zip.get_f32_vec(cursor + 2);
        source_downs_b_[st] = zip.get_f32_vec(cursor + 3);
        cursor += 4;

        load_resblock(cursor, stage_channels[st], source_rb_kernels[st], source_resblocks_[st]);
        for (int j = 0; j < 3; ++j) {
            load_resblock(cursor, stage_channels[st], rb_kernels[j], resblocks_[st][j]);
        }
    }

    if (cursor != 1445) {
        throw std::runtime_error("HiFTGenerator constant cursor mismatch: expected 1445, got " + std::to_string(cursor));
    }

    conv_post_w_   = zip.get_f32_vec(1445);
    conv_post_b_   = zip.get_f32_vec(1446);
    trim_fade_960_ = zip.get_f32_vec(1447);
}

void HiFTGenerator::run_snake_resblock(
    const SnakeResBlock& rb,
    float* x,
    int64_t T,
    const ThreadPool* pool
) const {
    const int64_t C = rb.channels;
    const int64_t K = rb.kernel_size;

    std::vector<float> xt(static_cast<size_t>(C * T));
    std::vector<float> conv_out(static_cast<size_t>(C * T));

    for (int s = 0; s < 3; ++s) {
        const int64_t d1 = rb.dilations[s];
        const int64_t pad1 = (K * d1 - d1) / 2;
        const int64_t pad2 = (K - 1) / 2;

        std::memcpy(xt.data(), x, static_cast<size_t>(C * T) * sizeof(float));
        snake_1d_inplace(xt.data(), rb.alpha1[s].data(), rb.inv_beta1[s].data(), C, T, pool);
        conv1d_f32(xt.data(), rb.conv1_w[s].data(), rb.conv1_b[s].data(), C, C, T, K, 1, pad1, d1, conv_out.data(), pool);

        snake_1d_inplace(conv_out.data(), rb.alpha2[s].data(), rb.inv_beta2[s].data(), C, T, pool);
        conv1d_f32(conv_out.data(), rb.conv2_w[s].data(), rb.conv2_b[s].data(), C, C, T, K, 1, pad2, 1, xt.data(), pool);

        for (int64_t i = 0; i < C * T; ++i) {
            x[i] += xt[i];
        }
    }
}

std::vector<float> HiFTGenerator::forward(
    const float* mel_80xT,
    int64_t T_mel,
    std::mt19937& rng,
    const ThreadPool* pool,
    bool deterministic_source
) const {
    // 1. ConvRNNF0Predictor: 5 layers of Conv1d(in, 512, k=3, pad=1) + ELU -> Linear(512, 1) -> abs
    std::vector<float> f0_cur(static_cast<size_t>(512 * T_mel));
    std::vector<float> f0_next(static_cast<size_t>(512 * T_mel));

    conv1d_f32(mel_80xT, f0_conv_w_[0].data(), f0_conv_b_[0].data(), 80, 512, T_mel, 3, 1, 1, 1, f0_cur.data(), pool);
    for (float& v : f0_cur) v = elu_scalar(v);

    for (int i = 1; i < 5; ++i) {
        conv1d_f32(f0_cur.data(), f0_conv_w_[i].data(), f0_conv_b_[i].data(), 512, 512, T_mel, 3, 1, 1, 1, f0_next.data(), pool);
        for (float& v : f0_next) v = elu_scalar(v);
        f0_cur.swap(f0_next);
    }

    std::vector<float> f0(static_cast<size_t>(T_mel));
    for (int64_t t = 0; t < T_mel; ++t) {
        float acc = f0_linear_b_;
        for (int c = 0; c < 512; ++c) {
            acc += f0_cur[c * T_mel + t] * f0_linear_w_[c];
        }
        f0[t] = std::fabs(acc);
    }

    // 2. SourceModuleHnNSF:
    //    Upsample F0 by 480x (`8 * 5 * 3 * 4`) -> `T_wav = T_mel * 480`
    const int64_t T_wav = T_mel * 480;
    float phase_rand[9] = {0.0f};
    if (!deterministic_source) {
        std::uniform_real_distribution<float> udist(-kPi, kPi);
        for (int h = 1; h < 9; ++h) phase_rand[h] = udist(rng);
    }

    std::normal_distribution<float> ndist(0.0f, 1.0f);
    double cum_phase[9] = {0.0};
    std::vector<float> s_source(static_cast<size_t>(T_wav));

    for (int64_t t = 0; t < T_wav; ++t) {
        const float f0_val = f0[t / 480];
        const bool voiced = (f0_val > 10.0f);
        const float uv = voiced ? 1.0f : 0.0f;
        const float noise_amp = uv * 0.003f + (1.0f - uv) * (0.1f / 3.0f);

        float acc = source_linear_b_;
        const double base_inc = static_cast<double>(f0_val) / 24000.0;
        for (int h = 0; h < 9; ++h) {
            cum_phase[h] = std::fmod(cum_phase[h] + base_inc * (h + 1), 1.0);
            float theta = static_cast<float>(cum_phase[h]) * kTwoPi + phase_rand[h];
            float sine_w = 0.1f * std::sin(theta);
            float n_val = deterministic_source ? 0.0f : (noise_amp * ndist(rng));
            float h_val = sine_w * uv + n_val;
            acc += h_val * source_linear_w_[h];
        }
        s_source[t] = std::tanh(acc);
    }

    // 3. 16-point STFT of `s_source` -> `s_stft` (`[18, T_stft]` where `T_stft = T_wav / 4 + 1 = T_mel * 120 + 1`)
    std::vector<float> s_stft;
    hift_stft_16(s_source.data(), T_wav, stft_window_16_.data(), s_stft);
    const int64_t T_stft = T_wav / 4 + 1;

    // 4. `conv_pre`: Conv1d(80, 512, k=7, pad=3) -> `[512, T_mel]`
    std::vector<float> x(static_cast<size_t>(512 * T_mel));
    conv1d_f32(mel_80xT, conv_pre_w_.data(), conv_pre_b_.data(), 80, 512, T_mel, 7, 1, 3, 1, x.data(), pool);

    // 5. Three upsample + source-fusion + multi-kernel ResBlock stages:
    //    Stage 0: 512 -> 256, rate=8, k=16, pad=4 => T0 = T_mel * 8
    //    Stage 1: 256 -> 128, rate=5, k=11, pad=3 => T1 = T_mel * 40
    //    Stage 2: 128 -> 64,  rate=3, k=7,  pad=2 (+ reflect_pad(1, 0)) => T2 = T_mel * 120 + 1 == T_stft!
    const int64_t in_ch_stage[3]  = {512, 256, 128};
    const int64_t out_ch_stage[3] = {256, 128, 64};
    const int64_t up_rates[3]     = {8, 5, 3};
    const int64_t up_kernels[3]   = {16, 11, 7};
    const int64_t up_pads[3]      = {4, 3, 2};

    const int64_t sdown_kernels[3] = {30, 6, 1};
    const int64_t sdown_strides[3] = {15, 3, 1};
    const int64_t sdown_pads[3]    = {7, 1, 0};

    int64_t T_cur = T_mel;
    for (int st = 0; st < 3; ++st) {
        const int64_t C_in = in_ch_stage[st];
        const int64_t C_out = out_ch_stage[st];

        for (float& v : x) v = leaky_relu_scalar(v, 0.1f);

        int64_t T_up = (T_cur - 1) * up_rates[st] - 2 * up_pads[st] + up_kernels[st];
        std::vector<float> x_up(static_cast<size_t>(C_out * T_up));
        conv_transpose1d_f32(
            x.data(), ups_w_[st].data(), ups_b_[st].data(),
            C_in, C_out, T_cur,
            up_kernels[st], up_rates[st], up_pads[st],
            x_up.data(), pool
        );

        // On final upsample stage (st == 2), apply `ReflectionPad1d((1, 0))` along time!
        // `T_up = (T_cur - 1) * 3 - 4 + 7 = 3 * T_cur = 120 * T_mel`.
        // Reflect-padding 1 sample on the left makes `T_up = 120 * T_mel + 1 == T_stft`!
        if (st == 2) {
            std::vector<float> x_padded(static_cast<size_t>(C_out * (T_up + 1)));
            for (int64_t c = 0; c < C_out; ++c) {
                const float* src = x_up.data() + c * T_up;
                float* dst = x_padded.data() + c * (T_up + 1);
                dst[0] = src[1]; // 1-sample left reflection pad
                std::memcpy(dst + 1, src, static_cast<size_t>(T_up) * sizeof(float));
            }
            x_up.swap(x_padded);
            T_up += 1;
        }

        // Source fusion: `si = source_resblocks[st](source_downs[st](s_stft))`
        std::vector<float> si(static_cast<size_t>(C_out * T_up));
        conv1d_f32(
            s_stft.data(), source_downs_w_[st].data(), source_downs_b_[st].data(),
            18, C_out, T_stft,
            sdown_kernels[st], sdown_strides[st], sdown_pads[st], 1,
            si.data(), pool
        );
        run_snake_resblock(source_resblocks_[st], si.data(), T_up, pool);

        for (int64_t i = 0; i < C_out * T_up; ++i) {
            x_up[i] += si[i];
        }

        // Average of 3 multi-receptive-field SnakeResBlocks (kernels 3, 7, 11)
        std::vector<float> xs_acc(static_cast<size_t>(C_out * T_up), 0.0f);
        std::vector<float> rb_in(static_cast<size_t>(C_out * T_up));
        for (int j = 0; j < 3; ++j) {
            std::memcpy(rb_in.data(), x_up.data(), static_cast<size_t>(C_out * T_up) * sizeof(float));
            run_snake_resblock(resblocks_[st][j], rb_in.data(), T_up, pool);
            for (int64_t i = 0; i < C_out * T_up; ++i) {
                xs_acc[i] += rb_in[i];
            }
        }
        const float inv3 = 1.0f / 3.0f;
        for (int64_t i = 0; i < C_out * T_up; ++i) {
            xs_acc[i] *= inv3;
        }

        x.swap(xs_acc);
        T_cur = T_up;
    }

    // 6. Final leaky_relu (default negative_slope = 0.01 in PyTorch F.leaky_relu(x)!) + conv_post(64, 18, 7, pad=3)
    for (float& v : x) v = leaky_relu_scalar(v, 0.01f);
    std::vector<float> post_out(static_cast<size_t>(18 * T_cur));
    conv1d_f32(x.data(), conv_post_w_.data(), conv_post_b_.data(), 64, 18, T_cur, 7, 1, 3, 1, post_out.data(), pool);

    // 7. Split into magnitude (0..8) and phase (9..17), compute real & imag, and run 16-point ISTFT
    std::vector<float> real_9xT(static_cast<size_t>(9 * T_cur));
    std::vector<float> imag_9xT(static_cast<size_t>(9 * T_cur));
    for (int k = 0; k < 9; ++k) {
        const float* mag_row = post_out.data() + k * T_cur;
        const float* phs_row = post_out.data() + (k + 9) * T_cur;
        float* re_row = real_9xT.data() + k * T_cur;
        float* im_row = imag_9xT.data() + k * T_cur;
        for (int64_t t = 0; t < T_cur; ++t) {
            float mag = std::min(100.0f, std::exp(mag_row[t]));
            float phs = std::sin(phs_row[t]);
            re_row[t] = mag * std::cos(phs);
            im_row[t] = mag * std::sin(phs);
        }
    }

    std::vector<float> wav;
    hift_istft_16(real_9xT.data(), imag_9xT.data(), T_cur, stft_window_16_.data(), wav);

    // 8. Clamp to [-0.99, 0.99] and apply 960-sample (40ms) `trim_fade`
    for (float& s : wav) {
        s = std::clamp(s, -0.99f, 0.99f);
    }
    const size_t fade_len = std::min(wav.size(), trim_fade_960_.size());
    for (size_t i = 0; i < fade_len; ++i) {
        wav[i] *= trim_fade_960_[i];
    }

    return wav;
}

} // namespace kitteni
