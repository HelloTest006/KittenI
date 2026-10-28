#include "kernels/nn_ops.hpp"
#include "kernels/cpu_features.hpp"
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace kitteni {

namespace {

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
__attribute__((target("avx512f")))
void rms_norm_avx512(const float* x, const float* weight, int64_t dim, float eps, float* out) {
    __m512 vsum = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        vsum = _mm512_fmadd_ps(vx, vx, vsum);
    }
    float sum_sq = _mm512_reduce_add_ps(vsum);
    for (; i < dim; ++i) sum_sq += x[i] * x[i];

    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
    __m512 vscale = _mm512_set1_ps(inv_rms);
    i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vw = _mm512_loadu_ps(weight + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(vx, vscale), vw));
    }
    for (; i < dim; ++i) out[i] = (x[i] * inv_rms) * weight[i];
}
#endif

} // namespace

void rms_norm_f32(
    const float* x,
    const float* weight,
    int64_t dim,
    float eps,
    float* out
) {
    const SimdIsa isa = get_active_isa();
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
    if (isa == SimdIsa::AVX512_VNNI) {
        rms_norm_avx512(x, weight, dim, eps, out);
        return;
    }
#endif
    float sum_sq = 0.0f;
    int64_t i = 0;
#if defined(__AVX2__)
    if (isa != SimdIsa::SCALAR) {
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            vsum = _mm256_fmadd_ps(vx, vx, vsum);
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 1));
        sum_sq = _mm_cvtss_f32(s128);
    }
#endif
    for (; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }

    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

    i = 0;
#if defined(__AVX2__)
    if (isa != SimdIsa::SCALAR) {
        __m256 vscale = _mm256_set1_ps(inv_rms);
        for (; i + 8 <= dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw));
        }
    }
#endif
    for (; i < dim; ++i) {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}

void layer_norm_f32(
    const float* x,
    const float* weight,
    const float* bias,
    int64_t dim,
    float eps,
    float* out
) {
    double sum = 0.0;
    for (int64_t i = 0; i < dim; ++i) sum += x[i];
    float mean = static_cast<float>(sum / static_cast<double>(dim));

    double var_sum = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        double d = static_cast<double>(x[i]) - mean;
        var_sum += d * d;
    }
    float inv_std = 1.0f / std::sqrt(static_cast<float>(var_sum / static_cast<double>(dim)) + eps);

    for (int64_t i = 0; i < dim; ++i) {
        float norm = (x[i] - mean) * inv_std;
        float w = weight ? weight[i] : 1.0f;
        float b = bias ? bias[i] : 0.0f;
        out[i] = norm * w + b;
    }
}

void rope_neox_inplace(
    float* q_or_k,
    int64_t num_heads,
    int64_t head_dim,
    int64_t pos,
    float rope_theta
) {
    const int64_t half = head_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        double freq = std::pow(static_cast<double>(rope_theta), -static_cast<double>(2 * i) / static_cast<double>(head_dim));
        double angle = static_cast<double>(pos) * freq;
        float cos_val = static_cast<float>(std::cos(angle));
        float sin_val = static_cast<float>(std::sin(angle));

        for (int64_t h = 0; h < num_heads; ++h) {
            float* vec = q_or_k + h * head_dim;
            float x0 = vec[i];
            float x1 = vec[i + half];
            vec[i]        = x0 * cos_val - x1 * sin_val;
            vec[i + half] = x1 * cos_val + x0 * sin_val;
        }
    }
}

void swiglu_f32(
    const float* gate,
    const float* up,
    int64_t dim,
    float* out
) {
    for (int64_t i = 0; i < dim; ++i) {
        float g = gate[i];
        float silu_g = g / (1.0f + std::exp(-g));
        out[i] = silu_g * up[i];
    }
}

void dense_linear_f32(
    const float* x,
    const float* weight,
    const float* bias,
    int64_t in_dim,
    int64_t out_dim,
    float* out
) {
    for (int64_t o = 0; o < out_dim; ++o) {
        const float* w_row = weight + o * in_dim;
        float acc = bias ? bias[o] : 0.0f;
        int64_t i = 0;
#if defined(__AVX2__)
        __m256 vacc = _mm256_setzero_ps();
        for (; i + 8 <= in_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(w_row + i);
            vacc = _mm256_fmadd_ps(vx, vw, vacc);
        }
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 1));
        acc += _mm_cvtss_f32(s128);
#endif
        for (; i < in_dim; ++i) {
            acc += x[i] * w_row[i];
        }
        out[o] = acc;
    }
}

} // namespace kitteni
