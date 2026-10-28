#include "kernels/ternary_gemv.hpp"
#include "kernels/cpu_features.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace kitteni {

void QuantizedActivation::quantize_from(const float* x, int64_t num_rows, int64_t in_dim) {
    if (in_dim % 128 != 0) {
        throw std::invalid_argument("QuantizedActivation requires in_dim divisible by 128");
    }
    rows = num_rows;
    in_features = in_dim;
    const int64_t groups_per_row = in_dim / 128;

    u8_data.resize(static_cast<size_t>(num_rows * in_dim));
    scales.resize(static_cast<size_t>(num_rows * groups_per_row));

    const SimdIsa isa = get_active_isa();

    for (int64_t r = 0; r < num_rows; ++r) {
        for (int64_t g = 0; g < groups_per_row; ++g) {
            const int64_t group_idx = r * groups_per_row + g;
            const float* gx = x + group_idx * 128;
            uint8_t* gu = u8_data.data() + group_idx * 128;

            float max_abs = 0.0f;
#if defined(__AVX2__)
            if (isa != SimdIsa::SCALAR) {
                __m256 vmax = _mm256_setzero_ps();
                const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
                for (int i = 0; i < 128; i += 8) {
                    __m256 v = _mm256_loadu_ps(gx + i);
                    vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, sign_mask));
                }
                __m128 hi = _mm256_extractf128_ps(vmax, 1);
                __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), hi);
                m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
                m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
                max_abs = _mm_cvtss_f32(m128);
            } else
#endif
            {
                for (int i = 0; i < 128; ++i) {
                    max_abs = std::max(max_abs, std::fabs(gx[i]));
                }
            }

            if (max_abs < 1e-12f) {
                scales[group_idx] = 0.0f;
                std::memset(gu, 128, 128);
                continue;
            }
            const float scale = max_abs / 127.0f;
            const float inv_scale = 127.0f / max_abs;
            scales[group_idx] = scale;

#if defined(__AVX2__)
            if (isa != SimdIsa::SCALAR) {
                __m256 vinv = _mm256_set1_ps(inv_scale);
                __m256 vbias = _mm256_set1_ps(128.0f);
                __m256 vmin = _mm256_set1_ps(1.0f);
                __m256 vmax_q = _mm256_set1_ps(255.0f);
                for (int i = 0; i < 128; i += 8) {
                    __m256 v = _mm256_loadu_ps(gx + i);
                    __m256 vq = _mm256_round_ps(_mm256_mul_ps(v, vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    vq = _mm256_add_ps(vq, vbias);
                    vq = _mm256_max_ps(vmin, _mm256_min_ps(vmax_q, vq));
                    __m256i vi = _mm256_cvtps_epi32(vq);
                    alignas(32) int32_t tmp[8];
                    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vi);
                    for (int k = 0; k < 8; ++k) {
                        gu[i + k] = static_cast<uint8_t>(tmp[k]);
                    }
                }
                continue;
            }
#endif
            for (int i = 0; i < 128; ++i) {
                int q = static_cast<int>(std::nearbyintf(gx[i] * inv_scale)) + 128;
                gu[i] = static_cast<uint8_t>(std::clamp(q, 1, 255));
            }
        }
    }
}

namespace {

inline int32_t dot_group128_u8_scalar(const uint8_t* g_packed32, const uint8_t* gu128) {
    int32_t acc = 0;
    for (int m = 0; m < 32; ++m) {
        uint8_t b = g_packed32[m];
        int32_t t0 = static_cast<int32_t>(b & 0x03U) - 1;
        int32_t t1 = static_cast<int32_t>((b >> 2) & 0x03U) - 1;
        int32_t t2 = static_cast<int32_t>((b >> 4) & 0x03U) - 1;
        int32_t t3 = static_cast<int32_t>((b >> 6) & 0x03U) - 1;

        acc += t0 * static_cast<int32_t>(gu128[m]);
        acc += t1 * static_cast<int32_t>(gu128[m + 32]);
        acc += t2 * static_cast<int32_t>(gu128[m + 64]);
        acc += t3 * static_cast<int32_t>(gu128[m + 96]);
    }
    return acc;
}

inline float dot_group128_f32_scalar(const uint8_t* g_packed32, const float* gx128) {
    float acc = 0.0f;
    for (int m = 0; m < 32; ++m) {
        uint8_t b = g_packed32[m];
        float t0 = static_cast<float>(static_cast<int>(b & 0x03U) - 1);
        float t1 = static_cast<float>(static_cast<int>((b >> 2) & 0x03U) - 1);
        float t2 = static_cast<float>(static_cast<int>((b >> 4) & 0x03U) - 1);
        float t3 = static_cast<float>(static_cast<int>((b >> 6) & 0x03U) - 1);

        acc += t0 * gx128[m] + t1 * gx128[m + 32] + t2 * gx128[m + 64] + t3 * gx128[m + 96];
    }
    return acc;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum128);
}

inline float hsum_ps_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    return _mm_cvtss_f32(sum128);
}

// -----------------------------------------------------------------------------
// Path 1: AVX2 (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`)
// -----------------------------------------------------------------------------
inline int32_t dot_group128_u8_avx2(const uint8_t* g_packed32, const uint8_t* gu128) {
    const __m256i mask2 = _mm256_set1_epi8(0x03);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g_packed32));

    __m256i w0 = _mm256_sub_epi8(_mm256_and_si256(packed, mask2), one8);
    __m256i w1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 2), mask2), one8);
    __m256i w2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 4), mask2), one8);
    __m256i w3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 6), mask2), one8);

    __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu128 + 0));
    __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu128 + 32));
    __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu128 + 64));
    __m256i x3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu128 + 96));

    __m256i p0 = _mm256_maddubs_epi16(x0, w0);
    __m256i p1 = _mm256_maddubs_epi16(x1, w1);
    __m256i p2 = _mm256_maddubs_epi16(x2, w2);
    __m256i p3 = _mm256_maddubs_epi16(x3, w3);

    __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3));
    __m256i sum32 = _mm256_madd_epi16(sum16, ones16);
    return hsum_epi32_avx2(sum32);
}

// -----------------------------------------------------------------------------
// Path 2: AVX_VNNI / AVX512VL+VNNI 256-bit (`vpdpbusd` over `__m256i`)
// Note: We use `avx512vl,avx512vnni` or `avxvnni` target attribute so GCC/Clang
// emits `vpdpbusd ymm, ymm, ymm`, accumulating 4 groups of 32 bytes directly into int32!
// -----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,avx512vl,avx512vnni")))
inline void gemv_range_avx_vnni(
    const TernaryMatrix& W,
    const QuantizedActivation& X_q,
    int64_t col_begin,
    int64_t col_end,
    float* Y
) {
    const int64_t groups_per_row = W.num_groups_per_row();
    const uint8_t* x_row_u8 = X_q.u8_data.data();
    const float* x_row_scales = X_q.scales.data();
    const __m256i mask2 = _mm256_set1_epi8(0x03);
    const __m256i one8 = _mm256_set1_epi8(1);

    for (int64_t out_col = col_begin; out_col < col_end; ++out_col) {
        const uint8_t* w_row_packed = W.packed_2bit.data() + out_col * groups_per_row * 32;
        const float* w_row_scales = W.scales.data() + out_col * groups_per_row;
        const int16_t* w_row_sums = W.group_sums.data() + out_col * groups_per_row;
        float total = 0.0f;

        for (int64_t g = 0; g < groups_per_row; ++g) {
            const float sx = x_row_scales[g];
            if (sx == 0.0f) continue;
            const uint8_t* gp = w_row_packed + g * 32;
            const uint8_t* gu = x_row_u8 + g * 128;

            __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gp));
            __m256i w0 = _mm256_sub_epi8(_mm256_and_si256(packed, mask2), one8);
            __m256i w1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 2), mask2), one8);
            __m256i w2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 4), mask2), one8);
            __m256i w3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 6), mask2), one8);

            __m256i acc = _mm256_setzero_si256();
            acc = _mm256_dpbusd_epi32(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 0)), w0);
            acc = _mm256_dpbusd_epi32(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 32)), w1);
            acc = _mm256_dpbusd_epi32(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 64)), w2);
            acc = _mm256_dpbusd_epi32(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 96)), w3);

            int32_t raw_dot = hsum_epi32_avx2(acc);
            int32_t unbiased_dot = raw_dot - 128 * static_cast<int32_t>(w_row_sums[g]);
            total += static_cast<float>(unbiased_dot) * (sx * w_row_scales[g]);
        }
        Y[out_col] = total;
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline void gemv_range_avx512_vnni(
    const TernaryMatrix& W,
    const QuantizedActivation& X_q,
    int64_t col_begin,
    int64_t col_end,
    float* Y
) {
    const int64_t groups_per_row = W.num_groups_per_row();
    const uint8_t* x_row_u8 = X_q.u8_data.data();
    const float* x_row_scales = X_q.scales.data();
    const __m512i mask2 = _mm512_set1_epi8(0x03);
    const __m512i one8  = _mm512_set1_epi8(1);

    // Process 2 output columns (64 packed weight bytes = 1 cache line = 1 __m512i load!) per group!
    int64_t out_col = col_begin;
    for (; out_col + 2 <= col_end; out_col += 2) {
        const uint8_t* w0_packed = W.packed_2bit.data() + (out_col + 0) * groups_per_row * 32;
        const uint8_t* w1_packed = W.packed_2bit.data() + (out_col + 1) * groups_per_row * 32;
        const float* w0_scales   = W.scales.data() + (out_col + 0) * groups_per_row;
        const float* w1_scales   = W.scales.data() + (out_col + 1) * groups_per_row;
        const int16_t* w0_sums   = W.group_sums.data() + (out_col + 0) * groups_per_row;
        const int16_t* w1_sums   = W.group_sums.data() + (out_col + 1) * groups_per_row;

        float total0 = 0.0f, total1 = 0.0f;
        for (int64_t g = 0; g < groups_per_row; ++g) {
            const float sx = x_row_scales[g];
            if (sx == 0.0f) continue;

            // Load 32 packed bytes of row 0 into low 256-bit lane and 32 packed bytes of row 1 into high 256-bit lane
            __m256i p_row0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0_packed + g * 32));
            __m256i p_row1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1_packed + g * 32));
            __m512i packed = _mm512_inserti64x4(_mm512_castsi256_si512(p_row0), p_row1, 1);

            __m512i w0 = _mm512_sub_epi8(_mm512_and_si512(packed, mask2), one8);
            __m512i w1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(packed, 2), mask2), one8);
            __m512i w2 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(packed, 4), mask2), one8);
            __m512i w3 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(packed, 6), mask2), one8);

            const uint8_t* gu = x_row_u8 + g * 128;
            // Broadcast each 32-byte activation sub-vector into both 256-bit halves of ZMM (1 cycle shufi64x2 / broadcast_i64x4)
            __m512i x0 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 0)));
            __m512i x1 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 32)));
            __m512i x2 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 64)));
            __m512i x3 = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 96)));

            __m512i acc = _mm512_setzero_si512();
            acc = _mm512_dpbusd_epi32(acc, x0, w0);
            acc = _mm512_dpbusd_epi32(acc, x1, w1);
            acc = _mm512_dpbusd_epi32(acc, x2, w2);
            acc = _mm512_dpbusd_epi32(acc, x3, w3);

            int32_t raw0 = hsum_epi32_avx2(_mm512_castsi512_si256(acc));
            int32_t raw1 = hsum_epi32_avx2(_mm512_extracti64x4_epi64(acc, 1));

            total0 += static_cast<float>(raw0 - 128 * static_cast<int32_t>(w0_sums[g])) * (sx * w0_scales[g]);
            total1 += static_cast<float>(raw1 - 128 * static_cast<int32_t>(w1_sums[g])) * (sx * w1_scales[g]);
        }
        Y[out_col + 0] = total0;
        Y[out_col + 1] = total1;
    }

    if (out_col < col_end) {
        gemv_range_avx_vnni(W, X_q, out_col, col_end, Y);
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline int32_t dot512_preunpacked(const uint8_t* gu, __m512i w01, __m512i w23) {
    __m512i acc = _mm512_setzero_si512();
    acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(gu + 0)), w01);
    acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(gu + 64)), w23);
    return _mm512_reduce_add_epi32(acc);
}

// 4-row unrolled AVX512_VNNI prefill helper for one output column
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline void gemm_col_avx512_vnni(
    const uint8_t* w_row_packed,
    const float* w_row_scales,
    const int16_t* w_row_sums,
    int64_t groups_per_row,
    int64_t in_features,
    int64_t out_features,
    int64_t out_col,
    const QuantizedActivation& X_q,
    float* Y
) {
    const int64_t num_rows = X_q.rows;
    const __m512i mask2 = _mm512_set1_epi8(0x03);
    const __m512i one8  = _mm512_set1_epi8(1);

    int64_t r = 0;
    for (; r + 4 <= num_rows; r += 4) {
        float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;
        const uint8_t* x0_u8 = X_q.u8_data.data() + (r + 0) * in_features;
        const uint8_t* x1_u8 = X_q.u8_data.data() + (r + 1) * in_features;
        const uint8_t* x2_u8 = X_q.u8_data.data() + (r + 2) * in_features;
        const uint8_t* x3_u8 = X_q.u8_data.data() + (r + 3) * in_features;

        const float* sx0 = X_q.scales.data() + (r + 0) * groups_per_row;
        const float* sx1 = X_q.scales.data() + (r + 1) * groups_per_row;
        const float* sx2 = X_q.scales.data() + (r + 2) * groups_per_row;
        const float* sx3 = X_q.scales.data() + (r + 3) * groups_per_row;

        for (int64_t g = 0; g < groups_per_row; ++g) {
            __m256i packed256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row_packed + g * 32));
            __m512i p_lo_hi = _mm512_inserti64x4(_mm512_castsi256_si512(packed256), _mm256_srli_epi16(packed256, 2), 1);
            __m512i w01 = _mm512_sub_epi8(_mm512_and_si512(p_lo_hi, mask2), one8);
            __m512i w23 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(p_lo_hi, 4), mask2), one8);

            const int32_t bias = 128 * static_cast<int32_t>(w_row_sums[g]);
            const float ws = w_row_scales[g];
            const size_t goff = static_cast<size_t>(g * 128);

            if (sx0[g] != 0.0f) total0 += static_cast<float>(dot512_preunpacked(x0_u8 + goff, w01, w23) - bias) * (sx0[g] * ws);
            if (sx1[g] != 0.0f) total1 += static_cast<float>(dot512_preunpacked(x1_u8 + goff, w01, w23) - bias) * (sx1[g] * ws);
            if (sx2[g] != 0.0f) total2 += static_cast<float>(dot512_preunpacked(x2_u8 + goff, w01, w23) - bias) * (sx2[g] * ws);
            if (sx3[g] != 0.0f) total3 += static_cast<float>(dot512_preunpacked(x3_u8 + goff, w01, w23) - bias) * (sx3[g] * ws);
        }

        Y[(r + 0) * out_features + out_col] = total0;
        Y[(r + 1) * out_features + out_col] = total1;
        Y[(r + 2) * out_features + out_col] = total2;
        Y[(r + 3) * out_features + out_col] = total3;
    }

        for (; r < num_rows; ++r) {
            const uint8_t* x_row_u8 = X_q.u8_data.data() + r * in_features;
            const float* x_row_scales = X_q.scales.data() + r * groups_per_row;
            float total = 0.0f;
            for (int64_t g = 0; g < groups_per_row; ++g) {
                const float sx = x_row_scales[g];
                if (sx == 0.0f) continue;
                __m256i packed256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row_packed + g * 32));
                __m512i p_lo_hi = _mm512_inserti64x4(_mm512_castsi256_si512(packed256), _mm256_srli_epi16(packed256, 2), 1);
                __m512i w01 = _mm512_sub_epi8(_mm512_and_si512(p_lo_hi, mask2), one8);
                __m512i w23 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(p_lo_hi, 4), mask2), one8);
                int32_t raw_dot = dot512_preunpacked(x_row_u8 + g * 128, w01, w23);
                int32_t unbiased_dot = raw_dot - 128 * static_cast<int32_t>(w_row_sums[g]);
                total += static_cast<float>(unbiased_dot) * (sx * w_row_scales[g]);
            }
            Y[r * out_features + out_col] = total;
        }
}
#endif

inline float dot_group128_f32_avx2(const uint8_t* g_packed32, const float* gx128) {
    const __m256i mask2 = _mm256_set1_epi32(0x03);
    const __m256i one32 = _mm256_set1_epi32(1);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int m = 0; m < 32; m += 8) {
        __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(g_packed32 + m));
        __m256i b32 = _mm256_cvtepu8_epi32(b8);

        __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(b32, mask2), one32));
        __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(b32, 2), mask2), one32));
        __m256 w2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(b32, 4), mask2), one32));
        __m256 w3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(b32, 6), mask2), one32));

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(gx128 + m),      w0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(gx128 + m + 32), w1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(gx128 + m + 64), w2, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(gx128 + m + 96), w3, acc3);
    }

    __m256 sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    return hsum_ps_avx2(sum);
}

#endif // x86

} // namespace

void ternary_matmul_quantized(
    const TernaryMatrix& W,
    const QuantizedActivation& X_q,
    float* Y,
    const ThreadPool* pool
) {
    const int64_t out_features = W.out_features;
    const int64_t groups_per_row = W.num_groups_per_row();
    const int64_t num_rows = X_q.rows;
    const SimdIsa isa = get_active_isa();

    auto compute_range = [&](int64_t col_begin, int64_t col_end, int /*tid*/) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
        if (isa == SimdIsa::AVX512_VNNI) {
            if (num_rows == 1) {
                // Single-token decode GEMV is memory-bandwidth bound: 256-bit EVEX/VNNI (`vpdpbusd ymm`)
                // avoids 512-bit cross-lane shuffles and 512-bit frequency downclocking while matching
                // exact 128-element group width (4x 32-byte lanes).
                gemv_range_avx_vnni(W, X_q, col_begin, col_end, Y);
            } else {
                // Multi-row prefill GEMM is compute-bound: 4-row unrolled 512-bit `vpdpbusd zmm` is 6x faster!
                for (int64_t out_col = col_begin; out_col < col_end; ++out_col) {
                    const uint8_t* w_row_packed = W.packed_2bit.data() + out_col * groups_per_row * 32;
                    const float* w_row_scales = W.scales.data() + out_col * groups_per_row;
                    const int16_t* w_row_sums = W.group_sums.data() + out_col * groups_per_row;
                    gemm_col_avx512_vnni(w_row_packed, w_row_scales, w_row_sums,
                                         groups_per_row, W.in_features, out_features, out_col, X_q, Y);
                }
            }
            return;
        }

        if (isa == SimdIsa::AVX_VNNI) {
            if (num_rows == 1) {
                gemv_range_avx_vnni(W, X_q, col_begin, col_end, Y);
            } else {
                for (int64_t r = 0; r < num_rows; ++r) {
                    QuantizedActivation single_row;
                    single_row.rows = 1;
                    single_row.in_features = W.in_features;
                    single_row.u8_data.assign(X_q.u8_data.begin() + r * W.in_features,
                                              X_q.u8_data.begin() + (r + 1) * W.in_features);
                    single_row.scales.assign(X_q.scales.begin() + r * groups_per_row,
                                             X_q.scales.begin() + (r + 1) * groups_per_row);
                    gemv_range_avx_vnni(W, single_row, col_begin, col_end, Y + r * out_features);
                }
            }
            return;
        }
#endif

#if defined(__AVX2__)
        if (isa == SimdIsa::AVX2) {
            if (num_rows == 1) {
                const uint8_t* x_row_u8 = X_q.u8_data.data();
                const float* x_row_scales = X_q.scales.data();

                for (int64_t out_col = col_begin; out_col < col_end; ++out_col) {
                    const uint8_t* w_row_packed = W.packed_2bit.data() + out_col * groups_per_row * 32;
                    const float* w_row_scales = W.scales.data() + out_col * groups_per_row;
                    const int16_t* w_row_sums = W.group_sums.data() + out_col * groups_per_row;

                    float total = 0.0f;
                    for (int64_t g = 0; g < groups_per_row; ++g) {
                        const float sx = x_row_scales[g];
                        if (sx == 0.0f) continue;
                        int32_t raw_dot = dot_group128_u8_avx2(w_row_packed + g * 32, x_row_u8 + g * 128);
                        int32_t unbiased_dot = raw_dot - 128 * static_cast<int32_t>(w_row_sums[g]);
                        total += static_cast<float>(unbiased_dot) * (sx * w_row_scales[g]);
                    }
                    Y[out_col] = total;
                }
                return;
            }

            const __m256i mask2 = _mm256_set1_epi8(0x03);
            const __m256i one8 = _mm256_set1_epi8(1);
            const __m256i ones16 = _mm256_set1_epi16(1);

            for (int64_t out_col = col_begin; out_col < col_end; ++out_col) {
                const uint8_t* w_row_packed = W.packed_2bit.data() + out_col * groups_per_row * 32;
                const float* w_row_scales = W.scales.data() + out_col * groups_per_row;
                const int16_t* w_row_sums = W.group_sums.data() + out_col * groups_per_row;

                int64_t r = 0;
                for (; r + 4 <= num_rows; r += 4) {
                    float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;
                    const uint8_t* x0_u8 = X_q.u8_data.data() + (r + 0) * W.in_features;
                    const uint8_t* x1_u8 = X_q.u8_data.data() + (r + 1) * W.in_features;
                    const uint8_t* x2_u8 = X_q.u8_data.data() + (r + 2) * W.in_features;
                    const uint8_t* x3_u8 = X_q.u8_data.data() + (r + 3) * W.in_features;

                    const float* sx0 = X_q.scales.data() + (r + 0) * groups_per_row;
                    const float* sx1 = X_q.scales.data() + (r + 1) * groups_per_row;
                    const float* sx2 = X_q.scales.data() + (r + 2) * groups_per_row;
                    const float* sx3 = X_q.scales.data() + (r + 3) * groups_per_row;

                    for (int64_t g = 0; g < groups_per_row; ++g) {
                        __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row_packed + g * 32));
                        __m256i w0 = _mm256_sub_epi8(_mm256_and_si256(packed, mask2), one8);
                        __m256i w1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 2), mask2), one8);
                        __m256i w2 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 4), mask2), one8);
                        __m256i w3 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 6), mask2), one8);

                        auto dot_with_unpacked = [&](const uint8_t* gu) -> int32_t {
                            __m256i p0 = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 0)), w0);
                            __m256i p1 = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 32)), w1);
                            __m256i p2 = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 64)), w2);
                            __m256i p3 = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gu + 96)), w3);
                            __m256i s16 = _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3));
                            return hsum_epi32_avx2(_mm256_madd_epi16(s16, ones16));
                        };

                        const int32_t bias = 128 * static_cast<int32_t>(w_row_sums[g]);
                        const float ws = w_row_scales[g];
                        const size_t goff = static_cast<size_t>(g * 128);

                        if (sx0[g] != 0.0f) total0 += static_cast<float>(dot_with_unpacked(x0_u8 + goff) - bias) * (sx0[g] * ws);
                        if (sx1[g] != 0.0f) total1 += static_cast<float>(dot_with_unpacked(x1_u8 + goff) - bias) * (sx1[g] * ws);
                        if (sx2[g] != 0.0f) total2 += static_cast<float>(dot_with_unpacked(x2_u8 + goff) - bias) * (sx2[g] * ws);
                        if (sx3[g] != 0.0f) total3 += static_cast<float>(dot_with_unpacked(x3_u8 + goff) - bias) * (sx3[g] * ws);
                    }

                    Y[(r + 0) * out_features + out_col] = total0;
                    Y[(r + 1) * out_features + out_col] = total1;
                    Y[(r + 2) * out_features + out_col] = total2;
                    Y[(r + 3) * out_features + out_col] = total3;
                }

                for (; r < num_rows; ++r) {
                    const uint8_t* x_row_u8 = X_q.u8_data.data() + r * W.in_features;
                    const float* x_row_scales = X_q.scales.data() + r * groups_per_row;
                    float total = 0.0f;
                    for (int64_t g = 0; g < groups_per_row; ++g) {
                        const float sx = x_row_scales[g];
                        if (sx == 0.0f) continue;
                        int32_t raw_dot = dot_group128_u8_avx2(w_row_packed + g * 32, x_row_u8 + g * 128);
                        int32_t unbiased_dot = raw_dot - 128 * static_cast<int32_t>(w_row_sums[g]);
                        total += static_cast<float>(unbiased_dot) * (sx * w_row_scales[g]);
                    }
                    Y[r * out_features + out_col] = total;
                }
            }
            return;
        }
#endif

        // Scalar fallback path
        for (int64_t out_col = col_begin; out_col < col_end; ++out_col) {
            const uint8_t* w_row_packed = W.packed_2bit.data() + out_col * groups_per_row * 32;
            const float* w_row_scales = W.scales.data() + out_col * groups_per_row;
            const int16_t* w_row_sums = W.group_sums.data() + out_col * groups_per_row;

            for (int64_t r = 0; r < num_rows; ++r) {
                const uint8_t* x_row_u8 = X_q.u8_data.data() + r * W.in_features;
                const float* x_row_scales = X_q.scales.data() + r * groups_per_row;
                float total = 0.0f;
                for (int64_t g = 0; g < groups_per_row; ++g) {
                    const float sx = x_row_scales[g];
                    if (sx == 0.0f) continue;
                    int32_t raw_dot = dot_group128_u8_scalar(w_row_packed + g * 32, x_row_u8 + g * 128);
                    int32_t unbiased_dot = raw_dot - 128 * static_cast<int32_t>(w_row_sums[g]);
                    total += static_cast<float>(unbiased_dot) * (sx * w_row_scales[g]);
                }
                Y[r * out_features + out_col] = total;
            }
        }
    };

    if (pool) {
        pool->parallel_for(out_features, compute_range);
    } else {
        compute_range(0, out_features, 0);
    }
}

void ternary_matmul_f32(
    const TernaryMatrix& W,
    const float* X,
    int64_t num_rows,
    float* Y,
    const ThreadPool* pool
) {
    const int64_t out_features = W.out_features;
    const int64_t groups_per_row = W.num_groups_per_row();
    const SimdIsa isa = get_active_isa();

    auto compute_range = [&](int64_t col_begin, int64_t col_end, int /*tid*/) {
        for (int64_t out_col = col_begin; out_col < col_end; ++out_col) {
            const uint8_t* w_row_packed = W.packed_2bit.data() + out_col * groups_per_row * 32;
            const float* w_row_scales = W.scales.data() + out_col * groups_per_row;

            for (int64_t r = 0; r < num_rows; ++r) {
                const float* x_row = X + r * W.in_features;
                float total = 0.0f;
                for (int64_t g = 0; g < groups_per_row; ++g) {
                    float group_dot = 0.0f;
#if defined(__AVX2__)
                    if (isa != SimdIsa::SCALAR) {
                        group_dot = dot_group128_f32_avx2(w_row_packed + g * 32, x_row + g * 128);
                    } else
#endif
                    {
                        group_dot = dot_group128_f32_scalar(w_row_packed + g * 32, x_row + g * 128);
                    }
                    total += group_dot * w_row_scales[g];
                }
                Y[r * out_features + out_col] = total;
            }
        }
    };

    if (pool) {
        pool->parallel_for(out_features, compute_range);
    } else {
        compute_range(0, out_features, 0);
    }
}

} // namespace kitteni
