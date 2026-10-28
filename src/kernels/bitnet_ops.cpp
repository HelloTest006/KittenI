#include "kernels/bitnet_ops.hpp"
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

inline int64_t round_up_64(int64_t n) {
    return (n + 63) & ~int64_t(63);
}

inline int32_t dot_u8_i8_scalar(const uint8_t* u8_vec, const int8_t* i8_vec, int64_t len) {
    int32_t sum = 0;
    for (int64_t i = 0; i < len; ++i) {
        sum += static_cast<int32_t>(u8_vec[i]) * static_cast<int32_t>(i8_vec[i]);
    }
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

inline int32_t dot_u8_i8_avx2(const uint8_t* u8_vec, const int8_t* i8_vec, int64_t padded_len) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i acc32 = _mm256_setzero_si256();
    for (int64_t i = 0; i < padded_len; i += 32) {
        __m256i vu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u8_vec + i));
        __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(i8_vec + i));
        __m256i p16 = _mm256_maddubs_epi16(vu, vi);
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(p16, ones16));
    }
    __m128i lo = _mm256_castsi256_si128(acc32);
    __m128i hi = _mm256_extracti128_si256(acc32, 1);
    __m128i s128 = _mm_add_epi32(lo, hi);
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1, 0, 3, 2)));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s128);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,avx512vl,avx512vnni")))
inline void bitlinear_range_avx_vnni(
    const BitNetWeight& W,
    const uint8_t* x_u8,
    const float* x_inv_scale,
    int64_t num_rows,
    int64_t o_begin,
    int64_t o_end,
    float* y
) {
    const int64_t out_dim = W.out_features;
    const int64_t padded_in = W.padded_in;
    for (int64_t o = o_begin; o < o_end; ++o) {
        const int8_t* w_row = W.codes_i8.data() + o * padded_in;
        const int32_t bias_cancel = 128 * W.row_sums[o];
        const float w_inv = W.inv_scales[o];
        const float b_val = W.bias.empty() ? 0.0f : W.bias[o];

        for (int64_t r = 0; r < num_rows; ++r) {
            const uint8_t* ur = x_u8 + r * padded_in;
            __m256i acc32 = _mm256_setzero_si256();
            for (int64_t i = 0; i < padded_in; i += 32) {
                __m256i vu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ur + i));
                __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row + i));
                acc32 = _mm256_dpbusd_epi32(acc32, vu, vi);
            }
            __m128i lo = _mm256_castsi256_si128(acc32);
            __m128i hi = _mm256_extracti128_si256(acc32, 1);
            __m128i s128 = _mm_add_epi32(lo, hi);
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1, 0, 3, 2)));
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(2, 3, 0, 1)));
            int32_t int_dot = _mm_cvtsi128_si32(s128) - bias_cancel;
            y[r * out_dim + o] = static_cast<float>(int_dot) * (x_inv_scale[r] * w_inv) + b_val;
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline void bitlinear_range_avx512_vnni(
    const BitNetWeight& W,
    const uint8_t* x_u8,
    const float* x_inv_scale,
    int64_t num_rows,
    int64_t o_begin,
    int64_t o_end,
    float* y
) {
    const int64_t out_dim = W.out_features;
    const int64_t padded_in = W.padded_in;
    for (int64_t o = o_begin; o < o_end; ++o) {
        const int8_t* w_row = W.codes_i8.data() + o * padded_in;
        const int32_t bias_cancel = 128 * W.row_sums[o];
        const float w_inv = W.inv_scales[o];
        const float b_val = W.bias.empty() ? 0.0f : W.bias[o];

        int64_t r = 0;
        for (; r + 4 <= num_rows; r += 4) {
            const uint8_t* u0 = x_u8 + (r + 0) * padded_in;
            const uint8_t* u1 = x_u8 + (r + 1) * padded_in;
            const uint8_t* u2 = x_u8 + (r + 2) * padded_in;
            const uint8_t* u3 = x_u8 + (r + 3) * padded_in;

            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            for (int64_t i = 0; i < padded_in; i += 64) {
                __m512i vw = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_row + i));
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u0 + i)), vw);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u1 + i)), vw);
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u2 + i)), vw);
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u3 + i)), vw);
            }

            int32_t d0 = _mm512_reduce_add_epi32(acc0) - bias_cancel;
            int32_t d1 = _mm512_reduce_add_epi32(acc1) - bias_cancel;
            int32_t d2 = _mm512_reduce_add_epi32(acc2) - bias_cancel;
            int32_t d3 = _mm512_reduce_add_epi32(acc3) - bias_cancel;

            y[(r + 0) * out_dim + o] = static_cast<float>(d0) * (x_inv_scale[r + 0] * w_inv) + b_val;
            y[(r + 1) * out_dim + o] = static_cast<float>(d1) * (x_inv_scale[r + 1] * w_inv) + b_val;
            y[(r + 2) * out_dim + o] = static_cast<float>(d2) * (x_inv_scale[r + 2] * w_inv) + b_val;
            y[(r + 3) * out_dim + o] = static_cast<float>(d3) * (x_inv_scale[r + 3] * w_inv) + b_val;
        }
        for (; r < num_rows; ++r) {
            const uint8_t* ur = x_u8 + r * padded_in;
            __m512i acc = _mm512_setzero_si512();
            for (int64_t i = 0; i < padded_in; i += 64) {
                acc = _mm512_dpbusd_epi32(
                    acc,
                    _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ur + i)),
                    _mm512_loadu_si512(reinterpret_cast<const __m512i*>(w_row + i))
                );
            }
            int32_t int_dot = _mm512_reduce_add_epi32(acc) - bias_cancel;
            y[r * out_dim + o] = static_cast<float>(int_dot) * (x_inv_scale[r] * w_inv) + b_val;
        }
    }
}

__attribute__((target("avx512f")))
inline void fmadd_row_avx512(float wk, const float* x_ptr, float* y_ptr, int64_t count) {
    int64_t i = 0;
    __m512 vwk = _mm512_set1_ps(wk);
    for (; i + 16 <= count; i += 16) {
        __m512 vx = _mm512_loadu_ps(x_ptr + i);
        __m512 vy = _mm512_loadu_ps(y_ptr + i);
        _mm512_storeu_ps(y_ptr + i, _mm512_fmadd_ps(vwk, vx, vy));
    }
    for (; i < count; ++i) {
        y_ptr[i] += wk * x_ptr[i];
    }
}
#endif

inline void fmadd_row_avx2(float wk, const float* x_ptr, float* y_ptr, int64_t count) {
    int64_t i = 0;
    __m256 vwk = _mm256_set1_ps(wk);
    for (; i + 8 <= count; i += 8) {
        __m256 vx = _mm256_loadu_ps(x_ptr + i);
        __m256 vy = _mm256_loadu_ps(y_ptr + i);
        _mm256_storeu_ps(y_ptr + i, _mm256_fmadd_ps(vwk, vx, vy));
    }
    for (; i < count; ++i) {
        y_ptr[i] += wk * x_ptr[i];
    }
}

#endif // x86

} // namespace

void BitNetWeight::init_from_packed_w4(
    const uint8_t* packed_u8,
    const float* scale_f32,
    const float* bias_f32,
    int64_t out_dim,
    int64_t in_ch,
    int64_t k_size,
    int64_t c_pad_left,
    int64_t c_pad_right
) {
    out_features = out_dim;
    in_channels = in_ch;
    kernel_size = k_size;
    in_features = in_ch * k_size;
    causal_pad_left = c_pad_left;
    causal_pad_right = c_pad_right;
    padded_in = round_up_64(in_features);

    codes_i8.assign(static_cast<size_t>(out_features * padded_in), 0);
    row_sums.assign(static_cast<size_t>(out_features), 0);
    inv_scales.resize(static_cast<size_t>(out_features));

    for (int64_t o = 0; o < out_features; ++o) {
        inv_scales[o] = 1.0f / scale_f32[o];
    }
    if (bias_f32) {
        bias.assign(bias_f32, bias_f32 + out_features);
    } else {
        bias.clear();
    }

    const int64_t total_numel = out_features * in_features;
    for (int64_t idx = 0; idx < total_numel; ++idx) {
        uint8_t b = packed_u8[idx >> 1];
        int8_t code = (idx & 1) ? (static_cast<int8_t>(b & 0x0FU) - 8)
                                : (static_cast<int8_t>(b >> 4) - 8);
        int64_t o = idx / in_features;
        int64_t i = idx % in_features;
        codes_i8[o * padded_in + i] = code;
        row_sums[o] += code;
    }
}

void BitNetWeight::init_from_dequantized_f32(
    const float* weight_f32,
    const float* bias_f32,
    int64_t out_dim,
    int64_t in_ch,
    int64_t k_size,
    int bits,
    int64_t c_pad_left,
    int64_t c_pad_right
) {
    out_features = out_dim;
    in_channels = in_ch;
    kernel_size = k_size;
    in_features = in_ch * k_size;
    causal_pad_left = c_pad_left;
    causal_pad_right = c_pad_right;
    padded_in = round_up_64(in_features);

    codes_i8.assign(static_cast<size_t>(out_features * padded_in), 0);
    row_sums.assign(static_cast<size_t>(out_features), 0);
    inv_scales.resize(static_cast<size_t>(out_features));

    if (bias_f32) {
        bias.assign(bias_f32, bias_f32 + out_features);
    } else {
        bias.clear();
    }

    const float qmax = static_cast<float>((1 << (bits - 1)) - 1);
    for (int64_t o = 0; o < out_features; ++o) {
        const float* w_row = weight_f32 + o * in_features;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < in_features; ++i) {
            max_abs = std::max(max_abs, std::fabs(w_row[i]));
        }
        if (max_abs < 1e-12f) {
            inv_scales[o] = 0.0f;
            continue;
        }
        const float scale = qmax / max_abs;
        inv_scales[o] = 1.0f / scale;
        int32_t rsum = 0;
        int8_t* dst_row = codes_i8.data() + o * padded_in;
        for (int64_t i = 0; i < in_features; ++i) {
            int q = static_cast<int>(std::nearbyintf(w_row[i] * scale));
            q = std::clamp(q, -128, 127);
            dst_row[i] = static_cast<int8_t>(q);
            rsum += q;
        }
        row_sums[o] = rsum;
    }
}

void bitlinear_forward(
    const BitNetWeight& W,
    const float* x,
    int64_t num_rows,
    float* y,
    const ThreadPool* pool
) {
    const int64_t in_dim = W.in_features;
    const int64_t out_dim = W.out_features;
    const int64_t padded_in = W.padded_in;
    const SimdIsa isa = get_active_isa();

    // 1. Quantize each row of `x` (`[num_rows, in_dim]`) to 8-bit biased uint8 `[1..255]` (`q + 128`)
    std::vector<uint8_t> x_u8(static_cast<size_t>(num_rows * padded_in), 128);
    std::vector<float> x_inv_scale(static_cast<size_t>(num_rows), 0.0f);

    for (int64_t r = 0; r < num_rows; ++r) {
        const float* xr = x + r * in_dim;
        uint8_t* ur = x_u8.data() + r * padded_in;

        float max_abs = 1e-5f;
        for (int64_t i = 0; i < in_dim; ++i) {
            max_abs = std::max(max_abs, std::fabs(xr[i]));
        }
        const float scale_x = 127.0f / max_abs;
        x_inv_scale[r] = 1.0f / scale_x;

        for (int64_t i = 0; i < in_dim; ++i) {
            int q = static_cast<int>(std::nearbyintf(xr[i] * scale_x));
            q = std::clamp(q, -128, 127);
            ur[i] = static_cast<uint8_t>(q + 128);
        }
    }

    // 2. Integer GEMV/GEMM across `out_dim` using active SIMD ISA
    auto compute_cols = [&](int64_t o_begin, int64_t o_end, int /*tid*/) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
        if (isa == SimdIsa::AVX512_VNNI) {
            bitlinear_range_avx512_vnni(W, x_u8.data(), x_inv_scale.data(), num_rows, o_begin, o_end, y);
            return;
        }
        if (isa == SimdIsa::AVX_VNNI) {
            bitlinear_range_avx_vnni(W, x_u8.data(), x_inv_scale.data(), num_rows, o_begin, o_end, y);
            return;
        }
#endif
        for (int64_t o = o_begin; o < o_end; ++o) {
            const int8_t* w_row = W.codes_i8.data() + o * padded_in;
            const int32_t bias_cancel = 128 * W.row_sums[o];
            const float w_inv = W.inv_scales[o];
            const float b_val = W.bias.empty() ? 0.0f : W.bias[o];

            for (int64_t r = 0; r < num_rows; ++r) {
                const uint8_t* ur = x_u8.data() + r * padded_in;
                int32_t raw_dot = 0;
#if defined(__AVX2__)
                if (isa == SimdIsa::AVX2) {
                    raw_dot = dot_u8_i8_avx2(ur, w_row, padded_in);
                } else
#endif
                {
                    raw_dot = dot_u8_i8_scalar(ur, w_row, in_dim);
                }
                int32_t int_dot = raw_dot - bias_cancel;
                y[r * out_dim + o] = static_cast<float>(int_dot) * (x_inv_scale[r] * w_inv) + b_val;
            }
        }
    };

    if (pool && out_dim >= 64) {
        pool->parallel_for(out_dim, compute_cols);
    } else {
        compute_cols(0, out_dim, 0);
    }
}

void bitconv1d_forward(
    const BitNetWeight& W,
    const float* x,
    int64_t T_in,
    float* y,
    const ThreadPool* pool
) {
    const int64_t C_in = W.in_channels;
    const int64_t C_out = W.out_features;
    const int64_t K = W.kernel_size;
    const int64_t pad_l = W.causal_pad_left + W.padding;
    const int64_t pad_r = W.causal_pad_right + W.padding;
    const int64_t T_pad = T_in + pad_l + pad_r;
    const int64_t T_out = (T_pad - W.dilation * (K - 1) - 1) / W.stride + 1;
    const SimdIsa isa = get_active_isa();

    std::vector<float> x_q(static_cast<size_t>(C_in * T_pad), 0.0f);
    for (int64_t t = 0; t < T_in; ++t) {
        const int64_t tp = t + pad_l;
        float max_abs = 1e-5f;
        for (int64_t c = 0; c < C_in; ++c) {
            max_abs = std::max(max_abs, std::fabs(x[c * T_in + t]));
        }
        const float scale = 127.0f / max_abs;
        const float inv_scale = 1.0f / scale;
        for (int64_t c = 0; c < C_in; ++c) {
            float v = x[c * T_in + t];
            float q = std::clamp(std::nearbyintf(v * scale), -128.0f, 127.0f);
            x_q[c * T_pad + tp] = q * inv_scale;
        }
    }

    auto compute_out = [&](int64_t o_begin, int64_t o_end, int /*tid*/) {
        for (int64_t o = o_begin; o < o_end; ++o) {
            const int8_t* w_codes = W.codes_i8.data() + o * W.padded_in;
            const float w_inv = W.inv_scales[o];
            const float b_val = W.bias.empty() ? 0.0f : W.bias[o];
            float* y_row = y + o * T_out;

            if (W.stride == 1 && W.dilation == 1) {
                for (int64_t t = 0; t < T_out; ++t) y_row[t] = 0.0f;
                for (int64_t c = 0; c < C_in; ++c) {
                    const float* x_ch = x_q.data() + c * T_pad;
                    const int8_t* w_ch = w_codes + c * K;
                    for (int64_t k = 0; k < K; ++k) {
                        const float wk = static_cast<float>(w_ch[k]);
                        if (wk == 0.0f) continue;
                        const float* x_ptr = x_ch + k;
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
                        if (isa == SimdIsa::AVX512_VNNI) {
                            fmadd_row_avx512(wk, x_ptr, y_row, T_out);
                            continue;
                        }
#endif
#if defined(__AVX2__)
                        if (isa != SimdIsa::SCALAR) {
                            fmadd_row_avx2(wk, x_ptr, y_row, T_out);
                            continue;
                        }
#endif
                        for (int64_t t = 0; t < T_out; ++t) {
                            y_row[t] += wk * x_ptr[t];
                        }
                    }
                }
                for (int64_t t = 0; t < T_out; ++t) {
                    y_row[t] = y_row[t] * w_inv + b_val;
                }
                continue;
            }

            for (int64_t t = 0; t < T_out; ++t) {
                const int64_t t_start = t * W.stride;
                float acc = 0.0f;
                for (int64_t c = 0; c < C_in; ++c) {
                    const float* x_ch = x_q.data() + c * T_pad + t_start;
                    const int8_t* w_ch = w_codes + c * K;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += x_ch[k * W.dilation] * static_cast<float>(w_ch[k]);
                    }
                }
                y_row[t] = acc * w_inv + b_val;
            }
        }
    };

    if (pool && C_out >= 32) {
        pool->parallel_for(C_out, compute_out);
    } else {
        compute_out(0, C_out, 0);
    }
}

void conv1d_f32(
    const float* x,
    const float* weight,
    const float* bias,
    int64_t in_channels,
    int64_t out_channels,
    int64_t T_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    float* y,
    const ThreadPool* pool
) {
    const int64_t T_out = (T_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const SimdIsa isa = get_active_isa();

    auto compute_out = [&](int64_t o_begin, int64_t o_end, int /*tid*/) {
        for (int64_t o = o_begin; o < o_end; ++o) {
            const float* w_out = weight + o * (in_channels * kernel_size);
            const float b_val = bias ? bias[o] : 0.0f;
            float* y_out = y + o * T_out;

            for (int64_t t = 0; t < T_out; ++t) y_out[t] = b_val;

            if (stride == 1) {
                for (int64_t c = 0; c < in_channels; ++c) {
                    const float* x_in = x + c * T_in;
                    const float* w_ch = w_out + c * kernel_size;
                    for (int64_t k = 0; k < kernel_size; ++k) {
                        const float wk = w_ch[k];
                        const int64_t k_offset = k * dilation - padding;
                        const int64_t t_start = std::max<int64_t>(0, -k_offset);
                        const int64_t t_end   = std::min<int64_t>(T_out, T_in - k_offset);
                        if (t_start >= t_end) continue;

                        const float* x_ptr = x_in + (t_start + k_offset);
                        float* y_ptr = y_out + t_start;
                        const int64_t count = t_end - t_start;
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
                        if (isa == SimdIsa::AVX512_VNNI) {
                            fmadd_row_avx512(wk, x_ptr, y_ptr, count);
                            continue;
                        }
#endif
#if defined(__AVX2__)
                        if (isa != SimdIsa::SCALAR) {
                            fmadd_row_avx2(wk, x_ptr, y_ptr, count);
                            continue;
                        }
#endif
                        for (int64_t i = 0; i < count; ++i) {
                            y_ptr[i] += wk * x_ptr[i];
                        }
                    }
                }
                continue;
            }

            // General stride > 1 path
            for (int64_t c = 0; c < in_channels; ++c) {
                const float* x_in = x + c * T_in;
                const float* w_ch = w_out + c * kernel_size;
                for (int64_t k = 0; k < kernel_size; ++k) {
                    const float wk = w_ch[k];
                    const int64_t k_offset = k * dilation - padding;
                    for (int64_t t = 0; t < T_out; ++t) {
                        int64_t in_t = t * stride + k_offset;
                        if (in_t >= 0 && in_t < T_in) {
                            y_out[t] += wk * x_in[in_t];
                        }
                    }
                }
            }
        }
    };

    if (pool && out_channels >= 16) {
        pool->parallel_for(out_channels, compute_out);
    } else {
        compute_out(0, out_channels, 0);
    }
}

void conv_transpose1d_f32(
    const float* x,
    const float* weight,
    const float* bias,
    int64_t in_channels,
    int64_t out_channels,
    int64_t T_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    float* y,
    const ThreadPool* pool
) {
    const int64_t T_out = (T_in - 1) * stride - 2 * padding + kernel_size;

    auto compute_out = [&](int64_t o_begin, int64_t o_end, int /*tid*/) {
        for (int64_t o = o_begin; o < o_end; ++o) {
            float* y_out = y + o * T_out;
            const float b_val = bias ? bias[o] : 0.0f;
            for (int64_t t = 0; t < T_out; ++t) y_out[t] = b_val;

            for (int64_t c = 0; c < in_channels; ++c) {
                const float* x_in = x + c * T_in;
                const float* w_co = weight + (c * out_channels + o) * kernel_size;
                for (int64_t k = 0; k < kernel_size; ++k) {
                    const float wk = w_co[k];
                    const int64_t k_off = k - padding;
                    for (int64_t t_in = 0; t_in < T_in; ++t_in) {
                        int64_t out_t = t_in * stride + k_off;
                        if (out_t >= 0 && out_t < T_out) {
                            y_out[out_t] += wk * x_in[t_in];
                        }
                    }
                }
            }
        }
    };

    if (pool && out_channels >= 16) {
        pool->parallel_for(out_channels, compute_out);
    } else {
        compute_out(0, out_channels, 0);
    }
}

} // namespace kitteni
