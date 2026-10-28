#include "kernels/ternary_unpack.hpp"
#include <stdexcept>
#include <cstring>
#include <array>

namespace kitteni {

void pack_trits_to_ternary_matrix(
    const int8_t* trits,
    const float* scales,
    int64_t out_features,
    int64_t in_features,
    TernaryMatrix& out
) {
    if (in_features % TernaryMatrix::kGroupSize != 0) {
        throw std::invalid_argument("in_features must be divisible by 128");
    }
    const int64_t groups_per_row = in_features / TernaryMatrix::kGroupSize;
    out.out_features = out_features;
    out.in_features = in_features;
    out.packed_2bit.resize(static_cast<size_t>(out_features * groups_per_row * 32));
    out.scales.resize(static_cast<size_t>(out_features * groups_per_row));
    out.group_sums.resize(static_cast<size_t>(out_features * groups_per_row));

    std::memcpy(out.scales.data(), scales, out.scales.size() * sizeof(float));

    for (int64_t r = 0; r < out_features; ++r) {
        for (int64_t g = 0; g < groups_per_row; ++g) {
            const int64_t group_idx = r * groups_per_row + g;
            const int8_t* g_trits = trits + group_idx * 128;
            uint8_t* g_packed = out.packed_2bit.data() + group_idx * 32;

            int16_t w_sum = 0;
            for (int m = 0; m < 32; ++m) {
                int8_t t0 = g_trits[m];
                int8_t t1 = g_trits[m + 32];
                int8_t t2 = g_trits[m + 64];
                int8_t t3 = g_trits[m + 96];
                w_sum += static_cast<int16_t>(t0 + t1 + t2 + t3);

                uint8_t c0 = static_cast<uint8_t>(t0 + 1) & 0x03U;
                uint8_t c1 = static_cast<uint8_t>(t1 + 1) & 0x03U;
                uint8_t c2 = static_cast<uint8_t>(t2 + 1) & 0x03U;
                uint8_t c3 = static_cast<uint8_t>(t3 + 1) & 0x03U;

                g_packed[m] = static_cast<uint8_t>(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
            }
            out.group_sums[group_idx] = w_sum;
        }
    }
}

void unpack_ternary_matrix_to_f32(
    const TernaryMatrix& mat,
    float* dst_f32
) {
    const int64_t groups_per_row = mat.num_groups_per_row();
    for (int64_t r = 0; r < mat.out_features; ++r) {
        for (int64_t g = 0; g < groups_per_row; ++g) {
            const int64_t group_idx = r * groups_per_row + g;
            const uint8_t* g_packed = mat.packed_2bit.data() + group_idx * 32;
            const float scale = mat.scales[group_idx];
            float* g_dst = dst_f32 + group_idx * 128;

            for (int m = 0; m < 32; ++m) {
                uint8_t b = g_packed[m];
                int8_t t0 = static_cast<int8_t>((b & 0x03U)) - 1;
                int8_t t1 = static_cast<int8_t>(((b >> 2) & 0x03U)) - 1;
                int8_t t2 = static_cast<int8_t>(((b >> 4) & 0x03U)) - 1;
                int8_t t3 = static_cast<int8_t>(((b >> 6) & 0x03U)) - 1;

                g_dst[m]      = static_cast<float>(t0) * scale;
                g_dst[m + 32] = static_cast<float>(t1) * scale;
                g_dst[m + 64] = static_cast<float>(t2) * scale;
                g_dst[m + 96] = static_cast<float>(t3) * scale;
            }
        }
    }
}

void unpack_ternary_matrix_to_bf16(
    const TernaryMatrix& mat,
    uint16_t* dst_bf16
) {
    const int64_t groups_per_row = mat.num_groups_per_row();
    for (int64_t r = 0; r < mat.out_features; ++r) {
        for (int64_t g = 0; g < groups_per_row; ++g) {
            const int64_t group_idx = r * groups_per_row + g;
            const uint8_t* g_packed = mat.packed_2bit.data() + group_idx * 32;
            // Match Python's `values.to(bf16) * scales.to(bf16)`
            const float scale_bf16 = bf16_to_fp32(fp32_to_bf16(mat.scales[group_idx]));
            uint16_t pos_val = fp32_to_bf16(scale_bf16);
            uint16_t neg_val = fp32_to_bf16(-scale_bf16);
            uint16_t zero_val = 0;
            const uint16_t lut[3] = {neg_val, zero_val, pos_val};

            uint16_t* g_dst = dst_bf16 + group_idx * 128;
            for (int m = 0; m < 32; ++m) {
                uint8_t b = g_packed[m];
                g_dst[m]      = lut[b & 0x03U];
                g_dst[m + 32] = lut[(b >> 2) & 0x03U];
                g_dst[m + 64] = lut[(b >> 4) & 0x03U];
                g_dst[m + 96] = lut[(b >> 6) & 0x03U];
            }
        }
    }
}

void load_tl2_to_ternary_matrix(
    const uint8_t* tl2_buf,
    int64_t rows_blocks,
    int64_t group_blocks,
    TernaryMatrix& out
) {
    constexpr int64_t kTl2Rows = 32;
    constexpr int64_t kGroup = 128;
    constexpr int64_t kTl2GroupBytes = 928;

    const int64_t out_features = rows_blocks * kTl2Rows;
    const int64_t in_features = group_blocks * kGroup;

    out.out_features = out_features;
    out.in_features = in_features;
    out.packed_2bit.resize(static_cast<size_t>(out_features * group_blocks * 32));
    out.scales.resize(static_cast<size_t>(out_features * group_blocks));
    out.group_sums.resize(static_cast<size_t>(out_features * group_blocks));

    // Process one (rb, gb) 928-byte block at a time (32 rows x 128 cols)
    // Cache-local stack buffer: 32 rows x 128 trits = 4096 bytes
    int8_t block_trits[kTl2Rows][kGroup];

    for (int64_t rb = 0; rb < rows_blocks; ++rb) {
        for (int64_t gb = 0; gb < group_blocks; ++gb) {
            const uint8_t* blk = tl2_buf + (rb * group_blocks + gb) * kTl2GroupBytes;

            // 1. First 64 bytes: 32 FP16 group scales for the 32 rows in this block
            for (int r = 0; r < kTl2Rows; ++r) {
                uint16_t h_scale = 0;
                std::memcpy(&h_scale, blk + r * 2, sizeof(uint16_t));
                // Note: Python tl2_emb4.py converts fp16 scale -> bf16 before multiplying
                float s_f32 = fp16_to_fp32(h_scale);
                float s_bf16 = bf16_to_fp32(fp32_to_bf16(s_f32));
                out.scales[(rb * kTl2Rows + r) * group_blocks + gb] = s_bf16;
            }

            size_t cursor = kTl2Rows * 2; // 64

            // 2. Four sub-blocks of 32 columns each (total 128 columns)
            for (int sub = 0; sub < 4; ++sub) {
                const uint8_t* segment = blk + cursor; // 200 bytes = 10 x 20
                cursor += 200;

                // Each of the 10 triples (t_idx = 0..9) has 20 bytes:
                //   bytes 0..15: packed 4-bit magnitudes for 32 rows (even row = low nibble, odd row = high nibble)
                //   bytes 16..19: 32 sign bits (4 bytes, little-endian bit order across the 32 rows)
                for (int t_idx = 0; t_idx < 10; ++t_idx) {
                    const uint8_t* seg_row = segment + t_idx * 20;
                    uint32_t sign_bits = 0;
                    std::memcpy(&sign_bits, seg_row + 16, sizeof(uint32_t));

                    for (int pair = 0; pair < 16; ++pair) {
                        uint8_t packed_mag = seg_row[pair];
                        int r_even = pair * 2;
                        int r_odd = pair * 2 + 1;

                        int mag_even = packed_mag & 0x0F;
                        int mag_odd = (packed_mag >> 4) & 0x0F;

                        bool neg_even = ((sign_bits >> r_even) & 1U) != 0;
                        bool neg_odd = ((sign_bits >> r_odd) & 1U) != 0;

                        int n_even = (neg_even ? -mag_even : mag_even) + 13;
                        int n_odd = (neg_odd ? -mag_odd : mag_odd) + 13;

                        int col_base = sub * 32 + t_idx * 3;
                        block_trits[r_even][col_base + 0] = static_cast<int8_t>((n_even / 9) - 1);
                        block_trits[r_even][col_base + 1] = static_cast<int8_t>(((n_even / 3) % 3) - 1);
                        block_trits[r_even][col_base + 2] = static_cast<int8_t>((n_even % 3) - 1);

                        block_trits[r_odd][col_base + 0] = static_cast<int8_t>((n_odd / 9) - 1);
                        block_trits[r_odd][col_base + 1] = static_cast<int8_t>(((n_odd / 3) % 3) - 1);
                        block_trits[r_odd][col_base + 2] = static_cast<int8_t>((n_odd % 3) - 1);
                    }
                }

                // Tail 16 bytes for columns 30 and 31 of this 32-column sub-block
                const uint8_t* tail_bytes = blk + cursor;
                cursor += 16;
                int col_tail = sub * 32 + 30;
                for (int pair = 0; pair < 16; ++pair) {
                    uint8_t tb = tail_bytes[pair];
                    uint8_t tail_even = tb & 0x0FU;
                    uint8_t tail_odd = (tb >> 4) & 0x0FU;

                    int r_even = pair * 2;
                    int r_odd = pair * 2 + 1;

                    block_trits[r_even][col_tail + 0] = static_cast<int8_t>(tail_even & 0x03U) - 1;
                    block_trits[r_even][col_tail + 1] = static_cast<int8_t>((tail_even >> 2) & 0x03U) - 1;

                    block_trits[r_odd][col_tail + 0] = static_cast<int8_t>(tail_odd & 0x03U) - 1;
                    block_trits[r_odd][col_tail + 1] = static_cast<int8_t>((tail_odd >> 2) & 0x03U) - 1;
                }
            }

            // Pack the 32x128 block into lane-interleaved 2-bit format
            for (int r = 0; r < kTl2Rows; ++r) {
                const int64_t global_r = rb * kTl2Rows + r;
                const int64_t group_idx = global_r * group_blocks + gb;
                uint8_t* g_packed = out.packed_2bit.data() + group_idx * 32;

                int16_t w_sum = 0;
                for (int m = 0; m < 32; ++m) {
                    int8_t t0 = block_trits[r][m];
                    int8_t t1 = block_trits[r][m + 32];
                    int8_t t2 = block_trits[r][m + 64];
                    int8_t t3 = block_trits[r][m + 96];
                    w_sum += static_cast<int16_t>(t0 + t1 + t2 + t3);

                    uint8_t c0 = static_cast<uint8_t>(t0 + 1) & 0x03U;
                    uint8_t c1 = static_cast<uint8_t>(t1 + 1) & 0x03U;
                    uint8_t c2 = static_cast<uint8_t>(t2 + 1) & 0x03U;
                    uint8_t c3 = static_cast<uint8_t>(t3 + 1) & 0x03U;
                    g_packed[m] = static_cast<uint8_t>(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
                }
                out.group_sums[group_idx] = w_sum;
            }
        }
    }
}

namespace {

struct Base3ByteLut {
    int8_t trits[256][5];
    constexpr Base3ByteLut() : trits{} {
        for (int b = 0; b < 256; ++b) {
            int v = b;
            for (int i = 0; i < 5; ++i) {
                trits[b][i] = static_cast<int8_t>((v % 3) - 1);
                v /= 3;
            }
        }
    }
};

constexpr Base3ByteLut kBase3Lut{};

} // namespace

void load_ternary158_to_ternary_matrix(
    const uint8_t* packed_base3,
    size_t packed_byte_len,
    const uint16_t* scales_raw,
    DType scale_dtype,
    int64_t out_features,
    int64_t in_features,
    TernaryMatrix& out
) {
    if (in_features % TernaryMatrix::kGroupSize != 0) {
        throw std::invalid_argument("in_features must be divisible by 128");
    }
    const size_t total_numel = static_cast<size_t>(out_features) * static_cast<size_t>(in_features);
    if (packed_byte_len * 5 < total_numel) {
        throw std::runtime_error("packed_base3 buffer too small for matrix dimensions");
    }

    const int64_t groups_per_row = in_features / TernaryMatrix::kGroupSize;
    const size_t total_groups = static_cast<size_t>(out_features * groups_per_row);

    std::vector<float> scales_f32(total_groups);
    if (scale_dtype == DType::BF16) {
        for (size_t i = 0; i < total_groups; ++i) {
            scales_f32[i] = bf16_to_fp32(scales_raw[i]);
        }
    } else if (scale_dtype == DType::F16) {
        for (size_t i = 0; i < total_groups; ++i) {
            scales_f32[i] = fp16_to_fp32(scales_raw[i]);
        }
    } else if (scale_dtype == DType::F32) {
        std::memcpy(scales_f32.data(), scales_raw, total_groups * sizeof(float));
    } else {
        throw std::runtime_error("Unsupported scale dtype in ternary158 loader");
    }

    // Decode base-3 stream row by row to avoid allocating a giant full-matrix trit buffer
    out.out_features = out_features;
    out.in_features = in_features;
    out.packed_2bit.resize(total_groups * 32);
    out.scales = std::move(scales_f32);
    out.group_sums.resize(total_groups);

    int8_t group_trits[128];
    size_t elem_idx = 0;

    for (size_t g = 0; g < total_groups; ++g) {
        for (int k = 0; k < 128; ++k, ++elem_idx) {
            size_t byte_idx = elem_idx / 5;
            size_t sub_idx = elem_idx % 5;
            group_trits[k] = kBase3Lut.trits[packed_base3[byte_idx]][sub_idx];
        }
        uint8_t* g_packed = out.packed_2bit.data() + g * 32;
        int16_t w_sum = 0;
        for (int m = 0; m < 32; ++m) {
            int8_t t0 = group_trits[m];
            int8_t t1 = group_trits[m + 32];
            int8_t t2 = group_trits[m + 64];
            int8_t t3 = group_trits[m + 96];
            w_sum += static_cast<int16_t>(t0 + t1 + t2 + t3);

            uint8_t c0 = static_cast<uint8_t>(t0 + 1) & 0x03U;
            uint8_t c1 = static_cast<uint8_t>(t1 + 1) & 0x03U;
            uint8_t c2 = static_cast<uint8_t>(t2 + 1) & 0x03U;
            uint8_t c3 = static_cast<uint8_t>(t3 + 1) & 0x03U;
            g_packed[m] = static_cast<uint8_t>(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
        }
        out.group_sums[g] = w_sum;
    }
}

void load_tq2_1_to_ternary_matrix(
    const uint8_t* tq2_1_blocks,
    int64_t out_features,
    int64_t in_features,
    TernaryMatrix& out
) {
    if (in_features % 256 != 0) {
        throw std::invalid_argument("TQ2_1 requires in_features divisible by 256");
    }
    const int64_t groups_per_row = in_features / 128;
    const int64_t blocks256_per_row = in_features / 256;
    const size_t total_groups = static_cast<size_t>(out_features * groups_per_row);

    out.out_features = out_features;
    out.in_features = in_features;
    out.packed_2bit.resize(total_groups * 32);
    out.scales.resize(total_groups);
    out.group_sums.resize(total_groups);

    // Each 68-byte TQ2_1 block holds 2 groups of 128 weights:
    //   bytes 0..31: group 0 packed_2bit (identical layout to our TernaryMatrix!)
    //   bytes 32..63: group 1 packed_2bit
    //   bytes 64..65: group 0 FP16 scale
    //   bytes 66..67: group 1 FP16 scale
    const int64_t total_blocks256 = out_features * blocks256_per_row;
    for (int64_t b = 0; b < total_blocks256; ++b) {
        const uint8_t* blk = tq2_1_blocks + b * 68;
        const int64_t g0 = b * 2;
        const int64_t g1 = b * 2 + 1;

        std::memcpy(out.packed_2bit.data() + g0 * 32, blk, 32);
        std::memcpy(out.packed_2bit.data() + g1 * 32, blk + 32, 32);

        uint16_t d0 = 0, d1 = 0;
        std::memcpy(&d0, blk + 64, 2);
        std::memcpy(&d1, blk + 66, 2);
        out.scales[g0] = fp16_to_fp32(d0);
        out.scales[g1] = fp16_to_fp32(d1);

        for (int64_t g : {g0, g1}) {
            const uint8_t* gp = out.packed_2bit.data() + g * 32;
            int16_t w_sum = 0;
            for (int m = 0; m < 32; ++m) {
                uint8_t v = gp[m];
                int8_t t0 = static_cast<int8_t>(v & 0x03U) - 1;
                int8_t t1 = static_cast<int8_t>((v >> 2) & 0x03U) - 1;
                int8_t t2 = static_cast<int8_t>((v >> 4) & 0x03U) - 1;
                int8_t t3 = static_cast<int8_t>((v >> 6) & 0x03U) - 1;
                w_sum += static_cast<int16_t>(t0 + t1 + t2 + t3);
            }
            out.group_sums[g] = w_sum;
        }
    }
}

} // namespace kitteni
