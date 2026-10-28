#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <span>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

namespace kitteni {

enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    U8 = 4,
    I16 = 5,
    U16 = 6,
    I32 = 7,
    U32 = 8,
    I64 = 9,
    BOOL = 10,
    TQ2_1 = 11,      // 256 trits in 68 bytes (64B codes + 2x FP16 scales)
    TL2 = 12,        // 32x128 ternary block in 928 bytes
    TERNARY_158 = 13 // 5 trits/byte + BF16 per-128 scale
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::I32:
        case DType::U32: return 4;
        case DType::F16:
        case DType::BF16:
        case DType::I16:
        case DType::U16: return 2;
        case DType::I8:
        case DType::U8:
        case DType::BOOL: return 1;
        case DType::I64: return 8;
        default: return 1;
    }
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I8: return "I8";
        case DType::U8: return "U8";
        case DType::I16: return "I16";
        case DType::U16: return "U16";
        case DType::I32: return "I32";
        case DType::U32: return "U32";
        case DType::I64: return "I64";
        case DType::BOOL: return "BOOL";
        case DType::TQ2_1: return "TQ2_1";
        case DType::TL2: return "TL2";
        case DType::TERNARY_158: return "TERNARY_158";
    }
    return "UNKNOWN";
}

// Fast IEEE-754 BF16 <-> FP32 conversions
inline float bf16_to_fp32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    // Round to nearest even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFFU;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

// Fast IEEE-754 FP16 <-> FP32 conversions
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal FP16
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFU;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFU;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000U;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t round_bit = 1U << (shift - 1);
        uint32_t sticky = (mant & (round_bit - 1)) ? 1U : 0U;
        uint32_t odd = (mant >> shift) & 1U;
        mant = (mant + round_bit - 1 + (odd | sticky)) >> shift;
        return static_cast<uint16_t>(sign | mant);
    } else if (exp == 0xFF - 127 + 15) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7C00U);
        return static_cast<uint16_t>(sign | 0x7C00U | (mant >> 13) | 1U);
    } else {
        uint32_t round_bit = 0x00001000U;
        uint32_t sticky = (mant & 0x00000FFFU) ? 1U : 0U;
        uint32_t odd = (mant >> 13) & 1U;
        mant += round_bit - 1 + (odd | sticky);
        if (mant & 0x00800000U) {
            mant = 0;
            exp++;
        }
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00U);
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }
}

struct TensorView {
    const uint8_t* data = nullptr;
    size_t byte_size = 0;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;

    size_t numel() const {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (int64_t d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    template <typename T>
    const T* ptr() const {
        return reinterpret_cast<const T*>(data);
    }
};

// Runtime Ternary Matrix format optimized for CPU SIMD GEMV/GEMM.
// Every 128-element group along `in_features` has:
// - 32 bytes of 2-bit codes (or 128 int8 signs {-1, 0, +1} in unpacked fast layout)
// - 1 float16/bfloat16/float32 scale gamma
// We support both compact 2-bit packed (4 trits/byte, 32 bytes per 128 group)
// and SIMD-ready int8/2-bit interleaved blocks.
struct TernaryMatrix {
    int64_t out_features = 0;
    int64_t in_features = 0;
    static constexpr int64_t kGroupSize = 128;

    // Compact 2-bit storage: [out_features, in_features / 4]
    // Each byte holds 4 trits in 2-bit fields:
    // byte m in [0..31] holds positions m, m+32, m+64, m+96 of the 128-group!
    // (Matches TQ2_1 lane-interleaved layout for 32-wide AVX2 / 16-wide NEON register unpacking!)
    // Code values: 0 -> -1, 1 -> 0, 2 -> +1
    std::vector<uint8_t> packed_2bit;

    // Per-group scales: [out_features, in_features / 128] stored as float32 for fast FMA
    std::vector<float> scales;

    // Precomputed group weight sum (sum of trits in {-1, 0, +1} per 128-group)
    // Useful when using unsigned activation offset tricks in VNNI/maddubs
    std::vector<int16_t> group_sums;

    int64_t num_groups_per_row() const {
        return in_features / kGroupSize;
    }

    size_t memory_bytes() const {
        return packed_2bit.size() + scales.size() * sizeof(float) + group_sums.size() * sizeof(int16_t);
    }
};

} // namespace kitteni
