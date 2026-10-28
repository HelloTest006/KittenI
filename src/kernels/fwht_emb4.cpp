#include "kernels/fwht_emb4.hpp"
#include "kernels/cpu_features.hpp"
#include "core/thread_pool.hpp"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace kitteni {

namespace {

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
__attribute__((target("avx512f")))
void fwht_inplace_avx512(float* vec, int64_t dim) {
    int64_t h = 1;
    for (; h < 16 && h < dim; h <<= 1) {
        for (int64_t i = 0; i < dim; i += (h << 1)) {
            for (int64_t j = i; j < i + h; ++j) {
                float x = vec[j];
                float y = vec[j + h];
                vec[j] = x + y;
                vec[j + h] = x - y;
            }
        }
    }
    for (; h < dim; h <<= 1) {
        for (int64_t i = 0; i < dim; i += (h << 1)) {
            for (int64_t j = i; j < i + h; j += 16) {
                __m512 vx = _mm512_loadu_ps(vec + j);
                __m512 vy = _mm512_loadu_ps(vec + j + h);
                _mm512_storeu_ps(vec + j, _mm512_add_ps(vx, vy));
                _mm512_storeu_ps(vec + j + h, _mm512_sub_ps(vx, vy));
            }
        }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(dim));
    if (dim >= 16) {
        __m512 vnorm = _mm512_set1_ps(norm);
        for (int64_t i = 0; i < dim; i += 16) {
            __m512 v = _mm512_loadu_ps(vec + i);
            _mm512_storeu_ps(vec + i, _mm512_mul_ps(v, vnorm));
        }
    } else {
        for (int64_t i = 0; i < dim; ++i) vec[i] *= norm;
    }
}
#endif

} // namespace

void fwht_inplace(float* vec, int64_t dim) {
    if (dim <= 0 || (dim & (dim - 1)) != 0) {
        throw std::invalid_argument("FWHT dimension must be a positive power of 2");
    }

    const SimdIsa isa = get_active_isa();
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
    if (isa == SimdIsa::AVX512_VNNI) {
        fwht_inplace_avx512(vec, dim);
        return;
    }
#endif

    int64_t h = 1;
#if defined(__AVX2__)
    if (isa != SimdIsa::SCALAR) {
        for (; h < 8 && h < dim; h <<= 1) {
            for (int64_t i = 0; i < dim; i += (h << 1)) {
                for (int64_t j = i; j < i + h; ++j) {
                    float x = vec[j];
                    float y = vec[j + h];
                    vec[j] = x + y;
                    vec[j + h] = x - y;
                }
            }
        }
        for (; h < dim; h <<= 1) {
            for (int64_t i = 0; i < dim; i += (h << 1)) {
                for (int64_t j = i; j < i + h; j += 8) {
                    __m256 vx = _mm256_loadu_ps(vec + j);
                    __m256 vy = _mm256_loadu_ps(vec + j + h);
                    _mm256_storeu_ps(vec + j, _mm256_add_ps(vx, vy));
                    _mm256_storeu_ps(vec + j + h, _mm256_sub_ps(vx, vy));
                }
            }
        }
        const float norm = 1.0f / std::sqrt(static_cast<float>(dim));
        if (dim >= 8) {
            __m256 vnorm = _mm256_set1_ps(norm);
            for (int64_t i = 0; i < dim; i += 8) {
                __m256 v = _mm256_loadu_ps(vec + i);
                _mm256_storeu_ps(vec + i, _mm256_mul_ps(v, vnorm));
            }
            return;
        }
        for (int64_t i = 0; i < dim; ++i) vec[i] *= norm;
        return;
    }
#endif

    for (; h < dim; h <<= 1) {
        for (int64_t i = 0; i < dim; i += (h << 1)) {
            for (int64_t j = i; j < i + h; ++j) {
                float x = vec[j];
                float y = vec[j + h];
                vec[j] = x + y;
                vec[j + h] = x - y;
            }
        }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int64_t i = 0; i < dim; ++i) {
        vec[i] *= norm;
    }
}

namespace {

std::vector<float> tensor_to_float_vec(const TensorView& tv) {
    size_t n = tv.numel();
    std::vector<float> out(n);
    if (tv.dtype == DType::F32) {
        std::memcpy(out.data(), tv.data, n * sizeof(float));
    } else if (tv.dtype == DType::BF16) {
        const uint16_t* p = tv.ptr<uint16_t>();
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(p[i]);
    } else if (tv.dtype == DType::F16) {
        const uint16_t* p = tv.ptr<uint16_t>();
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(p[i]);
    } else {
        throw std::runtime_error("Unsupported float tensor dtype in emb4 loader");
    }
    return out;
}

int ceil_log2(int x) {
    int bits = 0;
    int v = 1;
    while (v < x) {
        v <<= 1;
        bits++;
    }
    return bits;
}

} // namespace

void TieredEmb4Table::load(const SafetensorsFile& sf) {
    std::string emb_meta_str = sf.get_metadata("embedding_metadata");
    if (emb_meta_str.empty()) {
        throw std::runtime_error("Missing 'embedding_metadata' in TL2+emb4 safetensors file");
    }
    JsonValue emb_meta = parse_json(emb_meta_str);
    vocab_size_ = emb_meta.at("V").as_int();
    dim_ = emb_meta.at("D").as_int();

    const JsonValue& ks_val = emb_meta.at("Ks");
    JsonValue ks_parsed;
    if (ks_val.is_string()) {
        ks_parsed = parse_json(ks_val.as_string());
    } else {
        ks_parsed = ks_val;
    }
    const auto& ks_list = ks_parsed.as_array();
    size_t n_tiers = ks_list.size();

    TensorView tier_map_tv = sf.get_tensor("emb4/tier_map");
    if (static_cast<int64_t>(tier_map_tv.numel()) != vocab_size_) {
        throw std::runtime_error("emb4/tier_map length mismatch with vocab_size");
    }

    tier_map_.resize(static_cast<size_t>(vocab_size_));
    if (tier_map_tv.dtype == DType::U8 || tier_map_tv.dtype == DType::I8) {
        std::memcpy(tier_map_.data(), tier_map_tv.data, static_cast<size_t>(vocab_size_));
    } else if (tier_map_tv.dtype == DType::I32 || tier_map_tv.dtype == DType::U32) {
        const int32_t* p = tier_map_tv.ptr<int32_t>();
        for (int64_t i = 0; i < vocab_size_; ++i) tier_map_[i] = static_cast<uint8_t>(p[i]);
    } else {
        throw std::runtime_error("Unsupported emb4/tier_map dtype");
    }

    tiers_.clear();
    tiers_.resize(n_tiers);
    local_row_index_.assign(static_cast<size_t>(vocab_size_), -1);

    for (int64_t v = 0; v < vocab_size_; ++v) {
        uint8_t t = tier_map_[v];
        if (t >= n_tiers) {
            throw std::runtime_error("Invalid tier index in emb4/tier_map");
        }
        local_row_index_[v] = static_cast<int32_t>(tiers_[t].vocab_ids.size());
        tiers_[t].vocab_ids.push_back(static_cast<int32_t>(v));
    }

    for (size_t t = 0; t < n_tiers; ++t) {
        TierData& td = tiers_[t];
        td.codebook_size = static_cast<int>(ks_list[t].as_int());
        td.bits = ceil_log2(td.codebook_size);
        td.per_byte = 8 / td.bits;
        td.num_rows = static_cast<int64_t>(td.vocab_ids.size());

        std::string prefix = "emb4/t" + std::to_string(t) + "_";
        TensorView codes_tv = sf.get_tensor(prefix + "codes");
        td.packed_codes = codes_tv.data;
        td.packed_byte_len = codes_tv.byte_size;

        td.centers = tensor_to_float_vec(sf.get_tensor(prefix + "centers"));
        td.row_scales = tensor_to_float_vec(sf.get_tensor(prefix + "s"));
        td.col_scales = tensor_to_float_vec(sf.get_tensor(prefix + "t"));

        if (static_cast<int64_t>(td.centers.size()) != static_cast<int64_t>(td.codebook_size) * dim_) {
            throw std::runtime_error("emb4 centers shape mismatch for tier " + std::to_string(t));
        }
        if (static_cast<int64_t>(td.row_scales.size()) != td.num_rows) {
            throw std::runtime_error("emb4 row_scales shape mismatch for tier " + std::to_string(t));
        }
        if (static_cast<int64_t>(td.col_scales.size()) != dim_) {
            throw std::runtime_error("emb4 col_scales shape mismatch for tier " + std::to_string(t));
        }
    }
}

void TieredEmb4Table::lookup_row(int64_t token_id, float* out_f32, bool round_via_bf16) const {
    if (token_id < 0 || token_id >= vocab_size_) {
        throw std::out_of_range("Token ID out of range in TieredEmb4Table::lookup_row");
    }
    const uint8_t tier_idx = tier_map_[token_id];
    const int64_t local_row = local_row_index_[token_id];
    const TierData& td = tiers_[tier_idx];

    const int64_t start_code_idx = local_row * dim_;
    const int per_byte = td.per_byte;
    const int bits = td.bits;
    const uint8_t mask = static_cast<uint8_t>((1U << bits) - 1U);

    // 1. Unpack codes and gather column-specific centers:
    //    In Python: `np.take_along_axis(centres, codes, axis=0)` where centres is [K, D].
    //    So for column c, value = centres[code * D + c].
    if (bits == 8) {
        const uint8_t* row_codes = td.packed_codes + start_code_idx;
        for (int64_t c = 0; c < dim_; ++c) {
            uint8_t code = row_codes[c];
            out_f32[c] = td.centers[static_cast<size_t>(code) * static_cast<size_t>(dim_) + static_cast<size_t>(c)];
        }
    } else if (bits == 4) {
        // dim_ is 2048 (even), so start_code_idx is always byte-aligned
        const uint8_t* row_packed = td.packed_codes + (start_code_idx / 2);
        for (int64_t c = 0; c < dim_; c += 2) {
            uint8_t b = row_packed[c >> 1];
            uint8_t c0 = b & 0x0FU;
            uint8_t c1 = (b >> 4) & 0x0FU;
            out_f32[c]     = td.centers[static_cast<size_t>(c0) * static_cast<size_t>(dim_) + static_cast<size_t>(c)];
            out_f32[c + 1] = td.centers[static_cast<size_t>(c1) * static_cast<size_t>(dim_) + static_cast<size_t>(c + 1)];
        }
    } else {
        for (int64_t c = 0; c < dim_; ++c) {
            int64_t global_pos = start_code_idx + c;
            int64_t byte_idx = global_pos / per_byte;
            int sub_idx = static_cast<int>(global_pos % per_byte);
            uint8_t code = (td.packed_codes[byte_idx] >> (bits * sub_idx)) & mask;
            out_f32[c] = td.centers[static_cast<size_t>(code) * static_cast<size_t>(dim_) + static_cast<size_t>(c)];
        }
    }

    // 2. Rotate back via normalized Fast Walsh-Hadamard Transform (`@ hadamard.T`)
    fwht_inplace(out_f32, dim_);

    // 3. Undo column and row scales: `restored.mul_(col_scale).mul_(row_scale)`
    const float r_scale = td.row_scales[local_row];
    for (int64_t c = 0; c < dim_; ++c) {
        // Match Python's sequential `.mul_(col_scale).mul_(row_scale)`
        float v = (out_f32[c] * td.col_scales[c]) * r_scale;
        if (round_via_bf16) {
            v = bf16_to_fp32(fp32_to_bf16(v));
        }
        out_f32[c] = v;
    }
}

void TieredEmb4Table::compute_tied_logits(
    const float* hidden,
    float* out_logits,
    int64_t vocab_start,
    int64_t vocab_end,
    const ThreadPool* pool
) const {
    if (vocab_end < 0 || vocab_end > vocab_size_) {
        vocab_end = vocab_size_;
    }

    std::vector<float> u(static_cast<size_t>(dim_));

    for (size_t t_idx = 0; t_idx < tiers_.size(); ++t_idx) {
        const TierData& td = tiers_[t_idx];
        if (td.num_rows == 0) continue;

        // Since td.vocab_ids is strictly ascending, find the exact local row range [r_begin, r_end)
        auto it_begin = std::lower_bound(td.vocab_ids.begin(), td.vocab_ids.end(), static_cast<int32_t>(vocab_start));
        auto it_end   = std::lower_bound(td.vocab_ids.begin(), td.vocab_ids.end(), static_cast<int32_t>(vocab_end));
        const int64_t r_begin = static_cast<int64_t>(std::distance(td.vocab_ids.begin(), it_begin));
        const int64_t r_end   = static_cast<int64_t>(std::distance(td.vocab_ids.begin(), it_end));
        if (r_begin >= r_end) continue;

        // 1. Compute u = FWHT(col_scales * hidden)
        for (int64_t c = 0; c < dim_; ++c) {
            u[c] = td.col_scales[c] * hidden[c];
        }
        fwht_inplace(u.data(), dim_);

        // 2. Precompute weighted_centers[c, k] = centers[k, c] * u[c]
        const int K = td.codebook_size;
        std::vector<float> wc(static_cast<size_t>(dim_) * static_cast<size_t>(K));
        for (int k = 0; k < K; ++k) {
            const float* c_row = td.centers.data() + static_cast<size_t>(k) * static_cast<size_t>(dim_);
            for (int64_t c = 0; c < dim_; ++c) {
                wc[static_cast<size_t>(c) * static_cast<size_t>(K) + static_cast<size_t>(k)] = c_row[c] * u[c];
            }
        }

        // 3. Scan only rows in [r_begin, r_end), parallelized across threads if pool is provided
        const int bits = td.bits;
        const int per_byte = td.per_byte;
        const uint8_t mask = static_cast<uint8_t>((1U << bits) - 1U);

        auto scan_rows = [&](int64_t idx_start, int64_t idx_stop, int /*tid*/) {
            for (int64_t idx = idx_start; idx < idx_stop; ++idx) {
                const int64_t r = r_begin + idx;
                const int32_t vid = td.vocab_ids[r];
                const int64_t start_code_idx = r * dim_;
                float acc = 0.0f;

                if (bits == 8) {
                    const uint8_t* row_codes = td.packed_codes + start_code_idx;
                    for (int64_t c = 0; c < dim_; c += 4) {
                        acc += wc[static_cast<size_t>(c + 0) * static_cast<size_t>(K) + row_codes[c + 0]]
                             + wc[static_cast<size_t>(c + 1) * static_cast<size_t>(K) + row_codes[c + 1]]
                             + wc[static_cast<size_t>(c + 2) * static_cast<size_t>(K) + row_codes[c + 2]]
                             + wc[static_cast<size_t>(c + 3) * static_cast<size_t>(K) + row_codes[c + 3]];
                    }
                } else if (bits == 4) {
                    const uint8_t* row_packed = td.packed_codes + (start_code_idx / 2);
                    for (int64_t c = 0; c < dim_; c += 4) {
                        uint8_t b0 = row_packed[(c >> 1) + 0];
                        uint8_t b1 = row_packed[(c >> 1) + 1];
                        acc += wc[static_cast<size_t>(c + 0) * static_cast<size_t>(K) + (b0 & 0x0FU)]
                             + wc[static_cast<size_t>(c + 1) * static_cast<size_t>(K) + ((b0 >> 4) & 0x0FU)]
                             + wc[static_cast<size_t>(c + 2) * static_cast<size_t>(K) + (b1 & 0x0FU)]
                             + wc[static_cast<size_t>(c + 3) * static_cast<size_t>(K) + ((b1 >> 4) & 0x0FU)];
                    }
                } else {
                    for (int64_t c = 0; c < dim_; ++c) {
                        int64_t global_pos = start_code_idx + c;
                        uint8_t code = (td.packed_codes[global_pos / per_byte] >> (bits * (global_pos % per_byte))) & mask;
                        acc += wc[static_cast<size_t>(c) * static_cast<size_t>(K) + code];
                    }
                }

                out_logits[vid - vocab_start] = acc * td.row_scales[r];
            }
        };

        const int64_t count = r_end - r_begin;
        if (pool && count >= 64) {
            pool->parallel_for(count, scan_rows);
        } else {
            scan_rows(0, count, 0);
        }
    }
}

} // namespace kitteni
