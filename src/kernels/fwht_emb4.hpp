#pragma once

#include "kitteni/common.hpp"
#include "core/mmap_loader.hpp"
#include "core/json.hpp"
#include <vector>
#include <cstdint>

namespace kitteni {

// In-place normalized Fast Walsh-Hadamard Transform of length `dim` (power of 2).
// Equivalent to multiplying `vec` by the symmetric orthogonal matrix `H_dim / sqrt(dim)`.
void fwht_inplace(float* vec, int64_t dim);

// Represents the compressed 4-bit Hadamard-rotated tiered embedding table (`emb4/*`)
// from `model-tl2-emb4.safetensors`.
// Keeps the table compressed in memory (~170 MB) and supports:
// 1. Fast on-the-fly single-row dequantization (`lookup_row`) via FWHT (~3 us)
// 2. Fast compressed tied-embedding output projection (`tied_lm_head_logits`)
//    exploiting Hadamard orthogonality:
//      logits = E @ x = diag(s) (C @ H) diag(t) x = diag(s) (C @ (H (diag(t) x)))
//    so `H` is applied ONCE to `(t * x)` in O(D log D), followed by direct codebook gather!
class TieredEmb4Table {
public:
    struct TierData {
        int codebook_size = 0;
        int bits = 0;
        int per_byte = 0;
        int64_t num_rows = 0;
        const uint8_t* packed_codes = nullptr; // mmap view into emb4/t{i}_codes
        size_t packed_byte_len = 0;
        std::vector<float> centers;            // [codebook_size, dim] float32
        std::vector<float> row_scales;         // [num_rows] float32 (t{i}_s)
        std::vector<float> col_scales;         // [dim] float32 (t{i}_t)
        std::vector<int32_t> vocab_ids;        // [num_rows] global vocab ID for each local row
    };

    TieredEmb4Table() = default;

    // Initialize zero-copy from a mapped `model-tl2-emb4.safetensors` file
    void load(const SafetensorsFile& sf);

    int64_t vocab_size() const { return vocab_size_; }
    int64_t dim() const { return dim_; }
    size_t num_tiers() const { return tiers_.size(); }
    const TierData& tier(size_t idx) const { return tiers_.at(idx); }

    // Dequantize a single token row `token_id` into `out_f32[0..dim-1]`.
    // If `round_via_bf16` is true, rounds to BF16 exactly like Python's `restored.to(torch.bfloat16)`.
    void lookup_row(int64_t token_id, float* out_f32, bool round_via_bf16 = true) const;

    // Compute tied output logits `out_logits[0..vocab_size-1] = E @ hidden[0..dim-1]`
    // directly from the compressed 4-bit table without dequantizing E!
    // Uses Hadamard dual-space trick: only 1 FWHT per tier, then pure table lookup + accumulate.
    void compute_tied_logits(
        const float* hidden,
        float* out_logits,
        int64_t vocab_start = 0,
        int64_t vocab_end = -1,
        const class ThreadPool* pool = nullptr
    ) const;

private:
    int64_t vocab_size_ = 0;
    int64_t dim_ = 0;
    std::vector<uint8_t> tier_map_;        // [vocab_size] -> tier index
    std::vector<int32_t> local_row_index_; // [vocab_size] -> row index inside that tier
    std::vector<TierData> tiers_;
};

} // namespace kitteni
