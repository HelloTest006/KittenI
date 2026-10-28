#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace kitteni {

class Qwen3BPETokenizer {
public:
    Qwen3BPETokenizer() = default;

    // Load from the `lm/` directory containing `vocab.json`, `merges.txt`, and `added_tokens.json`
    void load(const std::string& lm_dir);

    // Tokenize UTF-8 text into Qwen3 token IDs (matching `tokenizer.encode(text, add_special_tokens=False)`)
    std::vector<int32_t> encode(std::string_view text) const;

    // Decode a sequence of token IDs back to UTF-8 string
    std::string decode(const std::vector<int32_t>& ids) const;

    size_t vocab_size() const { return vocab_to_id_.size(); }

private:
    std::unordered_map<std::string, int32_t> vocab_to_id_;
    std::unordered_map<int32_t, std::string> id_to_vocab_;

    // Key: `left + "\x01" + right` -> merge priority rank (0 = highest priority)
    std::unordered_map<std::string, int32_t> bpe_ranks_;

    // Byte-to-unicode mapping tables (GPT-2 / Qwen2 / Qwen3 standard byte-level BPE)
    std::string byte_to_unicode_[256];
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;

    void init_byte_tables();
    std::vector<std::string> pre_tokenize(std::string_view text) const;
    std::vector<int32_t> bpe_word(const std::string& byte_piece) const;
};

} // namespace kitteni
