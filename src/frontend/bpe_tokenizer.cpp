#include "frontend/bpe_tokenizer.hpp"
#include "core/json.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

namespace kitteni {

namespace {

void append_codepoint_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

struct CodepointSpan {
    uint32_t cp;
    size_t byte_offset;
    size_t byte_len;
};

std::vector<CodepointSpan> decode_utf8_spans(std::string_view sv) {
    std::vector<CodepointSpan> spans;
    spans.reserve(sv.size());
    size_t i = 0;
    while (i < sv.size()) {
        uint8_t b0 = static_cast<uint8_t>(sv[i]);
        uint32_t cp = 0;
        size_t len = 1;
        if ((b0 & 0x80U) == 0) {
            cp = b0;
            len = 1;
        } else if ((b0 & 0xE0U) == 0xC0U && i + 1 < sv.size()) {
            cp = ((b0 & 0x1FU) << 6) | (static_cast<uint8_t>(sv[i + 1]) & 0x3FU);
            len = 2;
        } else if ((b0 & 0xF0U) == 0xE0U && i + 2 < sv.size()) {
            cp = ((b0 & 0x0FU) << 12) |
                 ((static_cast<uint8_t>(sv[i + 1]) & 0x3FU) << 6) |
                 (static_cast<uint8_t>(sv[i + 2]) & 0x3FU);
            len = 3;
        } else if ((b0 & 0xF8U) == 0xF0U && i + 3 < sv.size()) {
            cp = ((b0 & 0x07U) << 18) |
                 ((static_cast<uint8_t>(sv[i + 1]) & 0x3FU) << 12) |
                 ((static_cast<uint8_t>(sv[i + 2]) & 0x3FU) << 6) |
                 (static_cast<uint8_t>(sv[i + 3]) & 0x3FU);
            len = 4;
        } else {
            cp = b0;
            len = 1;
        }
        spans.push_back({cp, i, len});
        i += len;
    }
    return spans;
}

bool is_newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

bool is_whitespace(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f') return true;
    if (cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000) {
        return true;
    }
    return false;
}

bool is_number(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp == 0x00B2 || cp == 0x00B3 || cp == 0x00B9 || (cp >= 0x00BC && cp <= 0x00BE)) return true;
    if (cp >= 0x0660 && cp <= 0x0669) return true; // Arabic-Indic digits
    if (cp >= 0x06F0 && cp <= 0x06F9) return true; // Extended Arabic-Indic
    if (cp >= 0x0966 && cp <= 0x096F) return true; // Devanagari digits
    if (cp >= 0xFF10 && cp <= 0xFF19) return true; // Fullwidth digits
    return false;
}

bool is_letter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0x00AA) return false;
    if (cp == 0x00AA || cp == 0x00B5 || cp == 0x00BA) return true;
    if (cp >= 0x00C0 && cp <= 0x02AF && cp != 0x00D7 && cp != 0x00F7) return true; // Latin-1, Ext A/B, IPA
    if (cp >= 0x0370 && cp <= 0x052F) return true; // Greek, Coptic, Cyrillic
    if (cp >= 0x0530 && cp <= 0x06FF && !is_number(cp)) return true; // Armenian, Hebrew, Arabic
    if (cp >= 0x0900 && cp <= 0x0DFF && !is_number(cp)) return true; // Devanagari & Indic scripts
    if (cp >= 0x0E00 && cp <= 0x0EFF) return true; // Thai, Lao
    if (cp >= 0x10A0 && cp <= 0x1FFF) return true; // Georgian, Hangul Jamo, Latin Ext Additional, Greek Ext
    if (cp >= 0x3040 && cp <= 0x31FF) return true; // Hiragana, Katakana, Bopomofo, Hangul Compat
    if (cp >= 0x3400 && cp <= 0x9FFF) return true; // CJK Unified Ideographs
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true; // Hangul Syllables
    if (cp >= 0xF900 && cp <= 0xFAFF) return true; // CJK Compatibility Ideographs
    return false;
}

uint32_t to_ascii_lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    return cp;
}

// Match `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at index `idx` in `spans`
size_t match_contraction(const std::vector<CodepointSpan>& spans, size_t idx) {
    if (idx + 1 >= spans.size() || spans[idx].cp != '\'') return 0;
    uint32_t c1 = to_ascii_lower(spans[idx + 1].cp);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        return 2;
    }
    if (idx + 2 < spans.size()) {
        uint32_t c2 = to_ascii_lower(spans[idx + 2].cp);
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
            return 3;
        }
    }
    return 0;
}

} // namespace

void Qwen3BPETokenizer::init_byte_tables() {
    bool direct[256] = {false};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
        std::string u8;
        append_codepoint_utf8(u8, cp);
        byte_to_unicode_[b] = u8;
        unicode_to_byte_[cp] = static_cast<uint8_t>(b);
    }
}

void Qwen3BPETokenizer::load(const std::string& lm_dir) {
    init_byte_tables();
    vocab_to_id_.clear();
    id_to_vocab_.clear();
    bpe_ranks_.clear();

    fs::path dir(lm_dir);
    fs::path vocab_path = dir / "vocab.json";
    fs::path merges_path = dir / "merges.txt";
    fs::path added_path = dir / "added_tokens.json";

    JsonValue vocab_json = parse_json_file(vocab_path.string());
    for (const auto& [token_str, id_val] : vocab_json.as_object()) {
        int32_t id = static_cast<int32_t>(id_val.as_int());
        vocab_to_id_[token_str] = id;
        id_to_vocab_[id] = token_str;
    }

    if (fs::exists(added_path)) {
        JsonValue added_json = parse_json_file(added_path.string());
        for (const auto& [token_str, id_val] : added_json.as_object()) {
            int32_t id = static_cast<int32_t>(id_val.as_int());
            vocab_to_id_[token_str] = id;
            id_to_vocab_[id] = token_str;
        }
    }

    std::ifstream mf(merges_path);
    if (!mf) {
        throw std::runtime_error("Cannot open merges.txt: " + merges_path.string());
    }
    std::string line;
    int32_t rank = 0;
    while (std::getline(mf, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string key = line.substr(0, sp) + "\x01" + line.substr(sp + 1);
        bpe_ranks_.emplace(std::move(key), rank++);
    }
}

std::vector<std::string> Qwen3BPETokenizer::pre_tokenize(std::string_view text) const {
    // Implements the exact Qwen2 / Qwen3 regex:
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    std::vector<CodepointSpan> spans = decode_utf8_spans(text);
    std::vector<std::string> pieces;
    size_t i = 0;
    const size_t n = spans.size();

    while (i < n) {
        // Alternative 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        size_t clen = match_contraction(spans, i);
        if (clen > 0) {
            size_t start_byte = spans[i].byte_offset;
            size_t end_byte = spans[i + clen - 1].byte_offset + spans[i + clen - 1].byte_len;
            pieces.emplace_back(text.substr(start_byte, end_byte - start_byte));
            i += clen;
            continue;
        }

        // Alternative 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_letter(spans[i].cp) ||
            (!is_newline(spans[i].cp) && !is_letter(spans[i].cp) && !is_number(spans[i].cp) &&
             i + 1 < n && is_letter(spans[i + 1].cp))) {
            size_t j = i + 1;
            while (j < n && is_letter(spans[j].cp)) {
                j++;
            }
            size_t start_byte = spans[i].byte_offset;
            size_t end_byte = spans[j - 1].byte_offset + spans[j - 1].byte_len;
            pieces.emplace_back(text.substr(start_byte, end_byte - start_byte));
            i = j;
            continue;
        }

        // Alternative 3: \p{N} (single digit per token!)
        if (is_number(spans[i].cp)) {
            pieces.emplace_back(text.substr(spans[i].byte_offset, spans[i].byte_len));
            i++;
            continue;
        }

        // Alternative 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        if (!is_whitespace(spans[i].cp) && !is_letter(spans[i].cp) && !is_number(spans[i].cp)) {
            size_t j = i + 1;
            while (j < n && !is_whitespace(spans[j].cp) && !is_letter(spans[j].cp) && !is_number(spans[j].cp)) {
                j++;
            }
            while (j < n && is_newline(spans[j].cp)) {
                j++;
            }
            size_t start_byte = spans[i].byte_offset;
            size_t end_byte = spans[j - 1].byte_offset + spans[j - 1].byte_len;
            pieces.emplace_back(text.substr(start_byte, end_byte - start_byte));
            i = j;
            continue;
        }
        if (spans[i].cp == ' ' && i + 1 < n &&
            !is_whitespace(spans[i + 1].cp) && !is_letter(spans[i + 1].cp) && !is_number(spans[i + 1].cp)) {
            size_t j = i + 2;
            while (j < n && !is_whitespace(spans[j].cp) && !is_letter(spans[j].cp) && !is_number(spans[j].cp)) {
                j++;
            }
            while (j < n && is_newline(spans[j].cp)) {
                j++;
            }
            size_t start_byte = spans[i].byte_offset;
            size_t end_byte = spans[j - 1].byte_offset + spans[j - 1].byte_len;
            pieces.emplace_back(text.substr(start_byte, end_byte - start_byte));
            i = j;
            continue;
        }

        // Alternatives 5, 6, 7: `\s*[\r\n]+|\s+(?!\S)|\s+`
        if (is_whitespace(spans[i].cp)) {
            size_t j = i;
            size_t last_newline_end = 0;
            while (j < n && is_whitespace(spans[j].cp)) {
                if (is_newline(spans[j].cp)) {
                    last_newline_end = j + 1;
                }
                j++;
            }
            if (last_newline_end > 0) {
                // Match `\s*[\r\n]+` up through the last newline in the whitespace run
                size_t start_byte = spans[i].byte_offset;
                size_t end_byte = spans[last_newline_end - 1].byte_offset + spans[last_newline_end - 1].byte_len;
                pieces.emplace_back(text.substr(start_byte, end_byte - start_byte));
                i = last_newline_end;
                continue;
            }
            // Pure non-newline whitespace run from i..j
            if (j < n && (j - i) > 1) {
                // `\s+(?!\S)` consumes all but the final space so the final space can prefix the next word/punct!
                size_t start_byte = spans[i].byte_offset;
                size_t end_byte = spans[j - 2].byte_offset + spans[j - 2].byte_len;
                pieces.emplace_back(text.substr(start_byte, end_byte - start_byte));
                i = j - 1;
                continue;
            }
            // Otherwise consume the entire whitespace run (`\s+`)
            size_t start_byte = spans[i].byte_offset;
            size_t end_byte = spans[j - 1].byte_offset + spans[j - 1].byte_len;
            pieces.emplace_back(text.substr(start_byte, end_byte - start_byte));
            i = j;
            continue;
        }

        // Fallback single character
        pieces.emplace_back(text.substr(spans[i].byte_offset, spans[i].byte_len));
        i++;
    }

    return pieces;
}

std::vector<int32_t> Qwen3BPETokenizer::bpe_word(const std::string& raw_piece) const {
    if (raw_piece.empty()) return {};

    // Convert raw bytes to ByteLevel unicode symbols
    std::vector<std::string> word;
    word.reserve(raw_piece.size());
    for (unsigned char b : raw_piece) {
        word.push_back(byte_to_unicode_[b]);
    }

    while (word.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_idx = 0;
        bool found = false;

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            std::string key = word[i] + "\x01" + word[i + 1];
            auto it = bpe_ranks_.find(key);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
                found = true;
            }
        }
        if (!found) break;

        const std::string first = word[best_idx];
        const std::string second = word[best_idx + 1];
        const std::string merged = first + second;

        std::vector<std::string> next_word;
        next_word.reserve(word.size() - 1);
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                next_word.push_back(merged);
                i += 2;
            } else {
                next_word.push_back(std::move(word[i]));
                i += 1;
            }
        }
        word = std::move(next_word);
    }

    std::vector<int32_t> ids;
    ids.reserve(word.size());
    for (const auto& sym : word) {
        auto it = vocab_to_id_.find(sym);
        if (it != vocab_to_id_.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::vector<int32_t> Qwen3BPETokenizer::encode(std::string_view text) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;

    auto pieces = pre_tokenize(text);
    for (const auto& piece : pieces) {
        auto piece_ids = bpe_word(piece);
        out.insert(out.end(), piece_ids.begin(), piece_ids.end());
    }
    return out;
}

std::string Qwen3BPETokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string byte_level_str;
    for (int32_t id : ids) {
        auto it = id_to_vocab_.find(id);
        if (it != id_to_vocab_.end()) {
            byte_level_str += it->second;
        }
    }
    auto spans = decode_utf8_spans(byte_level_str);
    std::string raw_utf8;
    raw_utf8.reserve(spans.size());
    for (const auto& s : spans) {
        auto it = unicode_to_byte_.find(s.cp);
        if (it != unicode_to_byte_.end()) {
            raw_utf8.push_back(static_cast<char>(it->second));
        } else {
            append_codepoint_utf8(raw_utf8, s.cp);
        }
    }
    return raw_utf8;
}

} // namespace kitteni
