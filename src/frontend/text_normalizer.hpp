#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cstddef>

namespace kitteni {

// Returns true when `text` contains a recognized `[emotion]` tag, `<vocal_event>` tag,
// or `(((emphasis)))` span, triggering `{emo: 1}` conditioning in KittenTTS 2.
bool has_expression_tag(std::string_view text);

// Count Unicode codepoints in UTF-8 string (matching Python `len(str)`)
size_t utf8_codepoint_len(std::string_view text);

// Normalize written text into spoken form for KittenTTS 2:
// - Protects `[emotion]`, `<event>`, and `(((emphasis)))` spans via private-use sentinels
// - Expands math/Greek symbols, version strings, currencies, percentages, ordinals, times, dates, units, numbers
// - Converts paired straight quotes, rewrites non-tag brackets to commas, and normalizes dashes/punctuation
std::string normalize_text(std::string_view text, bool normalize = true);

// Sentence-aware greedy chunker matching `kittenml.kittentts2.text.split_for_synthesis`:
// - Splits long text into chunks <= `max_chars` (default 380)
// - Folds undersized chunks (< `min_chars`, default 130) into neighbors up to `1.4 * max_chars`
// - Ensures every chunk ends with terminal/prosodic punctuation
std::vector<std::string> split_for_synthesis(
    std::string_view text,
    int max_chars = 380,
    int min_chars = 130
);

} // namespace kitteni
