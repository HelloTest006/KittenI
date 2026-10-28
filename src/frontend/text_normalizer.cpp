#include "frontend/text_normalizer.hpp"
#include <cstdint>
#include <regex>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

namespace kitteni {

namespace {

const std::unordered_set<std::string> kEmotionTags = {
    "mundane", "nervous", "tender", "angry", "excited",
    "stern", "sad", "contemplative", "surprised", "joyful"
};

const std::unordered_set<std::string> kVocalEventTags = {
    "pause", "sigh", "gasp", "laugh", "giggle",
    "sob", "scoff", "growl", "um", "gulp"
};

std::string to_lower_ascii(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    }
    return out;
}

std::string trim_ascii(std::string_view s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) b++;
    size_t e = s.size();
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) e--;
    return std::string(s.substr(b, e - b));
}

// Protect recognized `[emotion]`, `<event>`, and `(((emphasis)))` spans using `\xEE\x80\x80{idx}\xEE\x80\x81` (\uE000..\uE001)
std::pair<std::string, std::vector<std::string>> protect_tags(std::string_view input) {
    std::string out;
    std::vector<std::string> tags;
    size_t i = 0;
    const size_t n = input.size();

    while (i < n) {
        // Check `(((emphasis)))`
        if (i + 6 < n && input.substr(i, 3) == "(((") {
            size_t close = input.find(")))", i + 3);
            if (close != std::string_view::npos && (close - (i + 3)) >= 1 && (close - (i + 3)) <= 80) {
                std::string_view inner = input.substr(i + 3, close - (i + 3));
                if (inner.find_first_of("()\n") == std::string_view::npos) {
                    tags.emplace_back(input.substr(i, close + 3 - i));
                    out += "\xEE\x80\x80" + std::to_string(tags.size() - 1) + "\xEE\x80\x81";
                    i = close + 3;
                    continue;
                }
            }
        }
        // Check `[emotion]`
        if (input[i] == '[') {
            size_t close = input.find(']', i + 1);
            if (close != std::string_view::npos && (close - (i + 1)) <= 20) {
                std::string cand = to_lower_ascii(input.substr(i + 1, close - (i + 1)));
                if (kEmotionTags.count(cand)) {
                    tags.emplace_back(input.substr(i, close + 1 - i));
                    out += "\xEE\x80\x80" + std::to_string(tags.size() - 1) + "\xEE\x80\x81";
                    i = close + 1;
                    continue;
                }
            }
        }
        // Check `<event>`
        if (input[i] == '<') {
            size_t close = input.find('>', i + 1);
            if (close != std::string_view::npos && (close - (i + 1)) <= 20) {
                std::string cand = to_lower_ascii(input.substr(i + 1, close - (i + 1)));
                if (kVocalEventTags.count(cand)) {
                    tags.emplace_back(input.substr(i, close + 1 - i));
                    out += "\xEE\x80\x80" + std::to_string(tags.size() - 1) + "\xEE\x80\x81";
                    i = close + 1;
                    continue;
                }
            }
        }
        out.push_back(input[i]);
        i++;
    }
    return {out, tags};
}

std::string restore_tags(std::string text, const std::vector<std::string>& tags) {
    for (size_t i = 0; i < tags.size(); ++i) {
        std::string marker = "\xEE\x80\x80" + std::to_string(i) + "\xEE\x80\x81";
        size_t pos = 0;
        while ((pos = text.find(marker, pos)) != std::string::npos) {
            text.replace(pos, marker.size(), tags[i]);
            pos += tags[i].size();
        }
    }
    return text;
}

// Number to words (English)
const char* kOnes[] = {
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
};
const char* kTens[] = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
};
const char* kScales[] = {"", "thousand", "million", "billion", "trillion"};

std::string three_digits_to_words(int n) {
    if (n == 0) return "";
    std::string out;
    int h = n / 100;
    int rem = n % 100;
    if (h > 0) {
        out = std::string(kOnes[h]) + " hundred";
    }
    if (rem > 0) {
        if (!out.empty()) out += " ";
        if (rem < 20) {
            out += kOnes[rem];
        } else {
            out += kTens[rem / 10];
            if (rem % 10 != 0) {
                out += "-" + std::string(kOnes[rem % 10]);
            }
        }
    }
    return out;
}

std::string int_to_words(int64_t n) {
    if (n == 0) return "zero";
    if (n < 0) return "negative " + int_to_words(-n);

    if (n >= 100 && n <= 9999 && (n % 100 == 0) && (n % 1000 != 0)) {
        int64_t h = n / 100;
        if (h < 20) {
            return std::string(kOnes[h]) + " hundred";
        }
    }

    std::vector<std::string> parts;
    int scale_idx = 0;
    while (n > 0 && scale_idx < 5) {
        int chunk = static_cast<int>(n % 1000);
        if (chunk > 0) {
            std::string cw = three_digits_to_words(chunk);
            if (scale_idx > 0) cw += " " + std::string(kScales[scale_idx]);
            parts.push_back(std::move(cw));
        }
        n /= 1000;
        scale_idx++;
    }
    std::string result;
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        if (!result.empty()) result += " ";
        result += *it;
    }
    return result;
}

std::string float_str_to_words(const std::string& raw) {
    if (raw.empty()) return "";
    bool neg = (raw[0] == '-');
    std::string s = neg ? raw.substr(1) : raw;
    auto dot = s.find('.');
    std::string result;
    if (dot != std::string::npos) {
        std::string int_part = s.substr(0, dot);
        std::string dec_part = s.substr(dot + 1);
        std::string int_w = int_part.empty() ? "zero" : int_to_words(std::stoll(int_part));
        std::string dec_w;
        for (char c : dec_part) {
            if (c >= '0' && c <= '9') {
                if (!dec_w.empty()) dec_w += " ";
                dec_w += (c == '0') ? "zero" : kOnes[c - '0'];
            }
        }
        result = int_w + " point " + dec_w;
    } else {
        result = int_to_words(std::stoll(s));
    }
    return neg ? ("negative " + result) : result;
}

std::string ordinal_to_words(int64_t n) {
    std::string word = int_to_words(n);
    std::string prefix, last, joiner;
    auto dash = word.rfind('-');
    auto space = word.rfind(' ');
    if (dash != std::string::npos && (space == std::string::npos || dash > space)) {
        prefix = word.substr(0, dash);
        last = word.substr(dash + 1);
        joiner = "-";
    } else if (space != std::string::npos) {
        prefix = word.substr(0, space);
        last = word.substr(space + 1);
        joiner = " ";
    } else {
        last = word;
    }

    static const std::unordered_map<std::string, std::string> kOrdExceptions = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"four", "fourth"},
        {"five", "fifth"}, {"six", "sixth"}, {"seven", "seventh"}, {"eight", "eighth"},
        {"nine", "ninth"}, {"twelve", "twelfth"}
    };
    std::string last_ord;
    auto it = kOrdExceptions.find(last);
    if (it != kOrdExceptions.end()) {
        last_ord = it->second;
    } else if (!last.empty() && last.back() == 't') {
        last_ord = last + "h";
    } else if (!last.empty() && last.back() == 'e') {
        last_ord = last.substr(0, last.size() - 1) + "th";
    } else if (!last.empty() && last.back() == 'y') {
        last_ord = last.substr(0, last.size() - 1) + "ieth";
    } else {
        last_ord = last + "th";
    }
    return prefix.empty() ? last_ord : (prefix + joiner + last_ord);
}

void replace_all(std::string& s, std::string_view from, std::string_view to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string dashes_to_commas(std::string s) {
    replace_all(s, "\xE2\x80\x94", " , "); // em-dash —
    replace_all(s, "\xE2\x80\x93", " , "); // en-dash –
    replace_all(s, "--", " , ");

    s = std::regex_replace(s, std::regex(R"(,\s*,+)"), ", ");
    s = std::regex_replace(s, std::regex(R"(\s+([,.!?;:]))"), "$1");
    s = std::regex_replace(s, std::regex(R"(,\s*([.!?]))"), "$1");
    s = std::regex_replace(s, std::regex(R"(\s{2,})"), " ");
    s = std::regex_replace(s, std::regex(R"(^\s*,\s*)"), "");
    return trim_ascii(s);
}

std::string capitalize_and_terminate(std::string s) {
    if (s.empty()) return s;
    s = std::regex_replace(s, std::regex(R"(\s+([?!]))"), "$1");
    for (char& c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            if (c >= 'a' && c <= 'z') c = static_cast<char>(c - ('a' - 'A'));
            break;
        }
    }
    s = trim_ascii(s);
    if (s.empty()) return s;

    // Check if already terminated by . ! ? " ' ) ] or UTF-8 closing curly quotes \xE2\x80\x99 / \xE2\x80\x9D
    char back = s.back();
    if (back == '.' || back == '!' || back == '?' || back == '"' || back == '\'' || back == ')' || back == ']') {
        return s;
    }
    if (s.size() >= 3) {
        std::string_view tail3(s.data() + s.size() - 3, 3);
        if (tail3 == "\xE2\x80\x99" || tail3 == "\xE2\x80\x9D") {
            return s;
        }
    }
    // Strip trailing non-terminal marks [,;:-]
    while (!s.empty() && (s.back() == ',' || s.back() == ';' || s.back() == ':' || s.back() == '-' || s.back() == ' ')) {
        s.pop_back();
    }
    return s.empty() ? s : (s + ".");
}

} // namespace

bool has_expression_tag(std::string_view text) {
    auto [_, tags] = protect_tags(text);
    return !tags.empty();
}

size_t utf8_codepoint_len(std::string_view text) {
    size_t count = 0;
    for (unsigned char b : text) {
        if ((b & 0xC0U) != 0x80U) count++;
    }
    return count;
}

std::string normalize_text(std::string_view input, bool normalize) {
    auto [s, saved_tags] = protect_tags(input);

    // Collapse hard line wraps and missing space after punctuation
    s = std::regex_replace(s, std::regex(R"(\s*\n+\s*)"), " ");
    s = std::regex_replace(s, std::regex(R"(([a-z])([.!?;:])([A-Z]))"), "$1$2 $3");

    // Ellipsis -> ", "
    replace_all(s, "\xE2\x80\xA6", ", ");
    s = std::regex_replace(s, std::regex(R"((?:\.\s+){2,}\.|\.\.\.)"), ", ");

    // Common math glyphs
    static const std::pair<const char*, const char*> kMathSubs[] = {
        {"\xCE\xB1", " alpha "}, {"\xCE\xB2", " beta "}, {"\xCE\xB3", " gamma "},
        {"\xCE\xB4", " delta "}, {"\xCE\xBB", " lambda "}, {"\xCE\xBC", " mu "},
        {"\xCF\x80", " pi "}, {"\xCF\x83", " sigma "}, {"\xCF\x89", " omega "},
        {"\xE2\x89\xA0", " is not equal to "}, {"\xE2\x89\x88", " approximately equals "},
        {"\xE2\x89\xA4", " is less than or equal to "}, {"\xE2\x89\xA5", " is greater than or equal to "},
        {"\xC2\xB1", " plus or minus "}, {"\xC3\x97", " times "}, {"\xC3\xB7", " divided by "},
        {"=", " equals "}
    };
    for (const auto& [glyph, word] : kMathSubs) {
        replace_all(s, glyph, word);
    }

    // Version strings like v0.8 -> v 0 point 8
    {
        std::regex ver_re(R"(\b([vV])(\d+(?:\.\d+)+))");
        std::string out;
        size_t last = 0;
        for (std::sregex_iterator it(s.begin(), s.end(), ver_re), end; it != end; ++it) {
            out += s.substr(last, it->position() - last);
            std::string nums = (*it)[2].str();
            replace_all(nums, ".", " point ");
            out += (*it)[1].str() + " " + nums;
            last = it->position() + it->length();
        }
        out += s.substr(last);
        s = std::move(out);
    }

    // Reformat non-tag brackets `[x]`, `<x>`, `(x)` -> `, x,`
    s = std::regex_replace(s, std::regex(R"(\[([^\[\]\n]{1,400})\])"), ", $1,");
    s = std::regex_replace(s, std::regex(R"(<([^<>\n]{1,60})>)"), ", $1,");
    s = std::regex_replace(s, std::regex(R"(\(([^()\n]{1,400})\))"), ", $1,");
    while (true) {
        std::string next = std::regex_replace(s, std::regex(R"(,\s*([,.!?;:]))"), "$1");
        if (next == s) break;
        s = std::move(next);
    }

    s = dashes_to_commas(s);
    if (trim_ascii(s).empty()) {
        return restore_tags(s, saved_tags);
    }

    if (!normalize) {
        return restore_tags(capitalize_and_terminate(s), saved_tags);
    }

    // Paired straight quotes -> curly quotes “...”
    s = std::regex_replace(s, std::regex(R"xxx("([^"\n]*)")xxx"), "\xE2\x80\x9C$1\xE2\x80\x9D");

    // Expand currencies: $12.50, $100
    {
        std::regex curr_re(R"(\$\s*([\d,]+)(?:\.(\d{1,2}))?\b)");
        std::string out;
        size_t last = 0;
        for (std::sregex_iterator it(s.begin(), s.end(), curr_re), end; it != end; ++it) {
            out += s.substr(last, it->position() - last);
            std::string dollars_str = (*it)[1].str();
            dollars_str.erase(std::remove(dollars_str.begin(), dollars_str.end(), ','), dollars_str.end());
            int64_t d_val = std::stoll(dollars_str);
            std::string phrase = int_to_words(d_val) + (d_val == 1 ? " dollar" : " dollars");
            if ((*it)[2].matched) {
                std::string c_str = (*it)[2].str();
                if (c_str.size() == 1) c_str.push_back('0');
                int64_t c_val = std::stoll(c_str);
                if (c_val > 0) {
                    phrase += " and " + int_to_words(c_val) + (c_val == 1 ? " cent" : " cents");
                }
            }
            out += phrase;
            last = it->position() + it->length();
        }
        out += s.substr(last);
        s = std::move(out);
    }

    // Expand percentages: 50%, 3.5%
    {
        std::regex pct_re(R"((-?\d+(?:\.\d+)?)\s*%)");
        std::string out;
        size_t last = 0;
        for (std::sregex_iterator it(s.begin(), s.end(), pct_re), end; it != end; ++it) {
            out += s.substr(last, it->position() - last);
            out += float_str_to_words((*it)[1].str()) + " percent";
            last = it->position() + it->length();
        }
        out += s.substr(last);
        s = std::move(out);
    }

    // Expand ordinals: 1st, 2nd, 3rd, 4th...
    {
        std::regex ord_re(R"(\b(\d+)(st|nd|rd|th)\b)", std::regex::icase);
        std::string out;
        size_t last = 0;
        for (std::sregex_iterator it(s.begin(), s.end(), ord_re), end; it != end; ++it) {
            out += s.substr(last, it->position() - last);
            out += ordinal_to_words(std::stoll((*it)[1].str()));
            last = it->position() + it->length();
        }
        out += s.substr(last);
        s = std::move(out);
    }

    // Expand common abbreviations: Dr., Mr., Mrs., Ms., Prof.
    s = std::regex_replace(s, std::regex(R"(\bDr\.)"), "Doctor");
    s = std::regex_replace(s, std::regex(R"(\bMr\.)"), "Mister");
    s = std::regex_replace(s, std::regex(R"(\bMrs\.)"), "Misses");
    s = std::regex_replace(s, std::regex(R"(\bProf\.)"), "Professor");

    // Expand standalone numbers (skipping numbers inside sentinel `\xEE\x80\x80...\xEE\x80\x81`)
    {
        std::regex num_re(R"((?:\xEE\x80\x80\d+\xEE\x80\x81)|(?:\b\d[\d,]*(?:\.\d+)?\b))");
        std::string out;
        size_t last = 0;
        for (std::sregex_iterator it(s.begin(), s.end(), num_re), end; it != end; ++it) {
            out += s.substr(last, it->position() - last);
            std::string match_s = it->str();
            if (match_s.size() >= 3 && match_s.substr(0, 3) == "\xEE\x80\x80") {
                out += match_s; // Keep sentinel untouched!
            } else {
                match_s.erase(std::remove(match_s.begin(), match_s.end(), ','), match_s.end());
                out += float_str_to_words(match_s);
            }
            last = it->position() + it->length();
        }
        out += s.substr(last);
        s = std::move(out);
    }

    s = capitalize_and_terminate(dashes_to_commas(s));
    return restore_tags(s, saved_tags);
}

std::vector<std::string> split_for_synthesis(
    std::string_view text,
    int max_chars,
    int min_chars
) {
    std::string stripped = trim_ascii(text);
    if (stripped.empty()) return {};
    if (max_chars <= 0 || static_cast<int>(utf8_codepoint_len(stripped)) <= max_chars) {
        return {stripped};
    }

    // Split on sentence boundaries `(?<=[.!?])["’”\')\]]*\s+`
    std::vector<std::string> sentences;
    size_t start = 0;
    for (size_t i = 0; i < stripped.size(); ++i) {
        char c = stripped[i];
        if (c == '.' || c == '!' || c == '?') {
            size_t j = i + 1;
            while (j < stripped.size()) {
                if (stripped[j] == '"' || stripped[j] == '\'' || stripped[j] == ')' || stripped[j] == ']') {
                    j++;
                } else if (j + 2 < stripped.size() &&
                           (stripped.substr(j, 3) == "\xE2\x80\x99" || stripped.substr(j, 3) == "\xE2\x80\x9D")) {
                    j += 3;
                } else {
                    break;
                }
            }
            if (j < stripped.size() && (stripped[j] == ' ' || stripped[j] == '\t' || stripped[j] == '\n')) {
                size_t ws_end = j;
                while (ws_end < stripped.size() && (stripped[ws_end] == ' ' || stripped[ws_end] == '\t' || stripped[ws_end] == '\n')) {
                    ws_end++;
                }
                std::string sent = trim_ascii(std::string_view(stripped).substr(start, j - start));
                if (!sent.empty()) sentences.push_back(std::move(sent));
                start = ws_end;
                i = ws_end - 1;
            }
        }
    }
    if (start < stripped.size()) {
        std::string tail = trim_ascii(std::string_view(stripped).substr(start));
        if (!tail.empty()) sentences.push_back(std::move(tail));
    }

    std::vector<std::string> parts;
    std::string current;

    for (std::string sentence : sentences) {
        while (static_cast<int>(sentence.size()) > max_chars) {
            size_t cut = 0;
            for (const char* sep : {", ", "; ", " \xE2\x80\x94 "}) {
                auto found = sentence.substr(0, static_cast<size_t>(max_chars)).rfind(sep);
                if (found != std::string::npos) cut = std::max(cut, found);
            }
            cut = (cut > static_cast<size_t>(max_chars / 3)) ? (cut + 1) : static_cast<size_t>(max_chars);
            std::string head = trim_ascii(std::string_view(sentence).substr(0, cut));
            sentence = trim_ascii(std::string_view(sentence).substr(cut));
            if (!current.empty()) {
                parts.push_back(std::move(current));
                current.clear();
            }
            parts.push_back(std::move(head));
        }
        if (current.empty()) {
            current = std::move(sentence);
        } else if (static_cast<int>(current.size() + 1 + sentence.size()) <= max_chars) {
            current += " " + sentence;
        } else {
            parts.push_back(std::move(current));
            current = std::move(sentence);
        }
    }
    if (!current.empty()) {
        parts.push_back(std::move(current));
    }

    // Fold undersized chunks (< min_chars) into neighbors up to 1.4 * max_chars
    const int limit = static_cast<int>(max_chars * 1.4);
    bool changed = true;
    while (changed && parts.size() > 1 && min_chars > 0) {
        changed = false;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (static_cast<int>(parts[i].size()) >= min_chars) continue;
            bool prev_ok = (i > 0) && (static_cast<int>(parts[i - 1].size() + 1 + parts[i].size()) <= limit);
            bool next_ok = (i + 1 < parts.size()) && (static_cast<int>(parts[i].size() + 1 + parts[i + 1].size()) <= limit);
            if (prev_ok && (!next_ok || parts[i - 1].size() <= parts[i + 1].size())) {
                parts[i - 1] = trim_ascii(parts[i - 1]) + " " + trim_ascii(parts[i]);
                parts.erase(parts.begin() + static_cast<ptrdiff_t>(i));
                changed = true;
                break;
            } else if (next_ok) {
                parts[i] = trim_ascii(parts[i]) + " " + trim_ascii(parts[i + 1]);
                parts.erase(parts.begin() + static_cast<ptrdiff_t>(i + 1));
                changed = true;
                break;
            }
        }
    }

    // Ensure terminal punctuation on each chunk
    for (auto& c : parts) {
        c = trim_ascii(c);
        if (c.empty()) continue;
        char b = c.back();
        bool has_term = (b == '.' || b == '!' || b == '?' || b == ',' || b == ';' || b == ':' ||
                         b == '"' || b == '\'' || b == ')' || b == ']');
        if (!has_term && c.size() >= 3) {
            std::string_view tail3(c.data() + c.size() - 3, 3);
            if (tail3 == "\xE2\x80\x99" || tail3 == "\xE2\x80\x9D") has_term = true;
        }
        if (!has_term) c.push_back(',');
    }
    return parts;
}

} // namespace kitteni
