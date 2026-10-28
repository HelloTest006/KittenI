#include "core/json.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstdlib>

namespace kitteni {

namespace {

class Parser {
public:
    explicit Parser(std::string_view src) : src_(src), pos_(0) {}

    JsonValue parse() {
        skip_ws();
        JsonValue val = parse_value();
        skip_ws();
        return val;
    }

private:
    std::string_view src_;
    size_t pos_;

    void skip_ws() {
        while (pos_ < src_.size() && (src_[pos_] == ' ' || src_[pos_] == '\t' || src_[pos_] == '\n' || src_[pos_] == '\r')) {
            pos_++;
        }
    }

    char peek() const {
        if (pos_ >= src_.size()) throw std::runtime_error("Unexpected end of JSON input");
        return src_[pos_];
    }

    char get() {
        char c = peek();
        pos_++;
        return c;
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return JsonValue(parse_string());
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw std::runtime_error(std::string("Unexpected character in JSON: ") + c);
    }

    JsonValue parse_object() {
        get(); // '{'
        JsonValue::Object obj;
        skip_ws();
        if (peek() == '}') {
            get();
            return JsonValue(std::move(obj));
        }
        while (true) {
            skip_ws();
            if (peek() != '"') throw std::runtime_error("Expected string key in JSON object");
            std::string key = parse_string();
            skip_ws();
            if (get() != ':') throw std::runtime_error("Expected ':' after object key");
            JsonValue val = parse_value();
            obj.emplace(std::move(key), std::move(val));
            skip_ws();
            char next = get();
            if (next == '}') break;
            if (next != ',') throw std::runtime_error("Expected ',' or '}' in JSON object");
        }
        return JsonValue(std::move(obj));
    }

    JsonValue parse_array() {
        get(); // '['
        JsonValue::Array arr;
        skip_ws();
        if (peek() == ']') {
            get();
            return JsonValue(std::move(arr));
        }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            char next = get();
            if (next == ']') break;
            if (next != ',') throw std::runtime_error("Expected ',' or ']' in JSON array");
        }
        return JsonValue(std::move(arr));
    }

    static void append_utf8(std::string& out, uint32_t cp) {
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

    uint32_t parse_hex4() {
        if (pos_ + 4 > src_.size()) throw std::runtime_error("Truncated unicode escape in JSON");
        uint32_t val = 0;
        for (int i = 0; i < 4; ++i) {
            char c = src_[pos_++];
            val <<= 4;
            if (c >= '0' && c <= '9') val |= (c - '0');
            else if (c >= 'a' && c <= 'f') val |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val |= (c - 'A' + 10);
            else throw std::runtime_error("Invalid hex digit in unicode escape");
        }
        return val;
    }

    std::string parse_string() {
        get(); // '"'
        std::string out;
        while (pos_ < src_.size()) {
            char c = src_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= src_.size()) throw std::runtime_error("Untrminated escape in JSON string");
                char esc = src_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (pos_ + 2 <= src_.size() && src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
                                pos_ += 2;
                                uint32_t low = parse_hex4();
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: out.push_back(esc); break;
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated string in JSON");
    }

    JsonValue parse_bool() {
        if (src_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return JsonValue(true);
        }
        if (src_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return JsonValue(false);
        }
        throw std::runtime_error("Invalid boolean literal in JSON");
    }

    JsonValue parse_null() {
        if (src_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return JsonValue(nullptr);
        }
        throw std::runtime_error("Invalid null literal in JSON");
    }

    JsonValue parse_number() {
        size_t start = pos_;
        bool is_float = false;
        if (src_[pos_] == '-') pos_++;
        while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') pos_++;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            is_float = true;
            pos_++;
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') pos_++;
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            is_float = true;
            pos_++;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) pos_++;
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') pos_++;
        }
        std::string num_str(src_.substr(start, pos_ - start));
        if (is_float) {
            return JsonValue(std::strtod(num_str.c_str(), nullptr));
        } else {
            return JsonValue(static_cast<int64_t>(std::strtoll(num_str.c_str(), nullptr, 10)));
        }
    }
};

} // namespace

JsonValue parse_json(std::string_view input) {
    return Parser(input).parse();
}

JsonValue parse_json_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open JSON file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_json(ss.str());
}

} // namespace kitteni
