#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <variant>
#include <cstdint>
#include <stdexcept>

namespace kitteni {

struct JsonValue {
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;

    std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object> data;

    JsonValue() : data(nullptr) {}
    JsonValue(std::nullptr_t) : data(nullptr) {}
    JsonValue(bool b) : data(b) {}
    JsonValue(int64_t i) : data(i) {}
    JsonValue(int i) : data(static_cast<int64_t>(i)) {}
    JsonValue(double d) : data(d) {}
    JsonValue(std::string s) : data(std::move(s)) {}
    JsonValue(const char* s) : data(std::string(s)) {}
    JsonValue(Array a) : data(std::move(a)) {}
    JsonValue(Object o) : data(std::move(o)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_int() const { return std::holds_alternative<int64_t>(data); }
    bool is_number() const { return is_int() || std::holds_alternative<double>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    bool as_bool(bool default_val = false) const {
        if (auto* p = std::get_if<bool>(&data)) return *p;
        return default_val;
    }

    int64_t as_int(int64_t default_val = 0) const {
        if (auto* p = std::get_if<int64_t>(&data)) return *p;
        if (auto* p = std::get_if<double>(&data)) return static_cast<int64_t>(*p);
        if (auto* p = std::get_if<std::string>(&data)) return std::stoll(*p);
        return default_val;
    }

    double as_float(double default_val = 0.0) const {
        if (auto* p = std::get_if<double>(&data)) return *p;
        if (auto* p = std::get_if<int64_t>(&data)) return static_cast<double>(*p);
        if (auto* p = std::get_if<std::string>(&data)) return std::stod(*p);
        return default_val;
    }

    const std::string& as_string() const {
        if (auto* p = std::get_if<std::string>(&data)) return *p;
        throw std::runtime_error("JsonValue is not a string");
    }

    std::string value_string(const std::string& key, const std::string& default_val = "") const {
        if (auto* o = std::get_if<Object>(&data)) {
            auto it = o->find(key);
            if (it != o->end() && it->second.is_string()) return it->second.as_string();
        }
        return default_val;
    }

    int64_t value_int(const std::string& key, int64_t default_val = 0) const {
        if (auto* o = std::get_if<Object>(&data)) {
            auto it = o->find(key);
            if (it != o->end() && it->second.is_number()) return it->second.as_int();
        }
        return default_val;
    }

    double value_float(const std::string& key, double default_val = 0.0) const {
        if (auto* o = std::get_if<Object>(&data)) {
            auto it = o->find(key);
            if (it != o->end() && it->second.is_number()) return it->second.as_float();
        }
        return default_val;
    }

    bool value_bool(const std::string& key, bool default_val = false) const {
        if (auto* o = std::get_if<Object>(&data)) {
            auto it = o->find(key);
            if (it != o->end() && it->second.is_bool()) return it->second.as_bool();
        }
        return default_val;
    }

    const Array& as_array() const {
        if (auto* p = std::get_if<Array>(&data)) return *p;
        throw std::runtime_error("JsonValue is not an array");
    }

    const Object& as_object() const {
        if (auto* p = std::get_if<Object>(&data)) return *p;
        throw std::runtime_error("JsonValue is not an object");
    }

    bool contains(const std::string& key) const {
        if (auto* o = std::get_if<Object>(&data)) return o->find(key) != o->end();
        return false;
    }

    const JsonValue& at(const std::string& key) const {
        const auto& o = as_object();
        auto it = o.find(key);
        if (it == o.end()) throw std::runtime_error("Missing JSON key: " + key);
        return it->second;
    }

    const JsonValue& operator[](const std::string& key) const {
        return at(key);
    }

    const JsonValue& operator[](size_t idx) const {
        const auto& a = as_array();
        if (idx >= a.size()) throw std::out_of_range("JSON array index out of range");
        return a[idx];
    }

    size_t size() const {
        if (auto* a = std::get_if<Array>(&data)) return a->size();
        if (auto* o = std::get_if<Object>(&data)) return o->size();
        return 0;
    }
};

JsonValue parse_json(std::string_view input);
JsonValue parse_json_file(const std::string& path);

} // namespace kitteni
