#include "core/mmap_loader.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace kitteni {

MmapFile::MmapFile(const std::string& path) : path_(path) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Cannot open file for mmap: " + path);
    }
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        throw std::runtime_error("Cannot get file size: " + path);
    }
    size_ = static_cast<size_t>(fileSize.QuadPart);
    if (size_ == 0) {
        CloseHandle(hFile);
        throw std::runtime_error("Cannot mmap empty file: " + path);
    }
    HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        CloseHandle(hFile);
        throw std::runtime_error("CreateFileMapping failed for: " + path);
    }
    void* ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        throw std::runtime_error("MapViewOfFile failed for: " + path);
    }
    file_handle_ = hFile;
    map_handle_ = hMap;
    data_ = static_cast<const uint8_t*>(ptr);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Cannot open file for mmap: " + path);
    }
    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Invalid file size for mmap: " + path);
    }
    size_ = static_cast<size_t>(st.st_size);
    void* ptr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap failed for: " + path);
    }
    data_ = static_cast<const uint8_t*>(ptr);
#endif
}

MmapFile::~MmapFile() {
    close();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : path_(std::move(other.path_)),
      data_(other.data_),
      size_(other.size_)
#ifdef _WIN32
      , file_handle_(other.file_handle_),
      map_handle_(other.map_handle_)
#else
      , fd_(other.fd_)
#endif
{
    other.data_ = nullptr;
    other.size_ = 0;
#ifdef _WIN32
    other.file_handle_ = nullptr;
    other.map_handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        data_ = other.data_;
        size_ = other.size_;
#ifdef _WIN32
        file_handle_ = other.file_handle_;
        map_handle_ = other.map_handle_;
        other.file_handle_ = nullptr;
        other.map_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MmapFile::close() {
#ifdef _WIN32
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (map_handle_) {
        CloseHandle(static_cast<HANDLE>(map_handle_));
        map_handle_ = nullptr;
    }
    if (file_handle_) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
#else
    if (data_) {
        ::munmap(const_cast<uint8_t*>(data_), size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    size_ = 0;
}

namespace {

DType parse_safetensors_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    if (s == "I16") return DType::I16;
    if (s == "U16") return DType::U16;
    if (s == "I32") return DType::I32;
    if (s == "U32") return DType::U32;
    if (s == "I64") return DType::I64;
    if (s == "BOOL") return DType::BOOL;
    throw std::runtime_error("Unsupported safetensors dtype: " + s);
}

} // namespace

SafetensorsFile::SafetensorsFile(const std::string& path) : mmap_(path) {
    if (mmap_.size() < 8) {
        throw std::runtime_error("Safetensors file too small: " + path);
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, mmap_.data(), sizeof(uint64_t));
    if (8 + header_len > mmap_.size()) {
        throw std::runtime_error("Corrupt safetensors header length in: " + path);
    }
    data_offset_ = 8 + static_cast<size_t>(header_len);
    std::string_view header_sv(reinterpret_cast<const char*>(mmap_.data() + 8), static_cast<size_t>(header_len));
    JsonValue root = parse_json(header_sv);

    for (const auto& [key, val] : root.as_object()) {
        if (key == "__metadata__") {
            if (val.is_object()) {
                for (const auto& [mk, mv] : val.as_object()) {
                    if (mv.is_string()) {
                        metadata_[mk] = mv.as_string();
                    }
                }
            }
            continue;
        }
        TensorView tv;
        tv.dtype = parse_safetensors_dtype(val.at("dtype").as_string());
        for (const auto& dim_val : val.at("shape").as_array()) {
            tv.shape.push_back(dim_val.as_int());
        }
        const auto& offsets = val.at("data_offsets").as_array();
        if (offsets.size() != 2) {
            throw std::runtime_error("Invalid data_offsets for tensor: " + key);
        }
        size_t begin = static_cast<size_t>(offsets[0].as_int());
        size_t end = static_cast<size_t>(offsets[1].as_int());
        if (end < begin || data_offset_ + end > mmap_.size()) {
            throw std::runtime_error("Out-of-bounds data_offsets for tensor: " + key);
        }
        tv.data = mmap_.data() + data_offset_ + begin;
        tv.byte_size = end - begin;
        tensors_.emplace(key, std::move(tv));
    }
}

std::string SafetensorsFile::get_metadata(const std::string& key, const std::string& fallback) const {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? it->second : fallback;
}

bool SafetensorsFile::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

TensorView SafetensorsFile::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Missing tensor in safetensors: " + name);
    }
    return it->second;
}

std::vector<std::string> SafetensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [k, _] : tensors_) names.push_back(k);
    return names;
}

// Parse .npy buffer inside a .npz ZIP entry
namespace {

TensorView parse_npy_entry(const uint8_t* ptr, size_t len) {
    if (len < 10 || std::memcmp(ptr, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Invalid .npy magic header in NPZ archive");
    }
    uint8_t major = ptr[6];
    size_t header_len = 0;
    size_t offset = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        std::memcpy(&h16, ptr + 8, 2);
        header_len = h16;
        offset = 10;
    } else {
        uint32_t h32 = 0;
        std::memcpy(&h32, ptr + 8, 4);
        header_len = h32;
        offset = 12;
    }
    if (offset + header_len > len) {
        throw std::runtime_error("Truncated .npy header in NPZ archive");
    }
    std::string header(reinterpret_cast<const char*>(ptr + offset), header_len);
    TensorView tv;

    // Parse 'descr'
    auto descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) throw std::runtime_error("Missing 'descr' in .npy header");
    auto q1 = header.find('\'', descr_pos + 7);
    auto q2 = header.find('\'', q1 + 1);
    std::string descr = header.substr(q1 + 1, q2 - q1 - 1);
    if (descr.find("f4") != std::string::npos) tv.dtype = DType::F32;
    else if (descr.find("f2") != std::string::npos) tv.dtype = DType::F16;
    else if (descr.find("i8") != std::string::npos) tv.dtype = DType::I64;
    else if (descr.find("i4") != std::string::npos) tv.dtype = DType::I32;
    else if (descr.find("i2") != std::string::npos) tv.dtype = DType::I16;
    else if (descr.find("i1") != std::string::npos) tv.dtype = DType::I8;
    else if (descr.find("u1") != std::string::npos) tv.dtype = DType::U8;
    else throw std::runtime_error("Unsupported .npy dtype descr: " + descr);

    // Parse 'shape'
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) throw std::runtime_error("Missing 'shape' in .npy header");
    auto lp = header.find('(', shape_pos);
    auto rp = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    size_t idx = 0;
    while (idx < shape_str.size()) {
        while (idx < shape_str.size() && (shape_str[idx] == ' ' || shape_str[idx] == ',')) idx++;
        if (idx >= shape_str.size()) break;
        size_t next = idx;
        while (next < shape_str.size() && shape_str[next] >= '0' && shape_str[next] <= '9') next++;
        if (next > idx) {
            tv.shape.push_back(std::stoll(shape_str.substr(idx, next - idx)));
        }
        idx = next + 1;
    }
    if (tv.shape.empty()) tv.shape.push_back(1);

    size_t data_start = offset + header_len;
    tv.data = ptr + data_start;
    tv.byte_size = len - data_start;
    return tv;
}

} // namespace

NpzArchive::NpzArchive(const std::string& path) : mmap_(path) {
    const uint8_t* base = mmap_.data();
    size_t total = mmap_.size();
    size_t pos = 0;

    while (pos + 30 <= total) {
        uint32_t sig = 0;
        std::memcpy(&sig, base + pos, 4);
        if (sig != 0x04034B50U) break; // Local file header signature

        uint16_t flags = 0, method = 0, name_len = 0, extra_len = 0;
        uint32_t comp_size = 0, uncomp_size = 0;
        std::memcpy(&flags, base + pos + 6, 2);
        std::memcpy(&method, base + pos + 8, 2);
        std::memcpy(&comp_size, base + pos + 18, 4);
        std::memcpy(&uncomp_size, base + pos + 22, 4);
        std::memcpy(&name_len, base + pos + 26, 2);
        std::memcpy(&extra_len, base + pos + 28, 2);

        size_t header_end = pos + 30 + name_len + extra_len;
        if (header_end > total) break;

        std::string name(reinterpret_cast<const char*>(base + pos + 30), name_len);
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
            name.resize(name.size() - 4);
        }

        // Handle ZIP64 extra field if sizes are 0xFFFFFFFF
        uint64_t actual_comp = comp_size;
        if (comp_size == 0xFFFFFFFFU && extra_len >= 20) {
            std::memcpy(&actual_comp, base + pos + 30 + name_len + 12, 8);
        }

        if (header_end + actual_comp > total) break;

        if (method == 0) {
            arrays_[name] = parse_npy_entry(base + header_end, static_cast<size_t>(actual_comp));
        }
        pos = header_end + static_cast<size_t>(actual_comp);
    }
}

bool NpzArchive::has_array(const std::string& name) const {
    return arrays_.find(name) != arrays_.end();
}

TensorView NpzArchive::get_array(const std::string& name) const {
    auto it = arrays_.find(name);
    if (it == arrays_.end()) throw std::runtime_error("Missing array in NPZ: " + name);
    return it->second;
}

std::vector<std::string> NpzArchive::array_names() const {
    std::vector<std::string> names;
    for (const auto& [k, _] : arrays_) names.push_back(k);
    return names;
}

} // namespace kitteni
