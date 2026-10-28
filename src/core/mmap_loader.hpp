#pragma once

#include "kitteni/common.hpp"
#include "core/json.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace kitteni {

class MmapFile {
public:
    MmapFile() = default;
    explicit MmapFile(const std::string& path);
    ~MmapFile();

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
    void close();
};

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path);

    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    std::string get_metadata(const std::string& key, const std::string& fallback = "") const;

    bool has_tensor(const std::string& name) const;
    TensorView get_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    size_t file_size() const { return mmap_.size(); }
    const std::string& path() const { return mmap_.path(); }

private:
    MmapFile mmap_;
    size_t data_offset_ = 0;
    std::map<std::string, std::string> metadata_;
    std::map<std::string, TensorView> tensors_;
};

// Reader for uncompressed / stored .npz (ZIP of .npy arrays), used by voices/*.npz
class NpzArchive {
public:
    explicit NpzArchive(const std::string& path);

    bool has_array(const std::string& name) const;
    TensorView get_array(const std::string& name) const;
    std::vector<std::string> array_names() const;

private:
    MmapFile mmap_;
    std::map<std::string, TensorView> arrays_;
    // Storage for any deflate-compressed entries if needed
    std::map<std::string, std::vector<uint8_t>> decompressed_buffers_;
};

} // namespace kitteni
