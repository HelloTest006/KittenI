#pragma once

#include "core/json.hpp"
#include <string>
#include <filesystem>

namespace kitteni {

struct ResolvedRepo {
    std::filesystem::path root_dir;
    std::string repo_id;
    JsonValue config;
};

// Resolves a local path or HuggingFace repo ID (e.g. "KittenML/kitten-tts-2").
// Checks standard HuggingFace Hub cache first. If missing and `allow_download` is true,
// downloads the requested weight variant ("emb4", "packed", or "full") via curl.
ResolvedRepo resolve_repository(
    const std::string& model_or_path,
    const std::string& weights_variant = "emb4",
    const std::string& cache_dir = "",
    bool allow_download = true
);

// Ensure a specific relative file inside `repo` is present on disk, downloading from HF if needed.
std::filesystem::path ensure_repo_file(
    const ResolvedRepo& repo,
    const std::string& relative_path,
    bool allow_download = true
);

} // namespace kitteni
