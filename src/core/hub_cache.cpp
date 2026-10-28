#include "core/hub_cache.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace kitteni {

namespace {

fs::path default_hf_hub_cache() {
    if (const char* env_hub = std::getenv("HF_HUB_CACHE")) {
        if (*env_hub) return fs::path(env_hub);
    }
    if (const char* env_home = std::getenv("HF_HOME")) {
        if (*env_home) return fs::path(env_home) / "hub";
    }
#ifdef _WIN32
    if (const char* up = std::getenv("USERPROFILE")) {
        return fs::path(up) / ".cache" / "huggingface" / "hub";
    }
#else
    if (const char* home = std::getenv("HOME")) {
        return fs::path(home) / ".cache" / "huggingface" / "hub";
    }
#endif
    return fs::current_path() / ".cache" / "huggingface" / "hub";
}

std::string sanitize_repo_dir_name(const std::string& repo_id) {
    std::string out = "models--";
    for (char c : repo_id) {
        if (c == '/') out += "--";
        else out.push_back(c);
    }
    return out;
}

void download_hf_file(const std::string& repo_id, const std::string& rel_path, const fs::path& dest_path) {
    fs::create_directories(dest_path.parent_path());
    fs::path tmp_path = dest_path;
    tmp_path += ".partial";

    std::string url = "https://huggingface.co/" + repo_id + "/resolve/main/" + rel_path;
    std::string cmd = "curl.exe -L --fail --retry 3 -o \"" + tmp_path.string() + "\" \"" + url + "\"";
#ifndef _WIN32
    cmd = "curl -L --fail --retry 3 -o \"" + tmp_path.string() + "\" \"" + url + "\"";
#endif
    std::cerr << "[KittenI] Downloading " << repo_id << "/" << rel_path << " ...\n";
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::exists(tmp_path)) {
        throw std::runtime_error("Failed to download from HuggingFace: " + url);
    }
    if (fs::exists(dest_path)) {
        fs::remove(dest_path);
    }
    fs::rename(tmp_path, dest_path);
}

} // namespace

ResolvedRepo resolve_repository(
    const std::string& model_or_path,
    const std::string& weights_variant,
    const std::string& cache_dir,
    bool allow_download
) {
    fs::path candidate(model_or_path);
    if (fs::is_directory(candidate) && fs::exists(candidate / "config.json")) {
        ResolvedRepo res;
        res.root_dir = candidate;
        res.repo_id = "";
        res.config = parse_json_file((candidate / "config.json").string());
        return res;
    }

    std::string repo_id = model_or_path.find('/') != std::string::npos
                              ? model_or_path
                              : ("KittenML/" + model_or_path);

    fs::path hub_cache = cache_dir.empty() ? default_hf_hub_cache() : fs::path(cache_dir);
    fs::path model_cache_dir = hub_cache / sanitize_repo_dir_name(repo_id);
    fs::path snapshot_dir;

    // Check if HF Hub snapshot already exists
    fs::path refs_main = model_cache_dir / "refs" / "main";
    if (fs::exists(refs_main)) {
        std::ifstream f(refs_main);
        std::string rev;
        std::getline(f, rev);
        while (!rev.empty() && (rev.back() == '\n' || rev.back() == '\r' || rev.back() == ' ')) {
            rev.pop_back();
        }
        if (!rev.empty() && fs::is_directory(model_cache_dir / "snapshots" / rev)) {
            snapshot_dir = model_cache_dir / "snapshots" / rev;
        }
    }
    if (snapshot_dir.empty() && fs::is_directory(model_cache_dir / "snapshots")) {
        for (const auto& entry : fs::directory_iterator(model_cache_dir / "snapshots")) {
            if (entry.is_directory()) {
                snapshot_dir = entry.path();
                break;
            }
        }
    }
    if (snapshot_dir.empty()) {
        snapshot_dir = model_cache_dir / "snapshots" / "main";
        fs::create_directories(snapshot_dir);
    }

    fs::path config_path = snapshot_dir / "config.json";
    if (!fs::exists(config_path)) {
        if (!allow_download) {
            throw std::runtime_error("Model config.json not found in cache and offline mode enabled: " + config_path.string());
        }
        download_hf_file(repo_id, "config.json", config_path);
    }

    ResolvedRepo res;
    res.root_dir = snapshot_dir;
    res.repo_id = repo_id;
    res.config = parse_json_file(config_path.string());

    if (res.config.value_string("type") != "KITTEN2") {
        throw std::runtime_error("KittenI only supports new KITTEN2 ternary checkpoints, got type: " +
                                 res.config.value_string("type", "UNKNOWN"));
    }

    // Ensure LM config and requested weights file exist if allow_download is set
    std::string lm_dir = res.config.value_string("lm_dir", "lm");
    ensure_repo_file(res, lm_dir + "/config.json", allow_download);

    if (!weights_variant.empty()) {
        std::string weight_rel;
        if (weights_variant == "emb4") {
            weight_rel = res.config.value_string("lm_emb4", "lm/model-tl2-emb4.safetensors");
        } else if (weights_variant == "packed") {
            weight_rel = res.config.value_string("lm_packed", "lm/model-ternary.safetensors");
        } else if (weights_variant == "full") {
            weight_rel = lm_dir + "/model.safetensors";
        }
        if (!weight_rel.empty()) {
            ensure_repo_file(res, weight_rel, allow_download);
        }
    }

    return res;
}

fs::path ensure_repo_file(
    const ResolvedRepo& repo,
    const std::string& relative_path,
    bool allow_download
) {
    fs::path full = repo.root_dir / fs::path(relative_path);
    if (fs::exists(full)) {
        return full;
    }
    if (!allow_download || repo.repo_id.empty()) {
        throw std::runtime_error("Required model file missing: " + full.string());
    }
    download_hf_file(repo.repo_id, relative_path, full);
    return full;
}

} // namespace kitteni
