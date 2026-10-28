#include "kernels/cpu_features.hpp"
#include <thread>
#include <sstream>
#include <atomic>
#include <stdexcept>
#include <algorithm>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace kitteni {

namespace {

CpuCapabilities detect_cpu() {
    CpuCapabilities caps;
    unsigned int threads = std::thread::hardware_concurrency();
    caps.logical_cores = threads > 0 ? static_cast<int>(threads) : 1;

#if defined(__aarch64__) || defined(_M_ARM64)
    caps.neon = true;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 0);
    int max_id = info[0];
    if (max_id >= 1) {
        __cpuidex(info, 1, 0);
        caps.fma = (info[2] & (1 << 12)) != 0;
        caps.f16c = (info[2] & (1 << 29)) != 0;
    }
    if (max_id >= 7) {
        __cpuidex(info, 7, 0);
        caps.avx2 = (info[1] & (1 << 5)) != 0;
        caps.avx512f = (info[1] & (1 << 16)) != 0;
        caps.avx512bw = (info[1] & (1 << 30)) != 0;
        caps.avx512vl = (info[1] & (1U << 31)) != 0;
        caps.avx512_vnni = (info[2] & (1 << 11)) != 0;
        __cpuidex(info, 7, 1);
        caps.avx_vnni = (info[0] & (1 << 4)) != 0;
    }
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps.fma = (ecx & (1U << 12)) != 0;
        caps.f16c = (ecx & (1U << 29)) != 0;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps.avx2 = (ebx & (1U << 5)) != 0;
        caps.avx512f = (ebx & (1U << 16)) != 0;
        caps.avx512bw = (ebx & (1U << 30)) != 0;
        caps.avx512vl = (ebx & (1U << 31)) != 0;
        caps.avx512_vnni = (ecx & (1U << 11)) != 0;
    }
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        caps.avx_vnni = (eax & (1U << 4)) != 0;
    }
#endif
    return caps;
}

SimdIsa default_best_isa() {
    const auto& c = get_cpu_capabilities();
    if (c.avx512f && c.avx512bw && c.avx512_vnni) {
        return SimdIsa::AVX512_VNNI;
    }
    if (c.avx2 && c.avx_vnni) {
        return SimdIsa::AVX_VNNI;
    }
    if (c.avx2) {
        return SimdIsa::AVX2;
    }
    return SimdIsa::SCALAR;
}

std::atomic<SimdIsa>& active_isa_storage() {
    static std::atomic<SimdIsa> instance{default_best_isa()};
    return instance;
}

} // namespace

const CpuCapabilities& get_cpu_capabilities() {
    static const CpuCapabilities instance = detect_cpu();
    return instance;
}

bool is_isa_supported(SimdIsa isa) {
    const auto& c = get_cpu_capabilities();
    switch (isa) {
        case SimdIsa::AUTO:
        case SimdIsa::SCALAR:
            return true;
        case SimdIsa::AVX2:
            return c.avx2 && c.fma;
        case SimdIsa::AVX_VNNI:
            // Note: CPUs with AVX512_VNNI + AVX512VL also execute 256-bit VNNI (EVEX-encoded vpdpbusd),
            // while Alder Lake+ has VEX-encoded AVX_VNNI. We check `c.avx_vnni || (c.avx512_vnni && c.avx512vl)`.
            return c.avx2 && (c.avx_vnni || (c.avx512_vnni && c.avx512vl));
        case SimdIsa::AVX512_VNNI:
            return c.avx512f && c.avx512bw && c.avx512_vnni;
    }
    return false;
}

SimdIsa get_active_isa() {
    return active_isa_storage().load(std::memory_order_relaxed);
}

void set_active_isa(SimdIsa isa) {
    if (isa == SimdIsa::AUTO) {
        active_isa_storage().store(default_best_isa(), std::memory_order_relaxed);
        return;
    }
    if (!is_isa_supported(isa)) {
        throw std::runtime_error(std::string("Requested SIMD ISA is not supported by this CPU: ") + isa_name(isa));
    }
    active_isa_storage().store(isa, std::memory_order_relaxed);
}

const char* isa_name(SimdIsa isa) {
    switch (isa) {
        case SimdIsa::AUTO: return "auto";
        case SimdIsa::SCALAR: return "scalar";
        case SimdIsa::AVX2: return "avx2";
        case SimdIsa::AVX_VNNI: return "avx_vnni";
        case SimdIsa::AVX512_VNNI: return "avx512_vnni";
    }
    return "unknown";
}

SimdIsa parse_isa_name(const std::string& name) {
    std::string s = name;
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
        if (c == '-') c = '_';
    }
    if (s == "auto") return SimdIsa::AUTO;
    if (s == "scalar") return SimdIsa::SCALAR;
    if (s == "avx2") return SimdIsa::AVX2;
    if (s == "avx_vnni" || s == "avxvnni") return SimdIsa::AVX_VNNI;
    if (s == "avx512_vnni" || s == "avx512vnni" || s == "avx512") return SimdIsa::AVX512_VNNI;
    throw std::runtime_error("Unknown ISA name '" + name + "' (choose from: auto, scalar, avx2, avx_vnni, avx512_vnni)");
}

std::string cpu_capabilities_summary() {
    const auto& c = get_cpu_capabilities();
    std::ostringstream ss;
    ss << "Cores=" << c.logical_cores;
    if (c.avx2) ss << " AVX2";
    if (c.fma) ss << " FMA";
    if (c.f16c) ss << " F16C";
    if (c.avx_vnni) ss << " AVX_VNNI";
    if (c.avx512f) ss << " AVX512F";
    if (c.avx512bw) ss << " AVX512BW";
    if (c.avx512vl) ss << " AVX512VL";
    if (c.avx512_vnni) ss << " AVX512_VNNI";
    if (c.neon) ss << " NEON";
    ss << " [Active ISA: " << isa_name(get_active_isa()) << "]";
    return ss.str();
}

} // namespace kitteni
