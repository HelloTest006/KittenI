#pragma once

#include <string>
#include <cstdint>

namespace kitteni {

enum class SimdIsa : uint8_t {
    AUTO = 0,
    SCALAR = 1,
    AVX2 = 2,
    AVX_VNNI = 3,
    AVX512_VNNI = 4
};

struct CpuCapabilities {
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool avx_vnni = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512_vnni = false;
    bool neon = false;
    int logical_cores = 1;
};

const CpuCapabilities& get_cpu_capabilities();
std::string cpu_capabilities_summary();

// Returns true if the current hardware supports the requested ISA
bool is_isa_supported(SimdIsa isa);

// Get/set the active SIMD ISA used by all KittenI kernels (`AUTO` resolves to the highest supported ISA)
SimdIsa get_active_isa();
void set_active_isa(SimdIsa isa);

const char* isa_name(SimdIsa isa);
SimdIsa parse_isa_name(const std::string& name);

} // namespace kitteni
