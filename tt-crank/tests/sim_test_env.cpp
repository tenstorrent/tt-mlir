// Optionally points the tt-mlir runtime at ttsim before main(), so test cases
// that would otherwise need a physical device run against the simulator
// instead.
//
// Opt-in via the TT_KURBLA_USE_SIMULATOR env var ("1" / "true" / "on"). When
// unset, this TU is inert and tests run against whatever physical device the
// runtime can find. CMake injects the dir paths via TT_KURBLA_SIM_DIR /
// TT_KURBLA_TT_METAL_HOME (see cmake/Ttsim.cmake). The env contract mirrors
// what tt-metal's own .github/workflows/ttsim.yaml sets up.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#if !defined(TT_KURBLA_SIM_DIR) || !defined(TT_KURBLA_TT_METAL_HOME)
#error "sim_test_env.cpp requires CMake to inject TT_KURBLA_SIM_DIR and TT_KURBLA_TT_METAL_HOME"
#endif

namespace {

bool env_flag_is_set(const char *name) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return false;
    }
    return std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 || std::strcmp(raw, "on") == 0;
}

struct SimEnvSetter {
    SimEnvSetter() {
        if (!env_flag_is_set("TT_KURBLA_USE_SIMULATOR")) {
            return;
        }

        const std::string sim_lib = std::string(TT_KURBLA_SIM_DIR) + "/libttsim.so";

        // Overwrite=1 — a stale env var from the parent shell would otherwise
        // silently win and route us to the wrong simulator.
        ::setenv("TT_METAL_SIMULATOR_HOME", TT_KURBLA_SIM_DIR, 1);
        ::setenv("TT_METAL_SIMULATOR", sim_lib.c_str(), 1);
        // The runtime reads TT_METAL_RUNTIME_ROOT (see tt-metal rtoptions.cpp:292);
        // TT_METAL_HOME is the legacy alias used in shell scripts. Set both.
        ::setenv("TT_METAL_RUNTIME_ROOT", TT_KURBLA_TT_METAL_HOME, 1);
        ::setenv("TT_METAL_HOME", TT_KURBLA_TT_METAL_HOME, 1);
        // ttsim only supports slow dispatch and has SFPU gaps — see
        // tt-metal/.github/workflows/ttsim.yaml.
        ::setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
        ::setenv("TT_METAL_DISABLE_SFPLOADMACRO", "1", 1);

        std::cerr << "[sim_test_env] routing runtime through ttsim: " << sim_lib << "\n";
    }
};

// File-scope constant — runs before main, after which env vars are set for the
// rest of the process. The variable is unused; only the side effect matters.
[[maybe_unused]] const SimEnvSetter g_sim_env_setter{};

} // namespace
