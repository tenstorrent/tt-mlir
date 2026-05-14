// Optionally points the tt-mlir runtime at ttsim, so any consumer of
// libtt_kurbla.so (tests, Python wheel, future bindings) routes through the
// simulator instead of opening a physical device.
//
// Opt-in via the TT_KURBLA_USE_SIMULATOR env var ("1" / "true" / "on").
// CMake injects the dir paths via TT_KURBLA_SIM_DIR / TT_KURBLA_TT_METAL_HOME
// (see cmake/TTsim.cmake).
//
// The setenv() calls run at library-load time (before the function-local
// static in execution_payload.cpp's runtime_device() can be initialized), so
// tt-metal's first device open sees the simulator env.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#if !defined(TT_KURBLA_SIM_DIR) || !defined(TT_KURBLA_TT_METAL_HOME)
#error "sim_env.cpp requires CMake to inject TT_KURBLA_SIM_DIR and TT_KURBLA_TT_METAL_HOME"
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

        std::cerr << "[tt_kurbla] routing runtime through ttsim: " << sim_lib << "\n";
    }
};

[[maybe_unused]] const SimEnvSetter g_sim_env_setter{};

} // namespace
