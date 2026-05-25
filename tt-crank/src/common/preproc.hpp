#pragma once

#ifdef DEBUG
inline constexpr bool build_debug = true;
inline constexpr bool build_release = false;
#else
inline constexpr bool build_debug = false;
inline constexpr bool build_release = true;
#endif

// Feature switches section.
//
// Enables compile time and runtime enabling/disabling of features.
// It exposes <feature_name>_enabled() function which evaluates to true or false based on provided value (and provided
// env variable).
//
// It has 2 forms:
// - FS_WITH_ENABLER(name, value, env) allows turning on feature switch when provided env var is set, regardless of
//   provided default value.
// - FS_WITH_DISABLER(name, value, env) allows turning off feature switch when provided env var is set, regardless of
//   provided default value.
//
// Example usage:
//
// config.hpp: FS_WITH_DISABLER(feature_0, true, "TT_KURBLA_FEATURE_0_DISABLED")
//
// somewhere.cpp:
// void use_feature_0() {
//     if (!feature_0_enabled()) {
//         return;
//     }
//     ...
// }

// clang-format off
#ifndef FS_IMPL // FS definitions (config.hpp)

#define FS_WITH_ENABLER(name, value, env)  bool name##_enabled();
#define FS_WITH_DISABLER(name, value, env) bool name##_enabled();

#else  // FS implementations (config.cpp)
#include <cstdlib>

#define FS_WITH_ENABLER(name, value, env)                                                                              \
    static bool env_##name = std::getenv(env) != nullptr;                                                              \
    bool name##_enabled() { return value || env_##name; }

#define FS_WITH_DISABLER(name, value, env)                                                                             \
    static bool env_##name = std::getenv(env) != nullptr;                                                              \
    bool name##_enabled() { return value && !env_##name; }

#endif // FS_IMPL
// clang-format on
