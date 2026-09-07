// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
#ifndef FS_CONFIG_IMPL // FS definitions (config.hpp)

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

#endif // FS_CONFIG_IMPL

// Config section.
//
// Exposes <config_name>_config() function which evaluates to default value
// unless the env var is set, in which case it takes the env var value.
//
// Example usage:
//
// config.hpp: CONFIG_STR(some_str, "default_str_value", "TT_KURBLA_SOME_STR")
//
// somewhere.cpp: std::string dir = some_str_config();

#ifndef FS_CONFIG_IMPL // Config definitions (config.hpp)

#define CONFIG_STR(name, value, env) const char *name##_config();

#else  // Config implementations (config.cpp)

#define CONFIG_STR(name, value, env)                                                                                   \
    static const char *env_##name = std::getenv(env);                                                                  \
    const char *name##_config() { return env_##name != nullptr ? env_##name : value; }

#endif // FS_CONFIG_IMPL
// clang-format on
