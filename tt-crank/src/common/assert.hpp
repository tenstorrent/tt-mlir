// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//  Summary:
//
//  TT_ASSERT -> debug build only assert. Throws exception with condition,
//  location and backtrace.
//
//  TT_FATAL -> always asserts. Throws exception with condition, location and
//  backtrace.
//
//  TT_THROW -> always throws exception with condition, location and backtrace.
//
//  Environment variables:
//  TT_CRANK_DISABLE_BACKTRACE -> disables backtrace.
//  TT_CRANK_ASSERT_ABORT -> forces abort instead of throw.
//
//  This header is taken from tt-metal repo and slightly modified.
#pragma once

#include <format>
#include <type_traits>

#include <cstdio>
#include <cstdlib>
#include <string>

// NOLINTBEGIN

template <typename T>
    requires std::is_enum_v<T>
struct std::formatter<T> : std::formatter<std::underlying_type_t<T>> {
    auto format(T val, std::format_context &ctx) const {
        return std::formatter<std::underlying_type_t<T>>::format(static_cast<std::underlying_type_t<T>>(val), ctx);
    }
};

namespace tt::assert {

[[noreturn]] void tt_throw_impl(const char *file, int line, const char *assert_type, const char *condition_str,
                                std::string msg = "");

inline void tt_throw(char const *file, int line, char const *assert_type, char const *condition_str) {
    tt_throw_impl(file, line, assert_type, condition_str);
}

template <typename... Args>
inline void tt_throw(char const *file, int line, char const *assert_type, char const *condition_str,
                     std::format_string<Args const &...> fmt, Args const &...args) {
    tt_throw_impl(file, line, assert_type, condition_str, std::format(fmt, args...));
}

} // namespace tt::assert

// Adding do while around TT_ASSERT to allow flexible usage of the macro. More details can be found in Stack Overflow
// post:
// https://stackoverflow.com/questions/55933541/else-without-previous-if-error-when-defining-macro-with-arguments/55933720#55933720
#ifdef DEBUG
#ifndef TT_ASSERT
#define TT_ASSERT(condition, ...)                                                                                      \
    do {                                                                                                               \
        if (not(condition)) [[unlikely]] {                                                                             \
            tt::assert::tt_throw(__FILE__, __LINE__, "TT_ASSERT", #condition, __VA_ARGS__);                            \
            __builtin_unreachable();                                                                                   \
        }                                                                                                              \
    } while (0) // NOLINT(cppcoreguidelines-macro-usage)
#endif
#else
#define TT_ASSERT(condition, ...)                                                                                      \
    do {                                                                                                               \
        (void)(condition);                                                                                             \
    } while (0) // this was done to avoid the compiler flagging unused variables when building Release
#endif

#ifndef TT_THROW
#define TT_THROW(...)                                                                                                  \
    do {                                                                                                               \
        tt::assert::tt_throw(__FILE__, __LINE__, "TT_THROW", "tt::exception", __VA_ARGS__);                            \
        __builtin_unreachable();                                                                                       \
    } while (0)
#endif

#ifndef TT_FATAL
#define TT_FATAL(condition, ...)                                                                                       \
    do {                                                                                                               \
        if (not(condition)) [[unlikely]] {                                                                             \
            tt::assert::tt_throw(__FILE__, __LINE__, "TT_FATAL", #condition, __VA_ARGS__);                             \
            __builtin_unreachable();                                                                                   \
        }                                                                                                              \
    } while (0) // NOLINT(cppcoreguidelines-macro-usage)
#endif

// NOLINTEND
