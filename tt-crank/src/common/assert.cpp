// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "assert.hpp"
#include "config.hpp"

#include <cstdlib>
#include <cxxabi.h>
#include <execinfo.h>
#include <sstream>
#include <stdexcept>
#include <tt-logger/tt-logger.hpp>

// NOLINTBEGIN

namespace tt::assert {

static std::string demangle(const char *str) {
    size_t size = 0;
    int status = 0;
    std::string rt(256, '\0');
    if (1 == sscanf(str, "%*[^(]%*[^_]%255[^)+]", rt.data())) {
        char *v = abi::__cxa_demangle(rt.data(), nullptr, &size, &status);
        if (v) {
            std::string result(v);
            free(v);
            return result;
        }
    }
    return str;
}

// @brief Get the current call stack
// @param[out] bt Save Call Stack
// @param[in] size Maximum number of return layers
// @param[in] skip Skip the number of layers at the top of the stack
// NOLINTBEGIN(cppcoreguidelines-no-malloc)
std::vector<std::string> backtrace(size_t size = 64, size_t skip = 1) {
    std::vector<std::string> bt;
    bt.reserve(size - skip);
    void **array = static_cast<void **>(malloc((sizeof(void *) * size)));
    size_t s = static_cast<size_t>(::backtrace(array, static_cast<int>(size)));
    char **strings = backtrace_symbols(array, static_cast<int>(s));
    if (strings == nullptr) {
        fprintf(stderr, "backtrace_symbols error.\n"); // NOLINT(cppcoreguidelines-pro-type-vararg)
        free(array);                                   // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
        return bt;
    }
    for (size_t i = skip; i < s; ++i) {
        bt.push_back(demangle(strings[i]));
    }
    free(strings); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    free(array);   // NOLINT(bugprone-multi-level-implicit-pointer-conversion)

    return bt;
}
// NOLINTEND(cppcoreguidelines-no-malloc)

// @brief String to get current stack information
// @param[in] size Maximum number of stacks
// @param[in] skip Skip the number of layers at the top of the stack
// @param[in] prefix Output before stack information
std::string backtrace_to_string(size_t size = 64, size_t skip = 2, const std::string &prefix = "") {
    std::vector<std::string> bt = backtrace(size, skip);
    std::stringstream ss;
    for (const auto &line : bt) {
        ss << prefix << line << '\n';
    }
    return ss.str();
}

[[noreturn]] void tt_throw_impl(const char *file, int line, const char *assert_type, const char *condition_str,
                                std::string msg) {
    log_critical(tt::LogAlways, "{}: {}", assert_type, msg);

    if (assert_abort_enabled()) {
        abort();
    }

    std::stringstream trace_message_ss;
    trace_message_ss << assert_type << " @ " << file << ":" << line << ": " << condition_str << "\n";
    trace_message_ss << "info:\n" << msg << "\n";

    if (backtrace_enabled()) {
        trace_message_ss << "backtrace:\n";
        trace_message_ss << tt::assert::backtrace_to_string(100, 3, " --- ");
    }
    trace_message_ss << std::flush;
    throw std::runtime_error(trace_message_ss.str());
}

} // namespace tt::assert

// NOLINTEND
