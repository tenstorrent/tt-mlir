// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SUPPORT_LOGGER_H
#define TTMLIR_SUPPORT_LOGGER_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

namespace ttmlir {

// Log components for different components
enum class LogComponent { Optimizer, OpValidation, Allocator, Test, General };

// Log levels in order of verbosity
enum class LogLevel {
  Trace, // Most verbose, enabled only with TTMLIR_LOGGER_LEVEL=trace
  Debug, // Default level
  Fatal  // Fatal errors
};

// Define LLVM log component type strings
inline constexpr const char *getLogComponentStr(LogComponent type) {
  switch (type) {
  case LogComponent::Optimizer:
    return "optimizer";
  case LogComponent::OpValidation:
    return "op-validation";
  case LogComponent::Allocator:
    return "allocator";
  case LogComponent::Test:
    return "test";
  case LogComponent::General:
    return "general";
  }
  return "unknown";
}

// String representations of log levels
inline constexpr const char *getLogLevelStr(LogLevel level) {
  switch (level) {
  case LogLevel::Trace:
    return "TRACE";
  case LogLevel::Debug:
    return "DEBUG";
  case LogLevel::Fatal:
    return "FATAL";
  }
}

// Get LLVM color type for log level
inline llvm::raw_ostream::Colors getLogLevelColor(LogLevel level) {
  switch (level) {
  case LogLevel::Trace:
    return llvm::raw_ostream::CYAN;
  case LogLevel::Debug:
    return llvm::raw_ostream::GREEN;
  case LogLevel::Fatal:
    return llvm::raw_ostream::RED;
  }
  return llvm::raw_ostream::RESET;
}

// Get minimum log level from environment
inline LogLevel getMinLogLevel() {
  static LogLevel minLevel = []() {
    const char *env = std::getenv("TTMLIR_LOGGER_LEVEL");
    if (!env) {
      // Default to Debug level
      return LogLevel::Debug;
    }

    std::string level(env);
    // Convert to uppercase for case-insensitive comparison
    std::transform(level.begin(), level.end(), level.begin(), ::toupper);

    if (level == "TRACE") {
      return LogLevel::Trace;
    }

    // Any unrecognized value defaults to Debug
    return LogLevel::Debug;
  }();
  return minLevel;
}

// Check if a log level is enabled
inline bool isLogLevelEnabled(LogLevel level) {
  return static_cast<int>(level) >= static_cast<int>(getMinLogLevel());
}

// Get current timestamp
inline std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << '.'
     << std::setfill('0') << std::setw(3) << ms.count();
  return ss.str();
}

// Main logging macro that uses DEBUG_WITH_TYPE
#ifdef TTMLIR_ENABLE_DEBUG_LOGS
#define TTMLIR_LOG_FMT(logComponent, logLevel, /* fmt, args */...)             \
  DEBUG_WITH_TYPE(                                                             \
      ttmlir::getLogComponentStr(logComponent),                                \
      if (ttmlir::isLogLevelEnabled(logLevel)) {                               \
        auto &OS = llvm::dbgs();                                               \
        OS.enable_colors(true);                                                \
        OS << "[";                                                             \
        OS.changeColor(llvm::raw_ostream::GREEN, /*bold=*/true);               \
        OS << ttmlir::getCurrentTimestamp();                                   \
        OS.resetColor();                                                       \
        OS << "] [";                                                           \
        OS.changeColor(ttmlir::getLogLevelColor(logLevel), /*bold=*/true);     \
        OS << ttmlir::getLogLevelStr(logLevel);                                \
        OS.resetColor();                                                       \
        OS << "] [";                                                           \
        OS.changeColor(llvm::raw_ostream::MAGENTA, /*bold=*/true);             \
        OS << ttmlir::getLogComponentStr(logComponent);                        \
        OS.resetColor();                                                       \
        OS << "] " << llvm::formatv(__VA_ARGS__) << "\n";                      \
        if (logLevel == ttmlir::LogLevel::Fatal) {                             \
          abort();                                                             \
        }                                                                      \
      })
#else
#define TTMLIR_LOG_FMT(logComponent, logLevel, /* fmt, args */...) ((void)0)
#endif

// Public logging macros
#define TTMLIR_TRACE(component, /* fmt, args */...)                            \
  TTMLIR_LOG_FMT(component, ttmlir::LogLevel::Trace,                           \
                 /* fmt, args */ __VA_ARGS__)
#define TTMLIR_DEBUG(component, /* fmt, args */...)                            \
  TTMLIR_LOG_FMT(component, ttmlir::LogLevel::Debug,                           \
                 /* fmt, args */ __VA_ARGS__)
#define TTMLIR_FATAL(component, /* fmt, args */...)                            \
  TTMLIR_LOG_FMT(component, ttmlir::LogLevel::Fatal,                           \
                 /* fmt, args */ __VA_ARGS__)

} // namespace ttmlir

#endif // TTMLIR_SUPPORT_LOGGER_H
