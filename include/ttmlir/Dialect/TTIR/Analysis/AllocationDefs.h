// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATIONDEFS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATIONDEFS_H

// TODO(vroubtsov) temp workaround for #4304 not being available yet
#if defined(__has_include)
#if __has_include("ttmlir/Asserts.h")
#include "ttmlir/Asserts.h"
#endif
#endif

#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/Error.h"

// Define some convenience macros local to `Analysis`.

#define TT_ALLOC_DEBUG(/* fmt, args */...)                                     \
  TTMLIR_DEBUG(ttmlir::LogComponent::Allocator, __VA_ARGS__)

#define TT_ALLOC_TRACE(/* fmt, args */...)                                     \
  TTMLIR_TRACE(ttmlir::LogComponent::Allocator, __VA_ARGS__)

#define TT_ALLOC_ERROR(/* fmt, args */...)                                     \
  do {                                                                         \
    auto &OS = llvm::errs();                                                   \
    OS.enable_colors(true);                                                    \
    OS.changeColor(llvm::raw_ostream::RED, /*bold=*/true);                     \
    OS << llvm::formatv(__VA_ARGS__) << "\n";                                  \
    OS.resetColor();                                                           \
  } while (false)

// clang-format off
#ifndef TT_IMPL_ASSERT_LOC_INFO

# define TT_assert(condition)                            ((void)0)
# define TT_assertv(condition, /* message[, args] */...) ((void)0)

# define TT_assert_open_range(x, a, b)                   ((void)0)
# define TT_assert_limit(x, limit)                       ((void)0)
# define TT_assert_exclusive_range(x, a, b)              ((void)0)
# define TT_assert_inclusive_range(x, a, b)              ((void)0)

# define TT_debug(condition)                            ((void)0)
# define TT_debugv(condition, /* message[, args] */...) ((void)0)

# define TT_debug_open_range(x, a, b)                   ((void)0)
# define TT_debug_limit(x, limit)                       ((void)0)
# define TT_debug_exclusive_range(x, a, b)              ((void)0)
# define TT_debug_inclusive_range(x, a, b)              ((void)0)

#endif
// clang-format on

namespace mlir::tt::ttir {

template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> ordinal(Enum e) {
  return llvm::to_underlying(e);
}

// TODO
template <typename T, typename = void>
struct parse {
  // static T evaluate(llvm::StringRef s) { ... }
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATIONDEFS_H
