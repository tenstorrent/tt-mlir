// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===---------------------------------------------------------------------===//
/// @file
/// This header offers better assert macro definitions for the "assert
/// liberally" aspect of defensive programming.
/// @link https://llvm.org/docs/CodingStandards.html#assert-liberally.
///
/// Main value propositions:
///
/// - in common cases of simple binary comparisons like `TT_assert(x < y)`,
///   the runtime error message will contain not just the stringified image
///  `x < y` but also the individual runtime values of `x` and `y`. This is
///   often very helpful for understanding the root cause of the problem
///   on the spot.
///
/// - a common case of non-binary checks are for range conditions like
///   `a <= x && x < b`. These are supported directly with a family of
///   shortcuts `T_assert_limit()`, `TT_assert_inclusive_range()`, etc
///   that reduce runtime checking overhead to a single branch.
///
/// - macros are divided into two variants, with one variant (`TT_debug...`)
///   guaranteed to be elided in release builds and therefore suitable for
///   liberally asserting performance-critical code paths.
///
/// - runtime messaging can be further enhanced via `llvm::formatv()`-style
///   formatting of local code expressions, including user types with bespoke
///   streaming/print formatting. This is useful for print-style debugging.
///
/// @see TT_assert
/// @see TT_assertv
/// @see TT_assert_open_range
/// @see TT_assert_limit
/// @see TT_assert_exclusive_range
/// @see TT_assert_inclusive_range
///
/// @see TT_debug
///
/// @see test/unittests/Support/DeathTestAsserts.cpp for usage examples
//===---------------------------------------------------------------------===//

#ifndef TTMLIR_ASSERTS_H
#define TTMLIR_ASSERTS_H

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

namespace ttmlir::utils::asserts {

// This assert facility can be disabled by defining TT_ASSERT_DISABLE_ASSERTS
// which will result in void-valued definitions of all assert macros. This
// will happen by default if NDEBUG is defined, parallel to the standard
// <cassert> behavior.

#if !defined(TT_ASSERT_DISABLE_ASSERTS)
#if defined(NDEBUG)
#define TT_ASSERT_DISABLE_ASSERTS
#endif // NDEBUG
#endif // TT_ASSERT_DISABLE_ASSERTS

// The TT_debug*() family of macros are meant to be elided in "release" builds
// and are meant to be used in tight loops and similar situations where
// invariant checking overhead should be removed from the final product.
//
// Specifically, these debug macros are void-valued definitions unless
// TT_ASSERT_ENABLE_DEBUG_ASSERTS is defined. The latter happens by default when
// TT_BUILD_DEBUG is defined by `Debug` CMAKE_BUILD_TYPE.

#if !defined(TT_ASSERT_ENABLE_DEBUG_ASSERTS)
#if defined(TT_BUILD_DEBUG)
#define TT_ASSERT_ENABLE_DEBUG_ASSERTS
#endif // TT_BUILD_DEBUG
#endif // TT_ASSERT_ENABLE_DEBUG_ASSERTS

// The default behavior for a triggered assert is to print a message (to
// TT_ASSERT_REPORT_STREAM()) and std::abort the process. Redefine
// TT_ASSERT_FAILURE() to call a different function (or throw an exception, or
// do nothing) to customize this behavior.

#if !defined(TT_ASSERT_FAILURE)
#define TT_ASSERT_FAILURE() ::std::abort()
#endif // TT_ASSERT_FAILURE

// Redefine TT_ASSERT_REPORT_STREAM() to an expression that will return a
// reference to an I/O stream that is to receive the triggered assert message
// just before TT_ASSERT_FAILURE() action is taken. The default is
// `::llvm::errs()`.

#if !defined(TT_ASSERT_REPORT_STREAM)
#define TT_ASSERT_REPORT_STREAM() ::llvm::errs()
#endif // TT_ASSERT_REPORT_STREAM

//===---------------------------------------------------------------------===//
namespace impl {

// clang-format off

#define TT_IMPL_ASSERT_LOC_INFO       __FILE__,__LINE__,__PRETTY_FUNCTION__

#define TT_ASSERT_UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

#define TT_ASSERT_STRINGIZE(x)        TT_IMPL_STRINGIZE(x)
#define TT_IMPL_STRINGIZE(x)          #x

// clang-format on

template <typename T>
struct always_false : std::false_type {};

template <typename T>
constexpr bool always_false_v = always_false<T>::value;

//===---------------------------------------------------------------------===//

template <typename T, typename = void>
struct UnsignedCast {
  // Specializing for non-integral types in order to generate nicer compiler
  // error messages.
  static_assert(always_false_v<T>,
                "range asserts are only supported for integral types");
};

template <typename T>
struct UnsignedCast<T, std::enable_if_t<std::is_integral_v<T>>> {
  static auto evaluate(T x) -> std::make_unsigned_t<T> {
    return static_cast<std::make_unsigned_t<T>>(x);
  }
};
//===---------------------------------------------------------------------===//

template <typename T, typename Stream, typename = void>
struct is_streamable : std::false_type {};

template <typename T, typename Stream>
struct is_streamable<
    T, Stream,
    std::void_t<decltype(std::declval<Stream &>() << std::declval<T>())>>
    : std::true_type {};

template <typename T, typename Stream>
constexpr bool is_streamable_v = is_streamable<T, Stream>::value;

template <typename T, typename = void>
struct has_to_string : std::false_type {};

template <typename T>
struct has_to_string<T, std::void_t<decltype(to_string(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool has_to_string_v = has_to_string<T>::value;

//===---------------------------------------------------------------------===//

// A hook for printing values in assertion failure messages (e.g. strings/chars
// could be printed as quoted strings, etc).
template <typename T, typename Stream, typename = void>
struct PrintAdaptor {
  static constexpr bool enabled = false;
  static void evaluate(Stream &os, T &&obj) {
    static_assert(
        always_false_v<T>,
        "assert condition contains an expression type that isn't printable "
        "(wrap in parentheses to disable expression decomposition)");
  }
};

template <typename T, typename Stream>
struct PrintAdaptor<T, Stream, std::enable_if_t<is_streamable_v<T, Stream>>> {
  static constexpr bool enabled = true;
  static void evaluate(Stream &os, T &&obj) { os << obj; }
};

template <typename T, typename Stream>
struct PrintAdaptor<
    T, Stream,
    std::enable_if_t<not is_streamable_v<T, Stream> and has_to_string_v<T>>> {
  static constexpr bool enabled = true;
  static void evaluate(Stream &os, T &&obj) {
    using std::to_string;
    os << to_string(obj);
  }
};

// llvm raw streams have ambiguous overloads for std::nullptr_t.
template <typename T, typename Stream>
struct PrintAdaptor<
    T, Stream,
    std::enable_if_t<std::is_null_pointer_v<std::remove_reference_t<T>>>> {
  static constexpr bool enabled = true;
  static void evaluate(Stream &os, T &&obj) { os << "nullptr"; }
};

} // namespace impl

template <typename T, typename Stream>
inline constexpr bool is_printable_v = impl::PrintAdaptor<T, Stream>::enabled;

template <typename T, typename Stream>
void print(Stream &os, T &&obj) {
  impl::PrintAdaptor<T, Stream>::evaluate(os, std::forward<T>(obj));
}
//===---------------------------------------------------------------------===//

enum class BinaryOp {
  eq, // ==
  ne, // !=
  lt, // <
  le, // <=
  gt, // >
  ge, // >=
  undefined
};

static inline constexpr const char *kBinaryOpName[]{
    "==", "!=", "<", "<=", ">", ">=", "UNDEFINED"};

//===---------------------------------------------------------------------===//

// NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
template <typename LHS, typename RHS, BinaryOp Op>
struct BinaryExpr {
  LHS lhs;
  RHS rhs;
  bool result;

  template <typename Stream>
  void prefix(Stream &os, const char *file, int line, const char *func,
              const char *condition) const {
    os << file << ':' << line << ": " << func << ": Assertion `" << condition
       << "` failed";
    if constexpr (is_printable_v<LHS, Stream> && is_printable_v<RHS, Stream>) {
      os << ", was `";
      print(os, lhs);
      os << ' ' << kBinaryOpName[llvm::to_underlying(Op)] << ' ';
      print(os, rhs);
      os << '`';
    }
  }

  // Supports `TT_assert()`.
  template <typename Stream>
  void report(Stream &os, const char *file, int line, const char *func,
              const char *condition) const {
    prefix(os, file, line, func, condition);
    os << '\n';
  }

  // Supports `TT_assertv()`.
  template <typename Stream>
  void report(Stream &os, const char *file, int line, const char *func,
              const char *condition, const std::string &message) const {
    prefix(os, file, line, func, condition);
    os << ", " << message << '\n';
  }

  // Intercept chained expressions.

  // clang-format off
#define TT_IMPL_ASSERT_CHAINED_OP_HANDLER(op)                                  \
  template <typename T>                                                        \
  auto operator op(T) -> BinaryExpr<LHS, RHS, Op> const {                      \
    static_assert(                                                             \
        impl::always_false_v<T>,                                               \
        "chained comparisons are not supported inside assertions (e.g., wrap " \
        "`x < y && a < b` inside parentheses, e.g. `(x < y && a < b)`)");      \
  }

  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(==)
  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(!=)

  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(<)
  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(<=)
  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(>)
  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(>=)

  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(&&)
  TT_IMPL_ASSERT_CHAINED_OP_HANDLER(||)

#undef TT_IMPL_ASSERT_CHAINED_OP_HANDLER
  // clang-format on
};
// NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)
//===---------------------------------------------------------------------===//

// NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
template <typename LHS>
struct ExprLHS {
  explicit constexpr ExprLHS(LHS lhs) : lhs{lhs} {}

  // Intercept binary expr cases.

  // clang-format off
#define TT_IMPL_ASSERT_BINARY_OP_HANDLER(op, bop)                              \
  template <typename RHS>                                                      \
  constexpr friend auto operator op(ExprLHS &&exprLHS, RHS &&rhs)              \
      -> BinaryExpr<LHS, RHS, BinaryOp::bop> {                                 \
    return {exprLHS.lhs, rhs, static_cast<bool>(exprLHS.lhs op rhs)};          \
  }

  TT_IMPL_ASSERT_BINARY_OP_HANDLER(==, eq)
  TT_IMPL_ASSERT_BINARY_OP_HANDLER(!=, ne)

  TT_IMPL_ASSERT_BINARY_OP_HANDLER(<, lt)
  TT_IMPL_ASSERT_BINARY_OP_HANDLER(<=, le)
  TT_IMPL_ASSERT_BINARY_OP_HANDLER(>, gt)
  TT_IMPL_ASSERT_BINARY_OP_HANDLER(>=, ge)

#undef TT_IMPL_ASSERT_BINARY_OP_HANDLER

  // TODO(vroubtsov) |, &, ^ ?

#define TT_IMPL_ASSERT_BINARY_OP_HANDLER(op)                                   \
  template <typename RHS>                                                      \
  constexpr friend auto operator op(ExprLHS &&lhs, RHS &&rhs)                  \
      -> BinaryExpr<LHS, RHS, BinaryOp::undefined> {                           \
    static_assert(impl::always_false_v<RHS>,                                   \
                  "operators ||, && are not supported inside TT_assert() "     \
                  "assertions: wrap condition in parentheses");                \
  }

  TT_IMPL_ASSERT_BINARY_OP_HANDLER(||)
  TT_IMPL_ASSERT_BINARY_OP_HANDLER(&&)

#undef TT_IMPL_ASSERT_BINARY_OP_HANDLER
  // clang-format on

  // Intercept unary expr cases.

  template <typename Stream>
  void prefix(Stream &os, const char *file, int line, const char *func,
              const char *condition) const {
    os << file << ':' << line << ": " << func << ": Assertion `" << condition
       << "` failed";
  }

  // Supports `TT_assert()`.
  template <typename Stream>
  void report(Stream &os, const char *file, int line, const char *func,
              const char *condition) const {
    prefix(os, file, line, func, condition);
    os << '\n';
  }

  // Supports `TT_assertv()`.
  template <typename Stream>
  void report(Stream &os, const char *file, int line, const char *func,
              const char *condition, const std::string &message) const {
    prefix(os, file, line, func, condition);
    os << ", " << message << '\n';
  }

  LHS lhs;
};
// NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)
//===---------------------------------------------------------------------===//

struct ExprDecomposer {

  template <typename T>
  constexpr friend auto operator<=(ExprDecomposer &&,
                                   T &&lhs) -> ExprLHS<const T &> {
    return ExprLHS<const T &>{lhs};
  }
};
//===---------------------------------------------------------------------===//

template <typename T>
decltype(auto) unsigned_cast(T x) {
  return impl::UnsignedCast<T>::evaluate(x);
}
//===---------------------------------------------------------------------===//
// clang-format off

// These macro names spellings are intentional, they are intended to show up
// in assert failure messages.

// Single-branch `a <= x < b` check.
#define in_open_range(x, a, b)                                                 \
  (::ttmlir::utils::asserts::unsigned_cast((x) - (a)) <                        \
   ::ttmlir::utils::asserts::unsigned_cast((b) - (a)))

// Single-branch `a < x < b` check.
#define in_exclusive_range(x, a, b)                                            \
  (::ttmlir::utils::asserts::unsigned_cast((x) - (a) - 1) <                    \
   ::ttmlir::utils::asserts::unsigned_cast((b) - (a) - 1))

// Single-branch `a <= x <= b` check.
#define in_inclusive_range(x, a, b)                                            \
  (::ttmlir::utils::asserts::unsigned_cast((x) - (a)) <=                       \
   ::ttmlir::utils::asserts::unsigned_cast((b) - (a)))

//===---------------------------------------------------------------------===//
// TT_assert*.
//===---------------------------------------------------------------------===//

#if !defined(TT_ASSERT_DISABLE_ASSERTS)

# define TT_assert(condition)                                                  \
    do {                                                                       \
      if (TT_ASSERT_UNLIKELY(!(condition))) {                                  \
        (::ttmlir::utils::asserts::ExprDecomposer { } <= condition )           \
          .report(TT_ASSERT_REPORT_STREAM(), TT_IMPL_ASSERT_LOC_INFO, #condition); \
        TT_ASSERT_FAILURE();                                                   \
      }                                                                        \
    } while (false)                                                            \
    /* */

// TODO(vroubtsov) for now, 'message' is required; in c++20 there will be
// a standard-compliant way to make it optional and to nicely merge
// TT_assert and TT_assertv into a single macro; there doesn't seem to be
// a way to do this in c++17 without enabling gcc/clang extensions.

# define TT_assertv(condition, /* message[, args...] */...)                    \
    do {                                                                       \
      if (TT_ASSERT_UNLIKELY(!(condition))) {                                  \
        (::ttmlir::utils::asserts::ExprDecomposer { } <= condition )           \
          .report(TT_ASSERT_REPORT_STREAM(), TT_IMPL_ASSERT_LOC_INFO, #condition, \
                  llvm::formatv(__VA_ARGS__).str());                           \
        TT_ASSERT_FAILURE();                                                   \
      }                                                                        \
    } while (false)                                                            \
    /* */

// a <= x < b
# define TT_assert_open_range(x, a, b) \
    TT_assertv(in_open_range(x, a, b), "{} ({}) is not in [{}, {})", TT_ASSERT_STRINGIZE(x), x, a, b)

// 0 <= x < limit, convenience shortcut.
# define TT_assert_limit(x, limit) \
    TT_assertv(in_open_range(x, 0, limit), "{} ({}) is not in [0, {})", TT_ASSERT_STRINGIZE(x), x, limit)

// a < x < b
# define TT_assert_exclusive_range(x, a, b) \
    TT_assertv(in_exclusive_range(x, a, b), "{} ({}) is not in ({}, {})", TT_ASSERT_STRINGIZE(x), x, a, b)

// a <= x <= b
# define TT_assert_inclusive_range(x, a, b) \
    TT_assertv(in_inclusive_range(x, a, b), "{} ({}) is not in [{}, {}]", TT_ASSERT_STRINGIZE(x), x, a, b)

#else // TT_ASSERT_DISABLE_ASSERTS

# define TT_assert(condition)                             ((void)0)
# define TT_assertv(condition, /* message[, args] */...)  ((void)0)

# define TT_assert_open_range(x, a, b)                    ((void)0)
# define TT_assert_limit(x, limit)                        ((void)0)
# define TT_assert_exclusive_range(x, a, b)               ((void)0)
# define TT_assert_inclusive_range(x, a, b)               ((void)0)

#endif // TT_ASSERT_DISABLE_ASSERTS

//===---------------------------------------------------------------------===//
// TT_debug*.
//===---------------------------------------------------------------------===//

#if defined(TT_ASSERT_ENABLE_DEBUG_ASSERTS)

# define TT_debug(condition)                              TT_assert(condition)
# define TT_debugv(condition, /* message[, args] */...)   TT_assertv(condition, __VA_ARGS__)

# define TT_debug_open_range(x, a, b)                     TT_assert_open_range(x, a, b)
# define TT_debug_limit(x, limit)                         TT_assert_limit(x, limit)
# define TT_debug_exclusive_range(x, a, b)                TT_assert_exclusive_range(x, a, b)
# define TT_debug_inclusive_range(x, a, b)                TT_assert_inclusive_range(x, a, b)

#else // !TT_ASSERT_ENABLE_DEBUG_ASSERTS

# define TT_debug(condition)                              ((void)0)
# define TT_debugv(condition, /* message[, args] */...)   ((void)0)

# define TT_debug_open_range(x, a, b)                     ((void)0)
# define TT_debug_limit(x, limit)                         ((void)0)
# define TT_debug_exclusive_range(x, a, b)                ((void)0)
# define TT_debug_inclusive_range(x, a, b)                ((void)0)

#endif // TT_ASSERT_ENABLE_DEBUG_ASSERTS

// clang-format off

} // namespace ttmlir::utils::asserts
#endif // TTMLIR_ASSERTS_H
