// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_ASSERTS_H
#define TTMLIR_ASSERTS_H

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

// ............................................................................
namespace ttmlir::utils::asserts {

// The TT_debug*() family of macros are meant to be elided in "release" builds
// and are meant to be used in tight loops and similar situations where
// invariant checking overhead should be remoed from the final product.
//
// These debug macros are enabled when TT_ASSERT_ENABLE_DEBUG_CHECKS is defined
// which by default happens when TTMLIR_ENABLE_DEBUG_LOGS is defined.

#ifdef TTMLIR_ENABLE_DEBUG_LOGS
#define TT_ASSERT_ENABLE_DEBUG_CHECKS
#endif // TTMLIR_ENABLE_DEBUG_LOGS

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

// ............................................................................
namespace impl {

// clang-format off

#define TT_IMPL_ASSERT_LOC_INFO       __FILE__,__LINE__,__PRETTY_FUNCTION__

#define TT_ASSERT_UNLIKELY(condition) __builtin_expect (static_cast<bool>(condition), 0)

#define TT_ASSERT_STRINGIZE(x)        TT_IMPL_STRINGIZE(x)
#define TT_IMPL_STRINGIZE(x)          #x

// clang-format on

template <typename T>
struct always_false : std::false_type {};

// ............................................................................

template <typename T, typename = void>
struct UnsignedCast {
  // Specializing for non-integral types in order to generate nicer compiler
  // error messages.
  static_assert(always_false<T>::value,
                "range asserts are only supported for integral types");
};

template <typename T>
struct UnsignedCast<T, std::enable_if_t<std::is_integral_v<T>>> {
  static auto evaluate(T x) -> std::make_unsigned_t<T> {
    return static_cast<std::make_unsigned_t<T>>(x);
  }
};
// ............................................................................

template <typename T, typename Stream, typename = void>
struct is_streamable : std::false_type {};

template <typename T, typename Stream>
struct is_streamable<
    T, Stream,
    std::void_t<decltype(std::declval<Stream &>() << std::declval<T>())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_to_string : std::false_type {};

template <typename T>
struct has_to_string<T, std::void_t<decltype(to_string(std::declval<T>()))>>
    : std::true_type {};

// ............................................................................

// A hook for printing values in assertion failure messages (e.g. strings/chars
// could be printed as quoted strings, etc).
template <typename T, typename Stream, typename = void>
struct PrintAdaptor {
  static constexpr bool enabled = false;
  static void evaluate(Stream &os, T &&obj) {
    static_assert(
        always_false<T>::value,
        "assert condition contains an expression type that isn't printable "
        "(wrap in parentheses to disable expression decomposition)");
  }
};

template <typename T, typename Stream>
struct PrintAdaptor<T, Stream,
                    std::enable_if_t<is_streamable<T, Stream>::value>> {
  static constexpr bool enabled = true;
  static void evaluate(Stream &os, T &&obj) { os << obj; }
};

template <typename T, typename Stream>
struct PrintAdaptor<T, Stream,
                    std::enable_if_t<not is_streamable<T, Stream>::value and
                                     has_to_string<T>::value>> {
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
// ............................................................................
// clang-format off

enum class BinaryOp {
    eq,   // ==
    ne,   // !=
    lt,   // <
    le,   // <=
    gt,   // >
    ge,   // >=
    undefined
};

static inline constexpr const char * kBinaryOpName[] { "==", "!=", "<", "<=", ">", ">=", "UNDEFINED"};

// clang-format on
// ............................................................................

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
        impl::always_false<T>::value,                                          \
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
// ............................................................................

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
    static_assert(impl::always_false<RHS>::value,                              \
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

}; // end of class
// NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)
// ............................................................................

struct ExprDecomposer {

  // clang-format off
  template <typename T>
  constexpr friend auto operator<=(ExprDecomposer &&, T &&lhs) -> ExprLHS<const T &> {
    return ExprLHS<const T &>{lhs};
  }
  // clang-format on

}; // end of class
// ............................................................................

template <typename T>
decltype(auto) unsigned_cast(T x) {
  return impl::UnsignedCast<T>::evaluate(x);
}
// ............................................................................
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

// ............................................................................

#define TT_assert(condition)                                                  \
  do {                                                                        \
    if (!TT_ASSERT_UNLIKELY(condition)) {                                     \
      (::ttmlir::utils::asserts::ExprDecomposer { } <= condition )            \
        .report(TT_ASSERT_REPORT_STREAM(), TT_IMPL_ASSERT_LOC_INFO, #condition); \
      TT_ASSERT_FAILURE();                                                    \
    }                                                                         \
  } while (false)                                                             \
  /* */

// TODO(vroubtsov) for now, 'message' is required; in c++20 there will be
// a standard-compliant way to make it optional and to nicely merge
// TT_assert and TT_assertv into a single macro; there doesn't seem to be
// a way to do this in c++17 without enabling gcc/clang extensions.

#define TT_assertv(condition, /* message[, args...] */...)                    \
  do {                                                                        \
    if (!TT_ASSERT_UNLIKELY(condition)) {                                     \
      (::ttmlir::utils::asserts::ExprDecomposer { } <= condition )            \
        .report(TT_ASSERT_REPORT_STREAM(), TT_IMPL_ASSERT_LOC_INFO, #condition, \
                llvm::formatv(__VA_ARGS__).str());                            \
      TT_ASSERT_FAILURE();                                                    \
    }                                                                         \
  } while (false)                                                             \
  /* */

// ............................................................................

// a <= x < b
#define TT_assert_open_range(x, a, b) \
  TT_assertv(in_open_range(x, a, b), "{} ({}) is not in [{}, {})", TT_ASSERT_STRINGIZE(x), x, a, b)

// 0 <= x < b, convenience shortcut.
#define TT_assert_limit(x, limit) \
  TT_assertv(in_open_range(x, 0, limit), "{} ({}) is not in [0, {})", TT_ASSERT_STRINGIZE(x), x, limit)

// a < x < b
#define TT_assert_exclusive_range(x, a, b) \
  TT_assertv(in_exclusive_range(x, a, b), "{} ({}) is not in ({}, {})", TT_ASSERT_STRINGIZE(x), x, a, b)

// a <= x <= b
#define TT_assert_inclusive_range(x, a, b) \
  TT_assertv(in_inclusive_range(x, a, b), "{} ({}) is not in [{}, {}]", TT_ASSERT_STRINGIZE(x), x, a, b)

// ............................................................................

#ifdef TT_ASSERT_ENABLE_DEBUG_CHECKS

# define TT_debug(condition)                            TT_assert(condition)
# define TT_debugv(condition, /* message[, args] */...) TT_assertv(condition, __VA_ARGS__)

# define TT_debug_open_range(x, a, b)                   TT_assert_open_range(x, a, b)
# define TT_debug_limit(x, limit)                       TT_assert_limit(x, limit)
# define TT_debug_exclusive_range(x, a, b)              TT_assert_exclusive_range(x, a, b)
# define TT_debug_inclusive_range(x, a, b)              TT_assert_inclusive_range(x, a, b)

#else // !TT_ASSERT_ENABLE_DEBUG_CHECKS

# define TT_debug(condition)                            ((void)0)
# define TT_debugv(condition, /* message[, args] */...) ((void)0)

# define TT_debug_open_range(x, a, b)                   ((void)0)
# define TT_debug_limit(x, limit)                       ((void)0)
# define TT_debug_exclusive_range(x, a, b)              ((void)0)
# define TT_debug_inclusive_range(x, a, b)              ((void)0)

#endif // TT_ASSERT_ENABLE_DEBUG_CHECKS

// clang-format on
// ............................................................................

} // namespace ttmlir::utils::asserts
#endif // TTMLIR_ASSERTS_H
