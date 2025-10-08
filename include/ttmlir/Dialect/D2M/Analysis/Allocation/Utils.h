// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_UTILS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_UTILS_H

#include "ttmlir/Asserts.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLForwardCompat.h"

#include <tuple>
#include <type_traits>
#include <utility>

namespace mlir::tt::d2m::allocation {

inline bool debugEnabled() {
  return (llvm::DebugFlag &&
          ttmlir::isLogLevelEnabled(ttmlir::LogLevel::Debug));
}

#define TT_UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#define TT_DEBUG_ENABLED()                                                     \
  TT_UNLIKELY(mlir::tt::d2m::allocation::debugEnabled())

// Define some convenience macros local to `Analysis`.

#define TT_ALLOC_DEBUG(/* fmt, args */...)                                     \
  TTMLIR_DEBUG(::ttmlir::LogComponent::Allocator, __VA_ARGS__)

#define TT_ALLOC_TRACE(/* fmt, args */...)                                     \
  TTMLIR_TRACE(::ttmlir::LogComponent::Allocator, __VA_ARGS__)

#define TT_ALLOC_ERROR(/* fmt, args */...)                                     \
  do {                                                                         \
    auto &OS = ::llvm::errs();                                                 \
    OS.enable_colors(true);                                                    \
    OS.changeColor(::llvm::raw_ostream::RED, /*bold=*/true);                   \
    OS << llvm::formatv(__VA_ARGS__) << "\n";                                  \
    OS.resetColor();                                                           \
  } while (false)

/// Namespace-local shortcut for `llvm::to_underlying(e)`.
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> ordinal(Enum e) {
  return llvm::to_underlying(e);
}

namespace detail {
template <typename T, typename = void>
struct is_operation : std::false_type {};

template <typename T>
struct is_operation<T, std::void_t<decltype(std::declval<T>().getOperaton())>>
    : std::true_type {};
} // namespace detail

/// Helper for conditioning template parameters on being MLIR ops.
template <typename T>
constexpr bool is_operation_v = detail::is_operation<T>::value;

namespace detail {
template <typename Compare, typename... Fields>
class lexicographical_field_comparator : Compare {
public:
  lexicographical_field_comparator(std::tuple<Fields...> &&fields)
      : fields(std::move(fields)) {}

  template <typename T>
  constexpr bool operator()(const T &lhs, const T &rhs) const {
    using indexes = std::make_index_sequence<sizeof...(Fields)>;

    return compare(lhs, rhs, indexes{});
  }

private:
  template <typename T, std::size_t... I>
  bool compare(const T &lhs, const T &rhs, std::index_sequence<I...>) const {
    return (static_cast<const Compare &>(*this))(
        std::tie(lhs.*std::get<I>(fields)...),
        std::tie(rhs.*std::get<I>(fields)...));
  }

  std::tuple<Fields...> fields;
};
} // namespace detail

/// Helper function for creating a lexicographic comparator
/// from an ordered list of struct member pointers, e.g
/// @code
///   struct Priority {
///    std::size_t size;
///    std::int32_t k;
///   };
///   auto compare = make_lexicographical_field_comparator<std::less<>>(
///     &Priority::k,
///     &Priority::size
///   );
/// @endcode
template <typename Compare, typename... Fields>
auto make_lexicographical_field_comparator(Fields... fields) {
  return detail::lexicographical_field_comparator<Compare, Fields...>{
      std::make_tuple(fields...)};
}

namespace detail {

using std::begin;
using std::end;

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(begin(std::declval<T>())),
                                  decltype(end(std::declval<T>()))>>
    : std::true_type {};

} // namespace detail

template <typename T>
constexpr bool is_iterable_v = detail::is_iterable<T>::value;

namespace detail {
template <typename T>
struct AsShapePrintAdaptor {
  const T *obj;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const AsShapePrintAdaptor &adaptor) {
    bool first = true;
    for (const auto &dim : *adaptor.obj) {
      if (first) {
        first = false;
      } else {
        os << 'x';
      }
      os << dim;
    }

    return os;
  }
};
} // namespace detail

/// Syntactic sugar helper for printing shapes in `D0xD1x...` format. A "shape"
/// is loosely understood to be anything that is iterable. Usage:
/// @code
///     llvm::dbgs() << asShape(...) << ...
/// @endcode
template <typename T>
auto asShape(const T &obj)
    -> std::enable_if_t<is_iterable_v<T>, detail::AsShapePrintAdaptor<T>> {
  return {&obj};
}

/// Syntactic sugar helper for printing MLIR Value %-names with
/// the same IDs as generated by the stock IR tools.
///
/// Usage:
/// @code
///   void foo(func::FuncOp funcOp) {
///     [[maybe_unused]] AsOperandPrinter asOperand{funcOp};
///     ...
///     llvm::dbgs() << asOperand(...) << ...
/// @endcode
struct AsOperandPrinter {
  mlir::AsmState state;

  struct PrintAdaptor {
    AsOperandPrinter *parent;
    mlir::Operation *op = nullptr;
    mlir::Value *value = nullptr;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const PrintAdaptor &adaptor) {
      if (adaptor.op != nullptr) {
        if (adaptor.op->getNumResults() > 0) {
          adaptor.op->getResult(0).printAsOperand(os, adaptor.parent->state);
        } else {
          os << adaptor.op->getName();
        }
      } else {
        adaptor.value->printAsOperand(os, adaptor.parent->state);
      }
      return os;
    }
  };

  AsOperandPrinter(mlir::Operation *state) : state(state) {}

  PrintAdaptor operator()(mlir::Operation *op) { return {this, op, nullptr}; }
  PrintAdaptor operator()(mlir::Value &value) {
    return {this, nullptr, &value};
  }

  template <typename ConcreteOp>
  auto operator()(ConcreteOp op)
      -> std::enable_if_t<is_operation_v<ConcreteOp>, PrintAdaptor> {
    return this->operator()(op.getOperation());
  }
};

} // namespace mlir::tt::d2m::allocation

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_UTILS_H
