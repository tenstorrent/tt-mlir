// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_UTILS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_UTILS_H

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"

#include <tuple>
#include <type_traits>
#include <utility>

//===---------------------------------------------------------------------===//
namespace llvm {

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const BitVector &obj) {
  constexpr char digits[2] = {'0', '1'};

  os << '<';
  for (BitVector::size_type i = 0; i < obj.size(); ++i) {
    os << digits[obj.test(i)];
  }
  return os << '>';
}

} // namespace llvm
//===---------------------------------------------------------------------===//
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

//===---------------------------------------------------------------------===//

/// Namespace-local shortcut for `llvm::to_underlying(e)`.
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> ordinal(Enum e) {
  return llvm::to_underlying(e);
}
//===---------------------------------------------------------------------===//
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

//===---------------------------------------------------------------------===//
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
//===---------------------------------------------------------------------===//
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
//===---------------------------------------------------------------------===//

template <typename T>
SmallVector<T> concatToVector(const SmallVector<SmallVector<T>> &vs) {
  SmallVector<T> r;
  for (const auto &v : vs) {
    r.append(v.begin(), v.end());
  }
  return r;
}

template <typename T, typename... RangeTs>
auto concatToVector(RangeTs &&...ranges)
    -> std::enable_if_t<(sizeof...(RangeTs) > 1), SmallVector<T>> {
  return llvm::to_vector(
      llvm::concat<const T>(std::forward<RangeTs>(ranges)...));
}
//===---------------------------------------------------------------------===//

namespace detail {

template <char... Cs>
struct CharSeq {
  static void print(llvm::raw_ostream &os) { ((os << Cs), ...); }
};

template <typename T, typename SepSeq = CharSeq<>, typename OpenSeq = CharSeq<>,
          typename CloseSeq = CharSeq<>>
struct IterablePrintAdaptor {
  const T *obj;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const IterablePrintAdaptor &adaptor) {
    OpenSeq::print(os);
    {
      bool first = true;
      for (const auto &dim : *adaptor.obj) {
        if (first) {
          first = false;
        } else {
          SepSeq::print(os);
        }
        os << dim;
      }
    }
    CloseSeq::print(os);

    return os;
  }
};
} // namespace detail

/// Syntactic sugar helper for printing iterable containers in `[V0, V1, ...]`
/// format.
/// @code
///     llvm::dbgs() << asSeq(...) << ...
/// @endcode
template <typename T>
auto asSeq(const T &obj)
    -> std::enable_if_t<is_iterable_v<T>,
                        detail::IterablePrintAdaptor<
                            T, detail::CharSeq<',', ' '>, detail::CharSeq<'['>,
                            detail::CharSeq<']'>>> {
  return {&obj};
}

/// Syntactic sugar helper for printing shapes in `D0xD1x...` format. A "shape"
/// is loosely understood to be anything that is iterable. Usage:
/// @code
///     llvm::dbgs() << asShape(...) << ...
/// @endcode
template <typename T>
auto asShape(const T &obj)
    -> std::enable_if_t<is_iterable_v<T>,
                        detail::IterablePrintAdaptor<T, detail::CharSeq<'x'>>> {
  return {&obj};
}
//===---------------------------------------------------------------------===//
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

//===---------------------------------------------------------------------===//
// Shared helpers for the allocator and block factor analysis.
//===---------------------------------------------------------------------===//

/// @return `map` with all broadcast result expressions replaced with const-1
/// expression.
///
/// This could almost be a simple AffineMap::replace() but need to make sure
/// only complete `0`-result expressions are replaced, not other possible zero
/// const terms within result expression trees, however unlikely that seems.
inline AffineMap canonicalizeBroadcasts(AffineMap map) {
  auto *ctx = map.getContext();
  const auto replacement = mlir::getAffineConstantExpr(1, ctx);
  SmallVector<AffineExpr> exprs;

  for (auto expr : map.getResults()) {
    if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
      if (constExpr.getValue() == 0) {
        exprs.push_back(replacement);
        continue;
      }
    }
    exprs.push_back(expr);
  }

  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx);
}

/// Build a ShardLayoutAttr-backed MemRefType from explicit grid and shard
/// shapes.
inline MemRefType getStreamBufferType(ArrayRef<int64_t> gridShape,
                                      ArrayRef<int64_t> shardShape,
                                      Type elementType,
                                      ttcore::MemorySpaceAttr memSpaceAttr,
                                      uint32_t buffers) {
  TT_debug(gridShape.size() == shardShape.size());

  const SmallVector<int64_t> fullShape =
      concatToVector<int64_t>(gridShape, shardShape);
  const auto bufferLayout =
      ttcore::ShardLayoutAttr::get(shardShape, elementType, buffers);

  return MemRefType::get(fullShape, elementType, bufferLayout, memSpaceAttr);
}

/// @return the size in bytes of a stream buffer (must be in L1).
inline int64_t getStreamBufferSizeBytes(MemRefType bufferType,
                                        ttcore::DeviceAttr device) {
  TT_assertv(ttcore::getMemorySpace(bufferType) ==
                 ttcore::MemorySpace::DeviceL1,
             "stream buffers must be allocated in L1");
  return device.getMemrefSizeBytes(bufferType, 0, true);
}

/// Project the generic op's concatenated operand grid and shard shapes through
/// the inverse of its indexing maps, yielding iteration-space-aligned extents.
/// @return a (gridExtents, shardExtents) tuple.
inline std::tuple<SmallVector<int64_t>, SmallVector<int64_t>>
getGridAndShardExtents(d2m::GenericOp genericOp) {
  auto flatInverseMap = ttmlir::utils::concatInversePermutationMap(
      genericOp.getIndexingMapsValue(), /*reverse=*/false);

  return {flatInverseMap.compose(
              concatToVector(genericOp.getInputOutputOperandGridShapes())),
          flatInverseMap.compose(
              concatToVector(genericOp.getInputOutputOperandShardShapes()))};
}

/// Return a bitmask that indicates which of the dims are "participating"
/// (defined as dims that are used by any of the `genericOp`'s output indexing
/// map expressions).
inline llvm::BitVector getParticipatingDimMask(d2m::GenericOp genericOp) {
  TT_debug(genericOp.getOutputs().size() == 1u);
  AffineMap outputMap = genericOp.getIndexingMapsValue().back();

  const std::size_t rank = outputMap.getNumDims();

  llvm::BitVector mask(rank, false);
  for (std::size_t d = 0; d < rank; ++d) {
    mask[d] = outputMap.isFunctionOfDim(d);
  }
  return mask;
}

/// Calculate full "shard-only" blocking factors (defined as full blocking
/// factors divided by blocking factors).
inline SmallVector<int64_t> getShardBlockFactors(d2m::GenericOp genericOp) {
  SmallVector<int64_t> r = genericOp.getFullBlockFactors();
  SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();
  TT_debug(r.size() == blockFactors.size());
  for (std::size_t d = 0; d < r.size(); ++d) {
    TT_debugv(blockFactors[d] > 0, "unexpected block factor {} for dim {}",
              blockFactors[d], d);
    r[d] /= blockFactors[d];
  }

  return r;
}

} // namespace mlir::tt::d2m::allocation

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_UTILS_H
