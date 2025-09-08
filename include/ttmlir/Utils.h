// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_UTILS_H
#define TTMLIR_UTILS_H

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ttmlir::utils {

constexpr inline llvm::StringLiteral g_constEvalAttrName = "const_eval";
constexpr inline llvm::StringLiteral g_conv2dWeightAttrName =
    "ttir.conv2d_weight";
constexpr inline llvm::StringLiteral g_cpuHoistFuncCallAttrName =
    "ttir.cpu_hoist_call";
constexpr inline llvm::StringLiteral g_skipQdqCommuteAttrName =
    "ttir.skip_qdq_commute";

template <typename T>
T alignUp(const T val, const T alignment) {
  assert(alignment > 0);
  return ((val + alignment - 1) / alignment) * alignment;
}

template <typename Iter>
auto product(const Iter begin, const Iter end) ->
    typename std::iterator_traits<Iter>::value_type {
  using ValueType = typename std::iterator_traits<Iter>::value_type;
  return std::accumulate(begin, end, static_cast<ValueType>(1),
                         std::multiplies<ValueType>());
}

template <typename T>
inline mlir::SmallVector<T> calculateStrides(mlir::ArrayRef<T> shape,
                                             T elementSize = 1) {
  mlir::SmallVector<T> strides(shape.size());
  T stride = elementSize;
  assert(!shape.empty());
  for (std::int64_t i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

template <typename Vector, typename Fn>
inline void sample(const Vector &shape, Fn fn) {
  if (shape.size() == 0) {
    return;
  }
  llvm::SmallVector<std::int64_t> strides = calculateStrides(shape);
  llvm::SmallVector<std::int64_t, 8> index(shape.size());
  int64_t volume = shape[0] * strides[0];
  for (int64_t i = 0; i < volume; ++i) {
    for (unsigned j = 0; j < shape.size(); ++j) {
      index[j] = (i / strides[j]) % shape[j];
    }
    fn(index);
  }
}

template <typename Vector>
llvm::SmallVector<int64_t> evalShape(mlir::AffineMap map, Vector shape) {
  mlir::SmallVector<int64_t> lastIndex;
  for (auto dim : shape) {
    lastIndex.push_back(dim - 1);
  }

  auto result = map.compose(lastIndex);
  for (auto &dim : result) {
    dim += 1;
  }
  return result;
}

inline mlir::AffineMap
replaceAffineMapSymbols(mlir::AffineMap map, mlir::ArrayRef<int64_t> symbols) {
  assert(map.getNumSymbols() == symbols.size());

  mlir::SmallVector<mlir::AffineExpr> symReplacements;
  for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
    symReplacements.push_back(
        getAffineConstantExpr(symbols[i], map.getContext()));
  }

  mlir::SmallVector<mlir::AffineExpr> dimReplacements;
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    dimReplacements.push_back(getAffineDimExpr(i, map.getContext()));
  }

  unsigned numResultSyms = 0;
  return map.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                   map.getNumDims(), numResultSyms);
}

template <typename T>
T volume(mlir::ArrayRef<T> shape, T stride = 1) {
  return std::accumulate(shape.begin(), shape.end(), stride,
                         std::multiplies<T>());
}

// Returns a string that is the concatenation of the string representations of
// Range R elements interleaved with separator. Example: join({1, 2, 3}, ", ")
// -> "1, 2, 3"
template <typename Range>
std::string join(Range &&R, llvm::StringRef separator) {
  return llvm::join(
      llvm::map_range(R, [](auto &v) { return llvm::Twine(v).str(); }),
      separator);
}

// Prepacks `MlirAttribute`s stored in input array into a vector of
// `mlir::Attribute`s and then wraps that vector in `MlirAttribute` and returns
// it.
//
// This util function can be used as a helper to create an attribute from an
// array of attributes for any type defined like for example:
//
// def TT_IteratorTypeArrayAttr : TypedArrayAttrBase<TT_IteratorTypeAttr, "">;
//
// since these don't get any special Cpp class generated for them from
// tablegen.
//
// This is useful if you want to create a Pybind static geter in some attribute
// class `TT_XAttr` for which you have `TT_XArrayAttr` defined (as shown in
// example above). For example:
//
// py::class_<tt::CircularBufferAttributesAttr>(m,
//                                              "CircularBufferAttributesAttr")
//     .def_static(
//         "get",
//         [](MlirContext ctx, uint8_t cb_id, MlirAttribute core_range,
//            uint32_t total_size, uint32_t page_size, uint32_t data_format) {
//           return wrap(tt::CircularBufferAttributesAttr::get(
//               unwrap(ctx), static_cast<tt::CB>(cb_id),
//               mlir::cast<tt::CoreRangeAttr>(unwrap(core_range)), total_size,
//               page_size, static_cast<ttcore::DataType>(data_format)));
//         })
//     .def_static("get", [](MlirContext ctx,
//                           std::vector<MlirAttribute> attributesArray) {
//       return ::ttmlir::utils::wrapArrayOfMlirAttributesAsAttribute(ctx,
//           attributesArray);
//     });
inline MlirAttribute wrapArrayOfMlirAttributesAsAttribute(
    MlirContext ctx, std::vector<MlirAttribute> &attributesArray) {
  std::vector<mlir::Attribute> unwrappedAttributesArray;
  for (auto attr : attributesArray) {
    unwrappedAttributesArray.push_back(unwrap(attr));
  }
  return wrap(mlir::ArrayAttr::get(unwrap(ctx), unwrappedAttributesArray));
}

// Checks if the type of the given `mlir::Value` is a ranked tensor type.
inline bool isRankedTensor(mlir::Value v) {
  return mlir::isa<mlir::RankedTensorType>(v.getType());
}

// Returns the element received as a parameter. Useful as a callback for
// higher-order functions.
template <typename T>
inline T identity(T x) {
  return x;
}

// Returns a vector of indices `permutation` such that input[permutation[i]] ==
// output[i], for all i. Assumes that input and output have the same elements.
// Example:  input = [1, 2, 3], output = [3, 1, 2] -> [2, 0, 1]
template <typename T>
inline llvm::SmallVector<int64_t>
generatePermutation(llvm::ArrayRef<T> input, llvm::ArrayRef<T> output) {
  assert(input.size() == output.size());

  llvm::DenseMap<T, int64_t> indices;
  for (const auto [index, value] : llvm::enumerate(input)) {
    indices[value] = index;
  }
  llvm::SmallVector<int64_t> permutation;
  for (const T &dim : output) {
    permutation.push_back(indices[dim]);
  }
  return permutation;
}

// Returns a vector `output`, such that output[i] = input[permutation[i]], for
// all i. Assumes that permutation is a valid permutation of the indices of
// input. Example:  input = [1, 2, 3], permutation = [2, 0, 1] -> [3, 1, 2]
template <typename T>
inline llvm::SmallVector<T>
applyPermutation(llvm::ArrayRef<T> input, llvm::ArrayRef<int64_t> permutation) {
  assert(input.size() == permutation.size());

  llvm::SmallVector<T> output(input.size());

  llvm::transform(permutation, output.begin(),
                  [&](const int64_t i) { return input[i]; });

  return output;
}

// Returns a vector `inversePermutation`, such that
// inversePermutation[permutation[i]] = i, for all i. Assumes that permutation
// is a valid permutation of a range(0, permutation.size()). Example:
// permutation = [2, 0, 1] -> [1, 2, 0]
inline llvm::SmallVector<int64_t>
inversePermutation(llvm::ArrayRef<int64_t> permutation) {
  llvm::SmallVector<int64_t> inversePermutation(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    inversePermutation[permutation[i]] = i;
  }
  return inversePermutation;
}

// Returns a vector `broadcastShape`, such that each index i of inputShape
// multiplied by index i of broadcastShape is equal to index i of outputShape.
// Example:  inputShape = [1, 32, 1], outputShape = [1, 32, 16] -> [1, 1, 16]
template <typename T>
inline llvm::SmallVector<T>
getBroadcastDimensions(llvm::ArrayRef<int64_t> inputShape,
                       llvm::ArrayRef<int64_t> outputShape) {
  assert(inputShape.size() == outputShape.size() &&
         "Input and Output shape should match.");

  llvm::SmallVector<T> broadcastShape;
  for (size_t i = 0; i < outputShape.size(); i++) {
    T d = outputShape[i] / inputShape[i];
    broadcastShape.push_back(d);
  }

  return broadcastShape;
}

// For a given llvm::APInt value, returns it as a C++ integer type T.
template <typename T>
inline T integerAs(const llvm::APInt &value) {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(value.getSExtValue());
  } else {
    static_assert(std::is_unsigned_v<T>,
                  "T must be signed or unsigned integer type");
    return static_cast<T>(value.getZExtValue());
  }
}

// For a given mlir::Attribute attr, returns a pair of integers of type
// ReturnTy. If attr is an IntegerAttr, it's interpreted as a (value(attr),
// value(attr)) pair of values, where value(attr) is of type ScalarTy. If attr
// is a DenseArrayAttr<VectorElementTy> of size 2, it's interpreted as a
// (attr[0], attr[1]) pair of values. Otherwise, returns an error message.
template <typename ScalarTy, typename VectorElementTy = ScalarTy,
          typename ReturnTy = ScalarTy>
inline llvm::Expected<std::pair<ReturnTy, ReturnTy>>
getPairOfInteger(mlir::Attribute attr) {
  ReturnTy x{};
  ReturnTy y{};
  // If attr is IntgerAttr, it's interpreted as a (attr, attr) pair of values.
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    x = y = integerAs<ScalarTy>(value.getValue());
    // If attr is DenseArrayAttr, it's interpreted as a (attr[0], attr[1]) pair
    // of values if it has size 2.
  } else if (auto tuple = mlir::dyn_cast<
                 ::mlir::detail::DenseArrayAttrImpl<VectorElementTy>>(attr);
             tuple.size() == 2) {
    x = tuple[0];
    y = tuple[1];
    // Otherwise, it's an error.
  } else if (tuple) {
    return llvm::createStringError(
        "Expected integer or pair of integers, got tuple of size %lu",
        tuple.size());
  } else {
    return llvm::createStringError("Unexpected attribute type");
  }

  return std::make_pair(x, y);
}

/// Extracts a tuple of four integers from an mlir::Attribute.
/// - If `attr` is an IntegerAttr, it is interpreted as (value(attr),
/// value(attr), value(attr), value(attr))
/// - If `attr` is a DenseArrayAttr of size 2, it expands to (attr[0], attr[1],
/// attr[0], attr[1])
/// - If `attr` is a DenseArrayAttr of size 4, it is returned as (attr[0],
/// attr[1], attr[2], attr[3])
/// - Otherwise, returns an error.
template <typename ScalarTy, typename VectorElementTy = ScalarTy,
          typename ReturnTy = ScalarTy>
inline llvm::Expected<std::tuple<ReturnTy, ReturnTy, ReturnTy, ReturnTy>>
getQuadrupleOfInteger(mlir::Attribute attr) {
  // If attr is IntegerAttr, interpret it as (attr, attr, attr, attr)
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    ReturnTy v = integerAs<ScalarTy>(intAttr.getValue());
    return std::make_tuple(v, v, v, v);
  }

  auto tuple =
      mlir::dyn_cast<::mlir::detail::DenseArrayAttrImpl<VectorElementTy>>(attr);
  if (!tuple) {
    return llvm::createStringError("Unexpected attribute type");
  }

  // If attr is DenseArrayAttr, handle based on its size
  switch (tuple.size()) {
  case 2:
    return std::make_tuple(tuple[0], tuple[1], tuple[0], tuple[1]);
  case 4:
    return std::make_tuple(tuple[0], tuple[1], tuple[2], tuple[3]);
  default:
    return llvm::createStringError(
        "Expected integer, pair, or tuple of size 4, but got tuple of size %lu",
        tuple.size());
  }
}

namespace detail {
template <typename, typename = void>
struct is_leaf_type : std::true_type {};

template <typename T>
struct is_leaf_type<T, std::void_t<decltype(std::declval<T>().begin())>>
    : std::false_type {};

template <typename T>
constexpr bool is_leaf_type_v = is_leaf_type<T>::value;

template <typename T, typename = void>
struct get_value_type {
  using type = llvm::remove_cvref_t<T>;
};

template <typename T>
using get_value_type_t = typename get_value_type<T>::type;

template <typename T>
struct get_value_type<T, std::enable_if_t<!is_leaf_type_v<T>>> {
  using type = get_value_type_t<typename std::iterator_traits<
      decltype(std::declval<T>().begin())>::value_type>;
};

template <typename T, typename U>
std::enable_if_t<std::is_convertible_v<get_value_type_t<T>, U>>
append(llvm::SmallVector<U> &result, T &&value) {
  if constexpr (is_leaf_type_v<T>) {
    result.push_back(std::forward<T>(value));
  } else {
    for (auto &&v : value) {
      append(result, std::forward<decltype(v)>(v));
    }
  }
}
} // namespace detail

template <typename FirstTy, typename FallbackTy>
using type_or_fallback_t =
    std::conditional_t<std::is_void_v<FirstTy>, FallbackTy, FirstTy>;

template <typename ReturnTy = void, typename FirstTy, typename... RestTy>
llvm::SmallVector<
    type_or_fallback_t<ReturnTy, detail::get_value_type_t<FirstTy>>>
flatten(FirstTy &&first, RestTy &&...rest) {
  using TrueReturnTy =
      type_or_fallback_t<ReturnTy, detail::get_value_type_t<FirstTy>>;
  static_assert(
      (std::is_convertible_v<detail::get_value_type_t<FirstTy>, TrueReturnTy> &&
       ... &&
       std::is_convertible_v<detail::get_value_type_t<RestTy>, TrueReturnTy>));
  llvm::SmallVector<TrueReturnTy> result;
  detail::append(result, std::forward<FirstTy>(first));
  (detail::append(result, std::forward<RestTy>(rest)), ...);
  return result;
}

// Append a suffix to a location name if it's a NameLoc.
// If the location is not a NameLoc or suffix is empty, returns the original
// location.
inline mlir::Location appendLocationSuffix(mlir::Location loc,
                                           llvm::StringRef suffix) {
  if (suffix.empty() || !mlir::isa<mlir::NameLoc>(loc)) {
    return loc;
  }

  mlir::NameLoc nameLoc = mlir::cast<mlir::NameLoc>(loc);
  return mlir::NameLoc::get(
      mlir::StringAttr::get(loc.getContext(), nameLoc.getName().str() + suffix),
      loc);
}

// Extract the first n lines from a string.
inline std::string firstNLines(std::string str, int n) {
  std::unique_ptr<llvm::MemoryBuffer> memBuf =
      llvm::MemoryBuffer::getMemBuffer(str);
  llvm::line_iterator lineIt(*memBuf);
  std::string result;
  for (int i = 0; i < n && !lineIt.is_at_end(); ++i, ++lineIt) {
    result += *lineIt;
    result += "\n";
  }
  return result;
}

template <typename...>
constexpr bool always_false() {
  return false;
}

template <typename... ParentOps>
static mlir::Region *getRegionWithParentOfType(mlir::Operation *op) {
  mlir::Region *region = op->getParentRegion();
  mlir::Operation *parentOp = region->getParentOp();
  while (!mlir::isa<ParentOps...>(parentOp)) {
    region = parentOp->getParentRegion();
    if (!region) {
      break;
    }
    parentOp = region->getParentOp();
  }
  return region;
}

// Check if all users of srcOp are of UserOps types.
template <typename... UserOps>
inline bool allUsersOfType(mlir::Operation *srcOp) {
  auto check = [](mlir::Operation *op) { return mlir::isa<UserOps...>(op); };
  return llvm::all_of(srcOp->getResult(0).getUsers(), check);
}

// Count the number of users of a value.
inline size_t countUsers(mlir::Value value) {
  return std::distance(value.user_begin(), value.user_end());
}

inline bool isConstEvalFunc(mlir::Operation *op) {
  if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
    return funcOp->hasAttr(g_constEvalAttrName);
  }
  return false;
}

template <typename T, typename From>
T castContainer(const From &value) {
  return T(value.begin(), value.end());
};

namespace loop {

template <typename OpType>
OpType getOutermostLoopNest(mlir::Operation *op) {
  OpType opType = mlir::dyn_cast<OpType>(op);
  OpType maybeOuter = mlir::dyn_cast<OpType>(op->getParentOp());
  while (maybeOuter) {
    opType = maybeOuter;
    maybeOuter = mlir::dyn_cast<OpType>(maybeOuter->getParentOp());
  }
  return opType;
}

template <typename OpType>
OpType getOutermostLoopNest(mlir::Region *region) {
  return getOutermostLoopNest<OpType>(region->getParentOp());
}

template <typename OpType>
OpType getOutermostLoopNest(mlir::Block *block) {
  return getOutermostLoopNest<OpType>(block->getParent());
}

template <typename OpType>
OpType getOutermostLoopNest(mlir::OpOperand &use) {
  return getOutermostLoopNest<OpType>(use.getOwner());
}

template <typename OpType>
OpType getOutermostLoopNest(mlir::Value value) {
  return getOutermostLoopNest<OpType>(value.getParentRegion());
}

template <typename OpType>
OpType getOutermostLoopNest(mlir::ValueRange values) {
  assert(!values.empty());
  return getOutermostLoopNest<OpType>(values.front());
}

} // namespace loop

// Given a startng mlir::Value return a set of all values in the use-def
// chain. This chain is not topologically sorted, so the order of values in the
// result is not guaranteed. If you want to topologically sort the chain
// use topologicalSort.
inline llvm::SetVector<mlir::Value> getUseDefChain(mlir::Value start) {
  llvm::SetVector<mlir::Value> useDefChain;
  llvm::SmallVector<mlir::Value> worklist{start};
  llvm::SmallPtrSet<mlir::Value, 4> visited;

  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    useDefChain.insert(value);

    mlir::Operation *defOp = value.getDefiningOp();
    if (!defOp) {
      continue;
    }

    for (mlir::OpOperand &operand : defOp->getOpOperands()) {
      mlir::Value operandValue = operand.get();
      if (visited.contains(operandValue)) {
        continue;
      }
      visited.insert(operandValue);
      worklist.push_back(operandValue);
    }
  }

  return useDefChain;
}

// Given list of mlir::Value filter out block arguments.
inline llvm::SetVector<mlir::BlockArgument>
filterBlockArguments(llvm::ArrayRef<mlir::Value> values) {
  llvm::SetVector<mlir::BlockArgument> blockArgs;
  for (mlir::Value value : values) {
    if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      blockArgs.insert(blockArg);
    }
  }

  return blockArgs;
}

// Given list of mlir::Value filter out operations that define them.
// If value is not operation it is ignored.
inline llvm::SetVector<mlir::Operation *>
filterOperations(llvm::ArrayRef<mlir::Value> values) {
  llvm::SetVector<mlir::Operation *> ops;
  for (mlir::Value value : values) {
    if (auto *op = value.getDefiningOp()) {
      ops.insert(op);
    }
  }

  return ops;
}

using Coord = llvm::SmallVector<int64_t, 4>;
// Converts a linear device ID into its row-major N-D mesh coordinates.
inline Coord linearIdToCoord(size_t id, llvm::ArrayRef<int64_t> shape) {
  Coord coord(shape.size(), 0);
  for (size_t i = shape.size(); i-- > 0;) {
    coord[i] = id % shape[i];
    id /= shape[i];
  }
  return coord;
}

template <typename T>
llvm::SmallVector<llvm::SmallVector<T>>
denseElementsAttrTo2D(mlir::DenseElementsAttr attr) {
  auto shape = attr.getType().getShape();
  assert((shape.size() == 2) && "DenseElementsAttr must be 2D");
  size_t rows = shape[0];
  size_t columns = shape[1];
  assert((static_cast<size_t>(attr.getNumElements()) == rows * columns) &&
         "DenseElementsAttr size mismatch");
  auto values = attr.getValues<T>();
  auto it = values.begin();
  llvm::SmallVector<llvm::SmallVector<T>> result(rows,
                                                 llvm::SmallVector<T>(columns));
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < columns; ++c) {
      result[r][c] = *it++;
    }
  }
  return result;
}

inline llvm::StringRef getKernelName(mlir::Operation *op) {
  if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
    return funcOp.getSymName();
  }
  if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
    return funcOp.getSymName();
  }

  llvm_unreachable("Expected func op parent of op in call to getKernelName.");
}

} // namespace ttmlir::utils

#endif // TTMLIR_UTILS_H
