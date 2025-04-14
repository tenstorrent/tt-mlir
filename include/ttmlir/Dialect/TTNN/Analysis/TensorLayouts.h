// SPDX-FileCopyrightText: : © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_TENSORLAYOUTS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_TENSORLAYOUTS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

// DenseMap info for RankedTensorType
namespace llvm {
template <>
struct DenseMapInfo<mlir::RankedTensorType> {
  static mlir::RankedTensorType getEmptyKey() {
    static const intptr_t EmptyKeyVal = static_cast<intptr_t>(-1);
    return mlir::RankedTensorType(static_cast<mlir::TensorType::ImplType *>(
        reinterpret_cast<void *>(EmptyKeyVal)));
  }

  static mlir::RankedTensorType getTombstoneKey() {
    static const intptr_t TombstoneKeyVal = static_cast<intptr_t>(-2);
    return mlir::RankedTensorType(static_cast<mlir::TensorType::ImplType *>(
        reinterpret_cast<void *>(TombstoneKeyVal)));
  }

  static unsigned getHashValue(const mlir::RankedTensorType &val) {
    // For special keys, return their pointer values as hash
    if (val == getEmptyKey()) {
      return static_cast<unsigned>(-1);
    }

    if (val == getTombstoneKey()) {
      return static_cast<unsigned>(-2);
    }

    // For valid types, hash the shape, element type, and encoding
    return llvm::hash_combine(
        llvm::hash_combine_range(val.getShape().begin(), val.getShape().end()),
        val.getElementType(),
        val.getEncoding() ? val.getEncoding().getAsOpaquePointer() : nullptr);
  }

  static bool isEqual(const mlir::RankedTensorType &lhs,
                      const mlir::RankedTensorType &rhs) {
    // First, handle all special key combinations
    if (lhs == getEmptyKey()) {
      return rhs == getEmptyKey();
    }

    if (lhs == getTombstoneKey()) {
      return rhs == getTombstoneKey();
    }

    if (rhs == getEmptyKey() || rhs == getTombstoneKey()) {
      return false;
    }

    // Only compare shape, element type, and encoding for valid types
    return lhs.getShape().equals(rhs.getShape()) &&
           lhs.getElementType() == rhs.getElementType() &&
           lhs.getEncoding() == rhs.getEncoding();
  }
};
} // namespace llvm

namespace mlir::tt::ttnn {

enum class TensorMemoryLayoutIndex {
  Interleaved = 0,
  Sharded = 1,
  kNumValues = 2
};

enum class TensorDataLayoutIndex { RowMajor = 0, Tiled = 1, kNumValues = 2 };

// TensorType layouts categorized by scalar type, data layout (tile/RM) and
// memory layout (sharded/interleaved). We will not make distinction between
// different sharding strategies in this collection.
using TensorTypeLayouts = llvm::DenseMap<
    Type,
    std::array<std::array<std::vector<TTNNLayoutAttr>,
                          static_cast<std::size_t>(
                              TensorMemoryLayoutIndex::kNumValues)>,
               static_cast<std::size_t>(TensorDataLayoutIndex::kNumValues)>>;

using TensorTypeLayoutsMap =
    llvm::DenseMap<RankedTensorType, TensorTypeLayouts>;

// Helper function to convert Layout to TensorDataLayoutIndex as std::size_t
inline std::size_t getDataLayoutIndex(Layout layout) {
  return static_cast<std::size_t>(layout == Layout::Tile
                                      ? TensorDataLayoutIndex::Tiled
                                      : TensorDataLayoutIndex::RowMajor);
}

// Helper function to convert TensorMemoryLayout to TensorMemoryLayoutIndex as
// std::size_t
inline std::size_t getMemoryLayoutIndex(TensorMemoryLayout memLayout) {
  return static_cast<std::size_t>(isShardedMemoryLayout(memLayout)
                                      ? TensorMemoryLayoutIndex::Sharded
                                      : TensorMemoryLayoutIndex::Interleaved);
}

// Helper function to get all layouts for a given tensor type and scalar type
std::vector<TTNNLayoutAttr> getShardedLayoutsForTensorTypeAndScalarType(
    const TensorTypeLayoutsMap &tensorPossibleLayouts,
    RankedTensorType tensorType, Type scalarElementType);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_TENSORLAYOUTS_H
