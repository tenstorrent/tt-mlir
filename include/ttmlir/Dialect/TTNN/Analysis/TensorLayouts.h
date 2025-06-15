// SPDX-FileCopyrightText: : © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_TENSORLAYOUTS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_TENSORLAYOUTS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLForwardCompat.h"

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
        val.getElementType());
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
           lhs.getElementType() == rhs.getElementType();
  }
};
} // namespace llvm

namespace mlir::tt::ttnn {

enum class TensorMemoryLayoutIndex : std::size_t {
  Interleaved = 0,
  Sharded = 1,
  kNumValues = 2
};

enum class TensorPageLayout : std::size_t {
  RowMajor = 0,
  Tiled = 1,
  kNumValues = 2
};

// Tensor Type layouts categorized by memory layout (sharded/interleaved).
// For each memory layout, we have a vector of TTNNLayoutAttr.
using TensorTypeLayoutsForMemoryLayout =
    std::array<std::vector<TTNNLayoutAttr>,
               llvm::to_underlying(TensorMemoryLayoutIndex::kNumValues)>;

// Tensor type layouts categorized by page layout (tile/RM). For each page
// layout, we have a TensorTypeLayoutsForMemoryLayout.
using TensorTypeLayoutsForPageLayout =
    std::array<TensorTypeLayoutsForMemoryLayout,
               llvm::to_underlying(TensorPageLayout::kNumValues)>;

// Tensor type layouts categorized by scalar type. Key into the map is the
// scalar type (e.g. f32, bf16, etc.). Value is a
// TensorTypeLayoutsForPageLayout.
using TensorTypeLayoutsForScalarType =
    llvm::DenseMap<Type, TensorTypeLayoutsForPageLayout>;

// TensorType layouts categorized by scalar type, page layout (tile/RM) and
// memory layout (sharded/interleaved) combined.
// Key into the map is the tensor type (e.g. tensor<2x2xf32>). Value is a
// TensorTypeLayoutsForScalarType which contains the layouts for each scalar
// type, which in turn contains the layouts for each page layout (tile/RM) and
// memory layout combination.
using TensorTypeLayoutsMap =
    llvm::DenseMap<RankedTensorType, TensorTypeLayoutsForScalarType>;

// Helper function to convert Layout to TensorPageLayout as std::size_t
inline std::size_t getPageLayoutIndex(Layout layout) {
  assert((layout == Layout::Tile || layout == Layout::RowMajor) &&
         "Invalid layout type");
  return static_cast<std::size_t>(layout == Layout::Tile
                                      ? TensorPageLayout::Tiled
                                      : TensorPageLayout::RowMajor);
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

inline std::vector<TTNNLayoutAttr> getShardedLayoutsForPageLayout(
    std::size_t pageLayoutIdx,
    const TensorTypeLayoutsForPageLayout &tensorPossibleLayouts) {
  return tensorPossibleLayouts[pageLayoutIdx][static_cast<std::size_t>(
      TensorMemoryLayoutIndex::Sharded)];
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_TENSORLAYOUTS_H
