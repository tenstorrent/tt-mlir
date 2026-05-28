// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tt {
namespace ttnn {

// Search space for ttnn::Conv3dConfigAttr. Each non-empty SmallVector defines
// the candidate set for that field; an empty vector means the field is fixed
// to whatever the base config or default has.
//
struct Conv3dConfigSearchSpace {
  llvm::SmallVector<ttcore::DataType> weightsDtype;
  llvm::SmallVector<uint32_t> tOutBlock;
  llvm::SmallVector<uint32_t> wOutBlock;
  llvm::SmallVector<uint32_t> hOutBlock;
  llvm::SmallVector<uint32_t> cOutBlock;
  llvm::SmallVector<uint32_t> cInBlock;
  llvm::SmallVector<::mlir::tt::ttcore::GridAttr> computeWithStorageGridSize;

  Conv3dConfigSearchSpace() = default;

  bool isWeightsDtypeSetForSearch() const { return !weightsDtype.empty(); }
  bool isTOutBlockSetForSearch() const { return !tOutBlock.empty(); }
  bool isWOutBlockSetForSearch() const { return !wOutBlock.empty(); }
  bool isHOutBlockSetForSearch() const { return !hOutBlock.empty(); }
  bool isCOutBlockSetForSearch() const { return !cOutBlock.empty(); }
  bool isCInBlockSetForSearch() const { return !cInBlock.empty(); }
  bool isComputeWithStorageGridSizeSetForSearch() const {
    return !computeWithStorageGridSize.empty();
  }

  bool isAnyFieldSetForSearch() const {
    return isWeightsDtypeSetForSearch() || isTOutBlockSetForSearch() ||
           isWOutBlockSetForSearch() || isHOutBlockSetForSearch() ||
           isCOutBlockSetForSearch() || isCInBlockSetForSearch() ||
           isComputeWithStorageGridSizeSetForSearch();
  }
};

// Enumerates every Conv3dConfigAttr produced by taking the cartesian product
// of `space`'s candidate sets, layered over `baseConfig`: any field already
// set on `baseConfig` is treated as fixed and excluded from the search.
// `callback` fires once per candidate not rejected by `filterOut`.
//
// `filterOut` receives the produced attr and returns true to reject it. Use
// it for empirical legality rules — e.g. divisibility, h_out_block *
// w_out_block <= 256, minimum core utilization. Pass `{}` for no filter.
//
// Returns true if the search space had at least one field to enumerate over.
// When false, the callback never fires and the caller should typically fall
// back to `baseConfig`. When true, the callback may still fire zero times if
// `filterOut` rejected every candidate.
bool forEachConv3dConfig(
    Conv3dOp *op, Conv3dConfigAttr baseConfig,
    const Conv3dConfigSearchSpace &space,
    llvm::function_ref<bool(const Conv3dConfigAttr &)> filterOut,
    llvm::function_ref<void(Conv3dConfigAttr)> callback);

} // namespace ttnn
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H
