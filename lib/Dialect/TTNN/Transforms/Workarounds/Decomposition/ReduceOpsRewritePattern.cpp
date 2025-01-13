// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceOpsRewritePattern.h"

#include <algorithm>

namespace mlir::tt::ttnn::workarounds::decomposition {

llvm::SmallVector<int64_t>
getReduceDims(const std::optional<mlir::ArrayAttr> &dimArg) {
  llvm::SmallVector<int64_t, 4> reduceDims;
  if (!dimArg) {
    return reduceDims;
  }

  for (const mlir::Attribute &reduceDim : *dimArg) {
    reduceDims.push_back(mlir::cast<mlir::IntegerAttr>(reduceDim).getInt());
  }

  return reduceDims;
}

llvm::SmallVector<int64_t>
calculateNewReduceShape(RankedTensorType inputType,
                        const std::optional<mlir::ArrayAttr> &dimArg) {
  llvm::SmallVector<int64_t> outputShapeVec(inputType.getShape());
  llvm::SmallVector<int64_t> reduceDims = getReduceDims(dimArg);

  if (reduceDims.empty()) {
    // When reduce dimensions are not specified that means we are reducing over
    // all dimensions, so all dimensions of the output shape become 1.
    std::fill(outputShapeVec.begin(), outputShapeVec.end(), 1);
  } else {
    // Dimensions can be specified as negative numbers, so to calculate the
    // index in the output shape vector we need to sum them with the output
    // shape rank.
    int64_t outputShapeRank = static_cast<int64_t>(outputShapeVec.size());
    for (const int64_t reduceDim : reduceDims) {
      int64_t outputShapeIndex =
          reduceDim < 0 ? outputShapeRank + reduceDim : reduceDim;
      outputShapeVec[static_cast<size_t>(outputShapeIndex)] = 1;
    }
  }

  return outputShapeVec;
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
