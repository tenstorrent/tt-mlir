// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceOpsRewritePattern.h"

#include <algorithm>
#include <unordered_set>

namespace mlir::tt::ttnn::workarounds::decomposition {

llvm::SmallVector<int64_t>
getReduceDims(const std::optional<mlir::ArrayAttr> &dimArg) {
  llvm::SmallVector<int64_t> reduceDims;
  if (!dimArg.has_value()) {
    return reduceDims;
  }

  for (const mlir::Attribute &reduceDim : dimArg.value()) {
    reduceDims.push_back(mlir::cast<mlir::IntegerAttr>(reduceDim).getInt());
  }

  return reduceDims;
}

std::vector<int64_t>
calculateNewReduceShape(const RankedTensorType &inputType,
                        const std::optional<mlir::ArrayAttr> &dimArg) {
  std::vector<int64_t> outputShapeVec = inputType.getShape().vec();
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

mlir::ArrayAttr
createNewReduceDimArg(const RankedTensorType &inputType,
                      const std::optional<mlir::ArrayAttr> &dimArg) {
  llvm::SmallVector<int64_t> reduceDims = getReduceDims(dimArg);
  if (reduceDims.empty()) {
    return nullptr;
  }

  std::unordered_set<int64_t> uniqueReduceDims(reduceDims.begin(),
                                               reduceDims.end());
  if (uniqueReduceDims.size() == inputType.getShape().size()) {
    // In case when reduce is done over all dimensions of the input nullptr is
    // returned, because Metal supports reduce over all dimensions for any
    // tensor rank when reduce dimensions are not specified, but it doesn't
    // support reduce for tensors with rank larger than 2 when reduce
    // dimensions are specified.
    return nullptr;
  }

  return dimArg.value();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
