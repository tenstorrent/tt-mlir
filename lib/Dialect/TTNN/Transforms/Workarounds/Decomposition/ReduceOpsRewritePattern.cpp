// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceOpsRewritePattern.h"

#include <algorithm>
#include <unordered_set>

namespace mlir::tt::ttnn::workarounds::decomposition {

std::vector<int64_t>
getReduceDims(const std::optional<mlir::ArrayAttr> &dimArg) {
  std::vector<int64_t> reduceDims;
  if (!dimArg.has_value()) {
    return reduceDims;
  }

  for (const mlir::Attribute &reduceDim : dimArg.value()) {
    reduceDims.push_back(mlir::cast<mlir::IntegerAttr>(reduceDim).getInt());
  }

  return reduceDims;
}

std::vector<int64_t>
calculateNewReduceShape(const std::optional<mlir::ArrayAttr> &dimArg,
                        const RankedTensorType &inputType) {
  std::vector<int64_t> outputShapeVec = inputType.getShape().vec();
  std::vector<int64_t> reduceDims = getReduceDims(dimArg);

  if (reduceDims.empty()) {
    std::fill(outputShapeVec.begin(), outputShapeVec.end(), 1);
  } else {
    for (const int64_t reduceDim : reduceDims) {
      outputShapeVec[reduceDim < 0 ? outputShapeVec.size() -
                                         static_cast<size_t>(-reduceDim)
                                   : static_cast<size_t>(reduceDim)] = 1;
    }
  }

  return outputShapeVec;
}

mlir::ArrayAttr
calculateNewReduceDimArg(const RankedTensorType &inputType,
                         const std::optional<mlir::ArrayAttr> &dimArg) {
  std::vector<int64_t> reduceDims = getReduceDims(dimArg);
  if (reduceDims.empty()) {
    return nullptr;
  }

  std::unordered_set<int64_t> uniqueReduceDims(reduceDims.begin(),
                                               reduceDims.end());
  if (uniqueReduceDims.size() == inputType.getShape().size()) {
    return nullptr;
  }

  return dimArg.value();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
