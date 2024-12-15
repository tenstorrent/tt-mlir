// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceOpsRewritePattern.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

std::vector<int64_t>
calculateNewReduceShape(const std::optional<mlir::ArrayAttr> &dimArg,
                        const RankedTensorType &inputType) {
  std::vector<int64_t> outputShapeVec = inputType.getShape().vec();

  if (dimArg.has_value()) {
    for (const mlir::Attribute &reduceDim : dimArg.value()) {
      int64_t reduceDimInt = mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
      outputShapeVec[reduceDimInt < 0 ? outputShapeVec.size() -
                                            static_cast<size_t>(-reduceDimInt)
                                      : static_cast<size_t>(reduceDimInt)] = 1;
    }
  } else {
    for (size_t i = 0; i < outputShapeVec.size(); ++i) {
      outputShapeVec[i] = 1;
    }
  }

  return outputShapeVec;
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
