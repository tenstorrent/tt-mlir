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

} // namespace mlir::tt::ttnn::workarounds::decomposition
