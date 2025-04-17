// SPDX-FileCopyrightText: : © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

std::vector<TTNNLayoutAttr> getShardedLayoutsForTensorTypeAndScalarType(
    const TensorTypeLayoutsMap &tensorPossibleLayouts,
    RankedTensorType tensorType, Type scalarElementType) {

  auto iter = tensorPossibleLayouts.find(tensorType);
  assert(iter != tensorPossibleLayouts.end() &&
         "Tensor type not found in possible layouts");

  auto scalarLayoutsIter = iter->second.find(scalarElementType);
  assert(scalarLayoutsIter != iter->second.end() &&
         "Scalar type not found in possible layouts");

  auto layoutsForScalarType = scalarLayoutsIter->second;

  std::vector<TTNNLayoutAttr> layouts;
  for (size_t dataLayoutIdx = 0;
       dataLayoutIdx < static_cast<size_t>(TensorDataLayoutIndex::kNumValues);
       ++dataLayoutIdx) {
    const std::vector<TTNNLayoutAttr> &shardedLayouts =
        layoutsForScalarType[dataLayoutIdx][static_cast<size_t>(
            TensorMemoryLayoutIndex::Sharded)];

    layouts.insert(layouts.end(), shardedLayouts.begin(), shardedLayouts.end());
  }

  return layouts;
}

} // namespace mlir::tt::ttnn
