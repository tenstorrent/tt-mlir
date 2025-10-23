// SPDX-FileCopyrightText: : © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

std::vector<TTNNLayoutAttr> getShardedLayoutsForTensorTypeAndScalarType(
    const TensorTypeLayoutsMap &tensorPossibleLayouts,
    RankedTensorType tensorType, Type scalarElementType) {

  std::vector<TTNNLayoutAttr> layouts;
  for (size_t pageLayoutIdx = 0;
       pageLayoutIdx < static_cast<size_t>(TensorPageLayout::kNumValues);
       ++pageLayoutIdx) {
    auto shardedLayoutsForPageLayout =
        getShardedLayoutsForTensorTypeAndScalarType(
            tensorPossibleLayouts, tensorType, scalarElementType,
            pageLayoutIdx);
    layouts.insert(layouts.end(), shardedLayoutsForPageLayout.begin(),
                   shardedLayoutsForPageLayout.end());
  }

  return layouts;
}

std::vector<TTNNLayoutAttr> getShardedLayoutsForTensorTypeAndScalarType(
    const TensorTypeLayoutsMap &tensorPossibleLayouts,
    RankedTensorType tensorType, Type scalarElementType, size_t pageLayoutIdx) {

  auto iter = tensorPossibleLayouts.find(tensorType);
  assert(iter != tensorPossibleLayouts.end() &&
         "Tensor type not found in possible layouts");

  auto scalarLayoutsIter = iter->second.find(scalarElementType);
  assert(scalarLayoutsIter != iter->second.end() &&
         "Scalar type not found in possible layouts");

  auto layoutsForScalarType = scalarLayoutsIter->second;

  const std::vector<TTNNLayoutAttr> &shardedLayouts =
      layoutsForScalarType[pageLayoutIdx][static_cast<size_t>(
          TensorMemoryLayoutIndex::Sharded)];

  return shardedLayouts;
}

} // namespace mlir::tt::ttnn
