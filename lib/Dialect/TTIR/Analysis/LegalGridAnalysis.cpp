// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/LegalGridAnalysis.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir::tt::ttir {

bool mock_is_output_tensor_legal_for_op(Operation *op, LayoutAttr layout) {
  // Placeholder, needs to be replaced with a call the the TTNN op interface.
  return true;
}

bool tensor_shape_compatible_with_shard(Operation *op, LayoutAttr layout) {
  // These constraints are implemented seperatelly in every TTNN op.
  // Almost nothing seems to be shared between EVERY op, so is hard to have any
  // logic here without the risk of discarding a valid configuraiton or modeling
  // the constraint for each op. This logic may be offloaded to the TTNN op
  // interface.

  // For now we will check if the tilised tensor dims are divisible by the grid
  // dims. This will definitly discard possible valid configurations, but is a
  // start.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  int64_t MTiles = 1;
  if (tensorType.getRank() >= 2) {
    MTiles = (tensorShape.rbegin()[1] + 31) / 32;
  }

  int64_t KTIles = (tensorShape.back() + 31) / 32;

  int64_t gridR = layout.getGrid().getShape()[0];
  int64_t gridC = layout.getGrid().getShape()[1];

  return (MTiles % gridR == 0) && (KTIles % gridC == 0);
}

bool LegalGridAnalysis::applyOverrides() {
  // Lookup grid size overrides based on location information for current
  // operation.
  //

  // TODO(odjuricic): Need to override all params, not just grid size.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  LayoutAttr layout = mlir::cast<LayoutAttr>(tensorType.getEncoding());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  if (analysisInput.gridSizeOverrides && isa<NameLoc>(op->getLoc())) {
    StringRef loc_str_op_name = mlir::cast<NameLoc>(op->getLoc()).getName();
    auto gridOverride = analysisInput.gridSizeOverrides->find(loc_str_op_name);
    if (gridOverride != analysisInput.gridSizeOverrides->end()) {
      analysisResult.push_back(layout.withGrid(
          op->getContext(), tensorShape,
          GridAttr::get(op->getContext(),
                        ArrayRef<int64_t>(gridOverride->second))));
      analysisResult.push_back(layout.withGrid(
          op->getContext(), tensorShape,
          GridAttr::get(op->getContext(),
                        {gridOverride->second[0], gridOverride->second[1]})));
      return true;
    }
  }

  return false;
}

void LegalGridAnalysis::analysisImplementation() {
  // A first incomplete implementation of the LegalGridAnalysis.
  // This implementation is a placeholder and is meant to just enable testing of
  // other components.

  // Process only TTIR ops.
  if (not llvm::isa<TTIROp>(op)) {
    return;
  }
  // Skip operations that don't have output tensors.
  if (op->getNumResults() == 0) {
    return;
  }
  if (llvm::isa<ToLayoutOp>(op)) {
    return;
  }

  // Get output tensor type.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  LayoutAttr layout = mlir::cast<LayoutAttr>(tensorType.getEncoding());

  // L1 Interleaved (same as above).
  LayoutAttr l1Interleaved =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1);
  if (mock_is_output_tensor_legal_for_op(op, l1Interleaved)) {
    analysisResult.push_back(l1Interleaved);
  }

  // DRAM
  // No grid is set since the tensor is not sharded.
  // TODO(odjuricic): We need to set grid here since it will be used as the
  // compute gird. (not implemented in runtime atm)
  LayoutAttr dram =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceDRAM);
  if (mock_is_output_tensor_legal_for_op(op, dram)) {
    analysisResult.push_back(dram);
  }

  // L1 Sharded
  LayoutAttr shardedBase =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1);
  std::vector<LayoutAttr> shardedResults;

  // Block Sharded
  for (auto width = 2; width <= analysisInput.maxGrid.getShape()[0]; ++width) {
    for (auto height = 2; height <= analysisInput.maxGrid.getShape()[1];
         ++height) {
      shardedResults.push_back(shardedBase.withGrid(
          op->getContext(), tensorType,
          GridAttr::get(op->getContext(), {width, height})));
    }
  }

  auto numCores =
      analysisInput.maxGrid.getShape()[0] * analysisInput.maxGrid.getShape()[1];
  // Height Sharded
  // TODO(odjuricic): Missing affine mapping to actual grid. Need to check with
  // runtime implementation on what to produce here.
  for (auto height = 2; height <= numCores; ++height) {
    shardedResults.push_back(
        shardedBase.withGrid(op->getContext(), tensorType,
                             GridAttr::get(op->getContext(), {height, 1})));
  }

  // Width Sharded
  for (auto width = 2; width <= numCores; ++width) {
    shardedResults.push_back(
        shardedBase.withGrid(op->getContext(), tensorType,
                             GridAttr::get(op->getContext(), {1, width})));
  }

  // Filter layouts based on output tensor legality for current op.
  shardedResults.erase(
      std::remove_if(shardedResults.begin(), shardedResults.end(),
                     [this](LayoutAttr layout) {
                       return !tensor_shape_compatible_with_shard(op, layout) ||
                              !mock_is_output_tensor_legal_for_op(op, layout);
                     }),
      shardedResults.end());

  // Pick top largest sharded grids.
  std::sort(shardedResults.begin(), shardedResults.end(),
            [](LayoutAttr a, LayoutAttr b) {
              return a.getGrid().getShape()[0] * a.getGrid().getShape()[1] >
                     b.getGrid().getShape()[0] * b.getGrid().getShape()[1];
            });

  analysisResult.insert(
      analysisResult.end(), shardedResults.begin(),
      shardedResults.begin() +
          std::min(analysisInput.maxShardedGrids,
                   static_cast<int64_t>(shardedResults.size())));
}
} // namespace mlir::tt::ttir
