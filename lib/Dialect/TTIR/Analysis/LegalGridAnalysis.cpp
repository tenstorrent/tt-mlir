// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/LegalGridAnalysis.h"

namespace mlir::tt::ttir {

bool mock_is_output_tensor_legal_for_op(Operation *op, LayoutAttr layout) {
  // Placeholder, needs to be replaced with a call the the TTNN op interface.
  return true;
}

bool LegalGridAnalysis::applyOverrides() {
  // Lookup grid size overrides based on location information for current
  // operation.
  //

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

  // Get output tensor type.
  // TODO: This ignores multiple outputs...?
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  LayoutAttr layout = mlir::cast<LayoutAttr>(tensorType.getEncoding());

  // DRAM
  // No grid is set since the tensor is not sharded.
  // TODO: Is this a viable solution or should we have a grid?
  LayoutAttr dram =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceDRAM);
  analysisResult.push_back(dram);

  // L1 Interleaved (same as above)
  LayoutAttr l1Interleaved =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1);
  analysisResult.push_back(l1Interleaved);

  // L1 Sharded
  LayoutAttr shardedBase =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1);

  // Block Sharded
  for (auto width = 2; width <= analysisInput.maxGrid.getShape()[0]; ++width) {
    for (auto height = 2; height <= analysisInput.maxGrid.getShape()[1];
         ++height) {
      analysisResult.push_back(shardedBase.withGrid(
          op->getContext(), tensorType,
          GridAttr::get(op->getContext(), {width, height})));
    }
  }

  auto numCores =
      analysisInput.maxGrid.getShape()[0] * analysisInput.maxGrid.getShape()[1];
  // Height Sharded
  // TODO: Missing affine mapping to actual grid.
  // TODO: Can we have every shape of 1d grid? Probably not, need to check what
  // is divisible by grid sides.
  // TODO: Limit the number of options to some reasonable number.
  // TODO: Put all of this into the same loop.
  for (auto height = 2; height <= numCores; ++height) {
    analysisResult.push_back(
        shardedBase.withGrid(op->getContext(), tensorType,
                             GridAttr::get(op->getContext(), {height, 1})));
  }

  // Width Sharded
  for (auto width = 2; width <= numCores; ++width) {
    analysisResult.push_back(
        shardedBase.withGrid(op->getContext(), tensorType,
                             GridAttr::get(op->getContext(), {1, width})));
  }

  // Filter layouts based on output tensor legality for current op.
  analysisResult.erase(
      std::remove_if(analysisResult.begin(), analysisResult.end(),
                     [this](LayoutAttr layout) {
                       return !mock_is_output_tensor_legal_for_op(op, layout);
                     }),
      analysisResult.end());

  // TODO: Potetialy filter out tensors that dont fit into L1 at all.
}
} // namespace mlir::tt::ttir
