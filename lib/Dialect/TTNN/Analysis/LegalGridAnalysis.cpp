// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalGridAnalysis.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn {

bool mock_is_output_tensor_legal_for_op(Operation *op, tt::LayoutAttr layout) {
  // Placeholder, needs to be replaced with a call the the TTNN op interface.
  return true;
}

bool tensor_shape_compatible_with_shard(Operation *op, tt::LayoutAttr layout) {
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

bool cantChangeOutputLayout(Operation *op) {
  // Check if OP belongs to TTNN dialect.
  //
  if (!isa<TTNNDialect>(op->getDialect())) {
    return true;
  }

  if (llvm::isa<EmptyOp>(op)) {
    return true;
  }

  if (llvm::isa<ToLayoutOp>(op)) {
    return true;
  }

  return false;
}

bool LegalGridAnalysis::applyOverrides() {
  // Lookup layout overrides based on location information for current
  // operation.
  //

  if (not analysisInput.outputLayoutOverrides) {
    return false;
  }

  if (not isa<NameLoc>(op->getLoc())) {
    return false;
  }

  StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
  auto gridOverride = analysisInput.outputLayoutOverrides->find(opLocName);

  if (gridOverride == analysisInput.outputLayoutOverrides->end()) {
    return false;
  }

  OutputLayoutOverrideParams override = gridOverride->getValue();
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  tt::LayoutAttr layout = mlir::cast<tt::LayoutAttr>(tensorType.getEncoding());

  analysisResult.push_back(
      layout.withMemorySpace(op->getContext(), override.memorySpace)
          .withMemoryLayout(op->getContext(), override.memoryLayout)
          .withGrid(op->getContext(), tensorType,
                    GridAttr::get(op->getContext(),
                                  ArrayRef<int64_t>(override.grid))));

  return true;
}

void LegalGridAnalysis::analysisImplementation() {
  // A first incomplete implementation of the LegalGridAnalysis.
  // This implementation is a placeholder and is meant to just enable testing of
  // other components.

  // Skip operations that don't have output tensors.
  if (op->getNumResults() == 0) {
    return;
  }

  // Get output tensor type.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  tt::LayoutAttr layout = mlir::cast<tt::LayoutAttr>(tensorType.getEncoding());

  // Return existing layout if it is not possible to change it.
  if (cantChangeOutputLayout(op)) {
    analysisResult.push_back(layout);
    return;
  }

  // DRAM
  // No grid is set since the tensor is not sharded.
  // TODO(odjuricic): We need to set grid here since it will be used as the
  // compute gird. (not implemented in runtime atm)
  tt::LayoutAttr dram =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceDRAM)
          .withMemoryLayout(op->getContext(),
                            tt::TensorMemoryLayout::Interleaved)
          .withGrid(op->getContext(), tensorType,
                    GridAttr::get(op->getContext(),
                                  analysisInput.maxGrid.getShape()));
  if (mock_is_output_tensor_legal_for_op(op, dram)) {
    analysisResult.push_back(dram);
  }

  // L1 Interleaved (same as above).
  tt::LayoutAttr l1Interleaved =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1)
          .withMemoryLayout(op->getContext(),
                            tt::TensorMemoryLayout::Interleaved)
          .withGrid(op->getContext(), tensorType,
                    GridAttr::get(op->getContext(),
                                  analysisInput.maxGrid.getShape()));
  if (mock_is_output_tensor_legal_for_op(op, l1Interleaved)) {
    analysisResult.push_back(l1Interleaved);
  }

  // L1 Sharded
  tt::LayoutAttr shardedBase =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1);
  std::vector<tt::LayoutAttr> shardedResults;

  // Block Sharded
  for (auto width = 1; width <= analysisInput.maxGrid.getShape()[0]; ++width) {
    for (auto height = 1; height <= analysisInput.maxGrid.getShape()[1];
         ++height) {
      shardedResults.push_back(
          shardedBase
              .withGrid(op->getContext(), tensorType,
                        GridAttr::get(op->getContext(), {width, height}))
              .withMemoryLayout(op->getContext(),
                                tt::TensorMemoryLayout::BlockSharded));
    }
  }

  auto numCores =
      analysisInput.maxGrid.getShape()[0] * analysisInput.maxGrid.getShape()[1];
  // Height Sharded
  // TODO(odjuricic): Missing affine mapping to actual grid. Need to check with
  // runtime implementation on what to produce here.
  for (auto height = 1; height <= numCores; ++height) {
    shardedResults.push_back(
        shardedBase
            .withGrid(op->getContext(), tensorType,
                      GridAttr::get(op->getContext(), {height, 1}))
            .withMemoryLayout(op->getContext(),
                              tt::TensorMemoryLayout::HeightSharded));
  }

  // Width Sharded
  for (auto width = 1; width <= numCores; ++width) {
    shardedResults.push_back(
        shardedBase
            .withGrid(op->getContext(), tensorType,
                      GridAttr::get(op->getContext(), {1, width}))
            .withMemoryLayout(op->getContext(),
                              tt::TensorMemoryLayout::WidthSharded));
  }

  // Filter layouts based on output tensor legality for current op.
  shardedResults.erase(
      std::remove_if(shardedResults.begin(), shardedResults.end(),
                     [this](tt::LayoutAttr layout) {
                       return !tensor_shape_compatible_with_shard(op, layout) ||
                              !mock_is_output_tensor_legal_for_op(op, layout);
                     }),
      shardedResults.end());

  // Pick top largest sharded grids.
  std::sort(shardedResults.begin(), shardedResults.end(),
            [](tt::LayoutAttr a, tt::LayoutAttr b) {
              return a.getGrid().getGridVolume() > b.getGrid().getGridVolume();
            });

  analysisResult.insert(
      analysisResult.end(), shardedResults.begin(),
      shardedResults.begin() +
          std::min(analysisInput.maxShardedGrids,
                   static_cast<int64_t>(shardedResults.size())));
}
} // namespace mlir::tt::ttnn
