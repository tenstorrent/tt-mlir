// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalLayoutAnalysis.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn {

bool mockIsOutputTensorLegalForOp(Operation *op, TTNNLayoutAttr layout) {
  // Placeholder, needs to be replaced with a call the the TTNN op interface.
  return true;
}

bool tensorShapeCompatibleWithShard(Operation *op, TTNNLayoutAttr layout) {
  // These constraints are implemented seperatelly in every TTNN op.
  // Almost nothing seems to be shared between EVERY op, so is hard to have any
  // logic here without the risk of discarding a valid configuraiton or modeling
  // the constraint for each op.

  // For now just check if we have enough tiles to shard the tensor to the
  // desired grid. This is a safe heuristic that should be valid for all ops.

  if (not layout.hasShardedTensorMemoryLayout()) {
    return true;
  }

  if (layout.isTiled()) {
    RankedTensorType tensorType =
        mlir::cast<RankedTensorType>(op->getResult(0).getType());
    llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();
    llvm::SmallVector<int64_t, 2> tiledShape =
        layout.getTiledShape(tensorShape);
    llvm::ArrayRef<int64_t> gridShape = layout.getGrid().getShape();

    assert(tiledShape.size() == gridShape.size() &&
           "Tiled tensor shape and grid shape must have the same rank");

    for (size_t i = 0; i < tiledShape.size(); i++) {
      // We need to have at least as many tiles as the grid size.
      // Could also experiment with tiledShape[i] % gridShape[i] == 0, but need
      // more context.
      if (tiledShape[i] < gridShape[i]) {
        return false;
      }
    }
    return true;
  }

  // TODO(odjuricic): For row major there are no constraints on how the tensor
  // can be sharded. We need some kind of a heuristic to reduce the search
  // space.
  return true;
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

bool LegalLayoutAnalysis::applyOverrides() {
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
  auto overrideIt = analysisInput.outputLayoutOverrides->find(opLocName);

  if (overrideIt == analysisInput.outputLayoutOverrides->end()) {
    return false;
  }

  OutputLayoutOverrideParams layoutOverride = overrideIt->getValue();

  // If all layout parameters are set (except data type), we can skip analysis
  // and create the overriden layout. Otherwise, we need to perform analysis and
  // apply partial overrides.
  if (not layoutOverride.fullLayoutOverride()) {
    return false;
  }

  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  TTNNLayoutAttr layout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  GridAttr grid = GridAttr::get(op->getContext(),
                                ArrayRef<int64_t>(layoutOverride.grid.value()));

  // Create element type for the new layout.
  Type elementType = layout.getScalarElementType();
  if (layoutOverride.dataType.has_value()) {
    elementType = utils::createRowMajorTypeFromDtype(
        op->getContext(), layoutOverride.dataType.value());
  }

  if (layoutOverride.memoryLayout == Layout::Tile) {
    elementType = TileType::get(op->getContext(), elementType);
  }

  analysisResult.push_back(TTNNLayoutAttr::get(
      op->getContext(), tensorShape, elementType,
      layoutOverride.bufferType.value(), grid,
      TensorMemoryLayoutAttr::get(op->getContext(),
                                  layoutOverride.tensorMemoryLayout.value())));

  return true;
}

bool incompatibleWithOverride(
    const TTNNLayoutAttr &layout,
    const std::optional<OutputLayoutOverrideParams> &layoutOverride) {
  if (not layoutOverride.has_value()) {
    return false;
  }

  if (layoutOverride->grid.has_value()) {
    if (layout.getGrid().getShape()[0] != layoutOverride->grid.value()[0] ||
        layout.getGrid().getShape()[1] != layoutOverride->grid.value()[1]) {
      return true;
    }
  }
  if (layoutOverride->bufferType.has_value() &&
      layout.getBufferType() != layoutOverride->bufferType.value()) {
    return true;
  }
  if (layoutOverride->tensorMemoryLayout.has_value() &&
      layout.getMemLayout().getValue() !=
          layoutOverride->tensorMemoryLayout.value()) {
    return true;
  }
  if (layoutOverride->memoryLayout.has_value() &&
      layout.isTiled() !=
          (layoutOverride->memoryLayout.value() == Layout::Tile)) {
    return true;
  }
  return false;
}

void LegalLayoutAnalysis::analysisImplementation() {
  // Skip operations that don't have output tensors.
  if (op->getNumResults() == 0) {
    return;
  }

  if (!isa<RankedTensorType>(op->getResult(0).getType())) {
    return;
  }

  if (llvm::isa<ttnn::EmptyOp>(op)) {
    return;
  }

  // Get output tensor type.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();
  TTNNLayoutAttr layout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  // Return existing layout if it is not possible to change it.
  if (cantChangeOutputLayout(op)) {
    analysisResult.push_back(layout);
    return;
  }

  Type scalarElementType = layout.getScalarElementType();

  std::optional<OutputLayoutOverrideParams> override;

  // Check if we have an override for this op.
  if (isa<NameLoc>(op->getLoc())) {
    StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
    if (auto overrideIt = analysisInput.outputLayoutOverrides->find(opLocName);
        overrideIt != analysisInput.outputLayoutOverrides->end()) {
      override = overrideIt->getValue();
      if (override->dataType.has_value()) {
        scalarElementType = {utils::createRowMajorTypeFromDtype(
            op->getContext(), override->dataType.value())};
      }
    }
  }

  Type tileElementType = TileType::get(op->getContext(), scalarElementType);
  std::vector<TTNNLayoutAttr> shardedResults;

  bool rowMajorAllowed = analysisInput.rowMajorEnabled;
  if (override.has_value() && override->memoryLayout.has_value() &&
      override->memoryLayout.value() == Layout::RowMajor) {
    // Force allow row major if override is set.
    rowMajorAllowed = true;
  }

  // Generate both TILE and ROW_MAJOR layouts.
  for (Type elementType : {scalarElementType, tileElementType}) {
    if (not rowMajorAllowed && elementType == scalarElementType) {
      continue;
    }
    // DRAM
    analysisResult.push_back(TTNNLayoutAttr::get(
        op->getContext(), tensorShape, elementType, BufferType::DRAM,
        analysisInput.maxGrid,
        TensorMemoryLayoutAttr::get(op->getContext(),
                                    TensorMemoryLayout::Interleaved)));

    // L1 Interleaved (same as above).
    analysisResult.push_back(TTNNLayoutAttr::get(
        op->getContext(), tensorShape, elementType, BufferType::L1,
        analysisInput.maxGrid,
        TensorMemoryLayoutAttr::get(op->getContext(),
                                    TensorMemoryLayout::Interleaved)));

    // L1 Sharded
    TTNNLayoutAttr shardedBase =
        layout.withBufferType(op->getContext(), BufferType::L1)
            .withElementType(op->getContext(), elementType);

    assert(analysisInput.maxGrid.getShape().size() == 2 &&
           "Max device grid is expected to be 2D.");
    // Block Sharded
    for (int width = 1; width <= analysisInput.maxGrid.getShape()[0]; ++width) {
      for (int height = 1; height <= analysisInput.maxGrid.getShape()[1];
           ++height) {
        shardedResults.push_back(
            shardedBase
                .withGrid(op->getContext(), tensorType,
                          GridAttr::get(op->getContext(), {width, height}))
                .withMemoryLayout(op->getContext(),
                                  TensorMemoryLayout::BlockSharded));
      }
    }

    int64_t numCores = analysisInput.maxGrid.getGridVolume();
    // Height Sharded
    // TODO(odjuricic): Missing affine mapping to actual grid. Need to check
    // with runtime implementation on what to produce here.
    for (int height = 1; height <= numCores; ++height) {
      shardedResults.push_back(
          shardedBase
              .withGrid(op->getContext(), tensorType,
                        GridAttr::get(op->getContext(), {height, 1}))
              .withMemoryLayout(op->getContext(),
                                TensorMemoryLayout::HeightSharded));
    }

    // Width Sharded
    for (int width = 1; width <= numCores; ++width) {
      shardedResults.push_back(
          shardedBase
              .withGrid(op->getContext(), tensorType,
                        GridAttr::get(op->getContext(), {1, width}))
              .withMemoryLayout(op->getContext(),
                                TensorMemoryLayout::WidthSharded));
    }
  }

  // Filter layouts based on output tensor legality for current op.
  shardedResults.erase(
      std::remove_if(shardedResults.begin(), shardedResults.end(),
                     [this](TTNNLayoutAttr layout) {
                       return !tensorShapeCompatibleWithShard(op, layout) ||
                              !mockIsOutputTensorLegalForOp(op, layout);
                     }),
      shardedResults.end());

  // Pick top largest sharded grids.
  // This becomes a problem when we introduce row_major since an 8x8 tensor can
  // be sharded onto a 8x8 grid.
  std::sort(shardedResults.begin(), shardedResults.end(),
            [](TTNNLayoutAttr a, TTNNLayoutAttr b) {
              return a.getGrid().getGridVolume() > b.getGrid().getGridVolume();
            });

  analysisResult.insert(
      analysisResult.end(), shardedResults.begin(),
      shardedResults.begin() +
          std::min(analysisInput.maxShardedGrids,
                   static_cast<int64_t>(shardedResults.size())));

  // Apply partial layout overrides. Remove layouts that conflict with at least
  // one overriden param.
  if (override.has_value()) {
    auto shouldRemoveLayout =
        std::bind(incompatibleWithOverride, std::placeholders::_1, override);
    analysisResult.erase(std::remove_if(analysisResult.begin(),
                                        analysisResult.end(),
                                        shouldRemoveLayout),
                         analysisResult.end());
  }

  if (analysisResult.empty()) {
    op->emitError("No legal layout found for the operation.");
    assert(false && "At least one legal layout must be found.");
  }
}
} // namespace mlir::tt::ttnn
