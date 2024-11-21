// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalGridAnalysis.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include <cassert>
#include <cstdint>
#include <mlir/IR/Types.h>
#include <mlir/IR/Visitors.h>
#include <optional>

namespace mlir::tt::ttnn {

bool mock_is_output_tensor_legal_for_op(Operation *op, TTNNLayoutAttr layout) {
  // Placeholder, needs to be replaced with a call the the TTNN op interface.
  return true;
}

bool tensor_shape_compatible_with_shard(Operation *op, TTNNLayoutAttr layout) {
  // These constraints are implemented seperatelly in every TTNN op.
  // Almost nothing seems to be shared between EVERY op, so is hard to have any
  // logic here without the risk of discarding a valid configuraiton or modeling
  // the constraint for each op.

  // For now just check if we have enough tiles to shard the tensor to the
  // desired grid.

  if (not layout.hasShardedTensorMemoryLayout()) {
    return true;
  }

  if (layout.isTiled()) {
    RankedTensorType tensorType =
        mlir::cast<RankedTensorType>(op->getResult(0).getType());
    llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();
    auto tiledShape = layout.getTiledShape(tensorShape);
    auto gridShape = layout.getGrid().getShape();

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
  } else {
    // TODO(odjuricic): For row major there are no constraints on how the tensor
    // can be sharded. We need some kind of a heuristic to reduce the search
    // space.
    return true;
  }
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
  auto gridOverride = analysisInput.outputLayoutOverrides->find(opLocName);

  if (gridOverride == analysisInput.outputLayoutOverrides->end()) {
    return false;
  }

  OutputLayoutOverrideParams override = gridOverride->getValue();
  if (not override.fullLayoutOverride()) {
    // We cannot skip analysis.
    return false;
  }

  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  TTNNLayoutAttr layout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  GridAttr grid =
      GridAttr::get(op->getContext(), ArrayRef<int64_t>(override.grid.value()));

  // Create element type for the new layout.
  Type elementType = layout.getScalarElementType();
  if (override.dataType.has_value()) {
    elementType = {utils::createRowMajorTypeFromDtype(
        op->getContext(), override.dataType.value())};
  }

  if (override.memoryLayout == Layout::Tile) {
    elementType = TileType::get(op->getContext(), elementType);
  }

  // TODO rewrite like below.
  analysisResult.push_back(
      layout.withGrid(op->getContext(), tensorType, grid)
          .withBufferType(op->getContext(), override.bufferType.value())
          .withMemoryLayout(op->getContext(),
                            override.tensorMemoryLayout.value())
          .withElementType(op->getContext(), elementType));

  return true;
}

void LegalLayoutAnalysis::analysisImplementation() {
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
  auto tensorShape = tensorType.getShape();
  TTNNLayoutAttr layout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  // Return existing layout if it is not possible to change it.
  if (cantChangeOutputLayout(op)) {
    analysisResult.push_back(layout);
    return;
  }

  Type scalarElementType = layout.getScalarElementType();

  std::optional<OutputLayoutOverrideParams> override;

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
  if (override.has_value() and override->memoryLayout.has_value() and
      override->memoryLayout.value() == Layout::RowMajor) {
    // Force allow row major if override is set.
    rowMajorAllowed = true;
  }

  // Generate both TILE and ROW_MAJOR layouts.
  for (Type elementType : {scalarElementType, tileElementType}) {
    if (not rowMajorAllowed and elementType == scalarElementType) {
      continue;
    }
    // DRAM
    analysisResult.push_back(TTNNLayoutAttr::get(
        op->getContext(), tensorShape, elementType, BufferType::DRAM,
        analysisInput.maxGrid, TensorMemoryLayout::Interleaved));

    // L1 Interleaved (same as above).
    analysisResult.push_back(TTNNLayoutAttr::get(
        op->getContext(), tensorShape, elementType, BufferType::L1,
        analysisInput.maxGrid, TensorMemoryLayout::Interleaved));

    // L1 Sharded
    TTNNLayoutAttr shardedBase =
        layout.withBufferType(op->getContext(), BufferType::L1)
            .withElementType(op->getContext(), elementType);

    assert(analysisInput.maxGrid.getShape().size() == 2 &&
           "Max device grid is expected to be 2D.");
    // Block Sharded
    for (auto width = 1; width <= analysisInput.maxGrid.getShape()[0];
         ++width) {
      for (auto height = 1; height <= analysisInput.maxGrid.getShape()[1];
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
    for (auto height = 1; height <= numCores; ++height) {
      shardedResults.push_back(
          shardedBase
              .withGrid(op->getContext(), tensorType,
                        GridAttr::get(op->getContext(), {height, 1}))
              .withMemoryLayout(op->getContext(),
                                TensorMemoryLayout::HeightSharded));
    }

    // Width Sharded
    for (auto width = 1; width <= numCores; ++width) {
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
                       return !tensor_shape_compatible_with_shard(op, layout) ||
                              !mock_is_output_tensor_legal_for_op(op, layout);
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

  // Apply partial layout overrides.
  if (override.has_value()) {
    analysisResult.erase(
        std::remove_if(analysisResult.begin(), analysisResult.end(),
                       [override](TTNNLayoutAttr layout) {
                         bool keepLayout = true;
                         if (override->grid.has_value()) {
                           keepLayout &= layout.getGrid().getShape()[0] ==
                                             override->grid.value()[0] &&
                                         layout.getGrid().getShape()[1] ==
                                             override->grid.value()[1];
                         }
                         if (override->bufferType.has_value()) {
                           keepLayout &= layout.getBufferType() ==
                                         override->bufferType.value();
                         }
                         if (override->tensorMemoryLayout.has_value()) {
                           keepLayout &= layout.getMemLayout() ==
                                         override->tensorMemoryLayout.value();
                         }
                         if (override->memoryLayout.has_value()) {
                           keepLayout &=
                               layout.isTiled() ==
                               (override->memoryLayout.value() == Layout::Tile);
                         }
                         return !keepLayout;
                       }),
        analysisResult.end());
  }

  if (analysisResult.empty()) {
    op->emitError("No legal layout found for the operation.");
    assert(false && "At least one legal layout must be found.");
  }
}
} // namespace mlir::tt::ttnn
