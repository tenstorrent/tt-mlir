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
#include <mlir/IR/Types.h>
#include <optional>

namespace mlir::tt::ttnn {

bool mock_is_output_tensor_legal_for_op(Operation *op, tt::LayoutAttr layout) {
  // Placeholder, needs to be replaced with a call the the TTNN op interface.
  return true;
}

bool tensor_shape_compatible_with_shard(Operation *op, tt::LayoutAttr layout) {

  if (not layout.isTiled()) {
    // TODO: What do we do with ROW_MAJOR :)?
    return false;
  }

  // These constraints are implemented seperatelly in every TTNN op.
  // Almost nothing seems to be shared between EVERY op, so is hard to have any
  // logic here without the risk of discarding a valid configuraiton or modeling
  // the constraint for each op. This logic may be offloaded to the TTNN op
  // interface.

  // For now we will check if the tilised tensor dims are divisible by the grid
  // dims. This will definitly discard possible valid configurations, but is a
  // start.

  // TODO rewrite
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
  if (not override.allParamsSet()) {
    // We cannot skip analysis.
    return false;
  }

  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  tt::LayoutAttr layout = mlir::cast<tt::LayoutAttr>(tensorType.getEncoding());

  Type scalarElementType;
  if (override.dataType.has_value()) {
    scalarElementType = {utils::createRowMajorTypeFromDtype(
        op->getContext(), override.dataType.value())};

  } else {
    // No search space for data type for now. Just use default.
    scalarElementType = {layout.getScalarElementType()};
  }

  GridAttr grid =
      GridAttr::get(op->getContext(), ArrayRef<int64_t>(override.grid.value()));

  // Create element type for the new layout.
  Type elementType = scalarElementType;
  if (override.memoryLayout == Layout::Tile) {
    elementType = TileType::get(op->getContext(), elementType);
  }

  // TODO rewrite like below.
  analysisResult.push_back(
      layout.withGrid(op->getContext(), tensorType, grid)
          .withMemorySpace(op->getContext(), override.memorySpace.value())
          .withMemoryLayout(op->getContext(),
                            override.tensorMemoryLayout.value())
          .withElementType(op->getContext(), elementType));

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

  std::optional<OutputLayoutOverrideParams> override;

  if (isa<NameLoc>(op->getLoc())) {
    StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
    if (auto overrideIt = analysisInput.outputLayoutOverrides->find(opLocName);
        overrideIt != analysisInput.outputLayoutOverrides->end()) {
      override = overrideIt->getValue();
    }
  }

  // std::tuple<MemorySpace, TensorMemoryLayout, GridAttr> searchSpace;

  // // // Define the search space for every layout parameter.
  // std::vector<GridAttr> grids;
  // if (override.has_value() and override->grid.has_value()) {
  //   grids = {GridAttr::get(op->getContext(), override->grid.value())};
  // } else {
  //   assert(analysisInput.maxGrid.getShape().size() == 2 &&
  //          "Max device grid is expected to be 2D.");
  //   for (auto width = 1; width <= analysisInput.maxGrid.getShape()[0];
  //        ++width) {
  //     for (auto height = 1; height <= analysisInput.maxGrid.getShape()[1];
  //          ++height) {
  //       // Block sharded grids.
  //       grids.push_back(GridAttr::get(op->getContext(), {width, height}));
  //     }
  //   }
  //   for (uint64_t dim = 1; dim <= analysisInput.maxGrid.getGridVolume();
  //        ++dim) {
  //     // Width sharded grids.
  //     grids.push_back(
  //         GridAttr::get(op->getContext(), {1, static_cast<int64_t>(dim)}));
  //     // Height sharded grids.
  //     grids.push_back(
  //         GridAttr::get(op->getContext(), {static_cast<int64_t>(dim), 1}));
  //   }
  // }

  // std::vector<MemorySpace> memorySpaces;
  // if (override.has_value() and override->memorySpace.has_value()) {
  //   memorySpaces = {override->memorySpace.value()};
  // } else {
  //   memorySpaces = {MemorySpace::DeviceDRAM, MemorySpace::DeviceL1};
  // }

  // std::vector<tt::TensorMemoryLayout> tensorMemoryLayouts;
  // if (override.has_value() and override->tensorMemoryLayout.has_value()) {
  //   tensorMemoryLayouts = {override->tensorMemoryLayout.value()};
  // } else {
  //   // tt::TensorMemoryLayout::SingleBank not supported for now.
  //   tensorMemoryLayouts = {tt::TensorMemoryLayout::Interleaved,
  //                          tt::TensorMemoryLayout::BlockSharded,
  //                          tt::TensorMemoryLayout::HeightSharded,
  //                          tt::TensorMemoryLayout::WidthSharded};
  // }

  // std::vector<Layout> memoryLayouts;
  // if (override.has_value() and override->memoryLayout.has_value()) {
  //   memoryLayouts = {override->memoryLayout.value()};
  // } else {
  //   memoryLayouts = {Layout::RowMajor, Layout::Tile};
  // }

  // std::vector<Type> scalarElementType;
  // if (override.has_value() and override->dataType.has_value()) {
  //   scalarElementType = {utils::createRowMajorTypeFromDtype(
  //       op->getContext(), override->dataType.value())};

  // } else {
  //   // No search space for data type for now. Just use default.
  //   scalarElementType = {layout.getScalarElementType()};
  // }

  // for (Type elementType : scalarElementType) {
  //   for (Layout memoryLayout : memoryLayouts) {
  //     for (MemorySpace memorySpace : memorySpaces) {
  //       for (tt::TensorMemoryLayout tensorMemoryLayout : tensorMemoryLayouts)
  //       {
  //         for (GridAttr grid : grids) {
  //         }
  //       }
  //     }
  //   }
  // }

  Type scalarElementType;
  if (override.has_value() and override->dataType.has_value()) {
    scalarElementType = {utils::createRowMajorTypeFromDtype(
        op->getContext(), override->dataType.value())};

  } else {
    // No search space for data type for now. Just use default.
    scalarElementType = {layout.getScalarElementType()};
  }

  Type tileElementType = TileType::get(op->getContext(), scalarElementType);

  std::vector<tt::LayoutAttr> shardedResults;

  // Generate both TILE and ROW_MAJOR layouts for all possibilities.
  // for (Type elementType : {scalarElementType, tileElementType}) {
  for (Type elementType : {tileElementType}) {
    // DRAM
    analysisResult.push_back(
        tt::LayoutAttr::get(op->getContext(), tensorType,
                            MemorySpace::DeviceDRAM, analysisInput.maxGrid,
                            elementType, tt::TensorMemoryLayout::Interleaved));

    // L1 Interleaved (same as above).
    analysisResult.push_back(
        tt::LayoutAttr::get(op->getContext(), tensorType, MemorySpace::DeviceL1,
                            analysisInput.maxGrid, elementType,
                            tt::TensorMemoryLayout::Interleaved));

    // L1 Sharded
    tt::LayoutAttr shardedBase =
        layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1)
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
                                  tt::TensorMemoryLayout::BlockSharded));
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

  // Apply overrides.
  if (override.has_value()) {
    analysisResult.erase(
        std::remove_if(analysisResult.begin(), analysisResult.end(),
                       [override](tt::LayoutAttr layout) {
                         bool keepLayout = true;
                         if (override->grid.has_value()) {
                           keepLayout &= layout.getGrid().getShape()[0] ==
                                             override->grid.value()[0] &&
                                         layout.getGrid().getShape()[1] ==
                                             override->grid.value()[1];
                         }
                         if (override->memorySpace.has_value()) {
                           keepLayout &= layout.getMemorySpace() ==
                                         override->memorySpace.value();
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
                         return keepLayout;
                       }),
        analysisResult.end());
  }

  // TODO check if empty?
}
} // namespace mlir::tt::ttnn
