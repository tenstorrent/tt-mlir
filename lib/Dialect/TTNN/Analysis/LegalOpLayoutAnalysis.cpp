// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/Conv2dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include "mlir/Support/LLVM.h"
#include <algorithm>
#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn {

static bool cantChangeOutputLayout(Operation *op) {
  // Check if OP belongs to TTNN dialect.
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

bool LegalOpLayoutAnalysis::applyOverrides() {
  // Lookup layout overrides based on location information for current
  // operation.
  //

  if (!analysisInput.outputLayoutOverrides) {
    return false;
  }

  if (!isa<NameLoc>(op->getLoc())) {
    return false;
  }

  StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
  auto overrideIt = analysisInput.outputLayoutOverrides->find(opLocName);

  if (overrideIt == analysisInput.outputLayoutOverrides->end()) {
    return false;
  }

  OutputLayoutOverrideParams layoutOverride = overrideIt->getValue();

  // If all layout parameters are set (except data type), we can skip analysis
  // and create the overridden layout. Otherwise, we need to perform analysis
  // and apply partial overrides.
  if (!layoutOverride.fullLayoutOverride()) {
    return false;
  }

  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  TTNNLayoutAttr layout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  ttcore::GridAttr grid = ttcore::GridAttr::get(
      op->getContext(), ArrayRef<int64_t>(layoutOverride.grid.value()));

  // Create element type for the new layout.
  Type elementType = layout.getScalarElementType();
  if (layoutOverride.dataType.has_value()) {
    elementType = mlir::tt::ttcore::dataTypeToElementType(
        op->getContext(), layoutOverride.dataType.value());
  }

  if (layoutOverride.memoryLayout == Layout::Tile) {
    elementType = ttcore::TileType::get(elementType);
  }

  TTNNLayoutAttr newLayout = TTNNLayoutAttr::get(
      op->getContext(), tensorShape, elementType,
      layoutOverride.bufferType.value(), grid,
      TensorMemoryLayoutAttr::get(op->getContext(),
                                  layoutOverride.tensorMemoryLayout.value()));

  analysisResult.push_back({newLayout});

  return true;
}

bool incompatibleWithOverride(
    const OpConfig &config,
    const std::optional<OutputLayoutOverrideParams> &layoutOverride) {
  if (!layoutOverride.has_value()) {
    return false;
  }

  if (layoutOverride->grid.has_value()) {
    if (config.outputLayout.getGrid().getShape()[0] !=
            layoutOverride->grid.value()[0] ||
        config.outputLayout.getGrid().getShape()[1] !=
            layoutOverride->grid.value()[1]) {
      return true;
    }
  }
  if (layoutOverride->bufferType.has_value() &&
      config.outputLayout.getBufferType() !=
          layoutOverride->bufferType.value()) {
    return true;
  }
  if (layoutOverride->tensorMemoryLayout.has_value() &&
      config.outputLayout.getMemLayout().getValue() !=
          layoutOverride->tensorMemoryLayout.value()) {
    return true;
  }
  if (layoutOverride->memoryLayout.has_value() &&
      config.outputLayout.isTiled() !=
          (layoutOverride->memoryLayout.value() == Layout::Tile)) {
    return true;
  }
  return false;
}

// Skip operations that don't have nontrivial output tensors.
bool LegalOpLayoutAnalysis::isValidAnalysisTarget(Operation *op) {
  if (op->getNumResults() == 0) {
    return false;
  }
  if (!llvm::isa<mlir::RankedTensorType>(op->getResult(0).getType())) {
    return false;
  }
  if (llvm::isa<mlir::tt::ttnn::EmptyOp>(op)) {
    return false;
  }
  return true;
}

void LegalOpLayoutAnalysis::fillTTNNLayoutAttrs(TTNNLayoutAttr baseLayout) {
  Type scalarElementType = baseLayout.getScalarElementType();

  std::optional<OutputLayoutOverrideParams> override;

  // Check if we have an override for this op.
  if (isa<NameLoc>(op->getLoc())) {
    StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
    if (auto overrideIt = analysisInput.outputLayoutOverrides->find(opLocName);
        overrideIt != analysisInput.outputLayoutOverrides->end()) {
      override = overrideIt->getValue();
      if (override->dataType.has_value()) {
        scalarElementType = mlir::tt::ttcore::dataTypeToElementType(
            op->getContext(), override->dataType.value());
      }
    }
  }

  bool rowMajorAllowed = analysisInput.rowMajorEnabled;
  if (override.has_value() && override->memoryLayout.has_value() &&
      override->memoryLayout.value() == Layout::RowMajor) {
    // Force allow row major if override is set.
    rowMajorAllowed = true;
  }

  // Sharded layouts for row major and tile are kept separate so we can combine
  // them equally, and avoid having only RM or Tile layouts
  std::vector<TTNNLayoutAttr> shardedLayoutsRowMajor;
  std::vector<TTNNLayoutAttr> shardedLayoutsTile;
  std::vector<TTNNLayoutAttr> interleavedLayouts;

  // Find the entry for our tensor type and scalar type
  auto scalarTypeIt = analysisInput.possibleLayouts->find(scalarElementType);
  assert(scalarTypeIt != analysisInput.possibleLayouts->end() &&
         "Scalar type not found in all possible layouts");

  for (size_t pageLayoutIdx = 0;
       pageLayoutIdx < static_cast<size_t>(TensorPageLayout::kNumValues);
       ++pageLayoutIdx) {

    if (!rowMajorAllowed &&
        pageLayoutIdx == static_cast<size_t>(TensorPageLayout::RowMajor)) {
      continue;
    }

    // Insert interleaved layouts for current data layout
    const auto &interleavedLayoutsForDataLayout =
        scalarTypeIt->second[pageLayoutIdx][getMemoryLayoutIndex(
            TensorMemoryLayout::Interleaved)];

    interleavedLayouts.insert(interleavedLayouts.end(),
                              interleavedLayoutsForDataLayout.begin(),
                              interleavedLayoutsForDataLayout.end());

    // Insert sharded layouts for current data layout, block sharded will give
    // us unified index for all sharded layouts
    const std::vector<TTNNLayoutAttr> &shardedLayoutsForDataLayout =
        getShardedLayoutsForPageLayout(pageLayoutIdx, scalarTypeIt->second);

    if (pageLayoutIdx == getPageLayoutIndex(Layout::RowMajor)) {
      shardedLayoutsRowMajor.insert(shardedLayoutsRowMajor.end(),
                                    shardedLayoutsForDataLayout.begin(),
                                    shardedLayoutsForDataLayout.end());
    } else {
      shardedLayoutsTile.insert(shardedLayoutsTile.end(),
                                shardedLayoutsForDataLayout.begin(),
                                shardedLayoutsForDataLayout.end());
    }
  }

  // We will sort the layouts by grid volume, so we can take the largest ones
  // first. Possibly, they are all sorted by grid volume already, but we will
  // sort them again explicitly to be sure
  std::sort(shardedLayoutsRowMajor.begin(), shardedLayoutsRowMajor.end(),
            [](TTNNLayoutAttr a, TTNNLayoutAttr b) {
              return a.getGrid().getGridVolume() > b.getGrid().getGridVolume();
            });
  std::sort(shardedLayoutsTile.begin(), shardedLayoutsTile.end(),
            [](TTNNLayoutAttr a, TTNNLayoutAttr b) {
              return a.getGrid().getGridVolume() > b.getGrid().getGridVolume();
            });

  // Let's take maxShardedConfigs/2 from both row major and tile collections,
  // unless row major is not allowed, then we take all of the tile layouts
  size_t maxShardedConfigs = rowMajorAllowed
                                 ? (analysisInput.maxShardedConfigs / 2)
                                 : analysisInput.maxShardedConfigs;

  // If row major is not allowed, shardedLayoutsRowMajor vector will be empty,
  // so we will only take tile layouts
  shardedLayoutsRowMajor.resize(
      std::min(maxShardedConfigs, shardedLayoutsRowMajor.size()));
  shardedLayoutsTile.resize(
      std::min(maxShardedConfigs, shardedLayoutsTile.size()));

  std::vector<TTNNLayoutAttr> shardedLayouts;
  shardedLayouts.insert(shardedLayouts.end(), shardedLayoutsRowMajor.begin(),
                        shardedLayoutsRowMajor.end());
  shardedLayouts.insert(shardedLayouts.end(), shardedLayoutsTile.begin(),
                        shardedLayoutsTile.end());

  std::sort(shardedLayouts.begin(), shardedLayouts.end(),
            [](TTNNLayoutAttr a, TTNNLayoutAttr b) {
              return a.getGrid().getGridVolume() > b.getGrid().getGridVolume();
            });

  analysisResult.insert(analysisResult.end(), interleavedLayouts.begin(),
                        interleavedLayouts.end());
  analysisResult.insert(analysisResult.end(), shardedLayouts.begin(),
                        shardedLayouts.end());

  // Apply partial layout overrides. Remove layouts that conflict with at least
  // one overridden param.
  if (override.has_value()) {
    auto shouldRemoveLayout =
        std::bind(incompatibleWithOverride, std::placeholders::_1, override);
    analysisResult.erase(std::remove_if(analysisResult.begin(),
                                        analysisResult.end(),
                                        shouldRemoveLayout),
                         analysisResult.end());
  }
}

void LegalOpLayoutAnalysis::analysisImplementation() {
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
  TTNNLayoutAttr layout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  // Return existing layout if it is not possible to change it.
  if (cantChangeOutputLayout(op)) {
    analysisResult.push_back(layout);
    return;
  }

  fillTTNNLayoutAttrs(layout);

  if (analysisResult.empty()) {
    op->emitError("No legal layout found for the operation");
    llvm::llvm_unreachable_internal("No legal layout found for the operation.");
  }
}

} // namespace mlir::tt::ttnn
