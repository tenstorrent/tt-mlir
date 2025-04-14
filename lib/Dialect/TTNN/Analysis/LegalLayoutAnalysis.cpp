// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalLayoutAnalysis.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

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

void applyConv2dConfigOverrides(
    Operation *op, const Conv2dConfigOverrideParams &conv2dConfigOverrides,
    std::vector<OpConfig> &analysisResult) {
  // Apply conv2d config overrides to all legal (layout) configurations of
  // current op.
  // TODO(vkovacevic): Currently conv2d config overrides are applied without any
  // analysis, but will need to go through analysis in the future to check if
  // they are valid.
  //
  MLIRContext *context = op->getContext();
  DataType newDtype = conv2dConfigOverrides.dtype.value_or(DataType::BFloat16);
  DataType newWeightsDtype =
      conv2dConfigOverrides.weightsDtype.value_or(DataType::BFloat16);
  StringAttr newActivation =
      StringAttr::get(context, conv2dConfigOverrides.activation.value_or(""));
  uint32_t newInputChannelsAlignment =
      conv2dConfigOverrides.inputChannelsAlignment.value_or(32);
  bool newDeallocateActivation =
      conv2dConfigOverrides.deallocateActivation.value_or(false);
  bool newReallocateHaloOutput =
      conv2dConfigOverrides.reallocateHaloOutput.value_or(true);
  uint32_t newActBlockHOverride =
      conv2dConfigOverrides.actBlockHOverride.value_or(0);
  uint32_t newActBlockWDiv = conv2dConfigOverrides.actBlockWDiv.value_or(1);
  bool newReshardIfNotOptimal =
      conv2dConfigOverrides.reshardIfNotOptimal.value_or(false);
  bool newOverrideShardingConfig =
      conv2dConfigOverrides.overrideShardingConfig.value_or(false);
  ttnn::TensorMemoryLayoutAttr newShardLayout;
  if (conv2dConfigOverrides.shardLayout.has_value()) {
    newShardLayout = TensorMemoryLayoutAttr::get(
        context, conv2dConfigOverrides.shardLayout.value());
  }
  ttnn::CoreRangeSetAttr newCoreGrid =
      conv2dConfigOverrides.coreGrid.value_or(ttnn::CoreRangeSetAttr());
  bool newTransposeShards =
      conv2dConfigOverrides.transposeShards.value_or(false);
  Layout newOutputLayout =
      conv2dConfigOverrides.outputLayout.value_or(Layout::Tile);
  bool newPreprocessWeightsOnDevice =
      conv2dConfigOverrides.preprocessWeightsOnDevice.value_or(false);
  bool newAlwaysPreprocessWeights =
      conv2dConfigOverrides.alwaysPreprocessWeights.value_or(false);
  bool newEnableActDoubleBuffer =
      conv2dConfigOverrides.enableActDoubleBuffer.value_or(false);
  bool newEnableWeightsDoubleBuffer =
      conv2dConfigOverrides.enableWeightsDoubleBuffer.value_or(false);
  bool newEnableSplitReader =
      conv2dConfigOverrides.enableSplitReader.value_or(false);
  bool newEnableSubblockPadding =
      conv2dConfigOverrides.enableSubblockPadding.value_or(false);

  for (auto &opConfig : analysisResult) {
    assert(!opConfig.config &&
           "OpConfig should not have a config set before applying overrides");
    opConfig.config = Conv2dConfigAttr::get(
        context, newDtype, newWeightsDtype, newActivation,
        newInputChannelsAlignment, newDeallocateActivation,
        newReallocateHaloOutput, newActBlockHOverride, newActBlockWDiv,
        newReshardIfNotOptimal, newOverrideShardingConfig, newShardLayout,
        newCoreGrid, newTransposeShards, newOutputLayout,
        newPreprocessWeightsOnDevice, newAlwaysPreprocessWeights,
        newEnableActDoubleBuffer, newEnableWeightsDoubleBuffer,
        newEnableSplitReader, newEnableSubblockPadding);
  }
}

bool LegalLayoutAnalysis::applyOverrides() {
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
  // and create the overriden layout. Otherwise, we need to perform analysis and
  // apply partial overrides.
  if (!layoutOverride.fullLayoutOverride()) {
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
    elementType = mlir::tt::dataTypeToElementType(
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

  // Apply conv2d config overrides if they exist and op is Conv2d.
  if (analysisInput.conv2dConfigOverrides && isa<ttnn::Conv2dOp>(op)) {
    auto overrideConv2dIt =
        analysisInput.conv2dConfigOverrides->find(opLocName);
    if (overrideConv2dIt != analysisInput.conv2dConfigOverrides->end()) {
      applyConv2dConfigOverrides(op, overrideConv2dIt->getValue(),
                                 analysisResult);
    }
  }
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
        scalarElementType = mlir::tt::dataTypeToElementType(
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

  for (size_t dataLayoutIdx = 0;
       dataLayoutIdx < static_cast<size_t>(TensorDataLayoutIndex::kNumValues);
       ++dataLayoutIdx) {

    if (!rowMajorAllowed &&
        dataLayoutIdx == static_cast<size_t>(TensorDataLayoutIndex::RowMajor)) {
      continue;
    }

    // Insert interleaved layouts for current data layout
    const auto &interleavedLayoutsForDataLayout =
        scalarTypeIt->second[dataLayoutIdx][getMemoryLayoutIndex(
            TensorMemoryLayout::Interleaved)];

    interleavedLayouts.insert(interleavedLayouts.end(),
                              interleavedLayoutsForDataLayout.begin(),
                              interleavedLayoutsForDataLayout.end());

    // Insert sharded layouts for current data layout, block sharded will give
    // us unified index for all sharded layouts
    const auto &shardedLayoutsForDataLayout =
        scalarTypeIt->second[dataLayoutIdx][getMemoryLayoutIndex(
            TensorMemoryLayout::BlockSharded)];

    if (dataLayoutIdx == getDataLayoutIndex(Layout::RowMajor)) {
      shardedLayoutsRowMajor.insert(shardedLayoutsRowMajor.end(),
                                    shardedLayoutsForDataLayout.begin(),
                                    shardedLayoutsForDataLayout.end());
    } else {
      shardedLayoutsTile.insert(shardedLayoutsTile.end(),
                                shardedLayoutsForDataLayout.begin(),
                                shardedLayoutsForDataLayout.end());
    }
  }

  // Let's take maxShardedConfigs/2 from both row major and tile collections
  size_t maxShardedConfigs = analysisInput.maxShardedConfigs / 2;
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
  // one overriden param.
  if (override.has_value()) {
    auto shouldRemoveLayout =
        std::bind(incompatibleWithOverride, std::placeholders::_1, override);
    analysisResult.erase(std::remove_if(analysisResult.begin(),
                                        analysisResult.end(),
                                        shouldRemoveLayout),
                         analysisResult.end());
  }

  // Apply conv2d config overrides if they exist and op is Conv2d.
  if (auto opLoc = mlir::dyn_cast<NameLoc>(op->getLoc())) {
    StringRef opLocName = opLoc.getName().strref();
    if (analysisInput.conv2dConfigOverrides && isa<ttnn::Conv2dOp>(op)) {
      auto overrideConv2dIt =
          analysisInput.conv2dConfigOverrides->find(opLocName);
      if (overrideConv2dIt != analysisInput.conv2dConfigOverrides->end()) {
        applyConv2dConfigOverrides(op, overrideConv2dIt->getValue(),
                                   analysisResult);
      }
    }
  }

  if (analysisResult.empty()) {
    op->emitError("No legal layout found for the operation.");
    assert(false && "At least one legal layout must be found.");
  }
}
} // namespace mlir::tt::ttnn
