// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalLayoutAnalysis.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

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

void applyConv2dConfigOverrides(Operation *op,
                                const Conv2dConfigOverrideParams &overrides,
                                std::vector<OpConfig> &analysisResult) {
  // Apply conv2d config overrides to all legal (layout) configurations of
  // current op.
  // TODO(vkovacevic): Currently conv2d config overrides are applied without any
  // analysis, but will need to go through analysis in the future to check if
  // they are valid.
  //

  // vkovacevic: This is needed to get through a tt-metal assert in
  // prepare_conv2d_weights.cpp where `weight_tensor_.get_dtype() ==
  // weights_bias_dtype`.
  //
  MLIRContext *context = op->getContext();
  auto getBoolAttr = [context](bool value) {
    return BoolAttr::get(context, value);
  };

  DataType dtype = elementTypeToDataType(
      mlir::cast<RankedTensorType>(op->getOperand(0).getType())
          .getElementType());
  DataType weightsDtype = elementTypeToDataType(
      mlir::cast<RankedTensorType>(op->getOperand(1).getType())
          .getElementType());

  StringAttr activation =
      StringAttr::get(context, overrides.activation.value_or(""));
  uint32_t inputChannelsAlignment =
      overrides.inputChannelsAlignment.value_or(32);
  BoolAttr deallocateActivation =
      getBoolAttr(overrides.deallocateActivation.value_or(false));
  BoolAttr reallocateHaloOutput =
      getBoolAttr(overrides.reallocateHaloOutput.value_or(true));
  uint32_t actBlockHOverride = overrides.actBlockHOverride.value_or(0);
  uint32_t actBlockWDiv = overrides.actBlockWDiv.value_or(1);
  BoolAttr reshardIfNotOptimal =
      getBoolAttr(overrides.reshardIfNotOptimal.value_or(false));
  BoolAttr overrideShardingConfig =
      getBoolAttr(overrides.overrideShardingConfig.value_or(false));

  std::optional<ttnn::TensorMemoryLayout> shardLayout;
  if (overrides.shardLayout.has_value()) {
    shardLayout = overrides.shardLayout;
  }

  ttnn::CoreRangeSetAttr coreGrid =
      overrides.coreGrid.value_or(ttnn::CoreRangeSetAttr());
  BoolAttr transposeShards =
      getBoolAttr(overrides.transposeShards.value_or(false));
  Layout outputLayout = overrides.outputLayout.value_or(Layout::Tile);
  BoolAttr preprocessWeightsOnDevice =
      getBoolAttr(overrides.preprocessWeightsOnDevice.value_or(false));
  BoolAttr alwaysPreprocessWeights =
      getBoolAttr(overrides.alwaysPreprocessWeights.value_or(false));
  BoolAttr enableActDoubleBuffer =
      getBoolAttr(overrides.enableActDoubleBuffer.value_or(false));
  BoolAttr enableWeightsDoubleBuffer =
      getBoolAttr(overrides.enableWeightsDoubleBuffer.value_or(false));
  BoolAttr enableSplitReader =
      getBoolAttr(overrides.enableSplitReader.value_or(false));
  BoolAttr enableSubblockPadding =
      getBoolAttr(overrides.enableSubblockPadding.value_or(false));

  for (auto &opConfig : analysisResult) {
    assert(!opConfig.config &&
           "OpConfig should not have a config set before applying overrides");
    opConfig.config = Conv2dConfigAttr::get(
        context, dtype, weightsDtype, activation, inputChannelsAlignment,
        deallocateActivation, reallocateHaloOutput, actBlockHOverride,
        actBlockWDiv, reshardIfNotOptimal, overrideShardingConfig, shardLayout,
        coreGrid, transposeShards, outputLayout, preprocessWeightsOnDevice,
        alwaysPreprocessWeights, enableActDoubleBuffer,
        enableWeightsDoubleBuffer, enableSplitReader, enableSubblockPadding);
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

  // Apply conv2d config overrides.
  // If they do not exist, or they do not exist for a specific conv2d op, set
  // conv2d config with default values.
  //
  if (isa<ttnn::Conv2dOp>(op)) {
    Conv2dConfigOverrideParams conv2dConfigOverrides;
    if (analysisInput.conv2dConfigOverrides) {
      auto overrideConv2dIt =
          analysisInput.conv2dConfigOverrides->find(opLocName);
      if (overrideConv2dIt != analysisInput.conv2dConfigOverrides->end()) {
        conv2dConfigOverrides = overrideConv2dIt->getValue();
      }
    }
    applyConv2dConfigOverrides(op, conv2dConfigOverrides, analysisResult);
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
        scalarElementType = {mlir::tt::dataTypeToElementType(
            op->getContext(), override->dataType.value())};
      }
    }
  }

  bool rowMajorAllowed = analysisInput.rowMajorEnabled;
  if (override.has_value() && override->memoryLayout.has_value() &&
      override->memoryLayout.value() == Layout::RowMajor) {
    // Force allow row major if override is set.
    rowMajorAllowed = true;
  }

  std::vector<TTNNLayoutAttr> generatedLayouts =
      optimizer_utils::generateAllPossibleLayouts(
          op->getContext(), tensorType, analysisInput.maxGrid,
          scalarElementType,
          /*onlyShardedLayouts=*/false, analysisInput.maxShardedConfigs,
          rowMajorAllowed);

  analysisResult.insert(analysisResult.end(), generatedLayouts.begin(),
                        generatedLayouts.end());

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

  // Apply conv2d config overrides.
  // If they do not exist, or they do not exist for a specific conv2d op, set
  // conv2d config with default values.
  //
  if (auto opLoc = mlir::dyn_cast<NameLoc>(op->getLoc())) {
    StringRef opLocName = opLoc.getName().strref();
    if (isa<ttnn::Conv2dOp>(op)) {
      Conv2dConfigOverrideParams conv2dConfigOverrides;
      if (analysisInput.conv2dConfigOverrides) {
        auto overrideConv2dIt =
            analysisInput.conv2dConfigOverrides->find(opLocName);
        if (overrideConv2dIt != analysisInput.conv2dConfigOverrides->end()) {
          conv2dConfigOverrides = overrideConv2dIt->getValue();
        }
      }
      applyConv2dConfigOverrides(op, conv2dConfigOverrides, analysisResult);
    }
  }

  if (analysisResult.empty()) {
    op->emitError("No legal layout found for the operation.");
    assert(false && "At least one legal layout must be found.");
  }
}
} // namespace mlir::tt::ttnn
