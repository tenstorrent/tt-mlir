// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/Support/ErrorHandling.h"

#include <vector>

namespace mlir::tt::ttnn {

static bool isOpEnabledForAnalysis(Operation *op) {
  // Enable only for specific ops.
  if (llvm::isa<ttnn::Conv2dOp>(op)) {
    return true;
  }

  return false;
}

static void
applyConv2dConfigOverrides(ttnn::Conv2dOp op,
                           const Conv2dConfigOverrideParams &overrides,
                           std::vector<OpConfig> &analysisResult) {
  // Apply conv2d config overrides to all legal (layout) configurations of
  // current op.

  // If conv2d config is not set get default conv2d config.
  Conv2dConfigAttr conv2dConfigAttr = op.getConv2dConfigAttr();
  if (!conv2dConfigAttr) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "Conv2d config not set, using default");
    conv2dConfigAttr = Conv2dConfigAttr::get(op.getContext());
  }
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Conv2d config before overrides: {}", conv2dConfigAttr);
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Overrides: {}", overrides);

  if (overrides.weightsDtype.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withWeightsDtype(*overrides.weightsDtype);
  }

  if (overrides.activation.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withActivation(*overrides.activation);
  }

  if (overrides.deallocateActivation.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withDeallocateActivation(
        *overrides.deallocateActivation);
  }

  if (overrides.reallocateHaloOutput.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withReallocateHaloOutput(
        *overrides.reallocateHaloOutput);
  }

  if (overrides.actBlockHOverride.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withActBlockHOverride(*overrides.actBlockHOverride);
  }

  if (overrides.actBlockWDiv.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withActBlockWDiv(*overrides.actBlockWDiv);
  }

  if (overrides.reshardIfNotOptimal.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withReshardIfNotOptimal(
        *overrides.reshardIfNotOptimal);
  }

  if (overrides.overrideShardingConfig.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withOverrideShardingConfig(
        *overrides.overrideShardingConfig);
  }

  if (overrides.shardLayout.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withShardLayout(*overrides.shardLayout);
  }

  if (overrides.coreGrid.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withCoreGrid(*overrides.coreGrid);
  }

  if (overrides.transposeShards.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withTransposeShards(*overrides.transposeShards);
  }

  if (overrides.outputLayout.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withOutputLayout(*overrides.outputLayout);
  }

  if (overrides.enableActDoubleBuffer.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withEnableActDoubleBuffer(
        *overrides.enableActDoubleBuffer);
  }

  if (overrides.enableWeightsDoubleBuffer.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withEnableWeightsDoubleBuffer(
        *overrides.enableWeightsDoubleBuffer);
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Conv2d config after overrides: {}", conv2dConfigAttr);

  // Set overriden conv2d config for all OpConfigs.
  for (OpConfig &opConfig : analysisResult) {
    assert(opConfig.isAttrUninitialized() &&
           "OpConfig should not have a config set before applying overrides");
    opConfig.opSpecificAttrs = Conv2dAttrs{conv2dConfigAttr, std::nullopt};
  }
}

bool LegalOpConfigAnalysis::applyOverrides() {
  // For now, easiest way to initialize analysisResult is to copy the legal
  // configs here. Proper solution is that init() method is overridden in child
  // classes.
  analysisResult = analysisInput.legalConfigs;

  if (!isOpEnabledForAnalysis(op)) {
    return true;
  }

  if (!analysisInput.conv2dConfigOverrides) {
    return false;
  }

  if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(op)) {
    Conv2dConfigOverrideParams conv2dConfigOverrides;
    if (!isa<NameLoc>(op->getLoc())) {
      return false;
    }
    StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
    auto overrideConv2dIt =
        analysisInput.conv2dConfigOverrides->find(opLocName);
    if (overrideConv2dIt != analysisInput.conv2dConfigOverrides->end()) {
      conv2dConfigOverrides = overrideConv2dIt->getValue();
    }
    applyConv2dConfigOverrides(conv2dOp, conv2dConfigOverrides, analysisResult);

    // Conv2d config overrides were applied, return true if all config
    // parameters were overridden, therefore no need to search for legal
    // configs.
    return conv2dConfigOverrides.fullConfigOverride();
  }

  llvm::llvm_unreachable_internal("Unsupported op type");
}

void LegalOpConfigAnalysis::fillOpSpecificAttrs() {
  if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(op)) {
    assert(!analysisResult.empty() &&
           "Analysis result should not be empty after applying overrides");
    TTMLIR_TRACE(
        ttmlir::LogComponent::Optimizer,
        "Filling op specific attrs for conv2d op {}, starting with {} configs",
        conv2dOp, analysisResult.size());

    // It's possible that no conv2d config was applied in
    // applyConv2dConfigOverrides (e.g. when op does not have loc assigned) so
    // base config is empty.
    Conv2dConfigAttr conv2dConfigAttrBase =
        analysisResult.begin()->isAttrUninitialized()
            ? (conv2dOp.getConv2dConfigAttr()
                   ? conv2dOp.getConv2dConfigAttr()
                   : Conv2dConfigAttr::get(op->getContext()))
            : std::get<Conv2dAttrs>(analysisResult.begin()->opSpecificAttrs)
                  .conv2dConfig.value();

    // If weights dtype is not set, set it to the weight tensor dtype.
    if (!conv2dConfigAttrBase.getWeightsDtype().has_value()) {
      conv2dConfigAttrBase =
          conv2dConfigAttrBase.withWeightsDtype(ttcore::elementTypeToDataType(
              conv2dOp.getWeight().getType().getElementType()));
    }

    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "Op {} Base conv2d config: {}", conv2dOp.getLoc(),
                 conv2dConfigAttrBase);

    auto filterOut = [](const Conv2dConfigAttr &config) {
      //
      // Combinations that are invalid:
      // 1. reshard_if_not_optimal = true and shard_layout is not set.
      //
      return (config.hasReshardIfNotOptimal() &&
              config.getReshardIfNotOptimal().getValue() &&
              !config.hasShardLayout()) ||
             (config.getActBlockHOverride().value_or(0) < 64);
    };

    Conv2dConfigGenerator configGenerator(&conv2dOp, conv2dConfigAttrBase,
                                          searchSpace, filterOut);

    std::vector<OpConfig> newLegalConfigs;
    auto addConfigs = [&](const Conv2dConfigAttr &configAttr) {
      for (const OpConfig &existingOpConfig : analysisResult) {
        // Create a new OpConfig pairing the existing layout with the new conv
        // config.
        newLegalConfigs.emplace_back(existingOpConfig.outputLayout,
                                     Conv2dAttrs{configAttr, std::nullopt});
      }
    };

    if (configGenerator.searchDone()) {
      // If search is done before any configs are generated, we will just
      // put base config in all possible layouts. This way we are ensuring
      // dtype and weights_dtype will be set.
      addConfigs(conv2dConfigAttrBase);
    } else {
      // Otherwise, generate all possible configs and add them to the result.
      while (Conv2dConfigAttr configAttr = configGenerator.getNextConfig()) {
        addConfigs(configAttr);
      }
    }

    analysisResult = std::move(newLegalConfigs);
    TTMLIR_TRACE(
        ttmlir::LogComponent::Optimizer,
        "Filled op specific attrs for conv2d op {}, ending with {} configs",
        conv2dOp, analysisResult.size());

    return;
  }

  op->emitError("Unsupported op type");
  llvm::llvm_unreachable_internal("Unsupported op type");
}

void LegalOpConfigAnalysis::analysisImplementation() {
  if (!isOpEnabledForAnalysis(op)) {
    return;
  }

  fillOpSpecificAttrs();

  if (analysisResult.empty()) {
    op->emitError("No legal config found for the operation");
    llvm::llvm_unreachable_internal("No legal config found for the operation");
  }
}

} // namespace mlir::tt::ttnn
