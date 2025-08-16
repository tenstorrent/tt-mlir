// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

Conv2dConfigParams::Conv2dConfigParams(const Conv2dConfigAttr &attr,
                                       bool partial) {
  auto getOrDefaultOpt =
      [partial](auto value,
                auto defaultValue) -> std::optional<decltype(defaultValue)> {
    return value.has_value()
               ? value.value()
               : (partial
                      ? std::nullopt
                      : std::optional<decltype(defaultValue)>(defaultValue));
  };

  auto getOrDefaultBool =
      [partial](mlir::BoolAttr attr) -> std::optional<bool> {
    return attr ? std::optional<bool>(attr.getValue())
                : (partial ? std::nullopt : std::optional<bool>(false));
  };

  auto getOrDefaultString =
      [partial](mlir::StringAttr attr) -> std::optional<std::string> {
    return attr ? std::optional<std::string>(attr.getValue().str())
                : (partial ? std::nullopt : std::optional<std::string>(""));
  };

  weightsDtype = getOrDefaultOpt(attr.getWeightsDtype(),
                                 mlir::tt::ttcore::DataType::BFloat16);
  activation = getOrDefaultString(attr.getActivation());
  deallocateActivation = getOrDefaultBool(attr.getDeallocateActivation());
  reallocateHaloOutput = getOrDefaultBool(attr.getReallocateHaloOutput());
  actBlockHOverride = getOrDefaultOpt(attr.getActBlockHOverride(), uint32_t(0));
  actBlockWDiv = getOrDefaultOpt(attr.getActBlockWDiv(), uint32_t(1));
  reshardIfNotOptimal = getOrDefaultBool(attr.getReshardIfNotOptimal());
  overrideShardingConfig = getOrDefaultBool(attr.getOverrideShardingConfig());
  shardLayout =
      getOrDefaultOpt(attr.getShardLayout(), TensorMemoryLayout::HeightSharded);
  coreGrid = attr.getCoreGrid()
                 ? std::optional<CoreRangeSetAttr>(attr.getCoreGrid())
                 : std::nullopt;
  transposeShards = getOrDefaultBool(attr.getTransposeShards());
  outputLayout = getOrDefaultOpt(attr.getOutputLayout(), Layout::Tile);
  enableActDoubleBuffer = getOrDefaultBool(attr.getEnableActDoubleBuffer());
  enableWeightsDoubleBuffer =
      getOrDefaultBool(attr.getEnableWeightsDoubleBuffer());
  enableSplitReader = getOrDefaultBool(attr.getEnableSplitReader());
  inPlace = getOrDefaultBool(attr.getInPlace());
}

Conv2dConfigParams::Conv2dConfigParams(const Conv2dConfigParams &base,
                                       const Conv2dConfigParams &overrides) {
  // Copy base values first, then apply overrides
  *this = base;
  applyOverrides(overrides);
}

Conv2dConfigAttr
Conv2dConfigParams::buildConv2dConfigAttr(::mlir::MLIRContext *ctx) const {
  auto toStringAttr =
      [ctx](const std::optional<std::string> &str) -> mlir::StringAttr {
    return str.has_value() ? mlir::StringAttr::get(ctx, str.value()) : nullptr;
  };

  auto toBoolAttr = [ctx](const std::optional<bool> &b) -> mlir::BoolAttr {
    return b.has_value() ? mlir::BoolAttr::get(ctx, b.value()) : nullptr;
  };

  return Conv2dConfigAttr::get(
      ctx, weightsDtype, toStringAttr(activation),
      toBoolAttr(deallocateActivation), toBoolAttr(reallocateHaloOutput),
      actBlockHOverride, actBlockWDiv, toBoolAttr(reshardIfNotOptimal),
      toBoolAttr(overrideShardingConfig), shardLayout,
      coreGrid.has_value() ? coreGrid.value() : CoreRangeSetAttr{},
      toBoolAttr(transposeShards), outputLayout,
      toBoolAttr(enableActDoubleBuffer), toBoolAttr(enableWeightsDoubleBuffer),
      toBoolAttr(enableSplitReader), toBoolAttr(inPlace));
}

void Conv2dConfigParams::applyOverrides(const Conv2dConfigParams &overrides) {
  if (overrides.weightsDtype.has_value()) {
    weightsDtype = overrides.weightsDtype;
  }
  if (overrides.activation.has_value()) {
    activation = overrides.activation;
  }
  if (overrides.deallocateActivation.has_value()) {
    deallocateActivation = overrides.deallocateActivation;
  }
  if (overrides.reallocateHaloOutput.has_value()) {
    reallocateHaloOutput = overrides.reallocateHaloOutput;
  }
  if (overrides.actBlockHOverride.has_value()) {
    actBlockHOverride = overrides.actBlockHOverride;
  }
  if (overrides.actBlockWDiv.has_value()) {
    actBlockWDiv = overrides.actBlockWDiv;
  }
  if (overrides.reshardIfNotOptimal.has_value()) {
    reshardIfNotOptimal = overrides.reshardIfNotOptimal;
  }
  if (overrides.overrideShardingConfig.has_value()) {
    overrideShardingConfig = overrides.overrideShardingConfig;
  }
  if (overrides.shardLayout.has_value()) {
    shardLayout = overrides.shardLayout;
  }
  if (overrides.coreGrid.has_value()) {
    coreGrid = overrides.coreGrid;
  }
  if (overrides.transposeShards.has_value()) {
    transposeShards = overrides.transposeShards;
  }
  if (overrides.outputLayout.has_value()) {
    outputLayout = overrides.outputLayout;
  }
  if (overrides.enableActDoubleBuffer.has_value()) {
    enableActDoubleBuffer = overrides.enableActDoubleBuffer;
  }
  if (overrides.enableWeightsDoubleBuffer.has_value()) {
    enableWeightsDoubleBuffer = overrides.enableWeightsDoubleBuffer;
  }
  if (overrides.enableSplitReader.has_value()) {
    enableSplitReader = overrides.enableSplitReader;
  }
  if (overrides.inPlace.has_value()) {
    inPlace = overrides.inPlace;
  }
}

} // namespace mlir::tt::ttnn
