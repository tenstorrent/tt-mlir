// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::tt::ttnn {

Conv2dConfigParams::Conv2dConfigParams(Conv2dConfigAttr attr, bool partial) {
  auto getOrDefaultOpt =
      [partial](auto value,
                auto defaultValue) -> std::optional<decltype(defaultValue)> {
    return value || !partial ? std::make_optional(value.value_or(defaultValue))
                             : std::nullopt;
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
  actBlockHOverride = getOrDefaultOpt(attr.getActBlockHOverride(), 0);
  actBlockWDiv = getOrDefaultOpt(attr.getActBlockWDiv(), 1);
  reshardIfNotOptimal = getOrDefaultBool(attr.getReshardIfNotOptimal());
  overrideShardingConfig = getOrDefaultBool(attr.getOverrideShardingConfig());
  shardLayout = attr.getShardLayout();
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
      coreGrid.value_or(CoreRangeSetAttr()), toBoolAttr(transposeShards),
      outputLayout, toBoolAttr(enableActDoubleBuffer),
      toBoolAttr(enableWeightsDoubleBuffer), toBoolAttr(enableSplitReader),
      toBoolAttr(inPlace));
}
} // namespace mlir::tt::ttnn
