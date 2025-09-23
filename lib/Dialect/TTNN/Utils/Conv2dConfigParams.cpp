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

  weightsDtype = getOrDefaultOpt(attr.getWeightsDtype(),
                                 mlir::tt::ttcore::DataType::BFloat16);

  if (attr.getActivation()) {
    activation = attr.getActivation().getOpType();
    activationParams = attr.getActivation().getParams();
  } else {
    activation = std::nullopt;
    activationParams = std::nullopt;
  }

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
  inPlace = getOrDefaultBool(attr.getInPlace());
}

Conv2dConfigAttr
Conv2dConfigParams::buildConv2dConfigAttr(::mlir::MLIRContext *ctx) const {
  auto toBoolAttr = [ctx](const std::optional<bool> &b) -> mlir::BoolAttr {
    return b.has_value() ? mlir::BoolAttr::get(ctx, b.value()) : nullptr;
  };

  auto toUnaryAttr =
      [ctx](const std::optional<UnaryOpType> &act,
            const std::optional<std::vector<FloatAttr>> &actParams)
      -> UnaryWithParamAttr {
    if (act) {
      return UnaryWithParamAttr::get(
          ctx, *act, actParams.value_or(llvm::ArrayRef<FloatAttr>{}));
    }

    return nullptr;
  };

  return Conv2dConfigAttr::get(
      ctx, weightsDtype, toUnaryAttr(activation, activationParams),
      toBoolAttr(deallocateActivation), toBoolAttr(reallocateHaloOutput),
      actBlockHOverride, actBlockWDiv, toBoolAttr(reshardIfNotOptimal),
      toBoolAttr(overrideShardingConfig), shardLayout,
      coreGrid.value_or(CoreRangeSetAttr()), toBoolAttr(transposeShards),
      outputLayout, toBoolAttr(enableActDoubleBuffer),
      toBoolAttr(enableWeightsDoubleBuffer), toBoolAttr(inPlace));
}
} // namespace mlir::tt::ttnn
