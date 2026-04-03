// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LayerNormPostAllGatherMissingBiasRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult LayerNormPostAllGatherMissingBiasRewritePattern::matchAndRewrite(
    ttnn::LayerNormPostAllGatherOp srcOp, PatternRewriter &rewriter) const {
  // Only fire when weight is present and bias is absent.
  if (!srcOp.getWeight() || srcOp.getBias()) {
    return failure();
  }

  RankedTensorType weightType =
      mlir::cast<RankedTensorType>(srcOp.getWeight().getType());
  auto weightLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(weightType.getEncoding());

  auto biasShapeAttr =
      ttnn::ShapeAttr::get(rewriter.getContext(), weightType.getShape());
  auto biasDTypeAttr = ttcore::DataTypeAttr::get(
      rewriter.getContext(), weightLayoutAttr.getDataType());
  auto biasLayoutAttr = ttnn::LayoutAttr::get(rewriter.getContext(),
                                               weightLayoutAttr.getLayout());
  mlir::Value biasDevice =
      weightLayoutAttr.isDeviceBufferType()
          ? static_cast<mlir::Value>(
                ttnn::utils::getOrInsertDevice(rewriter, srcOp))
          : nullptr;
  ttnn::MemoryConfigAttr biasMemoryConfig =
      weightLayoutAttr.getMemLayout()
          ? ttnn::MemoryConfigAttr::get(
                rewriter.getContext(), weightLayoutAttr.getMemLayout(),
                ttnn::BufferTypeAttr::get(rewriter.getContext(),
                                          weightLayoutAttr.getBufferType()),
                std::nullopt)
          : nullptr;

  mlir::Value zeroBias =
      rewriter
          .create<ttnn::ZerosOp>(
              ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                                  "_bias_zeros"),
              weightType, biasDevice, biasShapeAttr, biasDTypeAttr,
              biasLayoutAttr, biasMemoryConfig)
          .getResult();

  rewriter.replaceOpWithNewOp<ttnn::LayerNormPostAllGatherOp>(
      srcOp, srcOp.getResult().getType(), srcOp.getInput(), srcOp.getStats(),
      srcOp.getWeight(), zeroBias, srcOp.getEpsilonAttr(),
      srcOp.getDtypeAttr(), srcOp.getMemoryConfigAttr(),
      srcOp.getComputeConfigAttr(), srcOp.getProgramConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
