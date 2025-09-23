// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Inserts a to_layout op if the value is not on host in row major layout.
// If is value type is block type we also need to cast it to bfloat16.
template <typename ConvOp>
mlir::Value moveToHostRowMajorIfNeeded(ConvOp op,
                                       mlir::TypedValue<RankedTensorType> value,
                                       mlir::PatternRewriter &rewriter,
                                       llvm::StringRef locSuffix) {
  mlir::RankedTensorType valueType = value.getType();
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(valueType.getEncoding());
  if (layoutAttr.getBufferType() == BufferType::SystemMemory &&
      !layoutAttr.isTiled()) {
    return nullptr;
  }

  ttcore::DataType currentDataType = layoutAttr.getDataType();
  ttcore::DataType desiredDataType = currentDataType;
  if (currentDataType == ttcore::DataType::BFP_BFloat8) {
    desiredDataType = ttcore::DataType::BFloat16;
  }

  return utils::createToLayoutOp(
      op, value, rewriter, Layout::RowMajor, BufferType::SystemMemory,
      /*targetTensorMemoryLayout=*/nullptr, desiredDataType, locSuffix);
}

mlir::RankedTensorType rewriteOutputToTile(mlir::RankedTensorType resultType) {
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(resultType.getEncoding());
  if (layoutAttr.isTiled()) {
    return nullptr;
  }

  // Clone the output type and change layout to tile.
  return utils::RankedTensorTypeFactory::create(resultType, Layout::Tile);
}

// Conv2d and ConvTranspose2d have 3 constraints:
// 1. Weight need to be on host in row_major.
// 2. Bias need to be on host in row_major if present.
// 3. Output of conv2d will be tile layout.
template <typename ConvOp>
class Conv2dRewritePattern : public mlir::OpRewritePattern<ConvOp> {
public:
  using mlir::OpRewritePattern<ConvOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value weight = moveToHostRowMajorIfNeeded(srcOp, srcOp.getWeight(),
                                                    rewriter, "_weight");
    mlir::Value bias = srcOp.getBias()
                           ? moveToHostRowMajorIfNeeded(srcOp, srcOp.getBias(),
                                                        rewriter, "_bias")
                           : nullptr;
    mlir::RankedTensorType outputType =
        rewriteOutputToTile(srcOp.getResult().getType());

    // No need to rewrite if nothing needs to be changed.
    if (!weight && !bias && !outputType) {
      return mlir::failure();
    }

    rewriter.modifyOpInPlace(srcOp, [&]() {
      if (weight) {
        srcOp.getWeightMutable().assign(weight);
      }
      if (bias) {
        srcOp.getBiasMutable().assign(bias);
      }
      if (outputType) {
        srcOp.getResult().setType(outputType);
      }
    });

    return mlir::success();
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DREWRITEPATTERN_H
