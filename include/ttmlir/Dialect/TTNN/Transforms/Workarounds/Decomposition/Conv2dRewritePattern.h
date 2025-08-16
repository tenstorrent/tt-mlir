// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

template <typename ConvOp>
mlir::Value moveToHostRowMajorIfNeeded(ConvOp op,
                                       mlir::TypedValue<RankedTensorType> value,
                                       mlir::PatternRewriter &rewriter,
                                       llvm::StringRef locSuffix) {
  mlir::RankedTensorType valueType =
      mlir::cast<mlir::RankedTensorType>(value.getType());
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

  ToLayoutOp layoutOp = utils::createToLayoutOp(
      op, value, rewriter, Layout::RowMajor, BufferType::SystemMemory,
      /*targetTensorMemoryLayout=*/std::nullopt, desiredDataType, locSuffix);
  return layoutOp.getResult();
}

template <typename ConvOp>
mlir::RankedTensorType rewriteOutputToTile(ConvOp op) {
  mlir::RankedTensorType outputType =
      mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());
  if (layoutAttr.isTiled()) {
    return nullptr;
  }

  // Create a new layout with the tile layout.
  return utils::RankedTensorTypeFactory::create(outputType, Layout::Tile);
}

// Conv2d and Conv2dTranspose currently have couple limitations:
// 1. Weight need to be on host in row_major.
// 2. Bias need to be on host in row_major if present.
// 3. Output of conv2d will be tile layout.
//
// This pattern rewrites the Conv2d/Conv2dTranspose operation to ensure
// that these conditions are met.
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
    mlir::RankedTensorType outputType = rewriteOutputToTile(srcOp);

    // No need to rewrite if no move to host is required.
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
