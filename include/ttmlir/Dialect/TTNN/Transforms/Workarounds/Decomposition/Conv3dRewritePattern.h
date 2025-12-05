// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

mlir::Value convertToLayoutIfNeeded(Conv3dOp op,
                                    mlir::TypedValue<RankedTensorType> tensor,
                                    Layout targetLayout,
                                    mlir::PatternRewriter &rewriter,
                                    llvm::StringRef suffix = "") {
  mlir::RankedTensorType tensorType = tensor.getType();
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  if (layoutAttr.getLayout() == targetLayout) {
    return nullptr;
  }

  return utils::createToLayoutOp(
      op, tensor, rewriter, targetLayout, layoutAttr.getBufferType(),
      layoutAttr.getMemLayout(), layoutAttr.getDataType(), suffix);
}

// Conv3d has 3 layout constraints:
// 1. Input must be in ROW_MAJOR layout.
// 2. Weight must be in TILE layout.
// 3. Bias must be in TILE layout.
class Conv3dRewritePattern : public mlir::OpRewritePattern<Conv3dOp> {
public:
  using mlir::OpRewritePattern<Conv3dOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Conv3dOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value input = convertToLayoutIfNeeded(
        srcOp, srcOp.getInput(), Layout::RowMajor, rewriter, "_input");

    mlir::Value weight = convertToLayoutIfNeeded(
        srcOp, srcOp.getWeight(), Layout::Tile, rewriter, "_weight");

    mlir::Value bias =
        srcOp.getBias()
            ? convertToLayoutIfNeeded(srcOp, srcOp.getBias(), Layout::Tile,
                                      rewriter, "_bias")
            : nullptr;

    // No need to rewrite if nothing needs to be changed
    if (!input && !weight && !bias) {
      return mlir::failure();
    }

    // Modify op in place with converted operands
    rewriter.modifyOpInPlace(srcOp, [&]() {
      if (input) {
        srcOp.getInputMutable().assign(input);
      }
      if (weight) {
        srcOp.getWeightMutable().assign(weight);
      }
      if (bias) {
        srcOp.getBiasMutable().assign(bias);
      }
    });

    return mlir::success();
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DREWRITEPATTERN_H
