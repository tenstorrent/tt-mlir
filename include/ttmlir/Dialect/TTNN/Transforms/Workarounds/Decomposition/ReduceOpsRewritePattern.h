// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

std::vector<int64_t>
calculateNewReduceShape(const std::optional<mlir::ArrayAttr> &dimArg,
                        const RankedTensorType &inputType);

template <typename ReduceOp>
class ReduceOpsRewritePattern : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
    if (srcOp.getKeepDim()) {
      return failure();
    }

    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(srcOp.getInput().getType());
    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult().getType());

    ReduceOp newReduceOp =
        createReduceOpWithKeepDim(srcOp, rewriter, inputType, outputType);

    if (outputType.getShape().size() < inputType.getShape().size()) {
      createReshapeOp(srcOp, newReduceOp, rewriter, outputType);
    } else {
      rewriter.replaceOp(srcOp, newReduceOp);
    }

    return success();
  }

private:
  ReduceOp createReduceOpWithKeepDim(ReduceOp &srcOp, PatternRewriter &rewriter,
                                     const RankedTensorType &inputType,
                                     const RankedTensorType &outputType) const {
    std::vector<int64_t> outputShapeVec =
        calculateNewReduceShape(srcOp.getDimArg(), inputType);

    RankedTensorType newOutputType = RankedTensorType::get(
        llvm::ArrayRef<int64_t>(outputShapeVec), outputType.getElementType(),
        outputType.getEncoding());

    return rewriter.create<ReduceOp>(srcOp.getLoc(), newOutputType,
                                     srcOp.getInput(), true /*keep_dim*/,
                                     srcOp.getDimArg().value_or(nullptr));
  }

  void createReshapeOp(ReduceOp &srcOp, ReduceOp &newReduceOp,
                       PatternRewriter &rewriter,
                       RankedTensorType &outputType) const {
    mlir::ArrayAttr shapeAttr = rewriter.getI32ArrayAttr(std::vector<int32_t>(
        outputType.getShape().begin(), outputType.getShape().end()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttnn::ReshapeOp>(
        srcOp, outputType, newReduceOp, shapeAttr);
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H
