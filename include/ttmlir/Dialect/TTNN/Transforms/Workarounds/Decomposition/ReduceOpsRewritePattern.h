// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Extracts reduce dimensions' values from the dimArg attribute. In case when
// dimArg is not specified, returns empty vector.
llvm::SmallVector<int64_t>
getReduceDims(const std::optional<mlir::ArrayAttr> &dimArg);

// Calculates the shape of the new Reduce op created in the workaround, based
// on the input shape and reducing dimensions.
llvm::SmallVector<int64_t>
calculateNewReduceShape(RankedTensorType inputType,
                        const std::optional<mlir::ArrayAttr> &dimArg);

// This workaround addresses the next Metal issue:
// https://github.com/tenstorrent/tt-metal/issues/13361
//
// TODO(mrakita): Remove this workaround once these Metal issues are fixed
// (tracked by https://github.com/tenstorrent/tt-mlir/issues/1624).
//
template <typename ReduceOp>
class ReduceOpsKeepDimRewritePattern : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
    if (srcOp.getKeepDim()) {
      return failure();
    }

    RankedTensorType inputType = srcOp.getInput().getType();
    RankedTensorType outputType = srcOp.getResult().getType();

    ReduceOp newReduceOp =
        createReduceOpWithKeepDim(srcOp, rewriter, inputType, outputType);

    // Metal TTNN implementation of Reduce ops doesn't yet support
    // keepDim=false. As a workaround, we convert Reduce ops to combination of
    // Reduce op with keepDim=true + Reshape op to remove the reduce dims so
    // that the rest of the graph is not affected. In case when this is not
    // needed (for example because type converters already promoted rank of the
    // op result) then we avoid adding unnecessary Reshape op.
    if (outputType.getShape().size() < inputType.getShape().size()) {
      replaceOpWithReshapeOp(srcOp, newReduceOp, rewriter, outputType);
    } else {
      rewriter.replaceOp(srcOp, newReduceOp);
    }

    return success();
  }

private:
  ReduceOp createReduceOpWithKeepDim(ReduceOp srcOp, PatternRewriter &rewriter,
                                     RankedTensorType inputType,
                                     RankedTensorType outputType) const {
    llvm::SmallVector<int64_t> outputShapeVec =
        calculateNewReduceShape(inputType, srcOp.getDimArg());

    TTNNLayoutAttr newOutputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(outputType.getEncoding())
            .withTensorShape(rewriter.getContext(), outputShapeVec);

    RankedTensorType newOutputType = RankedTensorType::get(
        outputShapeVec, outputType.getElementType(), newOutputLayoutAttr);

    return rewriter.create<ReduceOp>(srcOp.getLoc(), newOutputType,
                                     srcOp.getInput(), true /*keep_dim*/,
                                     srcOp.getDimArg().value_or(nullptr));
  }

  void replaceOpWithReshapeOp(ReduceOp srcOp, ReduceOp newReduceOp,
                              PatternRewriter &rewriter,
                              RankedTensorType outputType) const {
    mlir::ArrayAttr shapeAttr = rewriter.getI32ArrayAttr(
        llvm::SmallVector<int32_t>(outputType.getShape()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttnn::ReshapeOp>(
        srcOp, outputType, newReduceOp, shapeAttr);
  }
};

// This workaround addresses the next Metal issue:
// https://github.com/tenstorrent/tt-metal/issues/16118
//
// TODO(mrakita): Remove this workaround once these Metal issues are fixed
// (tracked by https://github.com/tenstorrent/tt-mlir/issues/1624).
//
template <typename ReduceOp>
class ReduceOpsAllDimsRewritePattern : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
    if (!srcOp.getDimArg() || srcOp.getDimArg()->empty()) {
      return failure();
    }

    llvm::SmallVector<int64_t> reduceDims = getReduceDims(srcOp.getDimArg());
    llvm::SmallSet<int64_t, 4> uniqueReduceDims(reduceDims.begin(),
                                                reduceDims.end());

    // Check if reduce is done over all dimensions of the input tensor.
    if (uniqueReduceDims.size() !=
        srcOp.getInput().getType().getShape().size()) {
      return failure();
    }

    // In case when reduce is done over all dimensions of the input we need to
    // unset the dimensions attribute, because Metal supports reduce over all
    // dimensions for any tensor rank when reduce dimensions are not specified,
    // but it doesn't support reduce for tensors with rank larger than 2 when
    // reduce dimensions are specified.
    rewriter.replaceOpWithNewOp<ReduceOp>(srcOp, srcOp.getResult().getType(),
                                          srcOp.getInput(), srcOp.getKeepDim(),
                                          nullptr);

    return success();
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H
