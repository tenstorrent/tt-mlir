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
#include <cmath>

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

// Metal TTNN implementation of Reduce ops doesn't yet support keepDim=false
// for every case. As a workaround, we convert Reduce ops to combination of
// Reduce op with keepDim=true + Reshape op to remove the reduce dims for
// unsupported use cases, so that the rest of the graph is not affected.

template <typename ReduceOp, bool keepDimUnsupported>
class ReduceOpsKeepDimRewritePattern : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
    if (!isWorkaroundRequired(srcOp)) {
      return failure();
    }

    RankedTensorType inputType = srcOp.getInput().getType();
    RankedTensorType outputType = srcOp.getResult().getType();

    ReduceOp newReduceOp =
        createReduceOpWithKeepDim(srcOp, rewriter, inputType, outputType);

    // In case when the output tensor has the same rank as the input tensor, we
    // avoid creating unnecessary Reshape op. This can happen when we are
    // reducing 1D tensor to scalar, since TTIR doesn't support scalars we
    // promote them to 1D tensors, so then we end up reducing 1D tensor to 1D
    // tensor and the reshape is not necessary.
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
            .withTensorShape(outputShapeVec);

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
        srcOp, outputType, newReduceOp, shapeAttr, /* memory_config */ nullptr);
  }

  // Determine if the workaround is required.
  bool isWorkaroundRequired(ReduceOp srcOp) const {
    // Workaround is necessary only for keepDim==false.
    if (srcOp.getKeepDim()) {
      return false;
    }

    // If ttnn reduce op doesn't support the 'keepDim' parameter then it keeps
    // the reduced dimension, so we have to apply the workaround.
    if (keepDimUnsupported) {
      return true;
    }

    // Otherwise we have to apply the workaround only in case of full tensor
    // reduction, because metal still has a bug with keepDim==false when all
    // dimensions are being reduced.
    if (!srcOp.getDimArg() || srcOp.getDimArg()->empty() ||
        srcOp.getDimArg()->size() ==
            srcOp.getInput().getType().getShape().size()) {
      return true;
    }

    return false;
  }
};

template <typename ReduceOp>
class ReduceOpsPadInputRewritePattern : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp srcOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(srcOp.getInput().getType());
    auto reductionDims = getReduceDims(srcOp.getDimArg());
    if (reductionDims.size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "only one dim is supported"); // TODO: Expand this to support
                                        // multiple dims. This is just for
                                        // initial implementation only.
    }
    auto reductionDim = reductionDims[0];

    auto shape = inputType.getShape();
    auto paddingArray = llvm::SmallVector<int32_t>();
    paddingArray.resize(2 * shape.size(), 0);

    auto paddingDimSize = shape[reductionDim];
    auto desiredDimSize = (paddingDimSize + 31) / 32 * 32;
    auto paddingSize = desiredDimSize - paddingDimSize;
    if (paddingSize <= 0) {
      return rewriter.notifyMatchFailure(srcOp, "padding size is negative");
    }
    paddingArray[2 * reductionDim + 1] = paddingSize;

    float paddingValue;
    if constexpr (std::is_same_v<ReduceOp, mlir::tt::ttnn::MaxOp>) {
      paddingValue = std::numeric_limits<float>::lowest();
    } else if constexpr (std::is_same_v<ReduceOp, mlir::tt::ttnn::MinOp>) {
      paddingValue = std::numeric_limits<float>::max();
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported reduce op type");
    }

    auto newShape = llvm::SmallVector<int64_t>(shape.begin(), shape.end());
    newShape[reductionDim] = desiredDimSize;
    auto resultType = RankedTensorType::get(
        newShape, inputType.getElementType(), inputType.getEncoding());

    auto PadOp = rewriter.create<mlir::tt::ttnn::PadOp>(
        srcOp.getLoc(), resultType, srcOp.getInput(),
        rewriter.getDenseI32ArrayAttr(paddingArray),
        rewriter.getFloatAttr(rewriter.getF32Type(), APFloat(paddingValue)),
        /* use_multicore */ rewriter.getBoolAttr(true),
        /* memory_config*/ nullptr);

    rewriter.replaceOpWithNewOp<ReduceOp>(
        srcOp, srcOp.getResult().getType(), PadOp.getResult(),
        rewriter.getBoolAttr(srcOp.getKeepDim()),
        srcOp.getDimArg().value_or(nullptr));

    return success();
  }
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCEOPSREWRITEPATTERN_H
