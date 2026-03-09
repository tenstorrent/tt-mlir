// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSLICEPROPAGATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static SmallVector<int32_t> toI32Vec(ArrayAttr attrs) {
  SmallVector<int32_t> result;
  for (auto attr : attrs) {
    result.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }
  return result;
}

static bool allUsersAreSlice(Operation *op) {
  return llvm::all_of(op->getUsers(), [](Operation *user) {
    return isa<ttnn::SliceStaticOp>(user);
  });
}

// Compute dimension mapping from inputShape to outputShape for a reshape.
// Each output dim maps to zero or more consecutive input dims whose product
// equals the output dim size.  A count of 0 means the output dim is an
// "inserted" size-1 dim with no corresponding input dim.
// Returns false if the reshape cannot be expressed this way.
static bool
computeDimMapping(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
                  SmallVector<std::pair<int64_t, int64_t>> &mapping) {
  mapping.clear();
  int64_t inIdx = 0;
  int64_t inputRank = inputShape.size();

  for (int64_t outIdx = 0; outIdx < static_cast<int64_t>(outputShape.size());
       ++outIdx) {
    int64_t target = outputShape[outIdx];

    // Size-1 output dim: either matches a size-1 input dim or is inserted.
    if (target == 1) {
      if (inIdx < inputRank && inputShape[inIdx] == 1) {
        mapping.push_back({inIdx, 1});
        ++inIdx;
      } else {
        mapping.push_back({inIdx, 0}); // inserted dim
      }
      continue;
    }

    int64_t product = 1;
    int64_t startIn = inIdx;
    while (inIdx < inputRank && product < target) {
      product *= inputShape[inIdx];
      ++inIdx;
    }
    if (product != target) {
      return false;
    }
    mapping.push_back({startIn, inIdx - startIn});
  }
  return inIdx == inputRank;
}

static bool hasNonTileAlignedInnerDims(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  if (rank < 2) {
    return false;
  }
  return (shape[rank - 1] % 32 != 0) || (shape[rank - 2] % 32 != 0);
}

class PropagateSliceThroughPermute
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SliceStaticOp op,
                                PatternRewriter &rewriter) const override {
    auto permuteOp = op.getInput().getDefiningOp<ttnn::PermuteOp>();
    if (!permuteOp || !allUsersAreSlice(permuteOp)) {
      return failure();
    }

    ArrayRef<int64_t> permutation = permuteOp.getPermutation();
    SmallVector<int64_t> invPerm =
        ttmlir::utils::inversePermutation(permutation);

    SmallVector<int32_t> begins = toI32Vec(op.getBegins());
    SmallVector<int32_t> ends = toI32Vec(op.getEnds());
    SmallVector<int32_t> step = toI32Vec(op.getStep());

    // Map slice params to pre-permute space using inverse permutation.
    SmallVector<int32_t> newBegins =
        ttmlir::utils::applyPermutation(ArrayRef<int32_t>(begins), invPerm);
    SmallVector<int32_t> newEnds =
        ttmlir::utils::applyPermutation(ArrayRef<int32_t>(ends), invPerm);
    SmallVector<int32_t> newStep =
        ttmlir::utils::applyPermutation(ArrayRef<int32_t>(step), invPerm);

    RankedTensorType prePermuteType = permuteOp.getInput().getType();
    SmallVector<int64_t> slicedPrePermuteShape;
    for (int64_t d = 0; d < static_cast<int64_t>(newBegins.size()); ++d) {
      slicedPrePermuteShape.push_back(
          llvm::divideCeil(newEnds[d] - newBegins[d], newStep[d]));
    }

    RankedTensorType slicedType = utils::RankedTensorTypeFactory::create(
        prePermuteType, slicedPrePermuteShape);

    auto newSliceOp = rewriter.create<ttnn::SliceStaticOp>(
        op.getLoc(), slicedType, permuteOp.getInput(),
        rewriter.getI32ArrayAttr(newBegins), rewriter.getI32ArrayAttr(newEnds),
        rewriter.getI32ArrayAttr(newStep));

    auto newPermuteOp = rewriter.create<ttnn::PermuteOp>(
        op.getLoc(), op.getType(), newSliceOp.getResult(), permutation,
        /*memory_config=*/nullptr, /*pad_value=*/llvm::APFloat(0.0f));

    rewriter.replaceOp(op, newPermuteOp.getResult());
    return success();
  }
};

class PropagateSliceThroughReshape
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SliceStaticOp op,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp = op.getInput().getDefiningOp<ttnn::ReshapeOp>();
    if (!reshapeOp || !allUsersAreSlice(reshapeOp)) {
      return failure();
    }

    RankedTensorType inputType = reshapeOp.getInput().getType();
    RankedTensorType outputType = reshapeOp.getResult().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    SmallVector<std::pair<int64_t, int64_t>> dimMapping;
    if (!computeDimMapping(inputShape, outputShape, dimMapping)) {
      return failure();
    }

    SmallVector<int32_t> begins = toI32Vec(op.getBegins());
    SmallVector<int32_t> ends = toI32Vec(op.getEnds());
    SmallVector<int32_t> step = toI32Vec(op.getStep());

    // Start with "no slicing" (full extent on every input dim).
    int64_t inputRank = inputShape.size();
    SmallVector<int32_t> newBegins(inputRank, 0);
    SmallVector<int32_t> newEnds(inputShape.begin(), inputShape.end());
    SmallVector<int32_t> newStep(inputRank, 1);
    SmallVector<int64_t> slicedInputShape(inputShape.begin(), inputShape.end());

    int64_t outputRank = outputShape.size();
    for (int64_t d = 0; d < outputRank; ++d) {
      int64_t sliceSize = llvm::divideCeil(ends[d] - begins[d], step[d]);

      if (sliceSize == outputShape[d]) {
        continue;
      }

      auto [startIn, count] = dimMapping[d];

      if (count == 0) {
        continue;
      }

      if (count != 1) {
        // Partial slice on a merged dim — can't commute.
        return failure();
      }

      newBegins[startIn] = begins[d];
      newEnds[startIn] = ends[d];
      newStep[startIn] = step[d];
      slicedInputShape[startIn] = sliceSize;
    }

    RankedTensorType slicedOutputType = op.getResult().getType();
    ArrayRef<int64_t> slicedOutputShape = slicedOutputType.getShape();

    RankedTensorType slicedInputType =
        utils::RankedTensorTypeFactory::create(inputType, slicedInputShape);

    auto newSliceOp = rewriter.create<ttnn::SliceStaticOp>(
        op.getLoc(), slicedInputType, reshapeOp.getInput(),
        rewriter.getI32ArrayAttr(newBegins), rewriter.getI32ArrayAttr(newEnds),
        rewriter.getI32ArrayAttr(newStep));

    SmallVector<int32_t> outShape32;
    for (int64_t s : slicedOutputShape) {
      outShape32.push_back(static_cast<int32_t>(s));
    }
    auto newReshapeOp = rewriter.create<ttnn::ReshapeOp>(
        op.getLoc(), slicedOutputType, newSliceOp.getResult(),
        rewriter.getI32ArrayAttr(outShape32),
        /*memory_config=*/nullptr);

    rewriter.replaceOp(op, newReshapeOp.getResult());
    return success();
  }
};

class PropagateSliceThroughRepeat
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SliceStaticOp op,
                                PatternRewriter &rewriter) const override {
    auto repeatOp = op.getInput().getDefiningOp<ttnn::RepeatOp>();
    if (!repeatOp || !allUsersAreSlice(repeatOp)) {
      return failure();
    }

    RankedTensorType repeatInputType = repeatOp.getInput().getType();
    RankedTensorType repeatOutputType = repeatOp.getResult().getType();
    ArrayRef<int64_t> repeatInputShape = repeatInputType.getShape();
    ArrayRef<int64_t> repeatOutputShape = repeatOutputType.getShape();
    int64_t rank = repeatInputShape.size();

    if (rank < 2) {
      return failure();
    }

    ShapeAttr repeatDimsAttr = repeatOp.getRepeatDims();
    ArrayRef<int64_t> repeatDims = repeatDimsAttr.getShape();
    SmallVector<int32_t> begins = toI32Vec(op.getBegins());
    SmallVector<int32_t> ends = toI32Vec(op.getEnds());
    SmallVector<int32_t> step = toI32Vec(op.getStep());
    SmallVector<int64_t> slicedInputShape;

    for (int64_t d = 0; d < rank; ++d) {
      int64_t sliceSize = (ends[d] - begins[d] + step[d] - 1) / step[d];
      if (repeatDims[d] != 1) {
        if (sliceSize != repeatOutputShape[d]) {
          return failure();
        }
        slicedInputShape.push_back(repeatInputShape[d]);
      } else {
        slicedInputShape.push_back(sliceSize);
      }
    }

    SmallVector<int32_t> newBegins, newEnds, newStep;
    for (int64_t d = 0; d < rank; ++d) {
      if (repeatDims[d] != 1) {
        newBegins.push_back(0);
        newEnds.push_back(static_cast<int32_t>(repeatInputShape[d]));
        newStep.push_back(1);
      } else {
        newBegins.push_back(begins[d]);
        newEnds.push_back(ends[d]);
        newStep.push_back(step[d]);
      }
    }

    RankedTensorType slicedType = utils::RankedTensorTypeFactory::create(
        repeatInputType, slicedInputShape);

    auto newSliceOp = rewriter.create<ttnn::SliceStaticOp>(
        op.getLoc(), slicedType, repeatOp.getInput(),
        rewriter.getI32ArrayAttr(newBegins), rewriter.getI32ArrayAttr(newEnds),
        rewriter.getI32ArrayAttr(newStep));

    SmallVector<int64_t> newRepeatOutShape;
    for (int64_t d = 0; d < rank; ++d) {
      newRepeatOutShape.push_back(slicedInputShape[d] * repeatDims[d]);
    }

    RankedTensorType newRepeatType =
        utils::RankedTensorTypeFactory::create(slicedType, newRepeatOutShape);

    auto newRepeatDimsAttr =
        ttnn::ShapeAttr::get(rewriter.getContext(), repeatDims);

    auto newRepeatOp = rewriter.create<ttnn::RepeatOp>(
        op.getLoc(), newRepeatType, newSliceOp.getResult(), newRepeatDimsAttr,
        /*memory_config=*/nullptr);

    rewriter.replaceOp(op, newRepeatOp.getResult());
    return success();
  }
};

template <typename EltwiseOpTy>
class PropagateSliceThroughEltwise
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern<ttnn::SliceStaticOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SliceStaticOp op,
                                PatternRewriter &rewriter) const override {
    auto eltwiseOp = op.getInput().getDefiningOp<EltwiseOpTy>();
    if (!eltwiseOp) {
      return failure();
    }

    auto outputType = cast<RankedTensorType>(eltwiseOp->getResult(0).getType());
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t rank = outputShape.size();

    SmallVector<int32_t> begins = toI32Vec(op.getBegins());
    SmallVector<int32_t> ends = toI32Vec(op.getEnds());
    SmallVector<int32_t> step = toI32Vec(op.getStep());

    // First pass: analyze only.
    struct SliceInfo {
      SmallVector<int32_t> begins, ends, step;
      SmallVector<int64_t> slicedShape;
      bool needSlice = false;
    };
    SmallVector<SliceInfo, 0> infos;
    int64_t numSlicedOperands = 0;

    for (uint32_t i = 0; i < eltwiseOp->getNumOperands(); ++i) {
      Value operand = eltwiseOp->getOperand(i);
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || operandType.getRank() != rank) {
        return failure();
      }
      ArrayRef<int64_t> operandShape = operandType.getShape();

      SliceInfo info;
      for (int64_t d = 0; d < rank; ++d) {
        if (operandShape[d] == 1) {
          info.begins.push_back(0);
          info.ends.push_back(1);
          info.step.push_back(1);
          info.slicedShape.push_back(1);
        } else if (operandShape[d] == outputShape[d]) {
          info.begins.push_back(begins[d]);
          info.ends.push_back(ends[d]);
          info.step.push_back(step[d]);
          info.slicedShape.push_back(
              llvm::divideCeil(ends[d] - begins[d], step[d]));
          if (begins[d] != 0 ||
              ends[d] != static_cast<int32_t>(operandShape[d]) ||
              step[d] != 1) {
            info.needSlice = true;
          }
        } else {
          return failure();
        }
      }
      if (ArrayRef<int64_t>(info.slicedShape) == operandShape) {
        info.needSlice = false;
      }
      if (info.needSlice) {
        numSlicedOperands++;
      }
      infos.push_back(std::move(info));
    }

    // Must slice at least one operand, otherwise nothing changes.
    if (numSlicedOperands == 0) {
      return failure();
    }

    // Second pass: create ops.
    SmallVector<Value> newOperands;
    for (uint32_t i = 0; i < eltwiseOp->getNumOperands(); ++i) {
      if (infos[i].needSlice) {
        auto operandType =
            cast<RankedTensorType>(eltwiseOp->getOperand(i).getType());
        auto slicedType = utils::RankedTensorTypeFactory::create(
            operandType, infos[i].slicedShape);
        auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
            op.getLoc(), slicedType, eltwiseOp->getOperand(i),
            rewriter.getI32ArrayAttr(infos[i].begins),
            rewriter.getI32ArrayAttr(infos[i].ends),
            rewriter.getI32ArrayAttr(infos[i].step));
        newOperands.push_back(sliceOp.getResult());
      } else {
        newOperands.push_back(eltwiseOp->getOperand(i));
      }
    }

    Operation *newEltwiseOp = rewriter.clone(*eltwiseOp);
    for (uint32_t i = 0; i < newOperands.size(); ++i) {
      newEltwiseOp->setOperand(i, newOperands[i]);
    }
    newEltwiseOp->getResult(0).setType(op.getResult().getType());

    rewriter.replaceOp(op, newEltwiseOp->getResult(0));
    return success();
  }
};

class RepeatReshapeAdjusting : public OpRewritePattern<ttnn::RepeatOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::RepeatOp op,
                                PatternRewriter &rewriter) const override {
    // Only fire if the repeat feeds into exactly one reshape.
    if (!op->hasOneUse()) {
      return failure();
    }
    auto reshapeOp = dyn_cast<ttnn::ReshapeOp>(*op->user_begin());
    if (!reshapeOp) {
      return failure();
    }

    RankedTensorType inputType = op.getInput().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    if (!hasNonTileAlignedInnerDims(inputShape)) {
      return failure();
    }

    ShapeAttr repeatDimsAttr = op.getRepeatDims();
    SmallVector<int64_t> dims(repeatDimsAttr.getShape().begin(),
                              repeatDimsAttr.getShape().end());

    // Compute the current output shape.
    SmallVector<int64_t> outputShape;
    for (int64_t d = 0; d < rank; ++d) {
      outputShape.push_back(inputShape[d] * dims[d]);
    }

    // Check if inner dims are already tile-aligned — nothing to do.
    if (outputShape[rank - 1] % 32 == 0 && outputShape[rank - 2] % 32 == 0) {
      return failure();
    }

    // Try to relocate outer-dim repeats to inner dims.
    SmallVector<int64_t> adjustedDims(dims);
    for (int64_t src = 0; src < rank - 2; ++src) {
      if (adjustedDims[src] <= 1 || inputShape[src] != 1) {
        continue;
      }
      for (int64_t tgt = rank - 2; tgt < rank; ++tgt) {
        if (inputShape[tgt] != 1 || adjustedDims[tgt] != 1) {
          continue;
        }
        adjustedDims[tgt] = adjustedDims[src];
        adjustedDims[src] = 1;
        break;
      }
    }

    // Check if anything actually changed.
    if (adjustedDims == SmallVector<int64_t>(dims)) {
      return failure();
    }

    // Build new repeat output shape.
    SmallVector<int64_t> newOutputShape;
    for (int64_t d = 0; d < rank; ++d) {
      newOutputShape.push_back(inputShape[d] * adjustedDims[d]);
    }

    RankedTensorType newRepeatType =
        utils::RankedTensorTypeFactory::create(inputType, newOutputShape);
    auto newRepeatDimsAttr =
        ttnn::ShapeAttr::get(rewriter.getContext(), adjustedDims);

    auto newRepeatOp = rewriter.create<ttnn::RepeatOp>(
        op.getLoc(), newRepeatType, op.getInput(), newRepeatDimsAttr,
        /*memory_config=*/nullptr);

    // The reshape stays the same — it takes the new repeat output
    // (different shape, same element count) and produces the same final shape.
    rewriter.replaceOp(op, newRepeatOp.getResult());
    return success();
  }
};

template <typename EltwiseOpTy>
class ReshapeElementwiseAdjusting : public OpRewritePattern<ttnn::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ReshapeOp op,
                                PatternRewriter &rewriter) const override {

    auto eltwiseOp = op.getInput().getDefiningOp<EltwiseOpTy>();
    if (!eltwiseOp || !eltwiseOp->hasOneUse()) {
      return failure();
    }
    llvm::errs() << "eltwiseOp: " << eltwiseOp->getLoc() << " "
                 << eltwiseOp->getName() << "\n";
    RankedTensorType finalType = op.getResult().getType();
    ArrayRef<int64_t> finalShape = finalType.getShape();
    int64_t finalRank = finalShape.size();
    auto paddedShape = ttnn::utils::getTilePaddedShape(finalShape);

    int64_t tiledVolume = ttmlir::utils::volume(llvm::ArrayRef(paddedShape));

    RankedTensorType eltwiseType =
        cast<RankedTensorType>(eltwiseOp->getResult(0).getType());
    ArrayRef<int64_t> eltwiseShape = eltwiseType.getShape();
    auto paddedEltwiseShape = ttnn::utils::getTilePaddedShape(eltwiseShape);
    int64_t tiledEltwiseVolume =
        ttmlir::utils::volume(llvm::ArrayRef(paddedEltwiseShape));

    if (tiledVolume >= tiledEltwiseVolume) {
      return failure();
    }

    SmallVector<std::pair<int64_t, int64_t>> dimMapping;
    if (!computeDimMapping(eltwiseShape, finalShape, dimMapping)) {
      return failure();
    }

    SmallVector<Value> newOperands;
    for (uint32_t i = 0; i < eltwiseOp->getNumOperands(); ++i) {
      Value operand = eltwiseOp->getOperand(i);
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType) {
        return failure();
      }
      ArrayRef<int64_t> operandShape = operandType.getShape();

      SmallVector<int64_t> operandTargetShape;
      bool valid = true;
      for (int64_t d = 0; d < finalRank; ++d) {
        auto [startIn, count] = dimMapping[d];

        if (count == 0) {
          // Inserted dim — operand gets size 1 (broadcast).
          operandTargetShape.push_back(1);
          continue;
        }

        if (count == 1) {
          // 1:1 mapping — operand keeps its size on this dim.
          operandTargetShape.push_back(operandShape[startIn]);
          continue;
        }

        // Merged dims — compute product of operand sizes on merged dims.
        int64_t operandProduct = 1;
        bool allOnes = true;
        for (int64_t k = startIn; k < startIn + count; ++k) {
          operandProduct *= operandShape[k];
          if (operandShape[k] != 1) {
            allOnes = false;
          }
        }

        if (allOnes) {
          // All broadcast — stays broadcast in target space.
          operandTargetShape.push_back(1);
        } else if (operandProduct == finalShape[d]) {
          // Exact match — operand fills the merged dim entirely.
          operandTargetShape.push_back(finalShape[d]);
        } else if (operandProduct == 1) {
          operandTargetShape.push_back(1);
        } else {
          // Partial: operand has some non-broadcast merged dims but
          // doesn't fill the target dim. Can't broadcast.
          valid = false;
          break;
        }
      }
      if (!valid) {
        return failure();
      }

      // Check if operand is already a reshape — adjust it instead of adding.
      auto existingReshape = operand.getDefiningOp<ttnn::ReshapeOp>();
      Value reshapeInput;
      if (existingReshape && existingReshape->hasOneUse()) {
        // Reuse the input of the existing reshape (skip the intermediate).
        reshapeInput = existingReshape.getInput();
      } else {
        reshapeInput = operand;
      }

      auto reshapeInputType = cast<RankedTensorType>(reshapeInput.getType());

      // Check if a reshape is actually needed.
      if (reshapeInputType.getShape() ==
          llvm::ArrayRef<int64_t>(operandTargetShape)) {
        newOperands.push_back(reshapeInput);
        continue;
      }
      if (ttmlir::utils::volume(reshapeInputType.getShape()) !=
          ttmlir::utils::volume(llvm::ArrayRef(operandTargetShape))) {
        return failure();
      }
      SmallVector<int32_t> targetShape32(operandTargetShape.begin(),
                                         operandTargetShape.end());
      auto newOperandType = utils::RankedTensorTypeFactory::create(
          reshapeInputType, operandTargetShape);
      auto newReshape = rewriter.create<ttnn::ReshapeOp>(
          op.getLoc(), newOperandType, reshapeInput,
          rewriter.getI32ArrayAttr(targetShape32),
          /*memory_config=*/nullptr);
      newOperands.push_back(newReshape.getResult());
    }

    // Create new eltwise in target space.
    Operation *newEltwise = rewriter.clone(*eltwiseOp);
    for (uint32_t i = 0; i < newOperands.size(); ++i) {
      newEltwise->setOperand(i, newOperands[i]);
    }
    newEltwise->getResult(0).setType(finalType);

    // Replace the reshape (not the eltwise) — the reshape's users
    // get the new eltwise's output directly.
    rewriter.replaceOp(op, newEltwise->getResult(0));

    // Original eltwise is now dead (had single use = the reshape we replaced).
    rewriter.eraseOp(eltwiseOp);
    return success();
  }
};

class TTNNSlicePropagation
    : public impl::TTNNSlicePropagationBase<TTNNSlicePropagation> {
public:
  using impl::TTNNSlicePropagationBase<
      TTNNSlicePropagation>::TTNNSlicePropagationBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<
        PropagateSliceThroughPermute, PropagateSliceThroughReshape,
        PropagateSliceThroughRepeat, PropagateSliceThroughEltwise<ttnn::AddOp>,
        PropagateSliceThroughEltwise<ttnn::MultiplyOp>,
        PropagateSliceThroughEltwise<ttnn::SubtractOp>,
        PropagateSliceThroughEltwise<ttnn::DivideOp>, RepeatReshapeAdjusting,
        ReshapeElementwiseAdjusting<ttnn::AddOp>,
        ReshapeElementwiseAdjusting<ttnn::MultiplyOp>,
        ReshapeElementwiseAdjusting<ttnn::SubtractOp>,
        ReshapeElementwiseAdjusting<ttnn::DivideOp>>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
