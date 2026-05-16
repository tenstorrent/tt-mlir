// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNMEMORYMANAGEMENT
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

struct DimGroup {
  int64_t inStart, inCount;
  int64_t outStart, outCount;
};

// Bidirectional dimension mapping that handles merges, splits, and
// merge+split combinations. Each group maps a range of consecutive input
// dims to a range of consecutive output dims with equal products.
// Inserted size-1 output dims (with no corresponding input dim) are
// represented as groups with inCount=0.
static bool computeDimGroupMapping(ArrayRef<int64_t> inputShape,
                                   ArrayRef<int64_t> outputShape,
                                   SmallVector<DimGroup> &groups) {
  groups.clear();
  int64_t inIdx = 0, outIdx = 0;
  int64_t inRank = inputShape.size(), outRank = outputShape.size();

  while (inIdx < inRank || outIdx < outRank) {
    if (outIdx < outRank && outputShape[outIdx] == 1 &&
        (inIdx >= inRank || inputShape[inIdx] != 1)) {
      groups.push_back({inIdx, 0, outIdx, 1});
      ++outIdx;
      continue;
    }

    if (inIdx >= inRank || outIdx >= outRank) {
      return false;
    }

    int64_t inStart = inIdx, outStart = outIdx;
    int64_t inProd = inputShape[inIdx++];
    int64_t outProd = outputShape[outIdx++];

    while (inProd != outProd) {
      if (inProd < outProd) {
        if (inIdx >= inRank) {
          return false;
        }
        inProd *= inputShape[inIdx++];
      } else {
        if (outIdx >= outRank) {
          return false;
        }
        outProd *= outputShape[outIdx++];
      }
    }
    groups.push_back({inStart, inIdx - inStart, outStart, outIdx - outStart});
  }
  return inIdx == inRank && outIdx == outRank;
}

static bool hasNonTileAlignedInnerDims(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  constexpr int64_t tileH = ttcore::TileType::getDefaultShape()[0];
  constexpr int64_t tileW = ttcore::TileType::getDefaultShape()[1];
  if (rank < 2) {
    return false;
  }
  return (shape[rank - 1] % tileW != 0) || (shape[rank - 2] % tileH != 0);
}

static bool hasOnlySingletonDimsBetween(ArrayRef<int64_t> shape, int64_t src,
                                        int64_t tgt) {
  assert(src < tgt && "expected source dim to be before target dim");
  return !llvm::any_of(llvm::seq<int64_t>(src + 1, tgt),
                       [&](int64_t dim) { return shape[dim] != 1; });
}

class PropagateSliceThroughPermute
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // This pattern propagates a slice through a permute.
  // For example:
  // permute(permutation=[3, 2, 0, 1]) [32, 1, 2, 4] -> [4, 2, 32, 1]
  // slice(dim=2, begin=0, end=16, step=1) [4, 2, 32, 1] -> [4, 2, 16, 1]
  // The result is equivalent to:
  // slice(dim=0, begin=0, end=16, step=1) [32, 1, 2, 4] -> [16, 1, 2, 4]
  // permute(permutation=[3, 2, 0, 1]) [16, 1, 2, 4] -> [4, 2, 16, 1]

  LogicalResult matchAndRewrite(ttnn::SliceStaticOp op,
                                PatternRewriter &rewriter) const override {
    auto permuteOp = op.getInput().getDefiningOp<ttnn::PermuteOp>();
    if (!permuteOp || !allUsersAreSlice(permuteOp)) {
      return failure();
    }

    // Slice can always be propagated through Permute.
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
        /*pad_value=*/permuteOp.getPadValue());

    rewriter.replaceOp(op, newPermuteOp.getResult());
    return success();
  }
};

class PropagateSliceThroughReshape
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // This pattern propagates a slice through a reshape when the sliced
  // dimensions map 1:1 across the reshape groups.
  // For example:
  // reshape [1, 8, 16, 1] -> [1, 8, 16]
  // slice(dim=1, begin=2, end=6, step=1) [1, 8, 16] -> [1, 4, 16]
  // The result is equivalent to:
  // slice(dim=1, begin=2, end=6, step=1) [1, 8, 16, 1] -> [1, 4, 16, 1]
  // reshape [1, 4, 16, 1] -> [1, 4, 16]

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

    // Reshape cannot be propagated through if the input and output shapes
    // are not compatible.
    SmallVector<DimGroup> groups;
    if (!computeDimGroupMapping(inputShape, outputShape, groups)) {
      return failure();
    }

    SmallVector<int32_t> begins = toI32Vec(op.getBegins());
    SmallVector<int32_t> ends = toI32Vec(op.getEnds());
    SmallVector<int32_t> step = toI32Vec(op.getStep());

    // Start with no slicing (full extent on every input dim).
    int64_t inputRank = inputShape.size();
    SmallVector<int32_t> newBegins(inputRank, 0);
    SmallVector<int32_t> newEnds(inputShape.begin(), inputShape.end());
    SmallVector<int32_t> newStep(inputRank, 1);
    SmallVector<int64_t> slicedInputShape(inputShape.begin(), inputShape.end());

    for (auto &group : groups) {
      if (group.inCount == 0) {
        continue;
      }

      bool anyPartial = false;
      for (int64_t d = group.outStart; d < group.outStart + group.outCount;
           ++d) {
        int64_t sliceSize = llvm::divideCeil(ends[d] - begins[d], step[d]);
        if (sliceSize != outputShape[d]) {
          anyPartial = true;
          break;
        }
      }

      if (!anyPartial) {
        continue;
      }
      // Cannot propagate partial slice through a multi-dimension group (merge
      // or split).
      if (group.inCount != 1 || group.outCount != 1) {
        return failure();
      }

      int64_t sliceSize = llvm::divideCeil(
          ends[group.outStart] - begins[group.outStart], step[group.outStart]);
      newBegins[group.inStart] = begins[group.outStart];
      newEnds[group.inStart] = ends[group.outStart];
      newStep[group.inStart] = step[group.outStart];
      slicedInputShape[group.inStart] = sliceSize;
    }

    RankedTensorType slicedOutputType = op.getResult().getType();
    ArrayRef<int64_t> slicedOutputShape = slicedOutputType.getShape();

    RankedTensorType slicedInputType =
        utils::RankedTensorTypeFactory::create(inputType, slicedInputShape);

    auto newSliceOp = rewriter.create<ttnn::SliceStaticOp>(
        op.getLoc(), slicedInputType, reshapeOp.getInput(),
        rewriter.getI32ArrayAttr(newBegins), rewriter.getI32ArrayAttr(newEnds),
        rewriter.getI32ArrayAttr(newStep));

    SmallVector<int32_t> outShape32(slicedOutputShape.begin(),
                                    slicedOutputShape.end());
    auto newReshapeOp = rewriter.create<ttnn::ReshapeOp>(
        op.getLoc(), slicedOutputType, newSliceOp.getResult(),
        rewriter.getI32ArrayAttr(outShape32));

    rewriter.replaceOp(op, newReshapeOp.getResult());
    return success();
  }
};

class PropagateSliceThroughRepeat
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // This pattern propagates a slice through a repeat if the slice does not
  // cut through any repeated dimension.
  // For example:
  // repeat(repeat_dims=[1, 2, 1]) [1, 16, 32] -> [1, 32, 32]
  // slice(dim=2, begin=0, end=16, step=1) [1, 32, 32] -> [1, 32, 16]
  // The result is equivalent to:
  // slice(dim=2, begin=0, end=16, step=1) [1, 16, 32] -> [1, 16, 16]
  // repeat(repeat_dims=[1, 2, 1]) [1, 16, 16] -> [1, 32, 16]

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
      int64_t sliceSize = llvm::divideCeil(ends[d] - begins[d], step[d]);
      if (repeatDims[d] != 1) {
        if (sliceSize != repeatOutputShape[d]) {
          // If a repeat is happening on a dimension that is being sliced,
          // propagation is not possible.
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
        op.getLoc(), newRepeatType, newSliceOp.getResult(), newRepeatDimsAttr);

    rewriter.replaceOp(op, newRepeatOp.getResult());
    return success();
  }
};

template <typename EltwiseOpTy>
class PropagateSliceThroughEltwise
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern<ttnn::SliceStaticOp>::OpRewritePattern;

  // This pattern propagates a slice through an elementwise op by slicing each
  // operand in a broadcast-aware way.
  // For example:
  // add([64, 64], [1, 64]) -> [64, 64]
  // slice(dim=0, begin=16, end=32, step=1) [64, 64] -> [16, 64]
  // The result is equivalent to:
  // slice(dim=0, begin=16, end=32, step=1) [64, 64] -> [16, 64]
  // keep [1, 64] unchanged (broadcast dim)
  // add([16, 64], [1, 64]) -> [16, 64]

  LogicalResult matchAndRewrite(ttnn::SliceStaticOp op,
                                PatternRewriter &rewriter) const override {
    auto eltwiseOp = op.getInput().getDefiningOp<EltwiseOpTy>();
    if (!eltwiseOp || !allUsersAreSlice(eltwiseOp)) {
      return failure();
    }

    auto outputType = cast<RankedTensorType>(eltwiseOp->getResult(0).getType());
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t rank = outputShape.size();

    SmallVector<int32_t> begins = toI32Vec(op.getBegins());
    SmallVector<int32_t> ends = toI32Vec(op.getEnds());
    SmallVector<int32_t> step = toI32Vec(op.getStep());

    struct SliceInfo {
      SmallVector<int32_t> begins, ends, step;
      SmallVector<int64_t> slicedShape;
      bool needSlice = false;
    };
    SmallVector<SliceInfo, 0> infos;

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
      infos.push_back(std::move(info));
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

  // This pattern relocates repeat factors from outer dims to inner dims
  // (under strict safety checks) to improve tile alignment before reshape.
  // For example:
  // repeat(repeat_dims=[8, 1, 1, 1]) [1, 1, 1, 64] -> [8, 1, 1, 64]
  // reshape [8, 1, 1, 64] -> [1, 1, 1, 512]
  // The result may be rewritten to:
  // repeat(repeat_dims=[1, 1, 1, 8]) [1, 1, 1, 64] -> [1, 1, 1, 512]
  // reshape [1, 1, 1, 512] -> [1, 1, 1, 512]

  LogicalResult matchAndRewrite(ttnn::RepeatOp op,
                                PatternRewriter &rewriter) const override {
    // This pattern realigns repeat to make inner dims tile-aligned.
    // Only fire if the repeat feeds only into reshapes.
    if (!llvm::all_of(op->getUsers(), [](Operation *user) {
          return isa<ttnn::ReshapeOp>(user);
        })) {
      return failure();
    }

    RankedTensorType inputType = op.getInput().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    ShapeAttr repeatDimsAttr = op.getRepeatDims();
    SmallVector<int64_t> dims(repeatDimsAttr.getShape().begin(),
                              repeatDimsAttr.getShape().end());

    ArrayRef<int64_t> outputShape = op.getResult().getType().getShape();

    // Check if inner dims are already tile-aligned — nothing to do.
    if (!hasNonTileAlignedInnerDims(outputShape)) {
      return failure();
    }

    // Try to relocate outer-dim repeats to inner dims.
    // This is only safe when all dimensions between source and target are
    // singleton, which preserves linearized element ordering.
    SmallVector<int64_t> adjustedDims(dims);
    for (int64_t src = 0; src < rank - 2; ++src) {
      if (adjustedDims[src] <= 1 || inputShape[src] != 1) {
        continue;
      }
      for (int64_t tgt = rank - 2; tgt < rank; ++tgt) {
        if (inputShape[tgt] != 1 || adjustedDims[tgt] != 1) {
          continue;
        }
        if (!hasOnlySingletonDimsBetween(inputShape, src, tgt)) {
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
        op.getLoc(), newRepeatType, op.getInput(), newRepeatDimsAttr);

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

  // This pattern pulls a reshape before an elementwise op by reshaping each
  // operand to the final target space, then rebuilding the eltwise op.
  // This is only done when the elementwise tiled volume is greater than the
  // reshape tiled volume.
  // For example: add([1024, 1, 1, 1], [1024, 1, 1, 1]) -> [1024, 1, 1, 1]
  // reshape [1024, 1, 1, 1] -> [1, 1, 32, 32]
  // The result is equivalent to:
  // reshape [1024, 1, 1, 1] -> [1, 1, 32, 32]
  // reshape [1024, 1, 1, 1] -> [1, 1, 32, 32]
  // add([1, 1, 32, 32], [1, 1, 32, 32]) -> [1, 1, 32, 32]

  LogicalResult matchAndRewrite(ttnn::ReshapeOp op,
                                PatternRewriter &rewriter) const override {

    auto eltwiseOp = op.getInput().getDefiningOp<EltwiseOpTy>();
    if (!eltwiseOp || !eltwiseOp->hasOneUse()) {
      return failure();
    }

    RankedTensorType finalType = op.getResult().getType();
    ArrayRef<int64_t> finalShape = finalType.getShape();
    auto paddedShape = ttnn::utils::getTilePaddedShape(finalShape);

    int64_t tiledVolume = ttmlir::utils::volume(llvm::ArrayRef(paddedShape));

    RankedTensorType eltwiseType =
        cast<RankedTensorType>(eltwiseOp->getResult(0).getType());
    ArrayRef<int64_t> eltwiseShape = eltwiseType.getShape();
    auto paddedEltwiseShape = ttnn::utils::getTilePaddedShape(eltwiseShape);
    int64_t tiledEltwiseVolume =
        ttmlir::utils::volume(llvm::ArrayRef(paddedEltwiseShape));

    // If the elementwise is already in target space, nothing to do.
    if (tiledVolume >= tiledEltwiseVolume) {
      return failure();
    }

    SmallVector<DimGroup> groups;
    if (!computeDimGroupMapping(eltwiseShape, finalShape, groups)) {
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
      for (auto &group : groups) {
        if (group.inCount == 0) {
          // Inserted dim — operand gets size 1 (broadcast).
          operandTargetShape.push_back(1);
          continue;
        }

        auto operandSub = operandShape.slice(group.inStart, group.inCount);
        int64_t operandProduct =
            std::accumulate(operandSub.begin(), operandSub.end(), 1LL,
                            std::multiplies<int64_t>());

        auto finalSub = finalShape.slice(group.outStart, group.outCount);
        int64_t finalProduct = std::accumulate(finalSub.begin(), finalSub.end(),
                                               1LL, std::multiplies<int64_t>());

        if (operandProduct == finalProduct) {
          operandTargetShape.append(finalSub.begin(), finalSub.end());
        } else if (operandProduct == 1) {
          operandTargetShape.append(group.outCount, 1);
        } else {
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
          rewriter.getI32ArrayAttr(targetShape32));
      newOperands.push_back(newReshape.getResult());
    }

    // Create new eltwise in target space.
    Operation *newEltwise = rewriter.clone(*eltwiseOp);
    for (uint32_t i = 0; i < newOperands.size(); ++i) {
      newEltwise->setOperand(i, newOperands[i]);
    }
    newEltwise->getResult(0).setType(finalType);

    // Replace the reshape — the reshape's users
    // get the new eltwise's output directly.
    rewriter.replaceOp(op, newEltwise->getResult(0));
    return success();
  }
};

class TTNNMemoryManagement
    : public impl::TTNNMemoryManagementBase<TTNNMemoryManagement> {
public:
  using impl::TTNNMemoryManagementBase<
      TTNNMemoryManagement>::TTNNMemoryManagementBase;

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
