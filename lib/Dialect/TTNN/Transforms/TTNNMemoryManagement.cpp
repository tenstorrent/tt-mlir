// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

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

static bool isFullSliceForDim(int32_t begin, int32_t end, int32_t step,
                              int64_t dimSize) {
  return begin == 0 && end == dimSize && step == 1;
}

static int64_t getTiledVolume(ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> paddedShape =
      ttnn::utils::getTilePaddedShape(shape);
  return ttmlir::utils::volume(llvm::ArrayRef<int64_t>(paddedShape));
}

static bool isLeadingSliceUser(ttnn::SliceStaticOp op) {
  for (Operation *user : op.getInput().getUsers()) {
    if (user == op.getOperation()) {
      continue;
    }
    if (user->getBlock() != op->getBlock()) {
      return false;
    }
    if (user->isBeforeInBlock(op.getOperation())) {
      return false;
    }
  }
  return true;
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
        /*memory_config=*/nullptr, /*pad_value=*/permuteOp.getPadValue());

    rewriter.replaceOp(op, newPermuteOp.getResult());
    return success();
  }
};

class HoistCommonReshapeAboveSlices
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // This pattern hoists a shared reshape above sibling slices in the special
  // case where each slice selects a single element from one dimension and the
  // following reshape simply drops that singleton dimension.
  // For example:
  // slice([1, 8190, 6, 3072], dim=2, begin=2, end=3) -> [1, 8190, 1, 3072]
  // reshape [1, 8190, 1, 3072] -> [1, 8190, 3072]
  // can be rewritten as:
  // reshape [1, 8190, 6, 3072] -> [1, 49140, 3072]
  // slice(dim=1, begin=2, end=49140, step=6) [1, 49140, 3072] -> [1, 8190,
  // 3072]
  LogicalResult matchAndRewrite(ttnn::SliceStaticOp op,
                                PatternRewriter &rewriter) const override {
    if (!isLeadingSliceUser(op)) {
      return failure();
    }

    RankedTensorType sourceType = op.getInput().getType();
    auto sourceLayout = dyn_cast<TTNNLayoutAttr>(sourceType.getEncoding());
    if (!sourceLayout || !sourceLayout.isTiled()) {
      return failure();
    }

    SmallVector<ttnn::SliceStaticOp> sliceOps;
    SmallVector<ttnn::ReshapeOp> reshapeOps;
    for (Operation *user : op.getInput().getUsers()) {
      auto sliceUser = dyn_cast<ttnn::SliceStaticOp>(user);
      if (!sliceUser || !sliceUser->hasOneUse() ||
          sliceUser->getBlock() != op->getBlock()) {
        return failure();
      }
      auto reshapeUser = dyn_cast<ttnn::ReshapeOp>(*sliceUser->user_begin());
      if (!reshapeUser || reshapeUser->getBlock() != op->getBlock()) {
        return failure();
      }
      sliceOps.push_back(sliceUser);
      reshapeOps.push_back(reshapeUser);
    }

    if (sliceOps.size() < 2) {
      return failure();
    }

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    RankedTensorType reshapedType = reshapeOps.front().getType();
    SmallVector<int32_t> firstBegins = toI32Vec(sliceOps.front().getBegins());
    SmallVector<int32_t> firstEnds = toI32Vec(sliceOps.front().getEnds());
    SmallVector<int32_t> firstStep = toI32Vec(sliceOps.front().getStep());

    SmallVector<int64_t> partialDims;
    for (int64_t dim = 0; dim < static_cast<int64_t>(sourceShape.size());
         ++dim) {
      if (!isFullSliceForDim(firstBegins[dim], firstEnds[dim], firstStep[dim],
                             sourceShape[dim])) {
        partialDims.push_back(dim);
      }
    }

    if (partialDims.size() != 1) {
      return failure();
    }

    int64_t slicedDim = partialDims.front();
    if (slicedDim == 0 || firstStep[slicedDim] != 1 ||
        firstEnds[slicedDim] - firstBegins[slicedDim] != 1) {
      return failure();
    }

    SmallVector<int64_t> expectedShape(sourceShape.begin(), sourceShape.end());
    expectedShape.erase(expectedShape.begin() + slicedDim);
    if (llvm::ArrayRef<int64_t>(expectedShape) != reshapedType.getShape()) {
      return failure();
    }

    SmallVector<int64_t> hoistedShape(sourceShape.begin(), sourceShape.end());
    hoistedShape[slicedDim - 1] *= hoistedShape[slicedDim];
    hoistedShape.erase(hoistedShape.begin() + slicedDim);

    int64_t oldVolume = 0;
    for (ttnn::SliceStaticOp sliceOp : sliceOps) {
      oldVolume += getTiledVolume(sliceOp.getType().getShape());
    }
    int64_t newVolume = getTiledVolume(hoistedShape);
    if (newVolume >= oldVolume) {
      return failure();
    }

    for (size_t i = 0; i < sliceOps.size(); ++i) {
      if (reshapeOps[i].getType() != reshapedType) {
        return failure();
      }

      SmallVector<int32_t> begins = toI32Vec(sliceOps[i].getBegins());
      SmallVector<int32_t> ends = toI32Vec(sliceOps[i].getEnds());
      SmallVector<int32_t> step = toI32Vec(sliceOps[i].getStep());

      for (int64_t dim = 0; dim < static_cast<int64_t>(sourceShape.size());
           ++dim) {
        bool isTargetDim = dim == slicedDim;
        if (isTargetDim) {
          if (step[dim] != 1 || ends[dim] - begins[dim] != 1) {
            return failure();
          }
          continue;
        }
        if (!isFullSliceForDim(begins[dim], ends[dim], step[dim],
                               sourceShape[dim])) {
          return failure();
        }
      }

      if (sliceOps[i].getType().getShape()[slicedDim] != 1) {
        return failure();
      }
    }

    RankedTensorType hoistedType =
        utils::RankedTensorTypeFactory::create(sourceType, hoistedShape);
    SmallVector<int32_t> hoistedShape32(hoistedShape.begin(),
                                        hoistedShape.end());
    auto hoistedReshape = rewriter.create<ttnn::ReshapeOp>(
        op.getLoc(), hoistedType, op.getInput(),
        rewriter.getI32ArrayAttr(hoistedShape32),
        /*memory_config=*/nullptr);

    for (size_t i = 0; i < sliceOps.size(); ++i) {
      SmallVector<int32_t> begins = toI32Vec(sliceOps[i].getBegins());
      SmallVector<int32_t> newBegins;
      SmallVector<int32_t> newEnds;
      SmallVector<int32_t> newStep;

      for (int64_t dim = 0; dim < static_cast<int64_t>(hoistedShape.size());
           ++dim) {
        if (dim < slicedDim - 1) {
          newBegins.push_back(0);
          newEnds.push_back(static_cast<int32_t>(hoistedShape[dim]));
          newStep.push_back(1);
          continue;
        }
        if (dim == slicedDim - 1) {
          newBegins.push_back(begins[slicedDim]);
          newEnds.push_back(static_cast<int32_t>(hoistedShape[dim]));
          newStep.push_back(static_cast<int32_t>(sourceShape[slicedDim]));
          continue;
        }
        newBegins.push_back(0);
        newEnds.push_back(static_cast<int32_t>(hoistedShape[dim]));
        newStep.push_back(1);
      }

      auto newSliceOp = rewriter.create<ttnn::SliceStaticOp>(
          reshapeOps[i].getLoc(), reshapeOps[i].getType(),
          hoistedReshape.getResult(), rewriter.getI32ArrayAttr(newBegins),
          rewriter.getI32ArrayAttr(newEnds), rewriter.getI32ArrayAttr(newStep));
      rewriter.replaceOp(reshapeOps[i], newSliceOp.getResult());
    }

    for (ttnn::SliceStaticOp sliceOp : sliceOps) {
      rewriter.eraseOp(sliceOp);
    }
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
    int64_t tiledVolume = getTiledVolume(finalShape);

    RankedTensorType eltwiseType =
        cast<RankedTensorType>(eltwiseOp->getResult(0).getType());
    ArrayRef<int64_t> eltwiseShape = eltwiseType.getShape();
    int64_t tiledEltwiseVolume = getTiledVolume(eltwiseShape);

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

    // Replace the reshape — the reshape's users
    // get the new eltwise's output directly.
    rewriter.replaceOp(op, newEltwise->getResult(0));
    return success();
  }
};

class PermuteRowMajorAdjusting : public OpRewritePattern<ttnn::PermuteOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::PermuteOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse()) {
      return failure();
    }

    Operation *user = *op->user_begin();
    ttnn::RepeatOp repeatUser = mlir::dyn_cast<ttnn::RepeatOp>(user);
    ttnn::ReshapeOp reshapeUser = mlir::dyn_cast<ttnn::ReshapeOp>(user);
    if (!reshapeUser) {
      if (!repeatUser || !repeatUser->hasOneUse()) {
        return failure();
      }
      reshapeUser = mlir::dyn_cast<ttnn::ReshapeOp>(*repeatUser->user_begin());
    }
    if (!reshapeUser) {
      return failure();
    }

    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto resultLayout =
        mlir::dyn_cast<ttnn::TTNNLayoutAttr>(resultType.getEncoding());
    if (!resultLayout || !resultLayout.isTiled()) {
      return failure();
    }

    auto paddedShape =
        ttnn::utils::getTilePaddedShape(op.getResult().getType().getShape());

    int64_t tiledVolume = ttmlir::utils::volume(llvm::ArrayRef(paddedShape));
    int64_t rmVolume =
        ttmlir::utils::volume(op.getResult().getType().getShape());

    RankedTensorType repeatResultType;
    ttnn::TTNNLayoutAttr repeatResultLayout;
    if (repeatUser) {
      repeatResultType =
          mlir::cast<RankedTensorType>(repeatUser.getResult().getType());
      repeatResultLayout =
          mlir::dyn_cast<ttnn::TTNNLayoutAttr>(repeatResultType.getEncoding());
      if (!repeatResultLayout || !repeatResultLayout.isTiled()) {
        return failure();
      }

      auto repeatPaddedShape =
          ttnn::utils::getTilePaddedShape(repeatResultType.getShape());
      tiledVolume =
          std::max(tiledVolume,
                   ttmlir::utils::volume(llvm::ArrayRef(repeatPaddedShape)));
      rmVolume =
          std::max(rmVolume, ttmlir::utils::volume(repeatResultType.getShape()));
    }

    // If the difference is less than 5MB, nothing to do.
    if (tiledVolume - rmVolume < 5LL * 1024LL * 1024LL) {
      return failure();
    }

    auto rowMajorLayout =
        resultLayout.withLayout(ttnn::Layout::RowMajor, resultType.getShape());
    if (rowMajorLayout == resultLayout) {
      return failure();
    }

    RankedTensorType rowMajorResultType =
        resultType.cloneWithEncoding(rowMajorLayout);

    RankedTensorType rowMajorRepeatResultType;
    if (repeatUser) {
      auto rowMajorRepeatLayout = repeatResultLayout.withLayout(
          ttnn::Layout::RowMajor, repeatResultType.getShape());
      rowMajorRepeatResultType =
          repeatResultType.cloneWithEncoding(rowMajorRepeatLayout);
    }

    auto reshapeResultType =
        mlir::cast<RankedTensorType>(reshapeUser.getResult().getType());
    auto reshapeResultLayout =
        mlir::dyn_cast<ttnn::TTNNLayoutAttr>(reshapeResultType.getEncoding());
    if (!reshapeResultLayout || !reshapeResultLayout.isTiled()) {
      return failure();
    }

    RankedTensorType rowMajorReshapeResultType;
    if (!repeatUser) {
      auto rowMajorReshapeLayout = reshapeResultLayout.withLayout(
          ttnn::Layout::RowMajor, reshapeResultType.getShape());
      rowMajorReshapeResultType =
          reshapeResultType.cloneWithEncoding(rowMajorReshapeLayout);
    }

    Value permuteInput = op.getInput();
    if (!permuteInput.getDefiningOp<ttnn::ToLayoutOp>()) {
      auto inputType = mlir::cast<RankedTensorType>(permuteInput.getType());
      auto inputLayout =
          mlir::dyn_cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());

      if (inputLayout && inputLayout.isTiled()) {
        auto rowMajorInput = utils::createToLayoutOp(
            op, mlir::cast<mlir::TypedValue<RankedTensorType>>(permuteInput),
            rewriter, Layout::RowMajor, inputLayout.getBufferType(),
            inputLayout.getMemLayout(), inputLayout.getDataType(),
            "_input_row_major");
        permuteInput = rowMajorInput.getResult();
      }
    }

    auto newPermuteOp = rewriter.create<ttnn::PermuteOp>(
        op.getLoc(), rowMajorResultType, permuteInput, op.getPermutation(),
        /*memory_config=*/nullptr, op.getPadValue());

    if (repeatUser) {
      auto newRepeatOp = rewriter.create<ttnn::RepeatOp>(
          repeatUser.getLoc(), rowMajorRepeatResultType, newPermuteOp.getResult(),
          repeatUser.getRepeatDims(), /*memory_config=*/nullptr);

      auto restoredRepeatLayout = utils::createToLayoutOp(
          repeatUser,
          mlir::cast<mlir::TypedValue<RankedTensorType>>(newRepeatOp.getResult()),
          rewriter, repeatResultLayout.getLayout(),
          repeatResultLayout.getBufferType(), repeatResultLayout.getMemLayout(),
          repeatResultLayout.getDataType(), "_restore_repeat_layout");

      auto newReshapeOp = rewriter.create<ttnn::ReshapeOp>(
          reshapeUser.getLoc(), reshapeResultType,
          restoredRepeatLayout.getResult(), reshapeUser.getShapeAttr(),
          reshapeUser.getMemoryConfigAttr());
      rewriter.replaceOp(reshapeUser, newReshapeOp.getResult());
      rewriter.eraseOp(repeatUser);
    } else {
      auto newReshapeOp = rewriter.create<ttnn::ReshapeOp>(
          reshapeUser.getLoc(), rowMajorReshapeResultType,
          newPermuteOp.getResult(), reshapeUser.getShapeAttr(),
          /*memory_config=*/nullptr);

      auto restoredLayout = utils::createToLayoutOp(
          reshapeUser,
          mlir::cast<mlir::TypedValue<RankedTensorType>>(
              newReshapeOp.getResult()),
          rewriter, reshapeResultLayout.getLayout(),
          reshapeResultLayout.getBufferType(),
          reshapeResultLayout.getMemLayout(),
          reshapeResultLayout.getDataType(), "_restore_layout");
      rewriter.replaceOp(reshapeUser, restoredLayout.getResult());
    }

    rewriter.eraseOp(op);

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
    patterns
        .add<PropagateSliceThroughPermute, PropagateSliceThroughReshape,
             HoistCommonReshapeAboveSlices, PropagateSliceThroughRepeat,
             PropagateSliceThroughEltwise<ttnn::AddOp>,
             PropagateSliceThroughEltwise<ttnn::MultiplyOp>,
             PropagateSliceThroughEltwise<ttnn::SubtractOp>,
             PropagateSliceThroughEltwise<ttnn::DivideOp>,
             RepeatReshapeAdjusting, ReshapeElementwiseAdjusting<ttnn::AddOp>,
             ReshapeElementwiseAdjusting<ttnn::MultiplyOp>,
             ReshapeElementwiseAdjusting<ttnn::SubtractOp>,
             ReshapeElementwiseAdjusting<ttnn::DivideOp>, PermuteRowMajorAdjusting>(&getContext());

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
