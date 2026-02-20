// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RepeatReshapeEltwiseSliceRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

static bool isElementwiseBinaryOp(Operation *op) {
  return isa<ttnn::AddOp, ttnn::MultiplyOp, ttnn::SubtractOp, ttnn::DivideOp>(
      op);
}

static bool hasNonTileAlignedInnerDims(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  if (rank < 2) {
    return false;
  }
  return (shape[rank - 1] % 32 != 0) || (shape[rank - 2] % 32 != 0);
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

// Move repeat factor from an outer dim to an inner (tile) dim when both the
// source and target dims have size 1 in the sliced shape.  Unlike the version
// in RepeatSliceReshapeRewritePattern, this skips the tile-alignment check
// because a reshape immediately follows the repeat and will fix alignment.
//
// Safe because: if both source and target dims are size 1 in the sliced
// shape, the repeat just broadcasts identical data along that dim.  The
// downstream reshape produces identical flat layout regardless of which
// dim was expanded.
static void relocateRepeatToInnerDims(ArrayRef<int64_t> slicedShape,
                                      SmallVector<int64_t> &dims,
                                      int64_t rank) {
  for (int64_t src = 0; src < rank - 2; ++src) {
    if (dims[src] <= 1 || slicedShape[src] != 1) {
      continue;
    }
    for (int64_t tgt = rank - 2; tgt < rank; ++tgt) {
      if (slicedShape[tgt] != 1 || dims[tgt] != 1) {
        continue;
      }
      dims[tgt] = dims[src];
      dims[src] = 1;
      break;
    }
  }
}

struct EltwiseSliceChain {
  ttnn::SliceStaticOp sliceOp;
  ttnn::ReshapeOp finalReshapeOp;
};

struct EltwiseGroup {
  Operation *eltwiseOp;
  int64_t reshapeOperandIdx;
  int64_t otherOperandIdx;
  Value otherInput;
  SmallVector<EltwiseSliceChain> chains;
};

LogicalResult RepeatReshapeEltwiseSliceRewritePattern::matchAndRewrite(
    ttnn::RepeatOp repeatOp, PatternRewriter &rewriter) const {

  RankedTensorType repeatInputType = repeatOp.getInput().getType();
  RankedTensorType repeatOutputType = repeatOp.getResult().getType();
  ArrayRef<int64_t> repeatInputShape = repeatInputType.getShape();
  ArrayRef<int64_t> repeatOutputShape = repeatOutputType.getShape();
  int64_t repeatRank = repeatInputShape.size();

  if (repeatRank < 2) {
    return failure();
  }

  if (!hasNonTileAlignedInnerDims(repeatOutputShape)) {
    return failure();
  }

  ShapeAttr repeatDimsAttr = repeatOp.getRepeatDims();
  ArrayRef<int64_t> repeatDims = repeatDimsAttr.getShape();

  // --- Step 1: RepeatOp must have exactly one ReshapeOp user. ---
  ttnn::ReshapeOp midReshapeOp = nullptr;
  for (Operation *user : repeatOp.getResult().getUsers()) {
    if (auto reshape = dyn_cast<ttnn::ReshapeOp>(user)) {
      if (midReshapeOp) {
        return failure();
      }
      midReshapeOp = reshape;
    }
  }
  if (!midReshapeOp) {
    return failure();
  }

  ArrayRef<int64_t> midReshapeOutShape =
      midReshapeOp.getResult().getType().getShape();
  int64_t reshapeRank = midReshapeOutShape.size();

  // --- Step 2: Compute dimension mapping (repeat output → reshape output). ---
  SmallVector<std::pair<int64_t, int64_t>> dimMapping;
  if (!computeDimMapping(repeatOutputShape, midReshapeOutShape, dimMapping)) {
    return failure();
  }

  // --- Step 3: Collect eltwise binary op users of the ReshapeOp. ---
  Value reshapeResult = midReshapeOp.getResult();
  SmallVector<EltwiseGroup> eltwiseGroups;

  for (Operation *user : reshapeResult.getUsers()) {
    if (!isElementwiseBinaryOp(user)) {
      continue;
    }

    EltwiseGroup group;
    group.eltwiseOp = user;
    group.reshapeOperandIdx = -1;
    group.otherOperandIdx = -1;

    for (int64_t i = 0; i < static_cast<int64_t>(user->getNumOperands()); ++i) {
      if (user->getOperand(i) == reshapeResult) {
        group.reshapeOperandIdx = i;
      } else if (isa<RankedTensorType>(user->getOperand(i).getType())) {
        group.otherOperandIdx = i;
      }
    }
    if (group.reshapeOperandIdx < 0 || group.otherOperandIdx < 0) {
      continue;
    }

    group.otherInput = user->getOperand(group.otherOperandIdx);
    auto otherType = cast<RankedTensorType>(group.otherInput.getType());
    if (otherType.getRank() != reshapeRank) {
      continue;
    }

    // Collect slice → reshape chains from this eltwise's output.
    Value eltwiseResult = user->getResult(0);
    for (Operation *su : eltwiseResult.getUsers()) {
      if (auto sliceOp = dyn_cast<ttnn::SliceStaticOp>(su)) {
        ttnn::ReshapeOp finalReshape = nullptr;
        bool valid = true;
        for (Operation *ssu : sliceOp.getResult().getUsers()) {
          if (auto r = dyn_cast<ttnn::ReshapeOp>(ssu)) {
            if (finalReshape) {
              valid = false;
              break;
            }
            finalReshape = r;
          } else {
            valid = false;
            break;
          }
        }
        if (valid && finalReshape) {
          group.chains.push_back({sliceOp, finalReshape});
        }
      }
    }

    if (!group.chains.empty()) {
      eltwiseGroups.push_back(std::move(group));
    }
  }

  if (eltwiseGroups.empty()) {
    return failure();
  }

  // --- Step 4: Rewrite each chain in each eltwise group. ---
  auto repeatInput = cast<TypedValue<RankedTensorType>>(repeatOp.getInput());
  bool anyRewritten = false;

  for (auto &group : eltwiseGroups) {
    Operation *eltwiseOp = group.eltwiseOp;
    Value otherInput = group.otherInput;
    auto otherInputType = cast<RankedTensorType>(otherInput.getType());
    ArrayRef<int64_t> otherInputShape = otherInputType.getShape();

    auto eltwiseOutType =
        cast<RankedTensorType>(eltwiseOp->getResult(0).getType());
    ArrayRef<int64_t> eltwiseOutShape = eltwiseOutType.getShape();

    for (auto &chain : group.chains) {
      // Parse slice attributes.
      SmallVector<int32_t> begins, ends, step;
      for (auto attr : chain.sliceOp.getBegins()) {
        begins.push_back(mlir::cast<IntegerAttr>(attr).getInt());
      }
      for (auto attr : chain.sliceOp.getEnds()) {
        ends.push_back(mlir::cast<IntegerAttr>(attr).getInt());
      }
      for (auto attr : chain.sliceOp.getStep()) {
        step.push_back(mlir::cast<IntegerAttr>(attr).getInt());
      }

      bool validSlice = true;
      SmallVector<int64_t> slicedRepeatInputShape(repeatInputShape.begin(),
                                                  repeatInputShape.end());
      SmallVector<int64_t> slicedOtherShape(otherInputShape.begin(),
                                            otherInputShape.end());

      SmallVector<int32_t> rptSliceBegins(repeatRank, 0);
      SmallVector<int32_t> rptSliceEnds;
      SmallVector<int32_t> rptSliceStep(repeatRank, 1);
      for (int64_t d = 0; d < repeatRank; ++d) {
        rptSliceEnds.push_back(static_cast<int32_t>(repeatInputShape[d]));
      }

      SmallVector<int32_t> otherSliceBegins(reshapeRank, 0);
      SmallVector<int32_t> otherSliceEnds;
      SmallVector<int32_t> otherSliceStep(reshapeRank, 1);
      for (int64_t d = 0; d < reshapeRank; ++d) {
        otherSliceEnds.push_back(static_cast<int32_t>(otherInputShape[d]));
      }

      bool needSliceOther = false;

      for (int64_t d = 0; d < reshapeRank; ++d) {
        int64_t sliceSize = (ends[d] - begins[d] + step[d] - 1) / step[d];
        if (sliceSize == eltwiseOutShape[d]) {
          continue;
        }

        auto [startIn, count] = dimMapping[d];

        if (count == 0) {
          if (otherInputShape[d] != 1) {
            otherSliceBegins[d] = begins[d];
            otherSliceEnds[d] = ends[d];
            otherSliceStep[d] = step[d];
            slicedOtherShape[d] = sliceSize;
            needSliceOther = true;
          } else {
            slicedOtherShape[d] = 1;
          }
          continue;
        }

        if (count != 1) {
          validSlice = false;
          break;
        }
        int64_t d5 = startIn;
        if (repeatDims[d5] != 1) {
          validSlice = false;
          break;
        }

        rptSliceBegins[d5] = begins[d];
        rptSliceEnds[d5] = ends[d];
        rptSliceStep[d5] = step[d];
        slicedRepeatInputShape[d5] = sliceSize;

        if (otherInputShape[d] != 1) {
          otherSliceBegins[d] = begins[d];
          otherSliceEnds[d] = ends[d];
          otherSliceStep[d] = step[d];
          slicedOtherShape[d] = sliceSize;
          needSliceOther = true;
        } else {
          slicedOtherShape[d] = 1;
        }
      }

      if (!validSlice) {
        continue;
      }

      SmallVector<int64_t> adjustedRepeatDims(repeatDims.begin(),
                                              repeatDims.end());
      relocateRepeatToInnerDims(slicedRepeatInputShape, adjustedRepeatDims,
                                repeatRank);

      SmallVector<int64_t> newRepeatOutShape;
      for (int64_t d = 0; d < repeatRank; ++d) {
        newRepeatOutShape.push_back(slicedRepeatInputShape[d] *
                                    adjustedRepeatDims[d]);
      }

      RankedTensorType finalType = chain.finalReshapeOp.getResult().getType();
      ArrayRef<int64_t> finalShape = finalType.getShape();
      int64_t finalRank = finalShape.size();

      int64_t finalElems = 1, repeatOutElems = 1;
      for (int64_t s : finalShape) {
        finalElems *= s;
      }
      for (int64_t s : newRepeatOutShape) {
        repeatOutElems *= s;
      }
      if (finalElems != repeatOutElems) {
        continue;
      }

      int64_t otherElems = 1;
      for (int64_t s : slicedOtherShape) {
        otherElems *= s;
      }

      SmallVector<int64_t> otherBroadcastShape(finalRank, 1);
      int64_t remaining = otherElems;
      for (int64_t d = finalRank - 1; d >= 0 && remaining > 1; --d) {
        if (remaining % finalShape[d] == 0) {
          otherBroadcastShape[d] = finalShape[d];
          remaining /= finalShape[d];
        }
      }
      if (remaining != 1) {
        continue;
      }

      // === Emit new ops at the original slice's position. ===
      rewriter.setInsertionPoint(chain.sliceOp);
      Location loc = chain.sliceOp.getLoc();

      // 1. Slice the repeat input.
      RankedTensorType slicedRptType = utils::RankedTensorTypeFactory::create(
          repeatInputType, slicedRepeatInputShape);
      auto newRptSlice = rewriter.create<ttnn::SliceStaticOp>(
          loc, slicedRptType, repeatInput,
          rewriter.getI32ArrayAttr(rptSliceBegins),
          rewriter.getI32ArrayAttr(rptSliceEnds),
          rewriter.getI32ArrayAttr(rptSliceStep));

      // 2. Repeat with relocated dims.
      RankedTensorType newRepeatOutType =
          utils::RankedTensorTypeFactory::create(slicedRptType,
                                                 newRepeatOutShape);
      auto newRepeatDimsAttr =
          ttnn::ShapeAttr::get(rewriter.getContext(), adjustedRepeatDims);
      auto newRepeat = rewriter.create<ttnn::RepeatOp>(
          loc, newRepeatOutType, newRptSlice.getResult(), newRepeatDimsAttr,
          /*memory_config=*/nullptr);

      // 3. Reshape repeat output directly to the final target shape.
      SmallVector<int32_t> finalShape32;
      for (int64_t s : finalShape) {
        finalShape32.push_back(static_cast<int32_t>(s));
      }
      auto newMidReshape = rewriter.create<ttnn::ReshapeOp>(
          loc, finalType, newRepeat.getResult(),
          rewriter.getI32ArrayAttr(finalShape32),
          /*memory_config=*/nullptr);

      // 4. Slice the other eltwise input (if needed), then reshape to
      //    broadcast-compatible shape with the final target.
      Value newOtherInput = otherInput;
      if (needSliceOther) {
        RankedTensorType slicedOtherType =
            utils::RankedTensorTypeFactory::create(otherInputType,
                                                   slicedOtherShape);
        auto newOtherSlice = rewriter.create<ttnn::SliceStaticOp>(
            loc, slicedOtherType, otherInput,
            rewriter.getI32ArrayAttr(otherSliceBegins),
            rewriter.getI32ArrayAttr(otherSliceEnds),
            rewriter.getI32ArrayAttr(otherSliceStep));
        newOtherInput = newOtherSlice.getResult();
      }

      auto curOtherType = cast<RankedTensorType>(newOtherInput.getType());
      ArrayRef<int64_t> curOtherShape = curOtherType.getShape();
      if (curOtherShape != ArrayRef<int64_t>(otherBroadcastShape)) {
        SmallVector<int32_t> otherBroadcast32;
        for (int64_t s : otherBroadcastShape) {
          otherBroadcast32.push_back(static_cast<int32_t>(s));
        }
        RankedTensorType otherBroadcastType =
            utils::RankedTensorTypeFactory::create(curOtherType,
                                                   otherBroadcastShape);
        auto otherReshape = rewriter.create<ttnn::ReshapeOp>(
            loc, otherBroadcastType, newOtherInput,
            rewriter.getI32ArrayAttr(otherBroadcast32),
            /*memory_config=*/nullptr);
        newOtherInput = otherReshape.getResult();
      }

      // 5. Clone the eltwise op with final-shaped operands.
      Operation *newEltwiseOp = rewriter.clone(*eltwiseOp);
      newEltwiseOp->setOperand(group.reshapeOperandIdx,
                               newMidReshape.getResult());
      newEltwiseOp->setOperand(group.otherOperandIdx, newOtherInput);
      newEltwiseOp->getResult(0).setType(finalType);

      rewriter.replaceOp(chain.finalReshapeOp, newEltwiseOp->getResult(0));
      rewriter.eraseOp(chain.sliceOp);
      anyRewritten = true;
    }

    // Clean up this eltwise if all its users are gone.
    if (eltwiseOp->getResult(0).use_empty()) {
      rewriter.eraseOp(eltwiseOp);
    }
  }

  if (!anyRewritten) {
    return failure();
  }

  // Clean up original ops if all users are gone.
  if (midReshapeOp.getResult().use_empty()) {
    rewriter.eraseOp(midReshapeOp);
  }
  if (repeatOp.getResult().use_empty()) {
    rewriter.eraseOp(repeatOp);
  }

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
