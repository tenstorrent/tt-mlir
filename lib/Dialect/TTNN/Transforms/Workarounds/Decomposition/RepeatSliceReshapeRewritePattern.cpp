// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RepeatSliceReshapeRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

static bool hasNonTileAlignedInnerDims(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  if (rank < 2) {
    return false;
  }
  return (shape[rank - 1] % 32 != 0) || (shape[rank - 2] % 32 != 0);
}

// When the sliced shape has size-1 dims, the repeat can be moved from a
// non-tile dim to an inner (tile) dim without changing the memory layout.
// This produces tile-aligned results.
//
// Safe because: if both source and target dims have size 1 in the sliced
// shape, the repeat just produces N copies of the same inner block,
// yielding identical data regardless of which dim is expanded.
static void relocateRepeatToInnerDims(ArrayRef<int64_t> slicedShape,
                                      SmallVector<int64_t> &dims,
                                      int64_t rank) {
  for (int64_t src = 0; src < rank - 2; ++src) {
    if (dims[src] <= 1 || slicedShape[src] != 1) {
      continue;
    }
    // Prefer rank-2 (second-to-last) first, then rank-1.
    for (int64_t tgt = rank - 2; tgt < rank; ++tgt) {
      if (slicedShape[tgt] != 1 || dims[tgt] != 1) {
        continue;
      }
      if ((dims[src] * slicedShape[tgt]) % 32 == 0) {
        dims[tgt] = dims[src];
        dims[src] = 1;
        break;
      }
    }
  }
}

struct SliceReshapeChain {
  ttnn::SliceStaticOp sliceOp;
  ttnn::ReshapeOp reshapeOp;
};

static std::optional<SliceReshapeChain>
tryBuildChain(ttnn::SliceStaticOp sliceOp) {
  SliceReshapeChain chain;
  chain.sliceOp = sliceOp;
  chain.reshapeOp = nullptr;

  for (Operation *user : sliceOp.getResult().getUsers()) {
    if (auto reshape = dyn_cast<ttnn::ReshapeOp>(user)) {
      if (chain.reshapeOp) {
        return std::nullopt;
      }
      chain.reshapeOp = reshape;
    } else {
      return std::nullopt;
    }
  }

  if (!chain.reshapeOp) {
    return std::nullopt;
  }
  return chain;
}

static bool analyzeChain(SliceReshapeChain &chain,
                         ArrayRef<int64_t> repeatInputShape,
                         ArrayRef<int64_t> repeatOutputShape,
                         ArrayRef<int64_t> repeatDims, int64_t rank,
                         SmallVector<int64_t> &slicedInputShape,
                         SmallVector<int64_t> &adjustedRepeatDims,
                         SmallVector<int32_t> &begins,
                         SmallVector<int32_t> &ends,
                         SmallVector<int32_t> &step) {
  begins.clear();
  ends.clear();
  step.clear();
  slicedInputShape.clear();

  for (auto attr : chain.sliceOp.getBegins()) {
    begins.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }
  for (auto attr : chain.sliceOp.getEnds()) {
    ends.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }
  for (auto attr : chain.sliceOp.getStep()) {
    step.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  for (int64_t d = 0; d < rank; ++d) {
    if (repeatDims[d] != 1) {
      int64_t sliceSize = (ends[d] - begins[d] + step[d] - 1) / step[d];
      if (sliceSize != repeatOutputShape[d]) {
        return false;
      }
      slicedInputShape.push_back(repeatInputShape[d]);
    } else {
      int64_t sliceSize = (ends[d] - begins[d] + step[d] - 1) / step[d];
      slicedInputShape.push_back(sliceSize);
    }
  }

  // Start with the original repeat dims, then try to relocate to inner dims.
  adjustedRepeatDims.assign(repeatDims.begin(), repeatDims.end());
  relocateRepeatToInnerDims(slicedInputShape, adjustedRepeatDims, rank);

  // Element-count consistency with the downstream reshape.
  RankedTensorType reshapeOutType = chain.reshapeOp.getResult().getType();
  ArrayRef<int64_t> reshapeOutShape = reshapeOutType.getShape();
  int64_t reshapeElements = 1;
  int64_t newElements = 1;
  for (int64_t s : reshapeOutShape) {
    reshapeElements *= s;
  }
  for (int64_t d = 0; d < rank; ++d) {
    newElements *= slicedInputShape[d] * adjustedRepeatDims[d];
  }
  return reshapeElements == newElements;
}

LogicalResult RepeatSliceReshapeRewritePattern::matchAndRewrite(
    ttnn::RepeatOp repeatOp, PatternRewriter &rewriter) const {

  RankedTensorType repeatInputType = repeatOp.getInput().getType();
  RankedTensorType repeatOutputType = repeatOp.getResult().getType();
  ArrayRef<int64_t> repeatInputShape = repeatInputType.getShape();
  ArrayRef<int64_t> repeatOutputShape = repeatOutputType.getShape();
  int64_t rank = repeatInputShape.size();

  if (rank < 2) {
    return failure();
  }

  if (!hasNonTileAlignedInnerDims(repeatOutputShape)) {
    return failure();
  }

  ShapeAttr repeatDimsAttr = repeatOp.getRepeatDims();
  ArrayRef<int64_t> repeatDims = repeatDimsAttr.getShape();

  if (repeatDims[rank - 1] != 1 || repeatDims[rank - 2] != 1) {
    return failure();
  }

  SmallVector<SliceReshapeChain> validChains;
  bool hasUnrewritableUser = false;

  for (Operation *user : repeatOp.getResult().getUsers()) {
    if (auto sliceOp = dyn_cast<ttnn::SliceStaticOp>(user)) {
      auto chain = tryBuildChain(sliceOp);
      if (chain) {
        validChains.push_back(std::move(*chain));
      } else {
        hasUnrewritableUser = true;
      }
    } else {
      hasUnrewritableUser = true;
    }
  }

  if (validChains.empty()) {
    return failure();
  }

  auto repeatInput = cast<TypedValue<RankedTensorType>>(repeatOp.getInput());
  bool anyRewritten = false;

  for (auto &chain : validChains) {
    SmallVector<int64_t> slicedInputShape;
    SmallVector<int64_t> adjustedRepeatDims;
    SmallVector<int32_t> begins, ends, step;

    if (!analyzeChain(chain, repeatInputShape, repeatOutputShape, repeatDims,
                      rank, slicedInputShape, adjustedRepeatDims, begins, ends,
                      step)) {
      hasUnrewritableUser = true;
      continue;
    }

    // Insert new ops at the original slice's position so the repeat result
    // is created right before its consumer, avoiding all repeat results
    // being alive simultaneously.
    rewriter.setInsertionPoint(chain.sliceOp);
    Location loc = chain.sliceOp.getLoc();

    // 1. Slice on the original (small) input.
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
        loc, slicedType, repeatInput, rewriter.getI32ArrayAttr(newBegins),
        rewriter.getI32ArrayAttr(newEnds), rewriter.getI32ArrayAttr(newStep));

    // 2. Repeat with relocated dims on the sliced tensor.
    SmallVector<int64_t> newRepeatOutShape;
    for (int64_t d = 0; d < rank; ++d) {
      newRepeatOutShape.push_back(slicedInputShape[d] *
                                  adjustedRepeatDims[d]);
    }

    RankedTensorType newRepeatType =
        utils::RankedTensorTypeFactory::create(slicedType, newRepeatOutShape);

    auto newRepeatDimsAttr =
        ttnn::ShapeAttr::get(rewriter.getContext(), adjustedRepeatDims);

    auto newRepeatOp = rewriter.create<ttnn::RepeatOp>(
        loc, newRepeatType, newSliceOp.getResult(), newRepeatDimsAttr,
        /*memory_config=*/nullptr);

    // 3. Reshape to the final output shape.
    RankedTensorType reshapeOutType = chain.reshapeOp.getResult().getType();
    ArrayRef<int64_t> reshapeOutShape = reshapeOutType.getShape();

    SmallVector<int32_t> reshapeShape32;
    for (int64_t s : reshapeOutShape) {
      reshapeShape32.push_back(static_cast<int32_t>(s));
    }

    auto newReshapeOp = rewriter.create<ttnn::ReshapeOp>(
        chain.reshapeOp.getLoc(), reshapeOutType, newRepeatOp.getResult(),
        rewriter.getI32ArrayAttr(reshapeShape32),
        /*memory_config=*/nullptr);

    rewriter.replaceOp(chain.reshapeOp, newReshapeOp.getResult());
    rewriter.eraseOp(chain.sliceOp);
    anyRewritten = true;
  }

  if (!anyRewritten) {
    return failure();
  }

  if (!hasUnrewritableUser && repeatOp.getResult().use_empty()) {
    rewriter.eraseOp(repeatOp);
  }

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
