// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/EmbeddingOpPadHiddenDimRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// TT-Metal Issue : https://github.com/tenstorrent/tt-metal/issues/43567
// True when the embedding hidden dim falls in the regression-prone region.
//
// Empirical sweep on a single Blackhole chip (vocab=256, seq_len=64, bf16
// weight, TILE_LAYOUT, DRAM) over ~160 tile_count values found a clean
// rule for the raw ttnn.embedding op:
//
//   PASS iff  hiddenDim < 8192  OR  hiddenDim % 2048 == 0
//
// equivalently in tile_count units (= hiddenDim / 32):
//
//   PASS iff  tile_count < 256  OR  tile_count % 64 == 0
//
// Below 256 tiles every value tested passed regardless of factorization;
// at and above 256 only multiples of 64 (320, 384, 448, 512, 576, ...)
// pass while everything else fails with PCC ~ 0.02.
//
// The padding strategy in matchAndRewrite rounds the hidden dim up to
// the next multiple of 2048, which is always in the "safe" set above
// 8192. The proper fix belongs in tt-metal's ttnn::embedding kernel; this
// rewrite is intended to be removed once that fix lands.
constexpr int64_t kSafeBelowHiddenDim = 8192;
constexpr int64_t kHiddenDimSafeMultiple = 2048;

bool isBadEmbeddingHiddenDim(int64_t hiddenDim) {
  if (hiddenDim <= 0 || hiddenDim % 32 != 0) {
    return false;
  }
  if (hiddenDim < kSafeBelowHiddenDim) {
    return false;
  }
  return (hiddenDim % kHiddenDimSafeMultiple) != 0;
}

// Rounds the hidden dim up to the next multiple of `kHiddenDimSafeMultiple`
// (= 2048 = 64 tiles), which is in the known-good region for hidden dims
// >= kSafeBelowHiddenDim.
int64_t nextSafeHiddenDim(int64_t hiddenDim) {
  return ((hiddenDim + kHiddenDimSafeMultiple - 1) / kHiddenDimSafeMultiple) *
         kHiddenDimSafeMultiple;
}

} // namespace

LogicalResult EmbeddingOpPadHiddenDimRewritePattern::matchAndRewrite(
    ttnn::EmbeddingOp srcOp, PatternRewriter &rewriter) const {
  auto weight =
      mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(srcOp.getWeight());
  mlir::RankedTensorType weightType = weight.getType();
  llvm::ArrayRef<int64_t> weightShape = weightType.getShape();
  int64_t hiddenDim = weightShape.back();

  if (!isBadEmbeddingHiddenDim(hiddenDim)) {
    return failure();
  }

  // The slice that restores the original hidden dim needs static
  // begins/ends, so bail out if the embedding result has any dynamic dim.
  mlir::RankedTensorType origResultType = srcOp.getResult().getType();
  llvm::ArrayRef<int64_t> origResultShape = origResultType.getShape();
  if (llvm::any_of(origResultShape, mlir::ShapedType::isDynamic)) {
    return failure();
  }

  int64_t paddedHiddenDim = nextSafeHiddenDim(hiddenDim);
  int64_t padAmount = paddedHiddenDim - hiddenDim;
  // padAmount is bounded by kHiddenDimSafeMultiple - 1 by construction,
  // so it always fits in i32.
  assert(padAmount > 0 && padAmount < kHiddenDimSafeMultiple &&
         "padAmount must be in (0, kHiddenDimSafeMultiple)");

  mlir::Location loc = srcOp.getLoc();

  // Build pad spec [low_0, high_0, low_1, high_1, ...] padding only the
  // last (hidden) dim on the high side.
  size_t weightRank = weightShape.size();
  llvm::SmallVector<int32_t> padding(weightRank * 2, 0);
  padding[weightRank * 2 - 1] = static_cast<int32_t>(padAmount);

  ttnn::PadOp paddedWeight = ttir_to_ttnn::utils::generatePad(
      weight, padding, rewriter,
      ttmlir::utils::appendLocationSuffix(loc, "_pad_weight"));

  // Build the result type for the embedding on the padded weight: same
  // leading shape as the original result, padded last dim.
  llvm::SmallVector<int64_t> paddedResultShape(origResultShape.begin(),
                                               origResultShape.end());
  paddedResultShape.back() = paddedHiddenDim;
  mlir::RankedTensorType paddedResultType =
      ttnn::utils::RankedTensorTypeFactory::create(origResultType,
                                                   paddedResultShape);

  auto paddedEmbedding = rewriter.create<ttnn::EmbeddingOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_padded"), paddedResultType,
      srcOp.getInput(), paddedWeight.getResult());

  // Slice the padded output back to the original hidden dim.
  llvm::SmallVector<int32_t> begins(origResultShape.size(), 0);
  llvm::SmallVector<int32_t> ends;
  ends.reserve(origResultShape.size());
  for (int64_t s : origResultShape) {
    ends.push_back(static_cast<int32_t>(s));
  }
  llvm::SmallVector<int32_t> step(origResultShape.size(), 1);

  rewriter.replaceOpWithNewOp<ttnn::SliceStaticOp>(
      srcOp, origResultType, paddedEmbedding.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(step));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
