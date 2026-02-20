// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PadHighDimRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <numeric>

namespace mlir::tt::ttnn::workarounds::decomposition {

static constexpr int64_t MIN_PAD_RANK = 4;
static constexpr int64_t MAX_PADDABLE_DIMS = 3;

LogicalResult
PadHighDimRewritePattern::matchAndRewrite(PadOp srcOp,
                                          PatternRewriter &rewriter) const {

  RankedTensorType inputType = srcOp.getInput().getType();
  int64_t rank = inputType.getRank();

  if (rank <= MIN_PAD_RANK) {
    return failure();
  }

  ArrayRef<int32_t> padding = srcOp.getPadding();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  auto dimHasPad = [&](int64_t d) {
    return padding[d * 2] != 0 || padding[d * 2 + 1] != 0;
  };

  SmallVector<int64_t> paddedDims, nonPaddedDims;
  for (int64_t d = 0; d < rank; ++d) {
    (dimHasPad(d) ? paddedDims : nonPaddedDims).push_back(d);
  }

  if (paddedDims.empty() || paddedDims.size() > MAX_PADDABLE_DIMS) {
    return failure();
  }

  // Permute so non-padded dims come first, padded dims last.
  SmallVector<int64_t> fwdPerm;
  fwdPerm.append(nonPaddedDims.begin(), nonPaddedDims.end());
  fwdPerm.append(paddedDims.begin(), paddedDims.end());

  auto permuteFwd = ttir_to_ttnn::utils::generatePermute(
      srcOp.getInput(), fwdPerm, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_perm_fwd"));

  // Reshape to merge all non-padded dims into one, reducing to
  // (1 + paddedDims.size())-D, which is at most 4D.
  int64_t mergedDim = std::accumulate(
      nonPaddedDims.begin(), nonPaddedDims.end(), static_cast<int64_t>(1),
      [&](int64_t acc, int64_t d) { return acc * inputShape[d]; });

  SmallVector<int64_t> squeezedShape;
  squeezedShape.push_back(mergedDim);
  for (int64_t d : paddedDims) {
    squeezedShape.push_back(inputShape[d]);
  }

  auto reshapeDown = ttir_to_ttnn::utils::generateReshape(
      permuteFwd.getResult(), squeezedShape, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_squeeze"));

  // Pad the reduced-rank tensor. Dim 0 (merged) has no padding.
  SmallVector<int32_t> reducedPadding = {0, 0};
  for (int64_t d : paddedDims) {
    reducedPadding.push_back(padding[d * 2]);
    reducedPadding.push_back(padding[d * 2 + 1]);
  }

  auto padReduced = ttir_to_ttnn::utils::generatePad(
      reshapeDown.getResult(), reducedPadding, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_pad"));

  // Reshape back to original rank, restoring non-padded dim sizes.
  ArrayRef<int64_t> paddedReducedShape =
      padReduced.getResult().getType().getShape();

  SmallVector<int64_t> unsqueezedShape;
  for (int64_t d : nonPaddedDims) {
    unsqueezedShape.push_back(inputShape[d]);
  }
  for (size_t i = 0; i < paddedDims.size(); ++i) {
    unsqueezedShape.push_back(paddedReducedShape[1 + i]);
  }

  auto reshapeUp = ttir_to_ttnn::utils::generateReshape(
      padReduced.getResult(), unsqueezedShape, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_unsqueeze"));

  // Inverse permute to restore original dimension order.
  SmallVector<int64_t> invPerm(rank);
  for (int64_t i = 0; i < rank; ++i) {
    invPerm[fwdPerm[i]] = i;
  }

  rewriter.replaceOp(srcOp, ttir_to_ttnn::utils::generatePermute(
                                reshapeUp.getResult(), invPerm, rewriter,
                                ttmlir::utils::appendLocationSuffix(
                                    srcOp.getLoc(), "_perm_inv")));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
