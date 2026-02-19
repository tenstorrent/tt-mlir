// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PadHighDimRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {

// ttnn::pad constraint: only the lowest 3 dimensions can be padded for rank > 4
// tensors.
static constexpr int64_t MAX_PADDABLE_LOW_DIMS = 3;

LogicalResult
PadHighDimRewritePattern::matchAndRewrite(PadOp srcOp,
                                          PatternRewriter &rewriter) const {

  RankedTensorType inputType = srcOp.getInput().getType();
  int64_t rank = inputType.getRank();

  if (rank <= 4) {
    return failure();
  }

  ArrayRef<int32_t> padding = srcOp.getPadding();
  int64_t lowestPaddableStart = rank - MAX_PADDABLE_LOW_DIMS;

  // Collect dims that have non-zero padding above the lowest 3.
  SmallVector<int64_t> highPaddedDims;
  for (int64_t d = 0; d < lowestPaddableStart; ++d) {
    if (padding[d * 2] != 0 || padding[d * 2 + 1] != 0) {
      highPaddedDims.push_back(d);
    }
  }

  if (highPaddedDims.empty()) {
    return failure();
  }

  // Collect low dims (in the lowest 3) that have zero padding — candidates for
  // swapping. Iterate in reverse so we prefer the highest-indexed dims first.
  // This is critical: tt-metal's update_original_shape only preserves the last
  // 2 dims of the 4D squeezed tensor when unsqueezing back to rank 5.
  SmallVector<int64_t> zeroLowDims;
  for (int64_t d = rank - 1; d >= lowestPaddableStart; --d) {
    if (padding[d * 2] == 0 && padding[d * 2 + 1] == 0) {
      zeroLowDims.push_back(d);
    }
  }

  if (zeroLowDims.size() < highPaddedDims.size()) {
    return failure();
  }

  // Build the forward permutation by swapping each problematic high dim with a
  // zero-padded low dim.
  SmallVector<int64_t> permutation(rank);
  std::iota(permutation.begin(), permutation.end(), 0);

  for (size_t i = 0; i < highPaddedDims.size(); ++i) {
    int64_t highDim = highPaddedDims[i];
    int64_t lowDim = zeroLowDims[i];
    std::swap(permutation[highDim], permutation[lowDim]);
  }

  // Permute the padding array to match the new dimension order.
  SmallVector<int32_t> permutedPadding(rank * 2);
  for (int64_t d = 0; d < rank; ++d) {
    int64_t srcDim = permutation[d];
    permutedPadding[d * 2] = padding[srcDim * 2];
    permutedPadding[d * 2 + 1] = padding[srcDim * 2 + 1];
  }

  // Permute input to move high-padded dims into the lowest 3.
  SmallVector<int64_t> permutedInputShape =
      ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
  RankedTensorType permutedInputType =
      utils::RankedTensorTypeFactory::create(inputType, permutedInputShape);

  auto forwardPermute = rewriter.create<PermuteOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_pad_permute_fwd"),
      permutedInputType, srcOp.getInput(),
      rewriter.getDenseI64ArrayAttr(permutation),
      /*memory_config=*/ttnn::MemoryConfigAttr(),
      /*pad_value=*/mlir::FloatAttr());

  // Pad on the (now-low) dimensions.
  SmallVector<int64_t> paddedShape(permutedInputShape);
  for (int64_t d = 0; d < rank; ++d) {
    paddedShape[d] += permutedPadding[d * 2] + permutedPadding[d * 2 + 1];
  }
  RankedTensorType paddedType =
      utils::RankedTensorTypeFactory::create(inputType, paddedShape);

  auto padOp = rewriter.create<PadOp>(
      srcOp.getLoc(), paddedType, forwardPermute.getResult(), permutedPadding,
      srcOp.getValue(), srcOp.getUseMulticore(), srcOp.getMemoryConfigAttr());

  // Inverse permute to restore original dimension order.
  SmallVector<int64_t> invPerm = ttmlir::utils::inversePermutation(permutation);

  rewriter.replaceOpWithNewOp<PermuteOp>(
      srcOp, srcOp.getResult().getType(), padOp.getResult(),
      rewriter.getDenseI64ArrayAttr(invPerm),
      /*memory_config=*/ttnn::MemoryConfigAttr(),
      /*pad_value=*/mlir::FloatAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
