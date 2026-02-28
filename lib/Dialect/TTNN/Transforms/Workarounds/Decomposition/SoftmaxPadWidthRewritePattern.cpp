// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SoftmaxPadWidthRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include <cmath>
#include <limits>

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Hardcoded circular buffer tile count used by the attention-optimized softmax
// kernel's large-kernel fallback in tt-metal.
constexpr int64_t LARGE_KERNEL_CB_LEN = 80;

// tt-metal's find_max_divisor: returns the largest divisor of `val`
// in [1..startMaxDiv], skipping 5 and 7.
int64_t findMaxDivisor(int64_t val, int64_t startMaxDiv) {
  for (int64_t d = startMaxDiv; d >= 1; --d) {
    if (d == 7 || d == 5) {
      continue;
    }
    if (val % d == 0) {
      return d;
    }
  }
  return 1;
}

bool isSoftmaxWidthSafe(int64_t widthTiles) {
  int64_t blockFP32 = findMaxDivisor(widthTiles, 4);
  int64_t blockFP16 = findMaxDivisor(widthTiles, 8);
  return (LARGE_KERNEL_CB_LEN % blockFP32 == 0) &&
         (LARGE_KERNEL_CB_LEN % blockFP16 == 0);
}

int64_t findSafeWidthTiles(int64_t widthTiles) {
  while (!isSoftmaxWidthSafe(widthTiles)) {
    ++widthTiles;
  }
  return widthTiles;
}

} // namespace

LogicalResult SoftmaxPadWidthRewritePattern::matchAndRewrite(
    ttnn::SoftmaxOp srcOp, PatternRewriter &rewriter) const {

  auto inputType = mlir::dyn_cast<RankedTensorType>(srcOp.getInput().getType());
  if (!inputType) {
    return failure();
  }

  int64_t rank = inputType.getRank();

  // The attention-optimized factory in tt-metal is selected only for rank-4
  // tensors with softmax on the last dimension.
  int32_t dim = srcOp.getDimension();
  if (dim < 0) {
    dim += rank;
  }
  if (rank != 4 || dim != rank - 1) {
    return failure();
  }

  int64_t dimSize = inputType.getShape()[dim];
  int64_t widthTiles =
      llvm::divideCeil(dimSize, static_cast<int64_t>(TILE_WIDTH));

  if (isSoftmaxWidthSafe(widthTiles)) {
    return failure();
  }

  int64_t safeWidthTiles = findSafeWidthTiles(widthTiles);
  int64_t paddedDimSize = safeWidthTiles * TILE_WIDTH;
  int64_t padAmount = paddedDimSize - dimSize;

  // Pad input with -inf so padded positions contribute zero after exp().
  SmallVector<int64_t> paddedShape(inputType.getShape());
  paddedShape[dim] = paddedDimSize;

  SmallVector<int32_t> padding(rank * 2, 0);
  padding[dim * 2 + 1] = static_cast<int32_t>(padAmount);

  auto paddedInputType =
      utils::RankedTensorTypeFactory::create(inputType, paddedShape);

  Value paddedInput =
      rewriter
          .create<PadOp>(
              srcOp.getLoc(), paddedInputType, srcOp.getInput(),
              rewriter.getDenseI32ArrayAttr(padding),
              rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()),
              /*use_multicore=*/rewriter.getBoolAttr(true),
              /*memory_config=*/nullptr)
          .getResult();

  // Run softmax on the padded tensor.
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(srcOp.getResult().getType());
  auto paddedResultType =
      utils::RankedTensorTypeFactory::create(resultType, paddedShape);

  Value paddedResult =
      rewriter
          .create<SoftmaxOp>(srcOp.getLoc(), paddedResultType, paddedInput,
                             srcOp.getDimension(), srcOp.getNumericStable(),
                             srcOp.getComputeConfigAttr())
          .getResult();

  // Slice back to the original shape.
  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends;
  for (int64_t s : inputType.getShape()) {
    ends.push_back(static_cast<int32_t>(s));
  }
  SmallVector<int32_t> steps(rank, 1);

  Value sliced =
      rewriter
          .create<SliceStaticOp>(srcOp.getLoc(), srcOp.getResult().getType(),
                                 paddedResult, rewriter.getI32ArrayAttr(begins),
                                 rewriter.getI32ArrayAttr(ends),
                                 rewriter.getI32ArrayAttr(steps))
          .getResult();

  rewriter.replaceOp(srcOp, sliced);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
