// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LayerNormPreAllGatherZeroPadRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult LayerNormPreAllGatherZeroPadRewritePattern::matchAndRewrite(
    ttnn::LayerNormPreAllGatherOp srcOp, PatternRewriter &rewriter) const {
  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  int64_t W = inputType.getShape().back();

  // Only fire when the last dimension is not tile-aligned.
  if (W % ttnn::TILE_WIDTH == 0) {
    return failure();
  }

  // Compute how many elements to pad on the high side of the last dimension.
  int64_t padAmount =
      ((W + ttnn::TILE_WIDTH - 1) / ttnn::TILE_WIDTH) * ttnn::TILE_WIDTH - W;

  // Build padding array: [0, 0, ..., 0, padAmount] — only pad last dim high.
  int64_t rank = inputType.getRank();
  llvm::SmallVector<int32_t> paddingValues(rank * 2, 0);
  paddingValues.back() = static_cast<int32_t>(padAmount);

  // Build the output type for the padded tensor.
  llvm::SmallVector<int64_t> paddedShape(inputType.getShape().begin(),
                                         inputType.getShape().end());
  paddedShape.back() = W + padAmount;
  RankedTensorType paddedInputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, paddedShape);

  // Insert ttnn.pad with value=0.0 to explicitly zero the tile padding area.
  // The kernel reads the full tile-padded width; without zeroing, garbage in
  // positions W..ceil(W/32)*32-1 would corrupt sum(x) and sum(x²).
  // N is now derived from logical_shape() in the kernel, so no slice is needed.
  auto padOp = rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_zero_pad"),
      paddedInputType, srcOp.getInput(),
      rewriter.getDenseI32ArrayAttr(paddingValues),
      rewriter.getF32FloatAttr(0.0f), rewriter.getBoolAttr(false),
      /*memory_config=*/nullptr);

  // If a residual input exists, pad it the same way so its shape matches the
  // padded input shape (required by the op verifier).
  mlir::Value paddedResidual = nullptr;
  if (srcOp.getResidualInput()) {
    RankedTensorType residualType =
        mlir::cast<RankedTensorType>(srcOp.getResidualInput().getType());
    llvm::SmallVector<int64_t> paddedResidualShape(
        residualType.getShape().begin(), residualType.getShape().end());
    paddedResidualShape.back() += padAmount;
    RankedTensorType paddedResidualType =
        ttnn::utils::RankedTensorTypeFactory::create(residualType,
                                                     paddedResidualShape);
    auto residualPadOp = rewriter.create<ttnn::PadOp>(
        ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                            "_residual_zero_pad"),
        paddedResidualType, srcOp.getResidualInput(),
        rewriter.getDenseI32ArrayAttr(paddingValues),
        rewriter.getF32FloatAttr(0.0f), rewriter.getBoolAttr(false),
        /*memory_config=*/nullptr);
    paddedResidual = residualPadOp.getResult();
  }

  // Replace with layer_norm_pre_all_gather using the zero-padded input.
  // The padded width is tile-aligned (W+padAmount) % ttnn::TILE_WIDTH == 0, so
  // this rewrite will not fire again on the newly created op.
  rewriter.replaceOpWithNewOp<ttnn::LayerNormPreAllGatherOp>(
      srcOp, srcOp.getResult().getType(), padOp.getResult(),
      /*residual_input=*/paddedResidual,
      /*recip=*/srcOp.getRecip(), srcOp.getDtypeAttr(),
      srcOp.getMemoryConfigAttr(), srcOp.getComputeConfigAttr(),
      srcOp.getProgramConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
