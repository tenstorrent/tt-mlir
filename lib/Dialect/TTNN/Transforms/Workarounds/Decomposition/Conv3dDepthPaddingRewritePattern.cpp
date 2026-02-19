// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv3dDepthPaddingRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {

// Conv3dOp input is NDHWC.
static constexpr int64_t DEPTH_DIM = 1;
static constexpr int64_t INPUT_RANK = 5;

LogicalResult Conv3dDepthPaddingRewritePattern::matchAndRewrite(
    Conv3dOp srcOp, PatternRewriter &rewriter) const {

  ArrayRef<int32_t> padding = srcOp.getPadding();
  int32_t depthPadding = padding[0];

  if (depthPadding == 0) {
    return failure();
  }

  // Explicitly pad the input along the depth dimension.
  RankedTensorType inputType = srcOp.getInput().getType();

  SmallVector<int32_t> inputPadding(INPUT_RANK * 2, 0);
  inputPadding[DEPTH_DIM * 2] = depthPadding;
  inputPadding[DEPTH_DIM * 2 + 1] = depthPadding;

  SmallVector<int64_t> paddedInputShape(inputType.getShape());
  paddedInputShape[DEPTH_DIM] += 2 * depthPadding;

  auto paddedInputType =
      RankedTensorType::get(paddedInputShape, inputType.getElementType(),
                            mlir::cast<TTNNLayoutAttr>(inputType.getEncoding())
                                .withTensorShape(paddedInputShape));

  auto paddedInput =
      rewriter.create<PadOp>(ttmlir::utils::appendLocationSuffix(
                                 srcOp.getInput().getLoc(), "_pad_depth"),
                             paddedInputType, srcOp.getInput(), inputPadding,
                             /*pad_value=*/mlir::APFloat(0.0f),
                             /*use_multicore=*/false,
                             /*memory_config=*/nullptr);

  // Update the op: set padded input, zero out depth padding, update
  // input_depth.
  rewriter.modifyOpInPlace(srcOp, [&]() {
    srcOp.getInputMutable().assign(paddedInput);

    SmallVector<int32_t> newPadding = {0, padding[1], padding[2]};
    srcOp.setPaddingAttr(rewriter.getDenseI32ArrayAttr(newPadding));

    srcOp.setInputDepthAttr(
        rewriter.getI32IntegerAttr(paddedInputShape[DEPTH_DIM]));
  });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
