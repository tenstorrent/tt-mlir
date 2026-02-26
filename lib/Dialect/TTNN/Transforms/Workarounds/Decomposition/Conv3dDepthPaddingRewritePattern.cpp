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
static constexpr int64_t HEIGHT_DIM = 2;
static constexpr int64_t WIDTH_DIM = 3;
static constexpr int64_t INPUT_RANK = 5;

LogicalResult Conv3dDepthPaddingRewritePattern::matchAndRewrite(
    Conv3dOp srcOp, PatternRewriter &rewriter) const {

  if (srcOp.getPaddingMode() != "zeros") {
    return failure();
  }

  ArrayRef<int32_t> padding = srcOp.getPadding();
  int32_t depthPad = padding[0];
  int32_t heightPad = padding[1];
  int32_t widthPad = padding[2];

  if (depthPad == 0 && heightPad == 0 && widthPad == 0) {
    return failure();
  }

  RankedTensorType inputType = srcOp.getInput().getType();

  SmallVector<int32_t> inputPadding(INPUT_RANK * 2, 0);
  inputPadding[DEPTH_DIM * 2] = depthPad;
  inputPadding[DEPTH_DIM * 2 + 1] = depthPad;
  inputPadding[HEIGHT_DIM * 2] = heightPad;
  inputPadding[HEIGHT_DIM * 2 + 1] = heightPad;
  inputPadding[WIDTH_DIM * 2] = widthPad;
  inputPadding[WIDTH_DIM * 2 + 1] = widthPad;

  SmallVector<int64_t> paddedInputShape(inputType.getShape());
  paddedInputShape[DEPTH_DIM] += 2 * depthPad;
  paddedInputShape[HEIGHT_DIM] += 2 * heightPad;
  paddedInputShape[WIDTH_DIM] += 2 * widthPad;

  auto paddedInputType =
      RankedTensorType::get(paddedInputShape, inputType.getElementType(),
                            mlir::cast<TTNNLayoutAttr>(inputType.getEncoding())
                                .withTensorShape(paddedInputShape));

  auto paddedInput =
      rewriter.create<PadOp>(ttmlir::utils::appendLocationSuffix(
                                 srcOp.getInput().getLoc(), "_pad_conv3d"),
                             paddedInputType, srcOp.getInput(), inputPadding,
                             /*pad_value=*/mlir::APFloat(0.0f),
                             /*use_multicore=*/false,
                             /*memory_config=*/nullptr);

  rewriter.modifyOpInPlace(srcOp, [&]() {
    srcOp.getInputMutable().assign(paddedInput);

    srcOp.setPaddingAttr(rewriter.getDenseI32ArrayAttr({0, 0, 0}));

    srcOp.setInputDepthAttr(
        rewriter.getI32IntegerAttr(paddedInputShape[DEPTH_DIM]));
    srcOp.setInputHeightAttr(
        rewriter.getI32IntegerAttr(paddedInputShape[HEIGHT_DIM]));
    srcOp.setInputWidthAttr(
        rewriter.getI32IntegerAttr(paddedInputShape[WIDTH_DIM]));
  });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
