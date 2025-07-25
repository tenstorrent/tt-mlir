// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/UnevenPoolPaddingRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

template <typename Pool2dOp>
LogicalResult UnevenPoolPaddingRewritePattern<Pool2dOp>::matchAndRewrite(
    Pool2dOp srcOp, PatternRewriter &rewriter) const {

  RankedTensorType inputType = srcOp.getInput().getType();
  ArrayRef<int32_t> padding = srcOp.getPadding();

  // If the padding size is 2, then this op is already legal in ttnn.
  if (padding.size() == 2) {
    return failure();
  }

  // The padding size for a Pool2dop must be 2 (even padding) or 4 (possibly
  // uneven padding). If we've reached this point and the padding size is
  // neither than something has gone wrong earlier in the compile flow.
  assert(padding.size() == 4 &&
         "Padding size must be 4 if this point is reached.");
  assert(inputType.getRank() == 4 && "Input type must be 4D.");

  // If the padding is size 4, it may be even. We will check for that and
  // replace the padding attribute with one of size 2 if it is even.
  bool isEvenPadding = padding[0] == padding[1] && padding[2] == padding[3];
  if (isEvenPadding) {
    rewriter.modifyOpInPlace(srcOp, [&]() { srcOp.setPadding({0, 0}); });
    return success();
  }

  // At this point we know that the padding is uneven. And so we must explicitly
  // put a PadOp in between the Pool2dOp and the original input.

  // The ttnn Pool2dOp may be in the format which takes the flattened input:
  // shape (1, 1, N*H*W, C) If this is the case we must insert a reshape op to
  // change the shape to (N, H, W, C) to apply the padding.

  bool isFlattenedInput = false;
  TypedValue<RankedTensorType> input = srcOp.getInput();
  if (inputType.getDimSize(0) == 1 && inputType.getDimSize(1) == 1 &&
      inputType.getDimSize(2) == srcOp.getBatchSize() * srcOp.getInputHeight() *
                                     srcOp.getInputWidth() &&
      inputType.getDimSize(3) == srcOp.getChannels()) {
    isFlattenedInput = true;
    SmallVector<int64_t> reshapeOutputShape = {
        srcOp.getBatchSize(), srcOp.getInputHeight(), srcOp.getInputWidth(),
        srcOp.getChannels()};
    input = ttir_to_ttnn::utils::generateReshape(input, reshapeOutputShape,
                                                 rewriter);
    inputType = input.getType();
  }

  SmallVector<int64_t> paddedShape(inputType.getShape());
  assert(paddedShape.size() == static_cast<size_t>(inputType.getRank()) &&
         "Padding size and input shape must have the same rank if the padding "
         "size is 4.");

  // Input to pad is NHWC, so we need to add the upper and lower padding to both
  // height and width.
  paddedShape[1] += padding[0] + padding[1];
  paddedShape[2] += padding[2] + padding[3];

  RankedTensorType paddedType = RankedTensorType::get(
      paddedShape, inputType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding())
          .withTensorShape(paddedShape));

  SmallVector<int32_t> padOpPadding(padding.size() * 2, 0);

  // The PadOp requires an upper/lower padding pair for each dimension. Since
  // the Pool2dOps are NHWC, we know that the middle four values of the padding
  // array are what must be populated.

  padOpPadding[2] = padding[0];
  padOpPadding[3] = padding[1];
  padOpPadding[4] = padding[2];
  padOpPadding[5] = padding[3];
  ttnn::PadOp padOp = rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(input.getLoc(), "pad"), paddedType,
      input, padOpPadding, /*pad_value=*/mlir::APFloat(0.0f),
      /*use_multicore=*/false,
      /*memory_config=*/nullptr);

  TypedValue<RankedTensorType> newPool2dInput = padOp.getResult();
  if (isFlattenedInput) {
    SmallVector<int64_t> flattenReshapeShape = {
        1, 1,
        padOp.getType().getDimSize(0) * padOp.getType().getDimSize(1) *
            padOp.getType().getDimSize(2),
        padOp.getType().getDimSize(3)};
    newPool2dInput = ttir_to_ttnn::utils::generateReshape(
        newPool2dInput, flattenReshapeShape, rewriter);
  }

  // Padding is applied explicitly by the PadOp, so we need to set the padding
  // to 0.
  DenseI32ArrayAttr newPaddingAttr = rewriter.getDenseI32ArrayAttr({0, 0});
  Pool2dOp newPool2dOp = rewriter.create<Pool2dOp>(
      srcOp.getLoc(), srcOp.getResult().getType(), newPool2dInput,
      srcOp.getBatchSizeAttr(), rewriter.getSI32IntegerAttr(paddedShape[1]),
      rewriter.getSI32IntegerAttr(paddedShape[2]), srcOp.getChannelsAttr(),
      srcOp.getKernelSizeAttr(), srcOp.getStrideAttr(), newPaddingAttr,
      srcOp.getDilationAttr(), srcOp.getMemoryConfigAttr(),
      srcOp.getAppliedShardSchemeAttr(), srcOp.getCeilModeAttr(),
      srcOp.getInPlaceHaloAttr());

  rewriter.replaceOp(srcOp, newPool2dOp);

  return success();
}

template class UnevenPoolPaddingRewritePattern<ttnn::MaxPool2dOp>;
template class UnevenPoolPaddingRewritePattern<ttnn::AvgPool2dOp>;

} // namespace mlir::tt::ttnn::workarounds::decomposition
