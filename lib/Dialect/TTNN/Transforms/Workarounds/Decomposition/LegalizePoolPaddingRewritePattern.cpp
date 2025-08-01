// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LegalizePoolPaddingRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

template <typename Pool2dOp>
LogicalResult LegalizePoolPaddingRewritePattern<Pool2dOp>::matchAndRewrite(
    Pool2dOp srcOp, PatternRewriter &rewriter) const {

  RankedTensorType inputType = srcOp.getInput().getType();
  ArrayRef<int32_t> padding = srcOp.getPadding();

  // If the padding size is 2, then this op is already legal in ttnn.
  if (padding.size() != 4) {
    return failure();
  }

  assert(inputType.getRank() == 4 && "Input type must be 4D.");
  int32_t paddingTop = padding[0];
  int32_t paddingLeft = padding[1];
  int32_t paddingBottom = padding[2];
  int32_t paddingRight = padding[3];

  if (paddingTop == paddingBottom && paddingLeft == paddingRight) {
    // If the padding is symmetric, we can use the 2D padding format.
    rewriter.modifyOpInPlace(srcOp, [&]() {
      srcOp.setPaddingAttr(
          rewriter.getDenseI32ArrayAttr({paddingTop, paddingLeft}));
    });
    return success();
  }

  // At this point we know that the padding is uneven. If ceil_mode = true, this
  // is supported in ttnn and we can return failure(). We return failure() since
  // the pattern we wish to match is uneven padding and ceil_mode = false. Thus
  // this is not a match.
  if (srcOp.getCeilMode()) {
    return failure();
  }

  // We must now explicate the padding as the padding is uneven and ceil_mode =
  // false. we must put a PadOp in between the Pool2dOp and the original input.

  // The ttnn Pool2dOp will be in the format which takes the flattened input:
  // shape (1, 1, N*H*W, C) (this is ensured by <Pool2dOp>::verify()). So we
  // must insert a reshape op to change the shape to (N, H, W, C) to apply the
  // padding.
  SmallVector<int64_t> reshapeOutputShape = {
      srcOp.getBatchSize(), srcOp.getInputHeight(), srcOp.getInputWidth(),
      srcOp.getChannels()};
  TypedValue<RankedTensorType> unflattenReshape =
      ttir_to_ttnn::utils::generateReshape(srcOp.getInput(), reshapeOutputShape,
                                           rewriter);
  RankedTensorType unflattenReshapeType = unflattenReshape.getType();

  SmallVector<int64_t> paddedShape(unflattenReshapeType.getShape());
  assert(paddedShape.size() ==
             static_cast<size_t>(unflattenReshapeType.getRank()) &&
         "Padding size and input shape must have the same rank if the padding "
         "size is 4.");

  // Input to pad is NHWC, so we need to add the upper and lower padding to both
  // height and width.
  paddedShape[1] += paddingTop + paddingBottom;
  paddedShape[2] += paddingLeft + paddingRight;

  RankedTensorType paddedType = RankedTensorType::get(
      paddedShape, unflattenReshapeType.getElementType(),
      mlir::cast<ttnn::TTNNLayoutAttr>(unflattenReshapeType.getEncoding())
          .withTensorShape(paddedShape));

  SmallVector<int32_t> padOpPadding(padding.size() * 2, 0);

  // The PadOp requires an upper/lower padding pair for each dimension. Since
  // the Pool2dOps are NHWC, we know that the middle four values of the padding
  // array are what must be populated.

  padOpPadding[2] = paddingTop;
  padOpPadding[3] = paddingBottom;
  padOpPadding[4] = paddingLeft;
  padOpPadding[5] = paddingRight;
  ttnn::PadOp padOp = rewriter.create<ttnn::PadOp>(
      ttmlir::utils::appendLocationSuffix(unflattenReshape.getLoc(), "pad"),
      paddedType, unflattenReshape, padOpPadding,
      /*pad_value=*/mlir::APFloat(0.0f),
      /*use_multicore=*/false,
      /*memory_config=*/nullptr);

  SmallVector<int64_t> flattenReshapeShape = {
      1, 1,
      padOp.getType().getDimSize(0) * padOp.getType().getDimSize(1) *
          padOp.getType().getDimSize(2),
      padOp.getType().getDimSize(3)};
  TypedValue<RankedTensorType> flattenPaddedInputReshape =
      ttir_to_ttnn::utils::generateReshape(padOp, flattenReshapeShape,
                                           rewriter);

  // Padding is applied explicitly by the PadOp, so we need to set the padding
  // to 0.
  DenseI32ArrayAttr newPaddingAttr = rewriter.getDenseI32ArrayAttr({0, 0});
  Pool2dOp newPool2dOp = rewriter.create<Pool2dOp>(
      srcOp.getLoc(), srcOp.getResult().getType(), flattenPaddedInputReshape,
      srcOp.getBatchSizeAttr(), rewriter.getSI32IntegerAttr(paddedShape[1]),
      rewriter.getSI32IntegerAttr(paddedShape[2]), srcOp.getChannelsAttr(),
      srcOp.getKernelSizeAttr(), srcOp.getStrideAttr(), newPaddingAttr,
      srcOp.getDilationAttr(), srcOp.getMemoryConfigAttr(),
      srcOp.getAppliedShardSchemeAttr(), srcOp.getCeilModeAttr(),
      srcOp.getInPlaceHaloAttr());

  rewriter.replaceOp(srcOp, newPool2dOp);

  return success();
}

template class LegalizePoolPaddingRewritePattern<ttnn::MaxPool2dOp>;
template class LegalizePoolPaddingRewritePattern<ttnn::AvgPool2dOp>;

} // namespace mlir::tt::ttnn::workarounds::decomposition
