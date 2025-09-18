// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/EmitPy/ConstantOp4DWorkaroundPattern.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNEMITPYWORKAROUNDS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

LogicalResult
workarounds::emitpy::ConstantOp4DWorkaroundPattern::matchAndRewrite(
    ttnn::ConstantOp constantOp, PatternRewriter &rewriter) const {
  auto resultType = constantOp.getResult().getType();
  auto shape = resultType.getShape();

  // Only apply workaround if tensor is not already 4D
  if (shape.size() == 4) {
    return failure();
  }

  // Only support up to 4D tensors
  if (shape.size() > 4) {
    return constantOp.emitError(
        "TTNNEmitPyWorkarounds: ConstantOp with more than 4 dimensions (" +
        std::to_string(shape.size()) + "D) is not supported");
  }

  Location loc = constantOp.getLoc();

  // Create 4D shape by padding with 1s on the left
  llvm::SmallVector<int64_t> targetShape(4, 1);
  std::copy_backward(shape.begin(), shape.end(), targetShape.end());

  // Reshape the value attribute to match the new 4D shape
  ElementsAttr originalValue = constantOp.getValueAttr();
  auto denseValue = mlir::cast<DenseElementsAttr>(originalValue);
  DenseElementsAttr reshapedValue = denseValue.reshape(
      RankedTensorType::get(targetShape, denseValue.getElementType()));

  // Create new constant op type with updated shape and encoding
  auto targetResultType = resultType.clone(targetShape);
  auto targetResultTypeEncoding =
      mlir::cast<TTNNLayoutAttr>(targetResultType.getEncoding());
  targetResultType = targetResultType.cloneWithEncoding(
      targetResultTypeEncoding.withTensorShape(targetShape));

  // Create new constant op with reshaped value
  auto newConstantOp = rewriter.create<ttnn::ConstantOp>(
      loc, targetResultType, constantOp.getDevice(), reshapedValue,
      constantOp.getDtypeAttr(), constantOp.getLayoutAttr(),
      constantOp.getMemoryConfigAttr());

  // Create reshape op to restore original shape
  auto reshapeOp = ttir_to_ttnn::utils::generateReshape(
      newConstantOp, shape, rewriter,
      ttmlir::utils::appendLocationSuffix(constantOp->getLoc(),
                                          "_reshapeConstant"));

  // Replace the original constant op with the reshaped result
  rewriter.replaceOp(constantOp, reshapeOp.getResult());

  return success();
}

// Pass implementation
class TTNNEmitPyWorkarounds
    : public impl::TTNNEmitPyWorkaroundsBase<TTNNEmitPyWorkarounds> {
public:
  using impl::TTNNEmitPyWorkaroundsBase<
      TTNNEmitPyWorkarounds>::TTNNEmitPyWorkaroundsBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<workarounds::emitpy::ConstantOp4DWorkaroundPattern>(context);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::tt::ttnn
