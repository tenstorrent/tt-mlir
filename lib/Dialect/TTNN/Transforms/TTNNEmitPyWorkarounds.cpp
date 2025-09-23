// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/EmitPy/ConstantOp4DWorkaroundPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/EmitPy/ConstantOpDataTypeWorkaroundPattern.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
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

DenseElementsAttr
workarounds::emitpy::ConstantOpDataTypeWorkaroundPattern::convertToFloat(
    DenseElementsAttr denseValue, PatternRewriter &rewriter) const {
  // If the value is already a float type, no need to convert.
  if (denseValue.getElementType().isFloat()) {
    return denseValue;
  }

  // Convert values from integer to float.
  llvm::SmallVector<float> convertedValues;
  for (const APInt &value : denseValue.getValues<APInt>()) {
    convertedValues.push_back(static_cast<float>(value.getSExtValue()));
  }

  // Create new dense elements attribute with float values.
  FloatType floatType = rewriter.getF32Type();
  RankedTensorType floatTensorType =
      RankedTensorType::get(denseValue.getType().getShape(), floatType);
  return DenseElementsAttr::get(floatTensorType,
                                llvm::ArrayRef<float>(convertedValues));
}

LogicalResult
workarounds::emitpy::ConstantOpDataTypeWorkaroundPattern::matchAndRewrite(
    ttnn::ConstantOp constantOp, PatternRewriter &rewriter) const {
  ElementsAttr value = constantOp.getValueAttr();
  auto denseValue = mlir::cast<DenseElementsAttr>(value);

  // If the value is already a float type, and the dtype attribute is float or
  // not set, no need to rewrite
  if (denseValue.getElementType().isFloat() && constantOp.getDtype() &&
      isFloat(constantOp.getDtype().value())) {
    return failure();
  }

  // Only support integer and float types for now.
  if (!denseValue.getElementType().isIntOrIndexOrFloat()) {
    return constantOp.emitError(
        "TTNNEmitPyWorkarounds: ConstantOp with non-integer and non-float data "
        "type is not supported");
  }

  // Rewrite the constant op to use float dtype and add a cast op after it.
  // We have three cases to handle:
  // 1. The dense value is float and dtype is integer:
  //   - Rewrite the constant op to use float dtype.
  //   - Add a cast op after it to convert the value to integer.
  // 2. The dense value is integer and dtype is integer:
  //   - Convert the dense value to float.
  //   - Rewrite the constant op to use converted float value and float dtype.
  //   - Add a cast op after it to convert the value to integer.
  // 3. The dense value is integer and dtype is float:
  //   - Convert the dense value to float.

  // Convert the dense value to float if it isn't float.
  DenseElementsAttr floatDenseAttr = convertToFloat(denseValue, rewriter);

  // Create new type and layout for the new constant op.
  FloatType floatType = rewriter.getF32Type();
  RankedTensorType resultType = constantOp.getResult().getType();
  RankedTensorType targetResultType =
      resultType.clone(resultType.getShape(), floatType);
  TTNNLayoutAttr targetLayoutAttr =
      mlir::cast<TTNNLayoutAttr>(resultType.getEncoding());
  targetResultType = targetResultType.cloneWithEncoding(
      targetLayoutAttr.withDataType(ttcore::DataType::Float32));

  // Create new constant op with float data and float dtype.
  auto newConstantOp = rewriter.create<ttnn::ConstantOp>(
      constantOp.getLoc(), targetResultType, constantOp.getDevice(),
      floatDenseAttr,
      isFloat(constantOp.getDtype().value())
          ? constantOp.getDtypeAttr()
          : ttcore::DataTypeAttr::get(rewriter.getContext(),
                                      ttcore::DataType::Float32),
      constantOp.getLayoutAttr(), constantOp.getMemoryConfigAttr());

  // If the original dtype is not float, we need to cast back to the original
  // dtype.
  if (!isFloat(constantOp.getDtype().value())) {
    // Get the target data type from the original constant op's layout.
    auto originalLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(resultType.getEncoding());
    ttcore::DataType targetDataType = originalLayoutAttr.getDataType();

    // Check whether we need to do the cast on device or host.
    if (constantOp.getDevice()) {
      // If the original constant op had a device attribute, do the cast on
      // device.
      auto typecastOp = rewriter.create<ttnn::TypecastOp>(
          ttmlir::utils::appendLocationSuffix(constantOp->getLoc(),
                                              "_emitPyConstantDTypeWorkaround"),
          resultType, newConstantOp.getResult(),
          ttcore::DataTypeAttr::get(rewriter.getContext(), targetDataType));

      // Replace the original constant op with the typecast result
      rewriter.replaceOp(constantOp, typecastOp.getResult());
    } else {
      // If no device attribute, do the cast on host.
      auto toDtypeOp = rewriter.create<ttnn::ToDTypeOp>(
          ttmlir::utils::appendLocationSuffix(constantOp->getLoc(),
                                              "_emitPyConstantDTypeWorkaround"),
          resultType, newConstantOp.getResult(),
          ttcore::DataTypeAttr::get(rewriter.getContext(), targetDataType));

      // Replace the original constant op with the typecast result
      rewriter.replaceOp(constantOp, toDtypeOp.getResult());
    }
  } else {
    // If the original dtype is float, no need to cast, just replace
    rewriter.replaceOp(constantOp, newConstantOp.getResult());
  }

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
    patterns.add<workarounds::emitpy::ConstantOpDataTypeWorkaroundPattern>(
        context);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::tt::ttnn
