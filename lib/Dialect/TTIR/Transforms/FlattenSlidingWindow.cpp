// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFLATTENSLIDINGWINDOW
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

ttir::ReshapeOp
generateTTIRReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                    ArrayRef<int64_t> newShape, PatternRewriter &rewriter) {
  // With reshape op, the output layout changes due to new output shape, hence
  // we need to create a new output layout attribute with the new shape.
  RankedTensorType inputType = input.getType();

  // Create a new output type for reshape operation with new shape and new
  // output layout.
  RankedTensorType outputType =
      RankedTensorType::get(newShape, inputType.getElementType());

  llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
  auto reshapeDPS = rewriter.create<tensor::EmptyOp>(
      input.getLoc(), outputType.getShape(), outputType.getElementType());

  return rewriter.create<ttir::ReshapeOp>(
      input.getLoc(), outputType, input, reshapeDPS,
      rewriter.getI32ArrayAttr(newShapeI32));
}

ttir::ReshapeOp
generateTTIRNHWFlatten(mlir::TypedValue<mlir::RankedTensorType> input,
                       PatternRewriter &rewriter) {
  llvm::ArrayRef<int64_t> shape = input.getType().getShape();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  llvm::SmallVector<int64_t> newShape = {1, 1, shape[0] * shape[1] * shape[2],
                                         shape[3]};
  return generateTTIRReshape(input, newShape, rewriter);
}
class ConvertToFlattenedConv2dPattern
    : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputTy = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputTy = op.getResult().getType();

    Value flattenedInput = generateTTIRNHWFlatten(
        mlir::cast<mlir::TypedValue<RankedTensorType>>(adaptor.getInput()),
        rewriter);

    // Convolution in ttnn returns a tensor in a flattened shape
    // (1 x 1 x N * H * W x C)
    llvm::ArrayRef<std::int64_t> outputShape = outputTy.getShape();
    llvm::SmallVector<std::int64_t, 4> flattenedOutputShape = {
        1, 1, outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};
    outputTy = mlir::cast<RankedTensorType>(getTypeConverter()->convertType(
        outputTy.cloneWith(flattenedOutputShape, outputTy.getElementType())));

    outputTy = mlir::RankedTensorType::get(flattenedOutputShape,
                                           outputTy.getElementType(),
                                           outputTy.getEncoding());

    auto FlattenedCompatInfoAttr = ttir::FlattenedCompatInfoAttr::get(
        getContext(), inputTy.getDimSize(3), outputTy.getDimSize(3),
        inputTy.getDimSize(0), inputTy.getDimSize(1), inputTy.getDimSize(2));

    auto newConvDPS = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputTy.getShape(), outputTy.getElementType());
    auto newConv = cast<ttir::Conv2dOp>(rewriter.clone(*op));
    newConv.setFlattenedCompatInfoAttr(FlattenedCompatInfoAttr);
    newConv->setOperand(0, flattenedInput);
    newConv.setDpsInitOperand(0, newConvDPS);
    newConv.getResult().setType(outputTy);

    Value output = generateTTIRReshape(newConv, outputShape, rewriter);

    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

namespace {
class ConvertToMaxPool2dFlattenedCompatOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());

    Value flattenedInput = generateTTIRNHWFlatten(
        mlir::cast<mlir::TypedValue<RankedTensorType>>(adaptor.getInput()),
        rewriter);

    auto outputType = op.getResult().getType();
    llvm::ArrayRef<std::int64_t> outputShape = outputType.getShape();

    llvm::SmallVector<int64_t> flattenedOutputShape{
        1, 1, outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};

    auto newOutputType = mlir::RankedTensorType::get(
        flattenedOutputShape, outputType.getElementType(),
        outputType.getEncoding());

    auto newPoolDPS = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), newOutputType.getShape(), newOutputType.getElementType());

    auto FlattenedCompatInfoAttr = ttir::FlattenedCompatInfoAttr::get(
        getContext(), inputType.getDimSize(3), outputType.getDimSize(3),
        inputType.getDimSize(0), inputType.getDimSize(1),
        inputType.getDimSize(2));

    auto newPool = cast<ttir::MaxPool2dOp>(rewriter.clone(*op));
    newPool.setFlattenedCompatInfoAttr(FlattenedCompatInfoAttr);
    newPool->setOperand(0, flattenedInput);
    newPool.setDpsInitOperand(0, newPoolDPS);
    newPool.getResult().setType(newOutputType);

    Value output = generateTTIRReshape(newPool, outputShape, rewriter);

    rewriter.replaceOp(op, output);

    return success();
  }
};
} // namespace

class TTIRFlattenSlidingWindow
    : public impl::TTIRFlattenSlidingWindowBase<TTIRFlattenSlidingWindow> {
public:
  using impl::TTIRFlattenSlidingWindowBase<
      TTIRFlattenSlidingWindow>::TTIRFlattenSlidingWindowBase;

  void runOnOperation() final {
    RewritePatternSet conversionPatterns(&getContext());
    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });
    conversionPatterns.add<ConvertToFlattenedConv2dPattern>(typeConverter,
                                                            &getContext());
    conversionPatterns
        .add<ConvertToMaxPool2dFlattenedCompatOpConversionPattern>(
            typeConverter, &getContext());
    FrozenRewritePatternSet conversionPatternSet(std::move(conversionPatterns));

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalOp<tensor::EmptyOp>(); // DPS operands are create with

    target.addDynamicallyLegalOp<ttir::Conv2dOp>([&](ttir::Conv2dOp op) {
      return op.getFlattenedCompatInfo().has_value();
    });
    target.addDynamicallyLegalOp<ttir::MaxPool2dOp>([&](ttir::MaxPool2dOp op) {
      return op.getFlattenedCompatInfo().has_value();
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(conversionPatternSet)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
