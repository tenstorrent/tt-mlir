// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

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

RankedTensorType getNHWFlattenedType(RankedTensorType unflattenedOutputType) {
  llvm::ArrayRef<int64_t> outputShape = unflattenedOutputType.getShape();
  assert(outputShape.size() == 4 && "Expecting 4D tensor");
  llvm::SmallVector<int64_t, 4> flattenedOutputShape = {
      1, 1, outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};

  return RankedTensorType::get(flattenedOutputShape,
                               unflattenedOutputType.getElementType());
}

ttir::ReshapeOp generateReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                                RankedTensorType outputType,
                                PatternRewriter &rewriter) {
  // We cannot pass the shape directly as the attribute as ttir::ReshapeOp
  // requires that the shape attribute is a 32-bit integer array attribute.
  // Construction the SmallVector allows us to cast it.
  return ttir::utils::createDPSOp<ttir::ReshapeOp>(
      rewriter, ttmlir::utils::appendLocationSuffix(input.getLoc(), "_reshape"),
      outputType, input,
      rewriter.getI32ArrayAttr(SmallVector<int32_t>(
          outputType.getShape().begin(), outputType.getShape().end())));
}

class ConvertToFlattenedConv2dPattern
    : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputType = op.getInput().getType();
    auto outputType = op.getResult().getType();

    Value flattenedInput = generateReshape(
        op.getInput(), getNHWFlattenedType(inputType), rewriter);

    auto flattenedCompatInfoAttr = ttir::FlattenedCompatInfoAttr::get(
        getContext(), inputType.getDimSize(0), inputType.getDimSize(1),
        inputType.getDimSize(2));

    auto newConv = ttir::utils::createDPSOp<ttir::Conv2dOp>(
        rewriter, op.getLoc(), getNHWFlattenedType(outputType), flattenedInput,
        adaptor.getWeight(), adaptor.getBias(), adaptor.getStride(),
        adaptor.getPadding(), adaptor.getDilation(), adaptor.getGroups(),
        flattenedCompatInfoAttr);

    Value output = generateReshape(newConv, outputType, rewriter);

    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

namespace {
template <typename Pooling2dOp>
class Pooling2dFlattenedCompatOpConversionPattern
    : public OpConversionPattern<Pooling2dOp> {
public:
  using OpConversionPattern<Pooling2dOp>::OpConversionPattern;
  using Adaptor = typename Pooling2dOp::Adaptor;

  LogicalResult
  matchAndRewrite(Pooling2dOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = op.getInput().getType();
    auto outputType = op.getResult().getType();

    Value flattenedInput = generateReshape(
        op.getInput(), getNHWFlattenedType(inputType), rewriter);

    auto flattenedCompatInfoAttr = ttir::FlattenedCompatInfoAttr::get(
        inputType.getContext(), inputType.getDimSize(0),
        inputType.getDimSize(1), inputType.getDimSize(2));

    Pooling2dOp newPool;
    if constexpr (std::is_same_v<Pooling2dOp, ttir::MaxPool2dOp>) {
      newPool = ttir::utils::createDPSOp<Pooling2dOp>(
          rewriter, op.getLoc(), getNHWFlattenedType(outputType),
          flattenedInput, adaptor.getKernel(), adaptor.getStride(),
          adaptor.getDilation(), adaptor.getPadding(), adaptor.getCeilMode(),
          flattenedCompatInfoAttr);
    } else if constexpr (std::is_same_v<Pooling2dOp, ttir::AvgPool2dOp>) {
      newPool = ttir::utils::createDPSOp<Pooling2dOp>(
          rewriter, op.getLoc(), getNHWFlattenedType(outputType),
          flattenedInput, adaptor.getKernel(), adaptor.getStride(),
          adaptor.getDilation(), adaptor.getPadding(), adaptor.getCeilMode(),
          adaptor.getCountIncludePad(), flattenedCompatInfoAttr);
    } else {
      llvm_unreachable("Pool2dOp must be AvgPool2dOp or MaxPool2dOp");
    }

    Value output = generateReshape(newPool, outputType, rewriter);

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
    conversionPatterns
        .add<ConvertToFlattenedConv2dPattern,
             Pooling2dFlattenedCompatOpConversionPattern<ttir::MaxPool2dOp>,
             Pooling2dFlattenedCompatOpConversionPattern<ttir::AvgPool2dOp>>(
            typeConverter, &getContext());
    FrozenRewritePatternSet conversionPatternSet(std::move(conversionPatterns));

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();

    target.addDynamicallyLegalOp<ttir::Conv2dOp>([&](ttir::Conv2dOp op) {
      return op.getFlattenedCompatInfo() != nullptr;
    });
    target.addDynamicallyLegalOp<ttir::MaxPool2dOp>([&](ttir::MaxPool2dOp op) {
      return op.getFlattenedCompatInfo() != nullptr;
    });
    target.addDynamicallyLegalOp<ttir::AvgPool2dOp>([&](ttir::AvgPool2dOp op) {
      return op.getFlattenedCompatInfo() != nullptr;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(conversionPatternSet)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
