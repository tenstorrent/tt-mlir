// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/Pooling.h"
#include "ttmlir/Conversion/TTIRToLinalg/Utils.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

#include <cstdint>

namespace mlir::tt::ttir_to_linalg {

//===----------------------------------------------------------------------===//
// Linalg conversion patterns
//===----------------------------------------------------------------------===//

namespace {
class MaxPool2dOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto input = adaptor.getInput();
    auto strides = adaptor.getStride();
    auto kernel = adaptor.getKernel();
    auto padding = adaptor.getPadding();
    auto dilation = adaptor.getDilation();

    // Expand stride if it contains only one element.
    auto stridesResult = ttmlir::utils::getPairOfInteger<int32_t>(strides);
    if (!stridesResult) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }
    auto [strideH, strideW] = *stridesResult;

    auto paddingResult = ttmlir::utils::getQuadrupleOfInteger<int32_t>(padding);
    if (!paddingResult) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }

    auto [paddingTop, paddingLeft, paddingBottom, paddingRight] =
        *paddingResult;

    // Expand kernel if it contains only one element.
    auto kernelResult = ttmlir::utils::getPairOfInteger<int32_t>(kernel);
    if (!kernelResult) {
      return rewriter.notifyMatchFailure(
          op, "kernel must be an integer or array attribute");
    }
    auto [kernelH, kernelW] = *kernelResult;

    auto dilationResult = ttmlir::utils::getPairOfInteger<int32_t>(dilation);
    if (!dilationResult) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }
    auto [dilationH, dilationW] = *dilationResult;

    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Handle flattened input.
    auto flatInfo = op.getFlattenedCompatInfoAttr();
    RankedTensorType savedFlatResultType;
    if (flatInfo) {
      savedFlatResultType = resultType;
      input = unflattenInput(input, flatInfo, rewriter, loc);
      inputType = cast<RankedTensorType>(input.getType());
      int64_t inH = flatInfo.getInputHeight();
      int64_t inW = flatInfo.getInputWidth();
      int64_t outH =
          (inH + paddingTop + paddingBottom - dilationH * (kernelH - 1) - 1) /
              strideH +
          1;
      int64_t outW =
          (inW + paddingLeft + paddingRight - dilationW * (kernelW - 1) - 1) /
              strideW +
          1;
      resultType = RankedTensorType::get({flatInfo.getBatchSize(), outH, outW,
                                          savedFlatResultType.getShape()[3]},
                                         savedFlatResultType.getElementType());
    }

    Type elementType = resultType.getElementType();
    int64_t inputH = inputType.getShape()[1];
    int64_t inputW = inputType.getShape()[2];

    // When ceil_mode is enabled, add extra padding to ensure the linalg op
    // produces enough output elements for the ceil-mode output shape.
    if (adaptor.getCeilMode()) {
      paddingBottom += calculateExtraPadding(
          inputH, kernelH, strideH, paddingTop, paddingBottom, dilationH);
      paddingRight += calculateExtraPadding(
          inputW, kernelW, strideW, paddingLeft, paddingRight, dilationW);
    }

    int64_t dilatedKernelH = (kernelH - 1) * dilationH + 1;
    int64_t dilatedKernelW = (kernelW - 1) * dilationW + 1;
    int64_t outputH =
        (inputH + paddingTop + paddingBottom - dilatedKernelH) / strideH + 1;
    int64_t outputW =
        (inputW + paddingLeft + paddingRight - dilatedKernelW) / strideW + 1;

    SmallVector<int64_t> actualResultShape(resultType.getShape());
    actualResultShape[1] = outputH;
    actualResultShape[2] = outputW;
    auto actualResultType =
        RankedTensorType::get(actualResultShape, elementType);

    // Max pooling identity: -inf for floats, INT_MIN for integers.
    Value negInfVal;
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      auto negInf = APFloat::getInf(floatType.getFloatSemantics(),
                                    /*Negative=*/true);
      negInfVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elementType, negInf));
    } else {
      auto intType = cast<IntegerType>(elementType);
      auto minVal = APInt::getSignedMinValue(intType.getWidth());
      negInfVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elementType, minVal));
    }

    // Pad input with -inf if needed.
    int64_t batch = inputType.getShape()[0];
    int64_t channels = inputType.getShape()[3];
    int64_t paddedH = inputH + paddingTop + paddingBottom;
    int64_t paddedW = inputW + paddingLeft + paddingRight;
    bool hasPadding = paddingTop > 0 || paddingBottom > 0 || paddingLeft > 0 ||
                      paddingRight > 0;

    Value paddedInput = input;
    if (hasPadding) {
      SmallVector<OpFoldResult> lowPad = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingTop),
          rewriter.getIndexAttr(paddingLeft), rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> highPad = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingBottom),
          rewriter.getIndexAttr(paddingRight), rewriter.getIndexAttr(0)};
      auto paddedType = RankedTensorType::get(
          {batch, paddedH, paddedW, channels}, elementType);
      paddedInput = rewriter.create<tensor::PadOp>(loc, paddedType, input,
                                                   lowPad, highPad, negInfVal);
    }

    // Create kernel tensor, strides, and dilations.
    auto kernelType =
        RankedTensorType::get({kernelH, kernelW}, rewriter.getF32Type());
    Value kernelTensor = rewriter.create<tensor::EmptyOp>(
        loc, kernelType.getShape(), kernelType.getElementType());

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        ArrayRef<int64_t>{strideH, strideW});
    auto dilationsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        ArrayRef<int64_t>{dilationH, dilationW});

    // Init output with -inf and run max pooling.
    Value outputInit = rewriter.create<tensor::EmptyOp>(
        loc, actualResultType.getShape(), elementType);
    Value poolOutput =
        rewriter.create<linalg::FillOp>(loc, negInfVal, outputInit)
            .getResult(0);

    auto poolOp = rewriter.create<linalg::PoolingNhwcMaxOp>(
        loc, TypeRange{actualResultType}, ValueRange{paddedInput, kernelTensor},
        ValueRange{poolOutput}, stridesAttr, dilationsAttr);
    Value result = poolOp.getResult(0);

    result = sliceResultToShape(result, resultType, rewriter, loc);

    if (flatInfo) {
      result = createTosaReshape(result, savedFlatResultType, rewriter, loc);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class AvgPool2dOpConversionPattern
    : public OpConversionPattern<ttir::AvgPool2dOp> {
public:
  using OpConversionPattern<ttir::AvgPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto strides = adaptor.getStride();
    auto kernel = adaptor.getKernel();
    auto padding = adaptor.getPadding();
    auto dilation = adaptor.getDilation();

    // Parse stride attribute.
    auto stridesResult = ttmlir::utils::getPairOfInteger<int32_t>(strides);
    if (!stridesResult) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }
    auto [strideH, strideW] = *stridesResult;

    // Parse padding attribute.
    auto paddingResult = ttmlir::utils::getQuadrupleOfInteger<int32_t>(padding);
    if (!paddingResult) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }
    auto [paddingTop, paddingLeft, paddingBottom, paddingRight] =
        *paddingResult;

    // Parse kernel attribute.
    auto kernelResult = ttmlir::utils::getPairOfInteger<int32_t>(kernel);
    if (!kernelResult) {
      return rewriter.notifyMatchFailure(
          op, "kernel must be an integer or array attribute");
    }
    auto [kernelH, kernelW] = *kernelResult;

    // Parse dilation attribute.
    auto dilationResult = ttmlir::utils::getPairOfInteger<int32_t>(dilation);
    if (!dilationResult) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }
    auto [dilationH, dilationW] = *dilationResult;

    bool countIncludePad = adaptor.getCountIncludePad();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Handle flattened input.
    auto flatInfo = op.getFlattenedCompatInfoAttr();
    RankedTensorType savedFlatResultType;
    if (flatInfo) {
      savedFlatResultType = resultType;
      input = unflattenInput(input, flatInfo, rewriter, loc);
      inputType = cast<RankedTensorType>(input.getType());
      int64_t inH = flatInfo.getInputHeight();
      int64_t inW = flatInfo.getInputWidth();
      int64_t outH =
          (inH + paddingTop + paddingBottom - dilationH * (kernelH - 1) - 1) /
              strideH +
          1;
      int64_t outW =
          (inW + paddingLeft + paddingRight - dilationW * (kernelW - 1) - 1) /
              strideW +
          1;
      resultType = RankedTensorType::get({flatInfo.getBatchSize(), outH, outW,
                                          savedFlatResultType.getShape()[3]},
                                         savedFlatResultType.getElementType());
    }

    Type elementType = resultType.getElementType();
    int64_t inputH = inputType.getShape()[1];
    int64_t inputW = inputType.getShape()[2];

    // When ceil_mode is enabled, add extra padding to ensure the linalg op
    // produces enough output elements for the ceil-mode output shape.
    int32_t extraPadBottom = 0, extraPadRight = 0;
    if (adaptor.getCeilMode()) {
      extraPadBottom = calculateExtraPadding(
          inputH, kernelH, strideH, paddingTop, paddingBottom, dilationH);
      extraPadRight = calculateExtraPadding(
          inputW, kernelW, strideW, paddingLeft, paddingRight, dilationW);
      paddingBottom += extraPadBottom;
      paddingRight += extraPadRight;
    }

    // Calculate output spatial dimensions.
    int64_t dilatedKernelH = (kernelH - 1) * dilationH + 1;
    int64_t dilatedKernelW = (kernelW - 1) * dilationW + 1;
    int64_t outputH =
        (inputH + paddingTop + paddingBottom - dilatedKernelH) / strideH + 1;
    int64_t outputW =
        (inputW + paddingLeft + paddingRight - dilatedKernelW) / strideW + 1;

    // Compute actual result shape (may differ from expected due to extra
    // padding).
    SmallVector<int64_t> actualResultShape(resultType.getShape());
    actualResultShape[1] = outputH;
    actualResultShape[2] = outputW;
    auto actualResultType =
        RankedTensorType::get(actualResultShape, elementType);

    // Use sum pooling + division for both count_include_pad cases.
    // The difference is only in how the divisor is computed:
    // - count_include_pad=true: constant divisor (kernel_h * kernel_w)
    // - count_include_pad=false: sum-pool a ones tensor to get per-position
    // counts
    int64_t batch = inputType.getShape()[0];
    int64_t channels = inputType.getShape()[3];
    int64_t paddedH = inputH + paddingTop + paddingBottom;
    int64_t paddedW = inputW + paddingLeft + paddingRight;

    bool hasPadding = paddingTop > 0 || paddingBottom > 0 || paddingLeft > 0 ||
                      paddingRight > 0;

    // Create zero constant for padding and fill operations.
    Value zeroVal;
    if (isa<FloatType>(elementType)) {
      zeroVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elementType, 0.0));
    } else {
      zeroVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elementType, 0));
    }

    // Create padding attributes.
    SmallVector<OpFoldResult> lowPad = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingTop),
        rewriter.getIndexAttr(paddingLeft), rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> highPad = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingBottom),
        rewriter.getIndexAttr(paddingRight), rewriter.getIndexAttr(0)};
    auto paddedType =
        RankedTensorType::get({batch, paddedH, paddedW, channels}, elementType);

    // Pad the input tensor if needed.
    Value paddedInput = input;
    if (hasPadding) {
      paddedInput = rewriter.create<tensor::PadOp>(loc, paddedType, input,
                                                   lowPad, highPad, zeroVal);
    }

    // Create the kernel tensor (shape only, values don't matter for pooling).
    auto linalgKernelType =
        RankedTensorType::get({kernelH, kernelW}, rewriter.getF32Type());
    Value kernelTensor = rewriter.create<tensor::EmptyOp>(
        loc, linalgKernelType.getShape(), linalgKernelType.getElementType());

    // Create strides and dilations attributes for linalg.pooling_nhwc_sum.
    auto linalgStridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        ArrayRef<int64_t>{strideH, strideW});
    auto dilationsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        ArrayRef<int64_t>{dilationH, dilationW});

    // Create output tensor initialized to zero for sum accumulation.
    Value sumOutputInit = rewriter.create<tensor::EmptyOp>(
        loc, actualResultType.getShape(), elementType);
    Value sumOutput =
        rewriter.create<linalg::FillOp>(loc, zeroVal, sumOutputInit)
            .getResult(0);

    // Perform sum pooling on input.
    auto sumPoolOp = rewriter.create<linalg::PoolingNhwcSumOp>(
        loc, TypeRange{actualResultType}, ValueRange{paddedInput, kernelTensor},
        ValueRange{sumOutput}, linalgStridesAttr, dilationsAttr);
    Value sumResult = sumPoolOp.getResult(0);

    // Compute divisor tensor by sum-pooling a binary mask of valid positions.
    // The mask shape depends on count_include_pad:
    // - count_include_pad=true:  ones cover input + user padding.
    // - count_include_pad=false: ones cover input only.
    // In both cases, ceil-mode extra padding positions get zeros.
    // Padding the mask with zeros and sum-pooling it counts exactly how many
    // valid elements fall into each sliding window.
    Value oneVal;
    if (isa<FloatType>(elementType)) {
      oneVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elementType, 1.0));
    } else {
      oneVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elementType, 1));
    }

    SmallVector<int64_t> onesShape;
    SmallVector<OpFoldResult> onesLowPad, onesHighPad;

    if (countIncludePad) {
      // Ones tensor covers input + user padding. Ceil extra padding gets zeros.
      int32_t userPadBottom = paddingBottom - extraPadBottom;
      int32_t userPadRight = paddingRight - extraPadRight;
      onesShape = {batch, inputH + paddingTop + userPadBottom,
                   inputW + paddingLeft + userPadRight, channels};
      onesLowPad = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
                    rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
      onesHighPad = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(extraPadBottom),
          rewriter.getIndexAttr(extraPadRight), rewriter.getIndexAttr(0)};
    } else {
      // Ones tensor covers input only. All padding gets zeros.
      onesShape = SmallVector<int64_t>(inputType.getShape());
      onesLowPad = lowPad;
      onesHighPad = highPad;
    }

    Value onesInit =
        rewriter.create<tensor::EmptyOp>(loc, onesShape, elementType);
    Value onesTensor =
        rewriter.create<linalg::FillOp>(loc, oneVal, onesInit).getResult(0);

    // Pad ones tensor with zeros.
    Value paddedOnes = rewriter.create<tensor::PadOp>(
        loc, paddedType, onesTensor, onesLowPad, onesHighPad, zeroVal);

    // Sum-pool the mask to count valid elements per window.
    Value countOutputInit = rewriter.create<tensor::EmptyOp>(
        loc, actualResultType.getShape(), elementType);
    Value countOutput =
        rewriter.create<linalg::FillOp>(loc, zeroVal, countOutputInit)
            .getResult(0);

    auto countPoolOp = rewriter.create<linalg::PoolingNhwcSumOp>(
        loc, TypeRange{actualResultType}, ValueRange{paddedOnes, kernelTensor},
        ValueRange{countOutput}, linalgStridesAttr, dilationsAttr);
    Value divisorTensor = countPoolOp.getResult(0);

    // Divide sum by divisor to get average.
    Value avgOutputInit = rewriter.create<tensor::EmptyOp>(
        loc, actualResultType.getShape(), elementType);
    auto divOp = rewriter.create<linalg::DivOp>(
        loc, actualResultType, ValueRange{sumResult, divisorTensor},
        avgOutputInit);
    Value result = divOp.getResult(0);

    result = sliceResultToShape(result, resultType, rewriter, loc);

    if (flatInfo) {
      result = createTosaReshape(result, savedFlatResultType, rewriter, loc);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class MaxPool2dWithIndicesOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dWithIndicesOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dWithIndicesOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dWithIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto strides = adaptor.getStride();
    auto kernel = adaptor.getKernel();
    auto padding = adaptor.getPadding();
    auto dilation = adaptor.getDilation();

    // Parse stride.
    auto stridesResult = ttmlir::utils::getPairOfInteger<int32_t>(strides);
    if (!stridesResult) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }
    auto [strideH, strideW] = *stridesResult;

    // Parse padding.
    auto paddingResult = ttmlir::utils::getQuadrupleOfInteger<int32_t>(padding);
    if (!paddingResult) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }
    auto [paddingTop, paddingLeft, paddingBottom, paddingRight] =
        *paddingResult;

    // Parse kernel.
    auto kernelResult = ttmlir::utils::getPairOfInteger<int32_t>(kernel);
    if (!kernelResult) {
      return rewriter.notifyMatchFailure(
          op, "kernel must be an integer or array attribute");
    }
    auto [kernelH, kernelW] = *kernelResult;

    // Parse dilation.
    auto dilationResult = ttmlir::utils::getPairOfInteger<int32_t>(dilation);
    if (!dilationResult) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }
    auto [dilationH, dilationW] = *dilationResult;

    auto inputType = cast<RankedTensorType>(input.getType());
    auto valuesResultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    auto indicesResultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResultIndices().getType()));
    assert(valuesResultType && "Values result type must be a ranked tensor.");
    assert(indicesResultType && "Indices result type must be a ranked tensor.");

    // Handle flattened input.
    auto flatInfo = op.getFlattenedCompatInfoAttr();
    RankedTensorType savedFlatValuesType, savedFlatIndicesType;
    if (flatInfo) {
      savedFlatValuesType = valuesResultType;
      savedFlatIndicesType = indicesResultType;
      input = unflattenInput(input, flatInfo, rewriter, loc);
      inputType = cast<RankedTensorType>(input.getType());
      int64_t inH = flatInfo.getInputHeight();
      int64_t inW = flatInfo.getInputWidth();
      int64_t outH =
          (inH + paddingTop + paddingBottom - dilationH * (kernelH - 1) - 1) /
              strideH +
          1;
      int64_t outW =
          (inW + paddingLeft + paddingRight - dilationW * (kernelW - 1) - 1) /
              strideW +
          1;
      valuesResultType =
          RankedTensorType::get({flatInfo.getBatchSize(), outH, outW,
                                 savedFlatValuesType.getShape()[3]},
                                savedFlatValuesType.getElementType());
      indicesResultType =
          RankedTensorType::get({flatInfo.getBatchSize(), outH, outW,
                                 savedFlatIndicesType.getShape()[3]},
                                indicesResultType.getElementType());
    }

    Type elementType = valuesResultType.getElementType();
    Type indexElementType = indicesResultType.getElementType();
    int64_t inputH = inputType.getShape()[1];
    int64_t inputW = inputType.getShape()[2];

    // Handle ceil_mode.
    if (adaptor.getCeilMode()) {
      paddingBottom += calculateExtraPadding(
          inputH, kernelH, strideH, paddingTop, paddingBottom, dilationH);
      paddingRight += calculateExtraPadding(
          inputW, kernelW, strideW, paddingLeft, paddingRight, dilationW);
    }

    int64_t dilatedKernelH = (kernelH - 1) * dilationH + 1;
    int64_t dilatedKernelW = (kernelW - 1) * dilationW + 1;
    int64_t outputH =
        (inputH + paddingTop + paddingBottom - dilatedKernelH) / strideH + 1;
    int64_t outputW =
        (inputW + paddingLeft + paddingRight - dilatedKernelW) / strideW + 1;

    SmallVector<int64_t> actualResultShape(valuesResultType.getShape());
    actualResultShape[1] = outputH;
    actualResultShape[2] = outputW;
    auto actualValuesType =
        RankedTensorType::get(actualResultShape, elementType);
    auto actualIndicesType =
        RankedTensorType::get(actualResultShape, indexElementType);

    // Max pooling identity: -inf for floats, INT_MIN for integers.
    Value negInfVal;
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      auto negInf = APFloat::getInf(floatType.getFloatSemantics(),
                                    /*Negative=*/true);
      negInfVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elementType, negInf));
    } else {
      auto intType = cast<IntegerType>(elementType);
      auto minVal = APInt::getSignedMinValue(intType.getWidth());
      negInfVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elementType, minVal));
    }

    // Pad input with -inf if needed.
    int64_t batch = inputType.getShape()[0];
    int64_t channels = inputType.getShape()[3];
    int64_t paddedH = inputH + paddingTop + paddingBottom;
    int64_t paddedW = inputW + paddingLeft + paddingRight;
    bool hasPadding = paddingTop > 0 || paddingBottom > 0 || paddingLeft > 0 ||
                      paddingRight > 0;

    Value paddedInput = input;
    if (hasPadding) {
      SmallVector<OpFoldResult> lowPad = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingTop),
          rewriter.getIndexAttr(paddingLeft), rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> highPad = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(paddingBottom),
          rewriter.getIndexAttr(paddingRight), rewriter.getIndexAttr(0)};
      auto paddedType = RankedTensorType::get(
          {batch, paddedH, paddedW, channels}, elementType);
      paddedInput = rewriter.create<tensor::PadOp>(loc, paddedType, input,
                                                   lowPad, highPad, negInfVal);
    }

    // Initialize outputs: values with -inf, indices with 0.
    Value zeroIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexElementType, 0));

    Value valuesInit = rewriter.create<tensor::EmptyOp>(
        loc, actualValuesType.getShape(), elementType);
    Value valuesOutput =
        rewriter.create<linalg::FillOp>(loc, negInfVal, valuesInit)
            .getResult(0);

    Value indicesInit = rewriter.create<tensor::EmptyOp>(
        loc, actualIndicesType.getShape(), indexElementType);
    Value indicesOutput =
        rewriter.create<linalg::FillOp>(loc, zeroIdx, indicesInit).getResult(0);

    // Build linalg.generic to compute max values and indices simultaneously.
    // Iteration domain: [N, outH, outW, C, kH, kW]
    //   - N, outH, outW, C are parallel (output dims)
    //   - kH, kW are reduction (kernel window dims)
    //
    // We cannot use an affine indexing map for the input because
    // strided/dilated access (e.g. d1*2 + d4) is non-invertible. Instead, the
    // input is accessed via tensor.extract inside the body using manually
    // computed coordinates.

    // Create a dummy input tensor for the kernel window dimensions so that
    // linalg.generic knows the reduction domain extents.
    auto windowType =
        RankedTensorType::get({kernelH, kernelW}, rewriter.getF32Type());
    Value windowTensor = rewriter.create<tensor::EmptyOp>(
        loc, windowType.getShape(), windowType.getElementType());

    auto *ctx = rewriter.getContext();
    unsigned numDims = 6; // N, outH, outW, C, kH, kW

    // Window: (n, oh, ow, c, kh, kw) -> (kh, kw)
    auto dimKH = rewriter.getAffineDimExpr(4);
    auto dimKW = rewriter.getAffineDimExpr(5);
    auto windowMap = AffineMap::get(numDims, 0, {dimKH, dimKW}, ctx);

    // Output: (n, oh, ow, c, kh, kw) -> (n, oh, ow, c)
    auto dimN = rewriter.getAffineDimExpr(0);
    auto dimOH = rewriter.getAffineDimExpr(1);
    auto dimOW = rewriter.getAffineDimExpr(2);
    auto dimC = rewriter.getAffineDimExpr(3);
    auto outputMap =
        AffineMap::get(numDims, 0, {dimN, dimOH, dimOW, dimC}, ctx);

    SmallVector<AffineMap> indexingMaps = {windowMap, outputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,  // N
        utils::IteratorType::parallel,  // outH
        utils::IteratorType::parallel,  // outW
        utils::IteratorType::parallel,  // C
        utils::IteratorType::reduction, // kH
        utils::IteratorType::reduction  // kW
    };

    // Hoist structured binding values for lambda capture (C++17 compat).
    int32_t sH = strideH, sW = strideW, dH = dilationH, dW = dilationW;
    int32_t pTop = paddingTop, pLeft = paddingLeft;
    int64_t inW = inputW;

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{actualValuesType, actualIndicesType},
        /*inputs=*/ValueRange{windowTensor},
        /*outputs=*/ValueRange{valuesOutput, indicesOutput}, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange bodyArgs) {
          // bodyArgs[0] = window element (unused), [1] = running max,
          // [2] = running index
          Value runningMax = bodyArgs[1];
          Value runningIdx = bodyArgs[2];

          // Get iteration indices.
          Value n = b.create<linalg::IndexOp>(bodyLoc, 0);
          Value oh = b.create<linalg::IndexOp>(bodyLoc, 1);
          Value ow = b.create<linalg::IndexOp>(bodyLoc, 2);
          Value c = b.create<linalg::IndexOp>(bodyLoc, 3);
          Value kh = b.create<linalg::IndexOp>(bodyLoc, 4);
          Value kw = b.create<linalg::IndexOp>(bodyLoc, 5);

          // Compute padded input coordinates as index type.
          // hIdx = oh * strideH + kh * dilationH
          // wIdx = ow * strideW + kw * dilationW
          Value strideHIdx = b.create<arith::ConstantIndexOp>(bodyLoc, sH);
          Value dilationHIdx = b.create<arith::ConstantIndexOp>(bodyLoc, dH);
          Value strideWIdx = b.create<arith::ConstantIndexOp>(bodyLoc, sW);
          Value dilationWIdx = b.create<arith::ConstantIndexOp>(bodyLoc, dW);

          Value hIdx = b.create<arith::AddIOp>(
              bodyLoc, b.create<arith::MulIOp>(bodyLoc, oh, strideHIdx),
              b.create<arith::MulIOp>(bodyLoc, kh, dilationHIdx));
          Value wIdx = b.create<arith::AddIOp>(
              bodyLoc, b.create<arith::MulIOp>(bodyLoc, ow, strideWIdx),
              b.create<arith::MulIOp>(bodyLoc, kw, dilationWIdx));

          // Extract the element from the padded input.
          Value curVal = b.create<tensor::ExtractOp>(
              bodyLoc, paddedInput, ValueRange{n, hIdx, wIdx, c});

          // Compare: is current value > running max?
          Value cmp;
          if (isa<FloatType>(elementType)) {
            cmp = b.create<arith::CmpFOp>(bodyLoc, arith::CmpFPredicate::OGT,
                                          curVal, runningMax);
          } else {
            cmp = b.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::sgt,
                                          curVal, runningMax);
          }

          // Compute the flattened index into the *original* (unpadded) input.
          // h_in = hIdx - paddingTop, w_in = wIdx - paddingLeft
          // flat_index = h_in * inputW + w_in
          Value padTopIdx = b.create<arith::ConstantIndexOp>(bodyLoc, pTop);
          Value padLeftIdx = b.create<arith::ConstantIndexOp>(bodyLoc, pLeft);
          Value inputWIdx = b.create<arith::ConstantIndexOp>(bodyLoc, inW);

          Value hIn = b.create<arith::SubIOp>(bodyLoc, hIdx, padTopIdx);
          Value wIn = b.create<arith::SubIOp>(bodyLoc, wIdx, padLeftIdx);
          Value flatIdxIndex = b.create<arith::AddIOp>(
              bodyLoc, b.create<arith::MulIOp>(bodyLoc, hIn, inputWIdx), wIn);

          // Cast flat index from index to the result index element type.
          Value flatIdx = b.create<arith::IndexCastOp>(
              bodyLoc, indexElementType, flatIdxIndex);

          // Select: if curVal > runningMax, use curVal and flatIdx.
          Value newMax =
              b.create<arith::SelectOp>(bodyLoc, cmp, curVal, runningMax);
          Value newIdx =
              b.create<arith::SelectOp>(bodyLoc, cmp, flatIdx, runningIdx);

          b.create<linalg::YieldOp>(bodyLoc, ValueRange{newMax, newIdx});
        });

    Value valuesResult = genericOp.getResult(0);
    Value indicesResult = genericOp.getResult(1);

    valuesResult =
        sliceResultToShape(valuesResult, valuesResultType, rewriter, loc);
    indicesResult =
        sliceResultToShape(indicesResult, indicesResultType, rewriter, loc);

    if (flatInfo) {
      valuesResult =
          createTosaReshape(valuesResult, savedFlatValuesType, rewriter, loc);
      indicesResult =
          createTosaReshape(indicesResult, savedFlatIndicesType, rewriter, loc);
    }

    rewriter.replaceOp(op, {valuesResult, indicesResult});
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TOSA conversion patterns
//===----------------------------------------------------------------------===//

namespace {
class GlobalAvgPool2dOpConversionPattern
    : public OpConversionPattern<ttir::GlobalAvgPool2dOp> {
public:
  using OpConversionPattern<ttir::GlobalAvgPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GlobalAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    int64_t inputHeight = inputType.getShape()[1];
    int64_t inputWidth = inputType.getShape()[2];

    // Global average pooling is equivalent to sum reduction over H,W followed
    // by division by (H * W). We use MeanOp-style implementation but reduce
    // only over dimensions 1 and 2 (H and W).

    // First, reduce along height dimension (dim 1).
    auto afterHeightReduceShape = resultType.getShape().vec();
    afterHeightReduceShape[1] = 1;
    afterHeightReduceShape[2] = inputWidth;
    auto heightReduceType = RankedTensorType::get(afterHeightReduceShape,
                                                  resultType.getElementType());

    auto heightAxisAttr = rewriter.getI32IntegerAttr(1);
    auto heightReduceResult = rewriter.create<tosa::ReduceSumOp>(
        loc, heightReduceType, input, heightAxisAttr);

    // Then, reduce along width dimension (dim 2).
    auto widthAxisAttr = rewriter.getI32IntegerAttr(2);
    auto widthReduceResult = rewriter.create<tosa::ReduceSumOp>(
        loc, resultType, heightReduceResult.getResult(), widthAxisAttr);

    // Divide by the total number of spatial elements (H * W).
    double spatialCount = static_cast<double>(inputHeight * inputWidth);
    auto elementType = resultType.getElementType();

    // Create a constant tensor with 1/spatialCount for division.
    SmallVector<int64_t> divisorShape(resultType.getRank(), 1);
    auto divisorType = RankedTensorType::get(divisorShape, elementType);

    DenseElementsAttr divisorAttr = DenseElementsAttr::get(
        divisorType, rewriter.getFloatAttr(elementType, 1.0 / spatialCount));
    auto divisorConst =
        rewriter.create<tosa::ConstOp>(loc, divisorType, divisorAttr);

    // Create shift tensor for tosa::MulOp (requires i8 tensor).
    auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
    auto shiftAttr =
        DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    // Multiply by reciprocal to get average.
    auto result = rewriter.create<tosa::MulOp>(
        loc, resultType, widthReduceResult.getResult(), divisorConst, shift);

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgPoolingPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<MaxPool2dOpConversionPattern, AvgPool2dOpConversionPattern,
               MaxPool2dWithIndicesOpConversionPattern>(typeConverter, ctx);
}

void populateTTIRToTosaPoolingPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<GlobalAvgPool2dOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt::ttir_to_linalg
