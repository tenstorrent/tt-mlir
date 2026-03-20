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
// Data types
//===----------------------------------------------------------------------===//

namespace {

struct PoolingAttrs {
  int32_t strideH, strideW;
  int32_t paddingTop, paddingLeft, paddingBottom, paddingRight;
  int32_t kernelH, kernelW;
  int32_t dilationH, dilationW;

  template <typename Adaptor>
  static FailureOr<PoolingAttrs> parse(Adaptor adaptor, Operation *op,
                                       ConversionPatternRewriter &rewriter) {
    auto strides =
        ttmlir::utils::getPairOfInteger<int32_t>(adaptor.getStride());
    if (!strides) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }

    auto padding =
        ttmlir::utils::getQuadrupleOfInteger<int32_t>(adaptor.getPadding());
    if (!padding) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }

    auto kernel = ttmlir::utils::getPairOfInteger<int32_t>(adaptor.getKernel());
    if (!kernel) {
      return rewriter.notifyMatchFailure(
          op, "kernel must be an integer or array attribute");
    }

    auto dilation =
        ttmlir::utils::getPairOfInteger<int32_t>(adaptor.getDilation());
    if (!dilation) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }

    auto [sH, sW] = *strides;
    auto [pT, pL, pB, pR] = *padding;
    auto [kH, kW] = *kernel;
    auto [dH, dW] = *dilation;
    return PoolingAttrs{sH, sW, pT, pL, pB, pR, kH, kW, dH, dW};
  }
};

/// Computed output dimensions from PoolingAttrs + input spatial dims +
/// ceil_mode.
struct OutputDims {
  int64_t outputH, outputW;
  int32_t padBottom, padRight; // Ceil-mode-adjusted padding.
};

//===----------------------------------------------------------------------===//
// Free helper functions
//===----------------------------------------------------------------------===//

/// Compute output spatial dimensions from raw input/padding values.
std::pair<int64_t, int64_t> computeSpatialOutputDims(int64_t inputH,
                                                     int64_t inputW,
                                                     const PoolingAttrs &attrs,
                                                     int32_t padBottom,
                                                     int32_t padRight) {
  int64_t dilatedKernelH = (attrs.kernelH - 1) * attrs.dilationH + 1;
  int64_t dilatedKernelW = (attrs.kernelW - 1) * attrs.dilationW + 1;
  int64_t outH =
      (inputH + attrs.paddingTop + padBottom - dilatedKernelH) / attrs.strideH +
      1;
  int64_t outW =
      (inputW + attrs.paddingLeft + padRight - dilatedKernelW) / attrs.strideW +
      1;
  return {outH, outW};
}

/// Compute output spatial dimensions and ceil-mode-adjusted padding.
OutputDims computeOutputDims(const PoolingAttrs &attrs,
                             RankedTensorType inputType, bool ceilMode) {
  int64_t inputH = inputType.getShape()[1];
  int64_t inputW = inputType.getShape()[2];
  int32_t padBottom = attrs.paddingBottom;
  int32_t padRight = attrs.paddingRight;

  if (ceilMode) {
    padBottom +=
        calculateExtraPadding(inputH, attrs.kernelH, attrs.strideH,
                              attrs.paddingTop, padBottom, attrs.dilationH);
    padRight +=
        calculateExtraPadding(inputW, attrs.kernelW, attrs.strideW,
                              attrs.paddingLeft, padRight, attrs.dilationW);
  }

  auto [outputH, outputW] =
      computeSpatialOutputDims(inputH, inputW, attrs, padBottom, padRight);
  return {outputH, outputW, padBottom, padRight};
}

/// Pad an NHWC input tensor. Derives padded shape from input + padding values.
/// Returns input unchanged if all padding values are zero.
Value padInput(Value input, Value padValue, int32_t padTop, int32_t padLeft,
               int32_t padBottom, int32_t padRight,
               ConversionPatternRewriter &rewriter, Location loc) {
  if (padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0) {
    return input;
  }
  auto inputType = cast<RankedTensorType>(input.getType());
  auto shape = inputType.getShape();
  auto paddedType =
      RankedTensorType::get({shape[0], shape[1] + padTop + padBottom,
                             shape[2] + padLeft + padRight, shape[3]},
                            inputType.getElementType());
  SmallVector<OpFoldResult> lowPad = {
      rewriter.getIndexAttr(0), rewriter.getIndexAttr(padTop),
      rewriter.getIndexAttr(padLeft), rewriter.getIndexAttr(0)};
  SmallVector<OpFoldResult> highPad = {
      rewriter.getIndexAttr(0), rewriter.getIndexAttr(padBottom),
      rewriter.getIndexAttr(padRight), rewriter.getIndexAttr(0)};
  return rewriter.create<tensor::PadOp>(loc, paddedType, input, lowPad, highPad,
                                        padValue);
}

/// Create a tensor filled with a scalar value.
Value createFilledTensor(ArrayRef<int64_t> shape, Type elementType,
                         Value fillVal, ConversionPatternRewriter &rewriter,
                         Location loc) {
  Value empty = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
  return rewriter.create<linalg::FillOp>(loc, fillVal, empty).getResult(0);
}

/// Create kernel window tensor for pooling ops. Element type and values are
/// unused by linalg.pooling_nhwc_*; only the shape matters.
Value createKernelTensor(const PoolingAttrs &attrs,
                         ConversionPatternRewriter &rewriter, Location loc) {
  auto kernelType = RankedTensorType::get({attrs.kernelH, attrs.kernelW},
                                          rewriter.getF32Type());
  return rewriter.create<tensor::EmptyOp>(loc, kernelType.getShape(),
                                          kernelType.getElementType());
}

/// Create kernel window tensor and strides/dilations attributes.
std::tuple<Value, DenseIntElementsAttr, DenseIntElementsAttr>
createLinalgPoolingAttrs(const PoolingAttrs &attrs,
                         ConversionPatternRewriter &rewriter, Location loc) {
  Value kernelTensor = createKernelTensor(attrs, rewriter, loc);
  auto stridesAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({2}, rewriter.getI64Type()),
      ArrayRef<int64_t>{attrs.strideH, attrs.strideW});
  auto dilationsAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({2}, rewriter.getI64Type()),
      ArrayRef<int64_t>{attrs.dilationH, attrs.dilationW});
  return {kernelTensor, stridesAttr, dilationsAttr};
}

/// Run linalg.pooling_nhwc_sum and return the result.
Value sumPool(const PoolingAttrs &attrs, Value input,
              RankedTensorType resultType, Value zeroVal,
              ConversionPatternRewriter &rewriter, Location loc) {
  auto [kernelTensor, stridesAttr, dilationsAttr] =
      createLinalgPoolingAttrs(attrs, rewriter, loc);
  Value output =
      createFilledTensor(resultType.getShape(), resultType.getElementType(),
                         zeroVal, rewriter, loc);
  return rewriter
      .create<linalg::PoolingNhwcSumOp>(
          loc, TypeRange{resultType}, ValueRange{input, kernelTensor},
          ValueRange{output}, stridesAttr, dilationsAttr)
      .getResult(0);
}

/// Create the identity value for max pooling: -inf for floats, INT_MIN for
/// integers.
Value createMaxPoolIdentity(Type elementType,
                            ConversionPatternRewriter &rewriter, Location loc) {
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    auto negInf =
        APFloat::getInf(floatType.getFloatSemantics(), /*Negative=*/true);
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, negInf));
  }
  auto intType = cast<IntegerType>(elementType);
  auto minVal = APInt::getSignedMinValue(intType.getWidth());
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(elementType, minVal));
}

/// Unflatten input for pooling and compute unflattened output spatial dims.
/// Mutates `input` to the unflattened NHWC tensor. Returns {batch, outH, outW}.
std::tuple<int64_t, int64_t, int64_t>
unflattenForPooling(Value &input, const PoolingAttrs &attrs,
                    ttir::FlattenedCompatInfoAttr flatInfo,
                    ConversionPatternRewriter &rewriter, Location loc) {
  input = unflattenInput(input, flatInfo, rewriter, loc);
  auto [outH, outW] = computeSpatialOutputDims(
      flatInfo.getInputHeight(), flatInfo.getInputWidth(), attrs,
      attrs.paddingBottom, attrs.paddingRight);
  return {flatInfo.getBatchSize(), outH, outW};
}

/// Build a pool output type by overriding the H and W dims of a result type.
RankedTensorType makePoolOutputType(RankedTensorType resultType,
                                    int64_t outputH, int64_t outputW) {
  SmallVector<int64_t> shape(resultType.getShape());
  shape[1] = outputH;
  shape[2] = outputW;
  return RankedTensorType::get(shape, resultType.getElementType());
}

//===----------------------------------------------------------------------===//
// Linalg conversion patterns
//===----------------------------------------------------------------------===//

class MaxPool2dOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto attrsOr = PoolingAttrs::parse(adaptor, op, rewriter);

    if (failed(attrsOr)) {
      return failure();
    }

    const auto &attrs = *attrsOr;

    Value input = adaptor.getInput();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    auto flatteningInfo = op.getFlattenedCompatInfoAttr();
    if (flatteningInfo) {
      auto [batch, outH, outW] =
          unflattenForPooling(input, attrs, flatteningInfo, rewriter, loc);
      resultType =
          RankedTensorType::get({batch, outH, outW, resultType.getShape()[3]},
                                resultType.getElementType());
    }

    auto inputType = cast<RankedTensorType>(input.getType());
    auto [outputH, outputW, padBottom, padRight] =
        computeOutputDims(attrs, inputType, adaptor.getCeilMode());

    Type elementType = resultType.getElementType();
    auto poolOutputType = makePoolOutputType(resultType, outputH, outputW);

    Value identity = createMaxPoolIdentity(elementType, rewriter, loc);

    Value paddedInput =
        padInput(input, identity, attrs.paddingTop, attrs.paddingLeft,
                 padBottom, padRight, rewriter, loc);

    Value poolOutput = createFilledTensor(poolOutputType.getShape(),
                                          elementType, identity, rewriter, loc);

    auto [kernelTensor, stridesAttr, dilationsAttr] =
        createLinalgPoolingAttrs(attrs, rewriter, loc);

    Value result = rewriter
                       .create<linalg::PoolingNhwcMaxOp>(
                           loc, TypeRange{poolOutputType},
                           ValueRange{paddedInput, kernelTensor},
                           ValueRange{poolOutput}, stridesAttr, dilationsAttr)
                       .getResult(0);

    result = sliceResultToShape(result, resultType, rewriter, loc);

    // If the flattening was applied, reshape back to flattened types.
    if (flatteningInfo) {
      result = createTosaReshape(
          result, cast<RankedTensorType>(op.getResult().getType()), rewriter,
          loc);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

class AvgPool2dOpConversionPattern
    : public OpConversionPattern<ttir::AvgPool2dOp> {
public:
  using OpConversionPattern<ttir::AvgPool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto attrsOr = PoolingAttrs::parse(adaptor, op, rewriter);
    if (failed(attrsOr)) {
      return failure();
    }

    const auto &attrs = *attrsOr;

    Value input = adaptor.getInput();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    if (!isa<FloatType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "avg_pool2d requires floating-point element type");
    }

    auto flatteningInfo = op.getFlattenedCompatInfoAttr();
    if (flatteningInfo) {
      auto [batch, outH, outW] =
          unflattenForPooling(input, attrs, flatteningInfo, rewriter, loc);
      resultType =
          RankedTensorType::get({batch, outH, outW, resultType.getShape()[3]},
                                resultType.getElementType());
    }

    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t inputH = inputType.getShape()[1];
    int64_t inputW = inputType.getShape()[2];
    auto [outputH, outputW, padBottom, padRight] =
        computeOutputDims(attrs, inputType, adaptor.getCeilMode());

    Type elementType = resultType.getElementType();
    int64_t batch = inputType.getShape()[0];
    int64_t channels = inputType.getShape()[3];
    auto poolOutputType = makePoolOutputType(resultType, outputH, outputW);

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 0.0));

    Value paddedInput =
        padInput(input, zero, attrs.paddingTop, attrs.paddingLeft, padBottom,
                 padRight, rewriter, loc);

    // Sum-pool the input.
    Value sumResult =
        sumPool(attrs, paddedInput, poolOutputType, zero, rewriter, loc);

    // Compute divisor by sum-pooling a binary mask of valid positions.
    // - count_include_pad=true:  ones cover input + user padding;
    //   ceil-mode extra padding positions get zeros.
    // - count_include_pad=false: ones cover input only;
    //   all padding positions get zeros.
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 1.0));

    int64_t paddedH = inputH + attrs.paddingTop + padBottom;
    int64_t paddedW = inputW + attrs.paddingLeft + padRight;
    auto paddedType =
        RankedTensorType::get({batch, paddedH, paddedW, channels}, elementType);

    SmallVector<int64_t> onesShape;
    SmallVector<OpFoldResult> onesLowPad, onesHighPad;

    if (adaptor.getCountIncludePad()) {
      // Ones tensor covers input + user padding. Ceil extra padding gets zeros.
      int32_t ceilExtraBottom = padBottom - attrs.paddingBottom;
      int32_t ceilExtraRight = padRight - attrs.paddingRight;
      onesShape = {batch, inputH + attrs.paddingTop + attrs.paddingBottom,
                   inputW + attrs.paddingLeft + attrs.paddingRight, channels};
      onesLowPad = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
                    rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
      onesHighPad = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(ceilExtraBottom),
          rewriter.getIndexAttr(ceilExtraRight), rewriter.getIndexAttr(0)};
    } else {
      // Ones tensor covers input only. All padding gets zeros.
      onesShape = SmallVector<int64_t>(inputType.getShape());
      onesLowPad = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(attrs.paddingTop),
          rewriter.getIndexAttr(attrs.paddingLeft), rewriter.getIndexAttr(0)};
      onesHighPad = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(padBottom),
                     rewriter.getIndexAttr(padRight), rewriter.getIndexAttr(0)};
    }

    Value onesTensor =
        createFilledTensor(onesShape, elementType, one, rewriter, loc);
    Value paddedOnes = rewriter.create<tensor::PadOp>(
        loc, paddedType, onesTensor, onesLowPad, onesHighPad, zero);

    // Sum-pool the mask to count valid elements per window.
    Value divisor =
        sumPool(attrs, paddedOnes, poolOutputType, zero, rewriter, loc);

    // Divide sum by count to get average.
    Value avgOutputInit = rewriter.create<tensor::EmptyOp>(
        loc, poolOutputType.getShape(), elementType);
    Value result = rewriter
                       .create<linalg::DivOp>(loc, poolOutputType,
                                              ValueRange{sumResult, divisor},
                                              avgOutputInit)
                       .getResult(0);

    result = sliceResultToShape(result, resultType, rewriter, loc);
    if (flatteningInfo) {
      result = createTosaReshape(
          result, cast<RankedTensorType>(op.getResult().getType()), rewriter,
          loc);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MaxPool2dWithIndicesOpConversionPattern
    : public OpConversionPattern<ttir::MaxPool2dWithIndicesOp> {
public:
  using OpConversionPattern<ttir::MaxPool2dWithIndicesOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dWithIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto attrsOr = PoolingAttrs::parse(adaptor, op, rewriter);
    if (failed(attrsOr)) {
      return failure();
    }
    const auto &attrs = *attrsOr;

    Value input = adaptor.getInput();
    auto valuesResultType = cast<RankedTensorType>(op.getResult().getType());
    auto indicesResultType =
        cast<RankedTensorType>(op.getResultIndices().getType());

    auto flatInfo = op.getFlattenedCompatInfoAttr();
    if (flatInfo) {
      auto [batch, outH, outW] =
          unflattenForPooling(input, attrs, flatInfo, rewriter, loc);
      valuesResultType = RankedTensorType::get(
          {batch, outH, outW, valuesResultType.getShape()[3]},
          valuesResultType.getElementType());
      indicesResultType = RankedTensorType::get(
          {batch, outH, outW, indicesResultType.getShape()[3]},
          indicesResultType.getElementType());
    }

    auto inputType = cast<RankedTensorType>(input.getType());
    Type elementType = valuesResultType.getElementType();
    Type indexElementType = indicesResultType.getElementType();
    int64_t inputW = inputType.getShape()[2];

    auto [outputH, outputW, padBottom, padRight] =
        computeOutputDims(attrs, inputType, adaptor.getCeilMode());

    auto poolValuesType =
        makePoolOutputType(valuesResultType, outputH, outputW);
    auto poolIndicesType =
        makePoolOutputType(indicesResultType, outputH, outputW);

    Value identity = createMaxPoolIdentity(elementType, rewriter, loc);
    Value paddedInput =
        padInput(input, identity, attrs.paddingTop, attrs.paddingLeft,
                 padBottom, padRight, rewriter, loc);

    Value zeroIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexElementType, 0));

    Value valuesOutput = createFilledTensor(
        poolValuesType.getShape(), elementType, identity, rewriter, loc);
    Value indicesOutput = createFilledTensor(
        poolIndicesType.getShape(), indexElementType, zeroIdx, rewriter, loc);

    // Build linalg.generic to compute max values and indices simultaneously.
    // Iteration domain: [N, outH, outW, C, kH, kW]
    //   - N, outH, outW, C are parallel (output dims)
    //   - kH, kW are reduction (kernel window dims)
    //
    // We cannot use an affine indexing map for the input because
    // strided/dilated access (e.g. d1*2 + d4) is non-invertible. Instead, the
    // input is accessed via tensor.extract inside the body using manually
    // computed coordinates.
    Value windowTensor = createKernelTensor(attrs, rewriter, loc);

    auto *ctx = rewriter.getContext();
    unsigned numDims = 6; // N, outH, outW, C, kH, kW

    // Window: (n, oh, ow, c, kh, kw) -> (kh, kw)
    auto windowMap = AffineMap::get(
        numDims, 0,
        {rewriter.getAffineDimExpr(4), rewriter.getAffineDimExpr(5)}, ctx);
    // Output: (n, oh, ow, c, kh, kw) -> (n, oh, ow, c)
    auto outputMap = AffineMap::get(
        numDims, 0,
        {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1),
         rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(3)},
        ctx);

    SmallVector<AffineMap> indexingMaps = {windowMap, outputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,  // N
        utils::IteratorType::parallel,  // outH
        utils::IteratorType::parallel,  // outW
        utils::IteratorType::parallel,  // C
        utils::IteratorType::reduction, // kH
        utils::IteratorType::reduction  // kW
    };

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{poolValuesType, poolIndicesType},
        /*inputs=*/ValueRange{windowTensor},
        /*outputs=*/ValueRange{valuesOutput, indicesOutput}, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange bodyArgs) {
          // bodyArgs[0] = window element (unused), [1] = running max,
          // [2] = running index
          Value runningMax = bodyArgs[1];
          Value runningIdx = bodyArgs[2];

          Value n = b.create<linalg::IndexOp>(bodyLoc, 0);
          Value oh = b.create<linalg::IndexOp>(bodyLoc, 1);
          Value ow = b.create<linalg::IndexOp>(bodyLoc, 2);
          Value c = b.create<linalg::IndexOp>(bodyLoc, 3);
          Value kh = b.create<linalg::IndexOp>(bodyLoc, 4);
          Value kw = b.create<linalg::IndexOp>(bodyLoc, 5);

          // hIdx = oh * strideH + kh * dilationH
          // wIdx = ow * strideW + kw * dilationW
          Value strideHVal =
              b.create<arith::ConstantIndexOp>(bodyLoc, attrs.strideH);
          Value dilationHVal =
              b.create<arith::ConstantIndexOp>(bodyLoc, attrs.dilationH);
          Value strideWVal =
              b.create<arith::ConstantIndexOp>(bodyLoc, attrs.strideW);
          Value dilationWVal =
              b.create<arith::ConstantIndexOp>(bodyLoc, attrs.dilationW);

          Value hIdx = b.create<arith::AddIOp>(
              bodyLoc, b.create<arith::MulIOp>(bodyLoc, oh, strideHVal),
              b.create<arith::MulIOp>(bodyLoc, kh, dilationHVal));
          Value wIdx = b.create<arith::AddIOp>(
              bodyLoc, b.create<arith::MulIOp>(bodyLoc, ow, strideWVal),
              b.create<arith::MulIOp>(bodyLoc, kw, dilationWVal));

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

          // Flattened index into the *original* (unpadded) input:
          // flat_index = (hIdx - paddingTop) * inputW + (wIdx - paddingLeft)
          Value padTopIdx =
              b.create<arith::ConstantIndexOp>(bodyLoc, attrs.paddingTop);
          Value padLeftIdx =
              b.create<arith::ConstantIndexOp>(bodyLoc, attrs.paddingLeft);
          Value inputWIdx = b.create<arith::ConstantIndexOp>(bodyLoc, inputW);

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

    Value valuesResult = sliceResultToShape(genericOp.getResult(0),
                                            valuesResultType, rewriter, loc);
    Value indicesResult = sliceResultToShape(genericOp.getResult(1),
                                             indicesResultType, rewriter, loc);
    if (flatInfo) {
      valuesResult = createTosaReshape(
          valuesResult, cast<RankedTensorType>(op.getResult().getType()),
          rewriter, loc);
      indicesResult = createTosaReshape(
          indicesResult,
          cast<RankedTensorType>(op.getResultIndices().getType()), rewriter,
          loc);
    }

    rewriter.replaceOp(op, {valuesResult, indicesResult});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TOSA conversion patterns
//===----------------------------------------------------------------------===//

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
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    int64_t inputHeight = inputType.getShape()[1];
    int64_t inputWidth = inputType.getShape()[2];

    // Global average pooling: reduce_sum over H and W, multiply by reciprocal.

    // First, reduce along height dimension (dim 1).
    auto afterHeightReduceShape = resultType.getShape().vec();
    afterHeightReduceShape[1] = 1;
    afterHeightReduceShape[2] = inputWidth;
    auto heightReduceType = RankedTensorType::get(afterHeightReduceShape,
                                                  resultType.getElementType());

    auto heightReduceResult = rewriter.create<tosa::ReduceSumOp>(
        loc, heightReduceType, input, rewriter.getI32IntegerAttr(1));

    // Then, reduce along width dimension (dim 2).
    auto widthReduceResult = rewriter.create<tosa::ReduceSumOp>(
        loc, resultType, heightReduceResult.getResult(),
        rewriter.getI32IntegerAttr(2));

    // Divide by the total number of spatial elements (H * W).
    double spatialCount = static_cast<double>(inputHeight * inputWidth);
    auto elementType = resultType.getElementType();

    // Create a constant tensor with 1/spatialCount for multiplication.
    SmallVector<int64_t> divisorShape(resultType.getRank(), 1);
    auto divisorType = RankedTensorType::get(divisorShape, elementType);
    DenseElementsAttr divisorAttr = DenseElementsAttr::get(
        divisorType, rewriter.getFloatAttr(elementType, 1.0 / spatialCount));
    auto divisorConst =
        rewriter.create<tosa::ConstOp>(loc, divisorType, divisorAttr);

    Value shift = createTosaMulShift(rewriter, loc);
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
