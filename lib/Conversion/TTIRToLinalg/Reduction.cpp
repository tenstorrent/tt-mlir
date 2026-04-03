// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/Reduction.h"
#include "ttmlir/Conversion/TTIRToLinalg/Utils.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

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
#include <numeric>

namespace mlir::tt::ttir_to_linalg {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

// Extract reduce dimensions from a TTIR_ReductionOp's dim_arg attribute.
// Normalizes negative dims. If dim_arg is absent (nullopt), returns all dims.
SmallVector<int64_t> getReduceDims(std::optional<ArrayAttr> dimArg,
                                   int64_t rank) {
  // nullopt means reduce all dims; empty array means reduce no dims (identity).
  if (!dimArg) {
    SmallVector<int64_t> allDims(rank);
    std::iota(allDims.begin(), allDims.end(), 0);
    return allDims;
  }
  SmallVector<int64_t> dims;
  for (auto dim : *dimArg) {
    int64_t d = cast<IntegerAttr>(dim).getInt();
    dims.push_back(d < 0 ? d + rank : d);
  }
  return dims;
}

// Compute the reduced shape and classify each dimension as parallel or
// reduction. Returns the reducedShape and populates iteratorTypes and
// outputExprs.
static SmallVector<int64_t>
computeReductionMeta(int64_t rank, ArrayRef<int64_t> inputShape,
                     ArrayRef<int64_t> reduceDims,
                     SmallVector<utils::IteratorType> &iteratorTypes,
                     SmallVector<AffineExpr> &outputExprs, MLIRContext *ctx) {
  SmallVector<int64_t> reducedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (llvm::is_contained(reduceDims, i)) {
      iteratorTypes.push_back(utils::IteratorType::reduction);
    } else {
      iteratorTypes.push_back(utils::IteratorType::parallel);
      reducedShape.push_back(inputShape[i]);
      outputExprs.push_back(getAffineDimExpr(i, ctx));
    }
  }
  return reducedShape;
}

// Reshape result to reinsert reduced dimensions as size-1 if keepDim is true.
static Value reshapeForKeepDim(Value result, RankedTensorType resultType,
                               int64_t rank, ArrayRef<int64_t> inputShape,
                               ArrayRef<int64_t> reduceDims, Location loc,
                               ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> keepDimShape;
  for (int64_t i = 0; i < rank; ++i) {
    keepDimShape.push_back(llvm::is_contained(reduceDims, i) ? 1
                                                             : inputShape[i]);
  }
  auto shapeType =
      tosa::shapeType::get(rewriter.getContext(), keepDimShape.size());
  auto shapeOp = rewriter.create<tosa::ConstShapeOp>(
      loc, shapeType, rewriter.getIndexTensorAttr(keepDimShape));
  return rewriter.create<tosa::ReshapeOp>(loc, resultType, result, shapeOp);
}

} // namespace

//===----------------------------------------------------------------------===//
// TOSA-based reduction patterns (Min, Max, Sum, Prod, Mean, ArgMax)
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Simple TOSA chain reduction (Min, Max, Sum, Prod)
//===----------------------------------------------------------------------===//

template <typename TTIROp, typename TosaReductionOp>
class SimpleReductionOpConversionPattern : public OpConversionPattern<TTIROp> {
public:
  using OpConversionPattern<TTIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROp op, typename TTIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    SmallVector<int64_t> dims = getReduceDims(op.getDimArg(), rank);
    bool keepDim = op.getKeepDim();

    Value result = createReductionOpChain<TosaReductionOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};

using MinOpConversionPattern =
    SimpleReductionOpConversionPattern<ttir::MinOp, tosa::ReduceMinOp>;
using MaxOpConversionPattern =
    SimpleReductionOpConversionPattern<ttir::MaxOp, tosa::ReduceMaxOp>;
using SumOpConversionPattern =
    SimpleReductionOpConversionPattern<ttir::SumOp, tosa::ReduceSumOp>;
using ProdOpConversionPattern =
    SimpleReductionOpConversionPattern<ttir::ProdOp, tosa::ReduceProductOp>;

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

class MeanOpConversionPattern : public OpConversionPattern<ttir::MeanOp> {
public:
  using OpConversionPattern<ttir::MeanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto inputElementType = inputType.getElementType();
    auto resultElementType = resultType.getElementType();
    if (isa<IntegerType>(inputElementType) &&
        isa<FloatType>(resultElementType)) {
      auto floatInputType =
          RankedTensorType::get(inputType.getShape(), resultElementType);
      auto intType = cast<IntegerType>(inputElementType);
      const bool useUnsignedCast =
          intType.isUnsigned() || intType.getWidth() == 1;
      if (useUnsignedCast) {
        input = rewriter.create<arith::UIToFPOp>(op.getLoc(), floatInputType,
                                                 input);
      } else {
        input = rewriter.create<arith::SIToFPOp>(op.getLoc(), floatInputType,
                                                 input);
      }
      inputType = cast<RankedTensorType>(input.getType());
    }

    SmallVector<int64_t> dims = getReduceDims(op.getDimArg(), rank);
    bool keepDim = op.getKeepDim();

    Value sum = createReductionOpChain<tosa::ReduceSumOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    int64_t numElements = 1;
    for (int64_t dim : dims) {
      numElements *= inputType.getShape()[dim];
    }

    DenseElementsAttr divisorAttr =
        createDenseElementsAttr(resultType, static_cast<double>(numElements));
    if (!divisorAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for mean");
    }

    auto divisor =
        rewriter.create<tosa::ConstOp>(op.getLoc(), resultType, divisorAttr);

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());

    auto divOp = rewriter.create<linalg::DivOp>(
        op.getLoc(), resultType, ValueRange{sum, divisor}, output.getResult());

    rewriter.replaceOp(op, divOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

// TOSA has no ArgMax reduction op, so we implement this using
// linalg::GenericOp with two output tensors (max values + max indices).
// After TTIRToTTIRDecomposition, ArgMaxOp arrives here with either no dim_arg
// (reduce all dimensions) or exactly 1 reduce dim.
// The tracked index is a linearized (flattened) index across reduce dimensions.
class ArgMaxOpConversionPattern : public OpConversionPattern<ttir::ArgMaxOp> {
public:
  using OpConversionPattern<ttir::ArgMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArgMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elementType = inputType.getElementType();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto dimArg = op.getDimArg();
    if (!(!dimArg || dimArg->size() <= 1)) {
      return rewriter.notifyMatchFailure(
          op, "Multi-dim argmax should have been decomposed.");
    }

    SmallVector<int64_t> reduceDims = getReduceDims(op.getDimArg(), rank);
    bool keepDim = op.getKeepDim();

    SmallVector<utils::IteratorType> iteratorTypes;
    SmallVector<AffineExpr> outputExprs;
    SmallVector<int64_t> reducedShape =
        computeReductionMeta(rank, inputType.getShape(), reduceDims,
                             iteratorTypes, outputExprs, rewriter.getContext());

    auto maxValuesType = RankedTensorType::get(reducedShape, elementType);
    auto maxIndicesType =
        RankedTensorType::get(reducedShape, rewriter.getI32Type());

    // Initialize max values to -inf (float) or INT_MIN (integer), and max
    // indices to 0.
    Value initMax;
    if (isa<FloatType>(elementType)) {
      auto negInfAttr = rewriter.getFloatAttr(
          elementType,
          APFloat::getInf(cast<FloatType>(elementType).getFloatSemantics(),
                          /*Negative=*/true));
      initMax =
          rewriter.create<arith::ConstantOp>(loc, elementType, negInfAttr);
    } else {
      auto intType = cast<IntegerType>(elementType);
      unsigned bitWidth = intType.getWidth();
      APInt minValue(bitWidth, /*val=*/0, /*isSigned=*/false);
      if (!intType.isUnsignedInteger()) {
        minValue = APInt::getSignedMinValue(bitWidth);
      }
      auto minAttr = rewriter.getIntegerAttr(elementType, minValue);
      initMax = rewriter.create<arith::ConstantOp>(loc, elementType, minAttr);
    }
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    Value maxValuesFilled =
        rewriter
            .create<linalg::FillOp>(
                loc, initMax,
                rewriter.create<tensor::EmptyOp>(loc, reducedShape, elementType)
                    .getResult())
            .getResult(0);
    Value maxIndicesFilled =
        rewriter
            .create<linalg::FillOp>(
                loc, zero,
                rewriter
                    .create<tensor::EmptyOp>(loc, reducedShape,
                                             rewriter.getI32Type())
                    .getResult())
            .getResult(0);

    AffineMap inputMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    AffineMap outputMap =
        AffineMap::get(rank, 0, outputExprs, rewriter.getContext());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{maxValuesType, maxIndicesType}, ValueRange{input},
        ValueRange{maxValuesFilled, maxIndicesFilled},
        SmallVector<AffineMap>{inputMap, outputMap, outputMap}, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value currentVal = args[0];
          Value currentMax = args[1];
          Value currentIdx = args[2];

          // Compute linearized index across reduce dimensions (row-major).
          Value linearIdx = nullptr;
          for (int64_t d : reduceDims) {
            Value idx = b.create<arith::IndexCastOp>(
                loc, b.getI32Type(), b.create<linalg::IndexOp>(loc, d));
            if (!linearIdx) {
              linearIdx = idx;
            } else {
              Value dimSize = b.create<arith::ConstantOp>(
                  loc, b.getI32Type(),
                  b.getI32IntegerAttr(inputType.getShape()[d]));
              linearIdx = b.create<arith::AddIOp>(
                  loc, b.create<arith::MulIOp>(loc, linearIdx, dimSize), idx);
            }
          }

          Value isGreater;
          if (isa<FloatType>(elementType)) {
            isGreater = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                                currentVal, currentMax)
                            .getResult();
          } else {
            auto intType = cast<IntegerType>(elementType);
            arith::CmpIPredicate pred = intType.isUnsignedInteger()
                                            ? arith::CmpIPredicate::ugt
                                            : arith::CmpIPredicate::sgt;
            isGreater =
                b.create<arith::CmpIOp>(loc, pred, currentVal, currentMax)
                    .getResult();
          }
          Value newMax =
              b.create<arith::SelectOp>(loc, isGreater, currentVal, currentMax);
          Value newIdx =
              b.create<arith::SelectOp>(loc, isGreater, linearIdx, currentIdx);
          b.create<linalg::YieldOp>(loc, ValueRange{newMax, newIdx});
        });

    Value result = genericOp.getResult(1);

    if (keepDim) {
      result = reshapeForKeepDim(result, resultType, rank, inputType.getShape(),
                                 reduceDims, loc, rewriter);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceAndOp / ReduceOrOp
//===----------------------------------------------------------------------===//

// Generic boolean reduction pattern used by both ReduceAnd and ReduceOr.
// Converts input to 0/1 values (nonzero → 1, zero → 0) using an elementwise
// linalg.generic, then applies tosa::ReduceMinOp (AND) or tosa::ReduceMaxOp
// (OR) via createReductionOpChain.
template <typename ReduceOp, typename TosaReductionOp>
class BooleanReductionOpConversionPattern
    : public OpConversionPattern<ReduceOp> {
public:
  using OpConversionPattern<ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, typename ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elementType = inputType.getElementType();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    SmallVector<int64_t> dims = getReduceDims(op.getDimArg(), rank);
    bool keepDim = op.getKeepDim();

    // Step 1: Convert input to 0/1 in the original element type.
    // nonzero → 1, zero → 0.
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> allParallel(rank,
                                                 utils::IteratorType::parallel);

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), elementType);

    auto booleanize = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{inputType}, ValueRange{input}, ValueRange{emptyTensor},
        SmallVector<AffineMap>{identityMap, identityMap}, allParallel,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value elem = args[0];

          Value isNonZero;
          Value zeroVal, oneVal;
          if (isa<FloatType>(elementType)) {
            zeroVal = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elementType, 0.0));
            oneVal = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elementType, 1.0));
            isNonZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                                elem, zeroVal);
          } else {
            zeroVal = b.create<arith::ConstantOp>(
                loc, b.getIntegerAttr(elementType, 0));
            oneVal = b.create<arith::ConstantOp>(
                loc, b.getIntegerAttr(elementType, 1));
            isNonZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                elem, zeroVal);
          }

          Value result =
              b.create<arith::SelectOp>(loc, isNonZero, oneVal, zeroVal);
          b.create<linalg::YieldOp>(loc, result);
        });

    // Step 2: Apply TOSA reduction on the 0/1 tensor.
    // ReduceMin for AND (min of all 1s = 1; any 0 → 0).
    // ReduceMax for OR (max of all 0s = 0; any 1 → 1).
    Value result = createReductionOpChain<TosaReductionOp>(
        booleanize.getResult(0), resultType, dims, keepDim, loc, rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};

using ReduceAndOpConversionPattern =
    BooleanReductionOpConversionPattern<ttir::ReduceAndOp, tosa::ReduceMinOp>;
using ReduceOrOpConversionPattern =
    BooleanReductionOpConversionPattern<ttir::ReduceOrOp, tosa::ReduceMaxOp>;

} // namespace

//===----------------------------------------------------------------------===//
// Linalg-based reduction patterns (CumSum)
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//

// Cumulative sum computes the running sum along a specified dimension.
// Since this is a scan operation (not a reduction), we cannot use the standard
// TOSA reduce operations. Instead, we unroll the scan at compile time,
// generating a sequence of slice+add+insert operations for each position
// along the scan dimension.
class CumSumOpConversionPattern : public OpConversionPattern<ttir::CumSumOp> {
public:
  using OpConversionPattern<ttir::CumSumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CumSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    int64_t dim = normalizeDim(op.getDim(), rank);

    int64_t dimSize = inputType.getShape()[dim];
    Type elementType = inputType.getElementType();

    // Create an initial output tensor filled with zeros.
    DenseElementsAttr zeroAttr = createDenseElementsAttr(resultType, 0.0);
    if (!zeroAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for cumsum");
    }
    Value output =
        rewriter.create<arith::ConstantOp>(loc, resultType, zeroAttr);

    // Compute the slice type (same shape but with dim size = 1).
    SmallVector<int64_t> sliceShape(inputType.getShape());
    sliceShape[dim] = 1;
    auto sliceType = RankedTensorType::get(sliceShape, elementType);

    // Create a zero-filled tensor for the running sum accumulator.
    DenseElementsAttr zeroSliceAttr = createDenseElementsAttr(sliceType, 0.0);
    Value runningSum =
        rewriter.create<arith::ConstantOp>(loc, sliceType, zeroSliceAttr);

    // Build the static sizes and strides for slice operations.
    SmallVector<OpFoldResult> staticSizes;
    SmallVector<OpFoldResult> staticStrides(rank, rewriter.getIndexAttr(1));
    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim) {
        staticSizes.push_back(rewriter.getIndexAttr(1));
      } else {
        staticSizes.push_back(rewriter.getIndexAttr(inputType.getShape()[i]));
      }
    }

    // Unroll the scan loop at compile time.
    for (int64_t idx = 0; idx < dimSize; ++idx) {
      SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
      offsets[dim] = rewriter.getIndexAttr(idx);

      Value inputSlice = rewriter.create<tensor::ExtractSliceOp>(
          loc, sliceType, input, offsets, staticSizes, staticStrides);

      auto emptySlice =
          rewriter.create<tensor::EmptyOp>(loc, sliceShape, elementType);
      auto addOp = rewriter.create<linalg::AddOp>(
          loc, sliceType, ValueRange{runningSum, inputSlice},
          emptySlice.getResult());
      runningSum = addOp.getResult(0);

      output = rewriter.create<tensor::InsertSliceOp>(
          loc, runningSum, output, offsets, staticSizes, staticStrides);
    }

    rewriter.replaceOp(op, output);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgReductionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<CumSumOpConversionPattern>(typeConverter, ctx);
}

void populateTTIRToTosaReductionPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<MinOpConversionPattern, MaxOpConversionPattern,
               SumOpConversionPattern, ProdOpConversionPattern,
               MeanOpConversionPattern, ArgMaxOpConversionPattern,
               ReduceAndOpConversionPattern, ReduceOrOpConversionPattern>(
      typeConverter, ctx);
}

} // namespace mlir::tt::ttir_to_linalg
