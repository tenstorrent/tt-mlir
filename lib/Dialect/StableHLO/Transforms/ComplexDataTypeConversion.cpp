// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOCOMPLEXDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Transpose helpers
// ---------------------------------------------------------------------------

// Move the trailing dim to the front: [..., 2] → [2, ...]
// perm = [rank-1, 0, 1, ..., rank-2]
static Value transposeTrailingToLeading(Location loc, Value input,
                                        OpBuilder &builder) {
  auto type = mlir::cast<RankedTensorType>(input.getType());
  int64_t rank = type.getRank();

  SmallVector<int64_t> perm;
  perm.push_back(rank - 1);
  for (int64_t i = 0; i < rank - 1; ++i) {
    perm.push_back(i);
  }

  SmallVector<int64_t> newShape;
  newShape.push_back(type.getShape()[rank - 1]);
  for (int64_t i = 0; i < rank - 1; ++i) {
    newShape.push_back(type.getShape()[i]);
  }

  auto newType = RankedTensorType::get(newShape, type.getElementType());
  return builder
      .create<mlir::stablehlo::TransposeOp>(loc, newType, input,
                                            builder.getDenseI64ArrayAttr(perm))
      .getResult();
}

// Move the leading dim to the trailing: [2, ...] → [..., 2]
// perm = [1, 2, ..., rank-1, 0]
static Value transposeLeadingToTrailing(Location loc, Value input,
                                        OpBuilder &builder) {
  auto type = mlir::cast<RankedTensorType>(input.getType());
  int64_t rank = type.getRank();

  SmallVector<int64_t> perm;
  for (int64_t i = 1; i < rank; ++i) {
    perm.push_back(i);
  }
  perm.push_back(0);

  SmallVector<int64_t> newShape;
  for (int64_t i = 1; i < rank; ++i) {
    newShape.push_back(type.getShape()[i]);
  }
  newShape.push_back(type.getShape()[0]);

  auto newType = RankedTensorType::get(newShape, type.getElementType());
  return builder
      .create<mlir::stablehlo::TransposeOp>(loc, newType, input,
                                            builder.getDenseI64ArrayAttr(perm))
      .getResult();
}

// ---------------------------------------------------------------------------
// Conversion patterns
// ---------------------------------------------------------------------------

// stablehlo.complex(lhs, rhs):
//   1. Unsqueeze lhs and rhs by prepending a size-1 dim → [1, ...]
//   2. Concatenate along dim 0                          → [2, ...]
//   3. Transpose dim 0 to trailing                      → [..., 2]
struct StablehloComplexToDecomposedPattern
    : public OpConversionPattern<mlir::stablehlo::ComplexOp> {
  using OpConversionPattern<mlir::stablehlo::ComplexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto lhsType = mlir::cast<RankedTensorType>(adaptor.getLhs().getType());

    // Step 1: unsqueeze by prepending dim of size 1 → [1, d0, d1, ...]
    SmallVector<int64_t> unsqueezedShape;
    unsqueezedShape.push_back(1);
    for (auto d : lhsType.getShape()) {
      unsqueezedShape.push_back(d);
    }
    auto unsqueezedType =
        RankedTensorType::get(unsqueezedShape, lhsType.getElementType());
    auto reshapedLhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc, unsqueezedType, adaptor.getLhs());
    auto reshapedRhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc, unsqueezedType, adaptor.getRhs());

    // Step 2: concatenate along dim 0 → [2, d0, d1, ...]
    SmallVector<int64_t> concatShape;
    concatShape.push_back(2);
    for (auto d : lhsType.getShape()) {
      concatShape.push_back(d);
    }
    auto concatType =
        RankedTensorType::get(concatShape, lhsType.getElementType());
    auto concatOp = rewriter.create<mlir::stablehlo::ConcatenateOp>(
        loc, concatType,
        ValueRange{reshapedLhs.getResult(), reshapedRhs.getResult()},
        /*dimension=*/0);

    // Step 3: transpose dim 0 to trailing → [d0, d1, ..., 2]
    auto transposed =
        transposeLeadingToTrailing(loc, concatOp.getResult(), rewriter);
    rewriter.replaceOp(op, transposed);
    return success();
  }
};

// stablehlo.real(operand: tensor<...x2xfN>):
//   1. Transpose trailing dim to front → [2, ...]
//   2. Slice dim 0 at [0, 1)           → [1, ...]
//   3. Reshape to squeeze the size-1   → [...]
struct StablehloRealToDecomposedPattern
    : public OpConversionPattern<mlir::stablehlo::RealOp> {
  using OpConversionPattern<mlir::stablehlo::RealOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::RealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: transpose [..., 2] → [2, ...]
    auto transposed =
        transposeTrailingToLeading(loc, adaptor.getOperand(), rewriter);
    auto transposedType = mlir::cast<RankedTensorType>(transposed.getType());
    int64_t rank = transposedType.getRank();
    auto transposedShape = transposedType.getShape();

    // Step 2: slice dim 0 at [0, 1) → real plane, shape [1, ...]
    SmallVector<int64_t> begins(rank, 0),
        ends(transposedShape.begin(), transposedShape.end()), steps(rank, 1);
    begins[0] = 0;
    ends[0] = 1;
    SmallVector<int64_t> sliceShape(transposedShape.begin(),
                                    transposedShape.end());
    sliceShape[0] = 1;
    auto sliceOp = rewriter.create<mlir::stablehlo::SliceOp>(
        loc, RankedTensorType::get(sliceShape, transposedType.getElementType()),
        transposed, rewriter.getDenseI64ArrayAttr(begins),
        rewriter.getDenseI64ArrayAttr(ends),
        rewriter.getDenseI64ArrayAttr(steps));

    // Step 3: squeeze size-1 leading dim away → [...]
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        op, resultType, sliceOp.getResult());
    return success();
  }
};

// stablehlo.imag(operand: tensor<...x2xfN>):
//   1. Transpose trailing dim to front → [2, ...]
//   2. Slice dim 0 at [1, 2)           → [1, ...]
//   3. Reshape to squeeze the size-1   → [...]
struct StablehloImagToDecomposedPattern
    : public OpConversionPattern<mlir::stablehlo::ImagOp> {
  using OpConversionPattern<mlir::stablehlo::ImagOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ImagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step 1: transpose [..., 2] → [2, ...]
    auto transposed =
        transposeTrailingToLeading(loc, adaptor.getOperand(), rewriter);
    auto transposedType = mlir::cast<RankedTensorType>(transposed.getType());
    int64_t rank = transposedType.getRank();
    auto transposedShape = transposedType.getShape();

    // Step 2: slice dim 0 at [1, 2) → imag plane, shape [1, ...]
    SmallVector<int64_t> begins(rank, 0),
        ends(transposedShape.begin(), transposedShape.end()), steps(rank, 1);
    begins[0] = 1;
    ends[0] = 2;
    SmallVector<int64_t> sliceShape(transposedShape.begin(),
                                    transposedShape.end());
    sliceShape[0] = 1;
    auto sliceOp = rewriter.create<mlir::stablehlo::SliceOp>(
        loc, RankedTensorType::get(sliceShape, transposedType.getElementType()),
        transposed, rewriter.getDenseI64ArrayAttr(begins),
        rewriter.getDenseI64ArrayAttr(ends),
        rewriter.getDenseI64ArrayAttr(steps));

    // Step 3: squeeze size-1 leading dim away → [...]
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        op, resultType, sliceOp.getResult());
    return success();
  }
};

// ConstantOp: no transpose needed.
// DenseElementsAttr stores complex values as interleaved (real, imag) pairs in
// row-major order, which already matches the [..., 2] trailing layout.
struct ConstantComplexConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConstantOp> {
  using OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::ComplexType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex element type");
    }

    auto newType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(resultType));
    auto denseAttr = mlir::cast<DenseElementsAttr>(op.getValue());

    // Interleaved (real, imag) pairs match [..., 2] row-major layout directly.
    SmallVector<APFloat> floatValues;
    for (auto complexVal : denseAttr.getValues<std::complex<APFloat>>()) {
      floatValues.push_back(complexVal.real());
      floatValues.push_back(complexVal.imag());
    }

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, newType, DenseElementsAttr::get(newType, floatValues));
    return success();
  }
};

// ReshapeOp on a complex result type:
//   1. Transpose trailing dim to front → [2, ...]
//   2. Reshape the spatial dims only   → [2, new_d0, new_d1, ...]
//   3. Transpose dim 0 back to trailing → [new_d0, new_d1, ..., 2]
struct ReshapeComplexConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using OpConversionPattern<mlir::stablehlo::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::ComplexType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex element type");
    }

    Location loc = op.getLoc();

    // Step 1: transpose [..., 2] → [2, ...]
    auto transposed =
        transposeTrailingToLeading(loc, adaptor.getOperand(), rewriter);

    // Step 2: reshape spatial dims, keeping leading '2' fixed.
    // Target spatial shape comes from the converted result type [..., 2],
    // which has the same trailing 2 and new spatial dims in front.
    auto newResultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(resultType));
    // newResultType is [...new_spatial..., 2]; build [2, ...new_spatial...]
    auto newResultShape = newResultType.getShape();
    SmallVector<int64_t> midShape;
    midShape.push_back(2);
    for (size_t i = 0; i < newResultShape.size() - 1; ++i) {
      midShape.push_back(newResultShape[i]);
    }
    auto midType =
        RankedTensorType::get(midShape, newResultType.getElementType());
    auto reshaped =
        rewriter.create<mlir::stablehlo::ReshapeOp>(loc, midType, transposed);

    // Step 3: transpose dim 0 back to trailing → [...new_spatial..., 2]
    auto result =
        transposeLeadingToTrailing(loc, reshaped.getResult(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// BroadcastInDimOp on a complex result type:
//   1. Transpose trailing dim to front on the operand → [2, ...]
//   2. BroadcastInDim with dim 0 mapped to new result's leading dim 0,
//      and all original dims shifted by +1              → [2,
//      ...new_spatial...]
//   3. Transpose dim 0 back to trailing                → [...new_spatial..., 2]
struct BroadcastInDimComplexConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::ComplexType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex element type");
    }

    Location loc = op.getLoc();

    // Step 1: transpose operand [..., 2] → [2, ...]
    auto transposedOperand =
        transposeTrailingToLeading(loc, adaptor.getOperand(), rewriter);

    // Step 2: broadcast into [2, ...new_spatial...].
    // Original broadcast_dimensions map operand spatial dims to result spatial
    // dims. After transposing, operand dim 0 is the '2' dim, and dims 1..N are
    // the original spatial dims. The new result is [2, ...new_spatial...], so
    // we map operand dim 0 → result dim 0, and each original dim d → result
    // dim d+1.
    auto newResultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(resultType));
    // newResultType is [...new_spatial..., 2]; build [2, ...new_spatial...]
    auto newResultShape = newResultType.getShape();
    SmallVector<int64_t> broadcastResultShape;
    broadcastResultShape.push_back(2);
    for (size_t i = 0; i < newResultShape.size() - 1; ++i) {
      broadcastResultShape.push_back(newResultShape[i]);
    }
    auto broadcastResultType = RankedTensorType::get(
        broadcastResultShape, newResultType.getElementType());

    auto dims = op.getBroadcastDimensions();
    SmallVector<int64_t> newDims;
    newDims.push_back(0); // operand dim 0 ('2') → result dim 0
    for (auto d : dims) {
      newDims.push_back(d + 1); // shift all original spatial dims by 1
    }

    auto broadcastOp = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, broadcastResultType, transposedOperand,
        rewriter.getDenseI64ArrayAttr(newDims));

    // Step 3: transpose dim 0 back to trailing → [...new_spatial..., 2]
    auto result =
        transposeLeadingToTrailing(loc, broadcastOp.getResult(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct StableHLOComplexDataTypeConversionPass
    : public impl::StableHLOComplexDataTypeConversionPassBase<
          StableHLOComplexDataTypeConversionPass> {
  using impl::StableHLOComplexDataTypeConversionPassBase<
      StableHLOComplexDataTypeConversionPass>::
      StableHLOComplexDataTypeConversionPassBase;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<BuiltinDialect>();

    target.addDynamicallyLegalOp<mlir::stablehlo::ConstantOp>(
        [](mlir::stablehlo::ConstantOp op) {
          auto resultType =
              mlir::cast<RankedTensorType>(op.getResult().getType());
          return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
        });

    target.addIllegalOp<mlir::stablehlo::ComplexOp>();
    target.addIllegalOp<mlir::stablehlo::RealOp>();
    target.addIllegalOp<mlir::stablehlo::ImagOp>();

    target.addDynamicallyLegalOp<mlir::stablehlo::ReshapeOp>(
        [](mlir::stablehlo::ReshapeOp op) {
          auto resultType =
              mlir::cast<RankedTensorType>(op.getResult().getType());
          return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
        });

    target.addDynamicallyLegalOp<mlir::stablehlo::BroadcastInDimOp>(
        [](mlir::stablehlo::BroadcastInDimOp op) {
          auto resultType =
              mlir::cast<RankedTensorType>(op.getResult().getType());
          return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
        });

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });
    // tensor<...xcomplex<fN>> → tensor<...x2xfN>: append trailing dim of 2.
    typeConverter.addConversion(
        [](RankedTensorType type) -> std::optional<Type> {
          auto complexTy =
              mlir::dyn_cast<mlir::ComplexType>(type.getElementType());
          if (!complexTy) {
            return std::nullopt;
          }
          auto floatTy =
              mlir::dyn_cast<mlir::FloatType>(complexTy.getElementType());
          if (!floatTy) {
            return std::nullopt;
          }
          SmallVector<int64_t> newShape(type.getShape());
          newShape.push_back(2);
          return RankedTensorType::get(newShape, floatTy);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantComplexConversionPattern>(typeConverter,
                                                   &getContext());
    patterns.add<StablehloComplexToDecomposedPattern>(typeConverter,
                                                      &getContext());
    patterns.add<StablehloRealToDecomposedPattern>(typeConverter,
                                                   &getContext());
    patterns.add<StablehloImagToDecomposedPattern>(typeConverter,
                                                   &getContext());
    patterns.add<ReshapeComplexConversionPattern>(typeConverter, &getContext());
    patterns.add<BroadcastInDimComplexConversionPattern>(typeConverter,
                                                         &getContext());

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::stablehlo
