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
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOCOMPLEXDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// ComplexDataTypeConversion overview
//===----------------------------------------------------------------------===//
// Runs after `stablehlo-complex-math-expander` (real-domain elementwise ops).
//
//  (1) Unpack complex dtypes — append trailing dim of 2: [re0, im0, re1, ...]
//        tensor<4xcomplex<f32>>    -->  tensor<4x2xf32>
//        tensor<3x4xcomplex<f32>>  -->  tensor<3x4x2xf32>
//      Affected: func args/returns, constants, reshape, broadcast_in_dim.
//
//  (2) Decompose complex/real/imag --> slice/concat/reshape.
//      Trailing dim of 2 is not tile-divisible; decompositions transiently
//      move it to the leading position before operating on it.
//
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// Transpose helpers
// ---------------------------------------------------------------------------

// Moves the trailing dimension to the front:
//   tensor<d0 x d1 x ... x dN>  -->  tensor<dN x d0 x d1 x ... x d(N-1)>
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

// Moves the leading dimension to the back:
//   tensor<d0 x d1 x ... x dN>  -->  tensor<d1 x d2 x ... x dN x d0>
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

// Decomposes complex(re, im) -> tensor<...x2xf32>:
//
//   re: tensor<d0x...xdN>    im: tensor<d0x...xdN>
//       |                            |
//    reshape                      reshape
//       |                            |
//   tensor<1xd0x...xdN>      tensor<1xd0x...xdN>
//       \                           /
//              concatenate(dim=0)
//                     |
//          tensor<2xd0x...xdN>   <- [re_slice, im_slice, ...]
//                     |
//           transposeLeadingToTrailing
//                     |
//          tensor<d0x...xdNx2>   <- unpacked complex layout
//
namespace {
class StablehloComplexToDecomposedPattern
    : public OpConversionPattern<mlir::stablehlo::ComplexOp> {
  using OpConversionPattern<mlir::stablehlo::ComplexOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto lhsType = mlir::cast<RankedTensorType>(adaptor.getLhs().getType());

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

    auto transposed =
        transposeLeadingToTrailing(loc, concatOp.getResult(), rewriter);
    rewriter.replaceOp(op, transposed);
    return success();
  }
};
} // namespace

// Decomposes real(x) / imag(x) -> tensor<...xf32>:
//
//   tensor<d0x...xdNx2>   <- unpacked complex layout
//          |
//   transposeTrailingToLeading
//          |
//   tensor<2xd0x...xdN>
//          |
//   slice(dim=0, offset=0 or 1, len=1)   <- 0=real, 1=imag
//          |
//   tensor<1xd0x...xdN>
//          |
//       reshape
//          |
//   tensor<d0x...xdN>     <- extracted component
namespace {
template <typename OpTy>
class StablehloRealImagToDecomposedPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  static constexpr int Offset =
      std::is_same_v<OpTy, mlir::stablehlo::RealOp> ? 0 : 1;

public:
  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpConversionPattern<OpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto transposed =
        transposeTrailingToLeading(loc, adaptor.getOperand(), rewriter);
    auto transposedType = mlir::cast<RankedTensorType>(transposed.getType());
    int64_t rank = transposedType.getRank();
    auto transposedShape = transposedType.getShape();

    SmallVector<int64_t> begins(rank, 0),
        ends(transposedShape.begin(), transposedShape.end()), steps(rank, 1);
    begins[0] = Offset;
    ends[0] = Offset + 1;
    SmallVector<int64_t> sliceShape(transposedShape.begin(),
                                    transposedShape.end());
    sliceShape[0] = 1;
    auto sliceOp = rewriter.create<mlir::stablehlo::SliceOp>(
        loc, RankedTensorType::get(sliceShape, transposedType.getElementType()),
        transposed, rewriter.getDenseI64ArrayAttr(begins),
        rewriter.getDenseI64ArrayAttr(ends),
        rewriter.getDenseI64ArrayAttr(steps));

    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        op, resultType, sliceOp.getResult());
    return success();
  }
};
} // namespace

// Rewrites ops that produce complex-typed tensors to operate on the equivalent
// unpacked real representation (trailing dim of size 2).
namespace {
template <typename OpTy>
class ComplexTypeDefaultConversionPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpConversionPattern<OpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newResultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<OpTy>(op, TypeRange{newResultType},
                                      adaptor.getOperands(),
                                      op.getProperties());
    return success();
  }
};
} // namespace

// Rewrites stablehlo::ConstantOp with complex-typed tensor results by
// unpacking each complex element into a pair of floats (real, imag) and
// producing a new constant over the equivalent real tensor type.
namespace {
class ComplexConstantOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConstantOp> {
  using OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

public:
  LogicalResult matchAndRewrite(
      mlir::stablehlo::ConstantOp op,
      OpConversionPattern<mlir::stablehlo::ConstantOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto newResultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    auto denseAttr = mlir::cast<DenseElementsAttr>(op.getValue());
    SmallVector<APFloat> floatValues;
    for (auto complexVal : denseAttr.getValues<std::complex<APFloat>>()) {
      floatValues.push_back(complexVal.real());
      floatValues.push_back(complexVal.imag());
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, newResultType, DenseElementsAttr::get(newResultType, floatValues));
    return success();
  }
};

// Rewrites stablehlo::BroadcastInDimOp with complex-typed tensor results
// by appending the trailing real/imag dimension to the broadcast dimensions
// and producing a new broadcast over the equivalent real tensor type.
class ComplexBroadcastInDimOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

public:
  LogicalResult matchAndRewrite(
      mlir::stablehlo::BroadcastInDimOp op,
      OpConversionPattern<mlir::stablehlo::BroadcastInDimOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto newResultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    auto dims = op.getBroadcastDimensions();
    SmallVector<int64_t> newDims(dims.begin(), dims.end());
    newDims.push_back(newResultType.getRank() - 1);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
        op, newResultType, adaptor.getOperand(),
        rewriter.getDenseI64ArrayAttr(newDims));
    return success();
  }
};

// Rewrites stablehlo::SliceOp with complex-typed tensor results by appending
// a full-range slice (0:2:1) for the trailing real/imag dimension.
class ComplexSliceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::SliceOp> {
  using OpConversionPattern<mlir::stablehlo::SliceOp>::OpConversionPattern;

public:
  LogicalResult matchAndRewrite(
      mlir::stablehlo::SliceOp op,
      OpConversionPattern<mlir::stablehlo::SliceOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto newResultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    SmallVector<int64_t> newStartIndices(op.getStartIndices());
    SmallVector<int64_t> newLimitIndices(op.getLimitIndices());
    SmallVector<int64_t> newStrides(op.getStrides());

    // adding [0:2:1] slice means "select all from the new trailing dimension"
    newStartIndices.push_back(0);
    newLimitIndices.push_back(2);
    newStrides.push_back(1);

    rewriter.replaceOpWithNewOp<mlir::stablehlo::SliceOp>(
        op, newResultType, adaptor.getOperand(),
        rewriter.getDenseI64ArrayAttr(newStartIndices),
        rewriter.getDenseI64ArrayAttr(newLimitIndices),
        rewriter.getDenseI64ArrayAttr(newStrides));
    return success();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Shardy sharding annotation helper
// ---------------------------------------------------------------------------

// When a complex-typed tensor gains a trailing dim of 2, its sharding
// annotation needs an extra dimension entry: closed, unsharded.
// Example: tensor<16x16xcomplex<f32>> with sharding [{}, {}]
//       -> tensor<16x16x2xf32>       with sharding [{}, {}, {}]
static mlir::sdy::TensorShardingPerValueAttr convertShardingsForComplexTypes(
    MLIRContext *ctx, mlir::sdy::TensorShardingPerValueAttr shardings,
    TypeRange originalTypes) {
  SmallVector<mlir::sdy::TensorShardingAttr> newShardings;
  for (auto [sharding, type] :
       llvm::zip_equal(shardings.getShardings(), originalTypes)) {
    auto rtt = mlir::dyn_cast<RankedTensorType>(type);
    if (rtt && mlir::isa<ComplexType>(rtt.getElementType())) {
      // Append a closed, empty DimensionShardingAttr for the trailing dim.
      SmallVector<mlir::sdy::DimensionShardingAttr> dims(
          sharding.getDimShardings().begin(),
          sharding.getDimShardings().end());
      dims.push_back(mlir::sdy::DimensionShardingAttr::get(
          ctx, /*axes=*/{}, /*isClosed=*/true));
      newShardings.push_back(mlir::sdy::TensorShardingAttr::get(
          ctx, sharding.getMeshOrRef(), dims, sharding.getReplicatedAxes(),
          sharding.getUnreducedAxes()));
    } else {
      newShardings.push_back(sharding);
    }
  }
  return mlir::sdy::TensorShardingPerValueAttr::get(ctx, newShardings);
}

// ---------------------------------------------------------------------------
// Shardy ManualComputation complex type conversion
// ---------------------------------------------------------------------------

// Converts sdy.manual_computation ops that have complex-typed operands,
// results, or region block arguments. Updates the op types, sharding
// annotations, and converts block arg types so that the existing complex
// decomposition patterns (ComplexOp, RealOp, ImagOp, etc.) can fire on
// ops inside the region.
namespace {
class ShardyManualComputationComplexConversionPattern
    : public OpConversionPattern<mlir::sdy::ManualComputationOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::sdy::ManualComputationOp op,
                  mlir::sdy::ManualComputationOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type converted = getTypeConverter()->convertType(type);
      if (!converted) {
        return failure();
      }
      newResultTypes.push_back(converted);
    }

    // Update sharding annotations for the new trailing dimension.
    auto newInShardings = convertShardingsForComplexTypes(
        op.getContext(), op.getInShardings(),
        op.getBody().getArgumentTypes());
    auto newOutShardings = convertShardingsForComplexTypes(
        op.getContext(), op.getOutShardings(), op.getResultTypes());

    // Build via OperationState so no implicit block is auto-created;
    // we then inline the original region and convert its block arg types.
    OperationState state(op.getLoc(),
                         mlir::sdy::ManualComputationOp::getOperationName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(newResultTypes);
    state.addAttribute(
        mlir::sdy::ManualComputationOp::getInShardingsAttrName(state.name),
        newInShardings);
    state.addAttribute(
        mlir::sdy::ManualComputationOp::getOutShardingsAttrName(state.name),
        newOutShardings);
    state.addAttribute(
        mlir::sdy::ManualComputationOp::getManualAxesAttrName(state.name),
        op.getManualAxesAttr());
    Region *newRegion = state.addRegion();
    rewriter.inlineRegionBefore(op.getBody(), *newRegion, newRegion->end());

    Operation *newOpBase = rewriter.create(state);

    // Convert block argument types (complex<f32> -> x2xf32).
    auto newOp = cast<mlir::sdy::ManualComputationOp>(newOpBase);
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(),
                                           *getTypeConverter()))) {
      return failure();
    }

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};
} // namespace

namespace {
struct StableHLOComplexDataTypeConversionPass
    : public impl::StableHLOComplexDataTypeConversionPassBase<
          StableHLOComplexDataTypeConversionPass> {
  using impl::StableHLOComplexDataTypeConversionPassBase<
      StableHLOComplexDataTypeConversionPass>::
      StableHLOComplexDataTypeConversionPassBase;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();

    auto isNotComplexType = [](mlir::Operation *op) {
      auto resultType =
          mlir::cast<RankedTensorType>(op->getResult(0).getType());
      return !mlir::isa<mlir::ComplexType>(resultType.getElementType());
    };

    auto hasComplexElementType = [](Type type) {
      auto rtt = mlir::dyn_cast<RankedTensorType>(type);
      return rtt && mlir::isa<ComplexType>(rtt.getElementType());
    };

    target.addDynamicallyLegalOp<
        mlir::stablehlo::ConstantOp, mlir::stablehlo::ReshapeOp,
        mlir::stablehlo::SliceOp, mlir::stablehlo::ConcatenateOp,
        mlir::stablehlo::BroadcastInDimOp>(isNotComplexType);

    target.addIllegalOp<mlir::stablehlo::ComplexOp, mlir::stablehlo::RealOp,
                        mlir::stablehlo::ImagOp>();

    target.addDynamicallyLegalOp<mlir::sdy::ManualComputationOp>(
        [&](mlir::sdy::ManualComputationOp op) {
          return llvm::none_of(op.getBody().getArgumentTypes(),
                               hasComplexElementType) &&
                 llvm::none_of(op.getResultTypes(), hasComplexElementType);
        });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
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
    patterns.add<
        ComplexBroadcastInDimOpConversionPattern,
        ComplexConstantOpConversionPattern, ComplexSliceOpConversionPattern,
        ComplexTypeDefaultConversionPattern<mlir::stablehlo::ConcatenateOp>,
        ComplexTypeDefaultConversionPattern<mlir::stablehlo::ReshapeOp>,
        ShardyManualComputationComplexConversionPattern,
        StablehloComplexToDecomposedPattern,
        StablehloRealImagToDecomposedPattern<mlir::stablehlo::RealOp>,
        StablehloRealImagToDecomposedPattern<mlir::stablehlo::ImagOp>>(
        typeConverter, &getContext());

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
