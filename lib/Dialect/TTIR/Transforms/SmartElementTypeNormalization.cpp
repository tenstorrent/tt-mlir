// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_SMARTELEMENTTYPENORMALIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class I64ToI32AndF64ToF32TypeConverter : public TypeConverter {
public:
  I64ToI32AndF64ToF32TypeConverter() {
    addConversion(
        [](mlir::RankedTensorType type) -> std::optional<RankedTensorType> {
          Type elementType = type.getElementType();

          // Tensor-of-tile types use TTCore TileType as element type; do not
          // run float/integer normalization on them (TileType has
          // FloatTypeInterface but getFloatSemantics is not fully implemented,
          // issue https://github.com/tenstorrent/tt-mlir/issues/5124 ).
          if (mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
            return type;
          }

          if (mlir::isa<mlir::quant::QuantizedType>(elementType)) {
            return type;
          }

          if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
            if (intType.getWidth() == 64) {
              Type newElementType = IntegerType::get(type.getContext(), 32);
              return mlir::RankedTensorType::get(
                  type.getShape(), newElementType, type.getEncoding());
            }
          }

          if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
            if (floatType.getWidth() == 64) {
              Type newElementType = mlir::Float32Type::get(type.getContext());
              return mlir::RankedTensorType::get(
                  type.getShape(), newElementType, type.getEncoding());
            }
          }

          return type;
        });
    // Pass-through for non-tensor types (e.g. memref in function signatures)
    // so convertType never returns null and UniformTypeRewriter does not
    // build a FunctionType with null types. Reject RankedTensorType so the
    // conversion above is used for tensor types.
    addConversion([](Type type) -> std::optional<Type> {
      if (mlir::isa<mlir::RankedTensorType>(type)) {
        return std::nullopt;
      }
      return type;
    });
  }
};

class I64AndF64ConstantOpAttrRewriter
    : public mlir::OpRewritePattern<tt::ttir::ConstantOp> {
public:
  using mlir::OpRewritePattern<tt::ttir::ConstantOp>::OpRewritePattern;

  I64AndF64ConstantOpAttrRewriter(const mlir::TypeConverter &converter,
                                  mlir::MLIRContext *ctx)
      : OpRewritePattern(ctx), converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(tt::ttir::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto attr = op.getValue();
    auto elementType = attr.getElementType();

    if (mlir::isa<DenseResourceElementsAttr>(attr)) {
      return rewriter.notifyMatchFailure(
          op, "DenseResourceElementsAttr conversion not supported");
    }

    auto newType = mlir::cast<mlir::ShapedType>(
        converter.convertType(attr.getShapedType()));
    if (newType.getElementType() == elementType) {
      return rewriter.notifyMatchFailure(op, "no conversion needed");
    }

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
      if (intType.getWidth() == 64) {
        llvm::SmallVector<mlir::APInt> intValues;
        for (mlir::APInt v : attr.getValues<mlir::APInt>()) {
          intValues.push_back(v.truncSSat(32));
        }
        auto newAttr = mlir::DenseElementsAttr::get(newType, intValues);
        rewriter.modifyOpInPlace(op, [&]() { op.setValueAttr(newAttr); });
        return success();
      }
    }

    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
      if (floatType.getWidth() == 64) {
        llvm::SmallVector<mlir::APFloat> floatValues;
        for (mlir::APFloat v : attr.getValues<mlir::APFloat>()) {
          float f = static_cast<float>(v.convertToDouble());
          floatValues.emplace_back(f);
        }
        auto newAttr = mlir::DenseElementsAttr::get(newType, floatValues);
        rewriter.modifyOpInPlace(op, [&]() { op.setValueAttr(newAttr); });
        return success();
      }
    }

    return rewriter.notifyMatchFailure(op, "no i64/f64 to convert");
  }

private:
  mlir::TypeConverter converter;
};

// Generic downstream propagation for unary tensor-manipulation ops.
// Propagation intentionally stops at typecast, which is an explicit type
// boundary.
struct PropagateUnaryTensorManipulationResultElementTypePattern
    : public mlir::RewritePattern {
  explicit PropagateUnaryTensorManipulationResultElementTypePattern(
      mlir::MLIRContext *ctx)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasTrait<mlir::OpTrait::OneOperand>() ||
        !op->hasTrait<mlir::OpTrait::OneResult>()) {
      return rewriter.notifyMatchFailure(op, "not unary single-result op");
    }

    if (!(op->hasTrait<TensorManipulation::Trait>() ||
          mlir::isa<BroadcastOp>(op)) ||
        mlir::isa<TypecastOp>(op)) {
      return rewriter.notifyMatchFailure(op, "not a propagation candidate");
    }

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }

    if (inputType.getElementType() == resultType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "already propagated");
    }

    auto newResultType = mlir::RankedTensorType::get(resultType.getShape(),
                                                     inputType.getElementType(),
                                                     resultType.getEncoding());
    rewriter.modifyOpInPlace(
        op, [&]() { op->getResult(0).setType(newResultType); });
    return mlir::success();
  }
};

// Propagate gather output element type from data input element type.
struct GatherResultTypePattern : public mlir::OpRewritePattern<GatherOp> {
  using mlir::OpRewritePattern<GatherOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(GatherOp op, mlir::PatternRewriter &rewriter) const override {
    auto inputType =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    if (inputType.getElementType() == resultType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "already propagated");
    }

    auto newResultType = mlir::RankedTensorType::get(resultType.getShape(),
                                                     inputType.getElementType(),
                                                     resultType.getEncoding());
    rewriter.replaceOpWithNewOp<GatherOp>(
        op, newResultType, op.getInput(), op.getStartIndices(),
        op.getOffsetDimsAttr(), op.getCollapsedSliceDimsAttr(),
        op.getOperandBatchingDimsAttr(), op.getStartIndicesBatchingDimsAttr(),
        op.getStartIndexMapAttr(), op.getIndexVectorDimAttr(),
        op.getSliceSizesAttr(), op.getIndicesAreSortedAttr());
    return mlir::success();
  }
};

// Align mixed elementwise-binary operand/result dtypes by replacing i1 with
// the non-i1 type. This normalizes ops like logical_and(i1, i32) -> i32.
struct AlignElementwiseBinaryTypesPattern : public mlir::RewritePattern {
  explicit AlignElementwiseBinaryTypesPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasTrait<ElementwiseBinary::Trait>() ||
        op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(op, "not a binary single-result op");
    }

    auto lhsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto rhsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }

    Type lhsElemType = lhsType.getElementType();
    Type rhsElemType = rhsType.getElementType();
    Type resultElemType = resultType.getElementType();

    Type targetElemType;
    auto accumulateTarget = [&](Type candidate) {
      if (candidate.isInteger(1)) {
        return true;
      }
      if (!targetElemType) {
        targetElemType = candidate;
        return true;
      }
      return targetElemType == candidate;
    };

    if (!accumulateTarget(lhsElemType) || !accumulateTarget(rhsElemType) ||
        !accumulateTarget(resultElemType) || !targetElemType) {
      return rewriter.notifyMatchFailure(
          op, "cannot infer a single non-i1 target element type");
    }

    if (lhsElemType == targetElemType && rhsElemType == targetElemType &&
        resultElemType == targetElemType) {
      return rewriter.notifyMatchFailure(op, "already aligned");
    }

    auto castOperandIfNeeded =
        [&](mlir::Value v, mlir::RankedTensorType operandType) -> mlir::Value {
      if (operandType.getElementType() == targetElemType) {
        return v;
      }
      auto castType = mlir::RankedTensorType::get(
          operandType.getShape(), targetElemType, operandType.getEncoding());
      return static_cast<mlir::Value>(
          rewriter.create<TypecastOp>(op->getLoc(), castType, v).getResult());
    };

    mlir::Value newLhs = castOperandIfNeeded(op->getOperand(0), lhsType);
    mlir::Value newRhs = castOperandIfNeeded(op->getOperand(1), rhsType);
    auto newResultType = mlir::RankedTensorType::get(
        resultType.getShape(), targetElemType, resultType.getEncoding());

    mlir::OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands({newLhs, newRhs});
    state.addAttributes(op->getAttrs());
    state.addTypes(newResultType);
    mlir::Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

// Normalize 32-bit integer element type to signless i32 so step 2 result types
// match step 3's type converter (avoids function result type vs return value
// mismatch).
static Type normalizeI32Signless(Type elementType, MLIRContext *ctx) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    if (intTy.getWidth() == 32) {
      return IntegerType::get(ctx, 32);
    }
  }
  return elementType;
}

// Rewrite comparison ops so the result type matches the input (lhs) type
// instead of i1.
template <typename ComparisonOp>
struct ComparisonResultTypePattern
    : public mlir::OpRewritePattern<ComparisonOp> {
  using mlir::OpRewritePattern<ComparisonOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ComparisonOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto lhsType = mlir::cast<mlir::RankedTensorType>(op.getLhs().getType());

    if (!resultType.getElementType().isInteger(1)) {
      return rewriter.notifyMatchFailure(op, "result is not i1");
    }
    if (resultType.getElementType() == lhsType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "result already matches lhs type");
    }

    Type elemType =
        normalizeI32Signless(lhsType.getElementType(), op.getContext());
    auto newResultType = mlir::RankedTensorType::get(
        resultType.getShape(), elemType, resultType.getEncoding());

    rewriter.replaceOpWithNewOp<ComparisonOp>(op, newResultType, op.getLhs(),
                                              op.getRhs());
    return mlir::success();
  }
};

// Rewrite logical_not so the result type matches the input type instead of i1.
struct LogicalNotResultTypePattern
    : public mlir::OpRewritePattern<LogicalNotOp> {
  using mlir::OpRewritePattern<LogicalNotOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LogicalNotOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto inputType =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());

    if (!resultType.getElementType().isInteger(1)) {
      return rewriter.notifyMatchFailure(op, "result is not i1");
    }
    if (resultType.getElementType() == inputType.getElementType()) {
      return rewriter.notifyMatchFailure(op,
                                         "result already matches input type");
    }

    Type elemType =
        normalizeI32Signless(inputType.getElementType(), op.getContext());
    auto newResultType = mlir::RankedTensorType::get(
        resultType.getShape(), elemType, resultType.getEncoding());

    rewriter.replaceOpWithNewOp<LogicalNotOp>(op, newResultType, op.getInput());
    return mlir::success();
  }
};

// Rewrite reduce_or so the result type matches the input type instead of i1.
struct ReduceOrResultTypePattern : public mlir::OpRewritePattern<ReduceOrOp> {
  using mlir::OpRewritePattern<ReduceOrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReduceOrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto inputType =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());

    if (!resultType.getElementType().isInteger(1)) {
      return rewriter.notifyMatchFailure(op, "result is not i1");
    }
    if (resultType.getElementType() == inputType.getElementType()) {
      return rewriter.notifyMatchFailure(op,
                                         "result already matches input type");
    }

    Type elemType =
        normalizeI32Signless(inputType.getElementType(), op.getContext());
    auto newResultType = mlir::RankedTensorType::get(
        resultType.getShape(), elemType, resultType.getEncoding());

    rewriter.replaceOpWithNewOp<ReduceOrOp>(op, newResultType, op.getInput(),
                                            op.getKeepDimAttr(),
                                            op.getDimArgAttr());
    return mlir::success();
  }
};

// Rewrite where so the condition tensor type matches the true/false value
// types.
struct WhereConditionTypePattern : public mlir::OpRewritePattern<WhereOp> {
  using mlir::OpRewritePattern<WhereOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(WhereOp op, mlir::PatternRewriter &rewriter) const override {
    auto condType = mlir::cast<mlir::RankedTensorType>(op.getFirst().getType());
    auto trueType =
        mlir::cast<mlir::RankedTensorType>(op.getSecond().getType());
    auto falseType =
        mlir::cast<mlir::RankedTensorType>(op.getThird().getType());

    if (!condType.getElementType().isInteger(1)) {
      return rewriter.notifyMatchFailure(op, "condition is not i1");
    }
    if (trueType != falseType) {
      return rewriter.notifyMatchFailure(op, "true/false types differ");
    }
    if (condType.getElementType() == trueType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "condition already matches true/false element type");
    }

    auto targetType = mlir::RankedTensorType::get(
        condType.getShape(), trueType.getElementType(), condType.getEncoding());
    auto condCast =
        rewriter.create<TypecastOp>(op.getLoc(), targetType, op.getFirst());

    rewriter.replaceOpWithNewOp<WhereOp>(op, op.getResult().getType(),
                                         condCast.getResult(), op.getSecond(),
                                         op.getThird());
    return mlir::success();
  }
};

// Step 3: Convert remaining i1 to bf16.
class I1ToBF16TypeConverter : public TypeConverter {
public:
  I1ToBF16TypeConverter() {
    addConversion(
        [](mlir::RankedTensorType type) -> std::optional<RankedTensorType> {
          Type elementType = type.getElementType();
          if (elementType.isInteger(1)) {
            Type newElementType = mlir::BFloat16Type::get(type.getContext());
            return mlir::RankedTensorType::get(type.getShape(), newElementType,
                                               type.getEncoding());
          }
          return type;
        });
    // Pass-through for non-tensor types so convertType never returns null.
    // Reject RankedTensorType so the conversion above is used for tensor types.
    addConversion([](Type type) -> std::optional<Type> {
      if (mlir::isa<mlir::RankedTensorType>(type)) {
        return std::nullopt;
      }
      return type;
    });
  }
};

class I1ConstantOpAttrRewriter
    : public mlir::OpRewritePattern<tt::ttir::ConstantOp> {
public:
  using mlir::OpRewritePattern<tt::ttir::ConstantOp>::OpRewritePattern;

  I1ConstantOpAttrRewriter(const mlir::TypeConverter &converter,
                           mlir::MLIRContext *ctx)
      : OpRewritePattern(ctx), converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(tt::ttir::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto attr = op.getValue();
    if (!attr.getElementType().isInteger(1)) {
      return rewriter.notifyMatchFailure(op, "not i1 constant");
    }
    if (mlir::isa<DenseResourceElementsAttr>(attr)) {
      return rewriter.notifyMatchFailure(
          op, "DenseResourceElementsAttr i1 conversion not supported");
    }
    auto newType = mlir::cast<mlir::ShapedType>(
        converter.convertType(attr.getShapedType()));
    llvm::SmallVector<mlir::APFloat> bf16Values;
    for (bool v : attr.getValues<bool>()) {
      mlir::APFloat bf16Value(mlir::APFloat::BFloat());
      bf16Value.convertFromAPInt(mlir::APInt(1, v ? 1 : 0), /*isSigned=*/false,
                                 mlir::APFloat::rmNearestTiesToEven);
      bf16Values.push_back(bf16Value);
    }
    auto newAttr = mlir::DenseElementsAttr::get(newType, bf16Values);
    rewriter.modifyOpInPlace(op, [&]() { op.setValueAttr(newAttr); });
    return success();
  }

private:
  mlir::TypeConverter converter;
};

struct SmartElementTypeNormalization
    : public impl::SmartElementTypeNormalizationBase<
          SmartElementTypeNormalization> {
  using impl::SmartElementTypeNormalizationBase<
      SmartElementTypeNormalization>::SmartElementTypeNormalizationBase;

  void runOnOperation() final {
    I64ToI32AndF64ToF32TypeConverter converter;
    mlir::GreedyRewriteConfig config;
    config.enableFolding(false);

    // Step 1: Convert i64->i32, f64->f32.
    {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<UniformTypeRewriter>(converter, &getContext());
      patterns.add<I64AndF64ConstantOpAttrRewriter>(converter, &getContext());
      if (failed(mlir::applyPatternsGreedily(getOperation(),
                                             std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    }

    // Step 2: Match comparison/where/logical_not/reduce_or output to input
    // types.
    {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<PropagateUnaryTensorManipulationResultElementTypePattern>(
          &getContext());
      patterns.add<GatherResultTypePattern>(&getContext());
      patterns.add<AlignElementwiseBinaryTypesPattern>(&getContext());
      patterns.add<ComparisonResultTypePattern<EqualOp>>(&getContext());
      patterns.add<ComparisonResultTypePattern<NotEqualOp>>(&getContext());
      patterns.add<ComparisonResultTypePattern<GreaterThanOp>>(&getContext());
      patterns.add<ComparisonResultTypePattern<GreaterEqualOp>>(&getContext());
      patterns.add<ComparisonResultTypePattern<LessThanOp>>(&getContext());
      patterns.add<ComparisonResultTypePattern<LessEqualOp>>(&getContext());
      patterns.add<LogicalNotResultTypePattern>(&getContext());
      patterns.add<ReduceOrResultTypePattern>(&getContext());
      patterns.add<WhereConditionTypePattern>(&getContext());

      if (failed(mlir::applyPatternsGreedily(getOperation(),
                                             std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    }

    // Step 3: Convert remaining i1->bf16.
    {
      I1ToBF16TypeConverter i1Converter;
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<UniformTypeRewriter>(i1Converter, &getContext());
      patterns.add<I1ConstantOpAttrRewriter>(i1Converter, &getContext());

      if (failed(mlir::applyPatternsGreedily(getOperation(),
                                             std::move(patterns), config))) {
        signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
