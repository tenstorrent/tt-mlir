// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_NORMALIZE64BITANDBOOLEANELEMENTTYPES
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class I64ToI32AndF64ToF32TypeConverter : public TypeConverter {
public:
  I64ToI32AndF64ToF32TypeConverter() {
    addConversion(
        [](mlir::RankedTensorType type) -> std::optional<RankedTensorType> {
          Type elementType = type.getElementType();

          if (mlir::isa<mlir::quant::QuantizedType>(elementType)) {
            return type;
          }

          if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
            if (intType.getWidth() == 64) {
              Type newElementType =
                  IntegerType::get(type.getContext(), 32,
                                   IntegerType::SignednessSemantics::Signed);
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

// Phase 3: Convert remaining i1 to i32.
class I1ToI32TypeConverter : public TypeConverter {
public:
  I1ToI32TypeConverter() {
    addConversion(
        [](mlir::RankedTensorType type) -> std::optional<RankedTensorType> {
          Type elementType = type.getElementType();
          if (elementType.isInteger(1)) {
            Type newElementType =
                IntegerType::get(type.getContext(), 32,
                                 IntegerType::SignednessSemantics::Signed);
            return mlir::RankedTensorType::get(type.getShape(), newElementType,
                                               type.getEncoding());
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
    llvm::SmallVector<mlir::APInt> intValues;
    for (bool v : attr.getValues<bool>()) {
      intValues.push_back(mlir::APInt(32, v ? 1 : 0, /*isSigned=*/true));
    }
    auto newAttr = mlir::DenseElementsAttr::get(newType, intValues);
    rewriter.modifyOpInPlace(op, [&]() { op.setValueAttr(newAttr); });
    return success();
  }

private:
  mlir::TypeConverter converter;
};

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

    auto newResultType = mlir::RankedTensorType::get(resultType.getShape(),
                                                     lhsType.getElementType(),
                                                     resultType.getEncoding());

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

    auto newResultType = mlir::RankedTensorType::get(resultType.getShape(),
                                                     inputType.getElementType(),
                                                     resultType.getEncoding());

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

    auto newResultType = mlir::RankedTensorType::get(resultType.getShape(),
                                                     inputType.getElementType(),
                                                     resultType.getEncoding());

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

struct Normalize64BitAndBooleanElementTypes
    : public impl::Normalize64BitAndBooleanElementTypesBase<
          Normalize64BitAndBooleanElementTypes> {
  using impl::Normalize64BitAndBooleanElementTypesBase<
      Normalize64BitAndBooleanElementTypes>::
      Normalize64BitAndBooleanElementTypesBase;

  void runOnOperation() final {
    I64ToI32AndF64ToF32TypeConverter converter;
    mlir::GreedyRewriteConfig config;
    config.enableFolding(false);

    // Phase 1: Convert i64->i32, f64->f32.
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

    // Phase 2: Match comparison/where/logical_not/reduce_or output to input
    // types.
    {
      mlir::RewritePatternSet patterns(&getContext());
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

    // Phase 3: Convert remaining i1->i32.
    {
      I1ToI32TypeConverter i1Converter;
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
