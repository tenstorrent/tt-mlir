// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_PREDICATETYPEALIGNMENT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Replace the element type of `tensorType`. When the tensor carries a TTNN
// layout encoding, the encoding's memref element type is updated in lock-step
// (preserving tile wrapping) so downstream passes that read it from the
// layout do not see stale types.
mlir::RankedTensorType
buildTensorWithElementType(mlir::RankedTensorType tensorType,
                           mlir::Type newScalarElemType) {
  auto encoding = tensorType.getEncoding();
  auto ttnnLayout = mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(encoding);
  if (!ttnnLayout) {
    return mlir::RankedTensorType::get(tensorType.getShape(), newScalarElemType,
                                       encoding);
  }
  mlir::Type newMemrefElemType = newScalarElemType;
  if (auto tileType =
          mlir::dyn_cast<ttcore::TileType>(ttnnLayout.getElementType())) {
    newMemrefElemType =
        ttcore::TileType::get(newScalarElemType, tileType.getShape());
  }
  return ttnn::utils::RankedTensorTypeFactory::create(tensorType,
                                                      newMemrefElemType);
}

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

    auto newResultType =
        buildTensorWithElementType(resultType, inputType.getElementType());
    rewriter.modifyOpInPlace(
        op, [&]() { op->getResult(0).setType(newResultType); });
    return mlir::success();
  }
};

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

    // i1 operands/results are promoted up to whichever non-i1 type the other
    // operands carry, so they don't count when checking for agreement.
    auto agreesWith = [](Type t, Type target) {
      return t.isInteger(1) || t == target;
    };

    Type firstNonI1;
    for (Type t : {lhsElemType, rhsElemType, resultElemType}) {
      if (!t.isInteger(1)) {
        firstNonI1 = t;
        break;
      }
    }
    if (!firstNonI1) {
      return rewriter.notifyMatchFailure(op, "no non-i1 target element type");
    }

    bool typesAgree = agreesWith(lhsElemType, firstNonI1) &&
                      agreesWith(rhsElemType, firstNonI1) &&
                      agreesWith(resultElemType, firstNonI1);

    Type targetElemType;
    if (typesAgree) {
      targetElemType = firstNonI1;
    } else if (!resultElemType.isInteger(1)) {
      // Cross-type binary like ttnn.eq(si32, si32) -> bf16: TTNN does the
      // typecast inside its kernel, but D2M needs a uniformly-typed compute
      // region, so align operands to the result type.
      targetElemType = resultElemType;
    } else {
      return rewriter.notifyMatchFailure(
          op, "cannot infer a single target element type");
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
      auto castType = buildTensorWithElementType(operandType, targetElemType);
      return static_cast<mlir::Value>(
          rewriter.create<TypecastOp>(op->getLoc(), castType, v).getResult());
    };

    mlir::Value newLhs = castOperandIfNeeded(op->getOperand(0), lhsType);
    mlir::Value newRhs = castOperandIfNeeded(op->getOperand(1), rhsType);
    auto newResultType = buildTensorWithElementType(resultType, targetElemType);

    mlir::OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands({newLhs, newRhs});
    state.addAttributes(op->getAttrs());
    state.addTypes(newResultType);
    mlir::Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

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

    auto newResultType =
        buildTensorWithElementType(resultType, lhsType.getElementType());

    rewriter.replaceOpWithNewOp<ComparisonOp>(op, newResultType, op.getLhs(),
                                              op.getRhs());
    return mlir::success();
  }
};

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

    auto newResultType =
        buildTensorWithElementType(resultType, inputType.getElementType());

    rewriter.replaceOpWithNewOp<LogicalNotOp>(op, newResultType, op.getInput());
    return mlir::success();
  }
};

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

    auto newResultType =
        buildTensorWithElementType(resultType, inputType.getElementType());

    rewriter.replaceOpWithNewOp<ReduceOrOp>(op, newResultType, op.getInput(),
                                            op.getKeepDimAttr(),
                                            op.getDimArgAttr());
    return mlir::success();
  }
};

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

    auto targetType =
        buildTensorWithElementType(condType, trueType.getElementType());
    auto condCast =
        rewriter.create<TypecastOp>(op.getLoc(), targetType, op.getFirst());

    rewriter.replaceOpWithNewOp<WhereOp>(op, op.getResult().getType(),
                                         condCast.getResult(), op.getSecond(),
                                         op.getThird());
    return mlir::success();
  }
};

struct PredicateTypeAlignment
    : public impl::PredicateTypeAlignmentBase<PredicateTypeAlignment> {
  using impl::PredicateTypeAlignmentBase<
      PredicateTypeAlignment>::PredicateTypeAlignmentBase;

  void runOnOperation() final {
    mlir::GreedyRewriteConfig config;
    config.enableFolding(false);

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PropagateUnaryTensorManipulationResultElementTypePattern>(
        &getContext());
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

    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns),
                                           config))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
