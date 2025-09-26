// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRREMOVEREDUNDANTCASTOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRRemoveRedundantCastOpsRewriter : public RewritePattern {
public:
  TTIRRemoveRedundantCastOpsRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Must have at least one operand and one result
    if (op->getNumOperands() == 0 || op->getNumResults() != 1) {
      return failure();
    }

    // Get the result and check if it's used in a convert op
    auto result = op->getResult(0);
    if (!result.hasOneUse()) {
      return failure();
    }

    auto convertUser = dyn_cast<ttir::TypecastOp>(*result.getUsers().begin());
    if (!convertUser) {
      return failure();
    }

    // Output convert: t2 -> t1
    mlir::RankedTensorType outerType = convertUser.getResult().getType();
    mlir::RankedTensorType innerType =
        mlir::cast<mlir::RankedTensorType>(convertUser.getOperand(0).getType());

    // Now check all operands of the op are convert ops from outerType ->
    // innerType
    SmallVector<Value> newOperands;
    Value emptyOp;
    for (auto operand : op->getOperands()) {
      auto *curOp = operand.getDefiningOp();
      if (!curOp) {
        return failure();
      }
      if (isa<ttir::EmptyOp>(operand.getDefiningOp())) {
        emptyOp = operand;
        continue;
      }

      auto convertOp = operand.getDefiningOp<ttir::TypecastOp>();
      if (!convertOp) {
        return failure();
      }

      if (convertOp.getOperand(0).getType() != outerType ||
          convertOp.getResult().getType() != innerType) {
        return failure();
      }

      newOperands.push_back(convertOp.getOperand(0));
    }

    auto output = rewriter.create<mlir::tt::ttir::EmptyOp>(
        op->getLoc(), outerType.getShape(), outerType.getElementType(),
        outerType.getEncoding());
    newOperands.push_back(output);

    // Clone the op with new operands and new type
    OperationState newOpState(op->getLoc(), op->getName());
    newOpState.addOperands(newOperands);
    newOpState.addAttributes(op->getAttrs());

    // Result type changes to outerType (original pre-convert)
    newOpState.addTypes(outerType);

    Operation *newOp = rewriter.create(newOpState);

    rewriter.replaceOp(convertUser, newOp->getResult(0));

    return llvm::success();
  }
};
} // namespace

namespace {
class TTIRRemoveRedundantCastOps
    : public impl::TTIRRemoveRedundantCastOpsBase<TTIRRemoveRedundantCastOps> {
public:
  using impl::TTIRRemoveRedundantCastOpsBase<
      TTIRRemoveRedundantCastOps>::TTIRRemoveRedundantCastOpsBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRRemoveRedundantCastOpsRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
