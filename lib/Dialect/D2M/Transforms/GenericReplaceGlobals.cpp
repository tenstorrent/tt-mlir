// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICREPLACEGLOBALS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenericReplaceGlobalsRewriter
    : public OpRewritePattern<ttcore::GetGlobalOp> {
public:
  using OpRewritePattern<ttcore::GetGlobalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttcore::GetGlobalOp op,
                                PatternRewriter &rewriter) const final {
    GenericOp generic = op->getParentOfType<GenericOp>();
    if (!generic) {
      return failure();
    }

    // Look up the global symbol definition to get its index
    auto global = SymbolTable::lookupNearestSymbolFrom<ttcore::GlobalOp>(
        op, op.getSymNameAttr());
    if (!global) {
      return op.emitError("Global symbol not found: ") << op.getSymNameAttr();
    }

    std::optional<int32_t> index = global.getIndex();
    if (!index) {
      return op.emitError("Global must have a valid index attribute");
    }

    Value operand = generic.getOperand(*index);
    rewriter.replaceAllUsesWith(op, operand);
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace {
class D2MGenericReplaceGlobals
    : public impl::D2MGenericReplaceGlobalsBase<D2MGenericReplaceGlobals> {
public:
  using impl::D2MGenericReplaceGlobalsBase<
      D2MGenericReplaceGlobals>::D2MGenericReplaceGlobalsBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());

    // Add legal dialects
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalDialect<d2m::D2MDialect>();

    target.addDynamicallyLegalOp<ttcore::GetGlobalOp>(
        [&](ttcore::GetGlobalOp op) {
          return !(op->getParentOfType<GenericOp>());
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenericReplaceGlobalsRewriter>(&getContext());

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
