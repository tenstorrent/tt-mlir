// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

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

    auto global = SymbolTable::lookupNearestSymbolFrom<ttcore::GlobalOp>(
        op, op.getSymNameAttr());
    assert(global);

    std::optional<int32_t> index = global.getIndex();
    assert(index);

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
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenericReplaceGlobalsRewriter>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
