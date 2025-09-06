// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICREPLACEGLOBALS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericReplaceGlobalsRewriter : public OpRewritePattern<ttcore::GetGlobalOp> {
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
class TTIRGenericReplaceGlobals
    : public impl::TTIRGenericReplaceGlobalsBase<
          TTIRGenericReplaceGlobals> {
public:
  using impl::TTIRGenericReplaceGlobalsBase<
      TTIRGenericReplaceGlobals>::TTIRGenericReplaceGlobalsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericReplaceGlobalsRewriter>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttir
