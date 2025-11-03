// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
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
    assert(generic); // dynamically legal op check ensures this is not null

    auto global = SymbolTable::lookupNearestSymbolFrom<ttcore::GlobalOp>(
        op, op.getSymNameAttr());
    if (!global) {
      op.emitError("Global symbol not found: ") << op.getSymNameAttr();
      return failure();
    }

    std::optional<int32_t> index = global.getIndex();
    if (!index) {
      op.emitError("Global must have a valid index attribute");
      return failure();
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
    target.addLegalDialect<
        arith::ArithDialect, BuiltinDialect, func::FuncDialect,
        linalg::LinalgDialect, memref::MemRefDialect, scf::SCFDialect,
        tensor::TensorDialect, ttcore::TTCoreDialect, d2m::D2MDialect>();

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
