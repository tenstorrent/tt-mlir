// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOLegalizeComposite.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCOMPOSITETOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct LegalizeStableHLOCompositeToTTIR
    : public ttir::impl::LegalizeStableHLOCompositeToTTIRBase<
          LegalizeStableHLOCompositeToTTIR> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateStableHLOCompositeLegalizationPatterns(context, patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace {
class StableHLOToTTIRCompositeOpGeluConversionPattern
    : public OpRewritePattern<mlir::stablehlo::CompositeOp> {

  using OpRewritePattern<mlir::stablehlo::CompositeOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                                PatternRewriter &rewriter) const override {

    // Check legality of the conversion.
    LogicalResult err = checkConversionLegality(srcOp, rewriter);
    if (failed(err)) {
      return err;
    }

    // Check that the composite op has exactly one result
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    ttir::utils::replaceOpWithNewDPSOp<ttir::GeluOp>(
        rewriter, srcOp,
        mlir::cast<RankedTensorType>(srcOp->getResult(0).getType()),
        srcOp->getOperands());

    return success();
  }

private:
  LogicalResult checkConversionLegality(mlir::stablehlo::CompositeOp &srcOp,
                                        PatternRewriter &rewriter) const {
    if (srcOp.getName() == "tt.gelu" || srcOp.getName() == "tt.gelu_tanh") {
      return success();
    }
    return rewriter.notifyMatchFailure(srcOp,
                                       "CompositeOp must be tt.gelu for now.");
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeStableHLOCompositeToTTIRPass() {
  return std::make_unique<LegalizeStableHLOCompositeToTTIR>();
}

void populateStableHLOCompositeLegalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<StableHLOToTTIRCompositeOpGeluConversionPattern>(context);
}
} // namespace mlir::tt
