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

template <typename TargetOp>
class StableHLOToTTIRCompositeOpConversionPattern
    : public OpRewritePattern<mlir::stablehlo::CompositeOp> {
public:
  StableHLOToTTIRCompositeOpConversionPattern(mlir::MLIRContext *context,
                                              const char *opName)
      : OpRewritePattern<mlir::stablehlo::CompositeOp>(context),
        opName(opName) {}

  LogicalResult matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                                PatternRewriter &rewriter) const override {
    if (srcOp.getName() != opName) {
      return rewriter.notifyMatchFailure(
          srcOp, ("CompositeOp must be " + std::string(opName) + ".").c_str());
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }
    ttir::utils::replaceOpWithNewDPSOp<TargetOp>(
        rewriter, srcOp,
        mlir::cast<RankedTensorType>(srcOp->getResult(0).getType()),
        srcOp->getOperands());
    return success();
  }

private:
  const char *opName;
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeStableHLOCompositeToTTIRPass() {
  return std::make_unique<LegalizeStableHLOCompositeToTTIR>();
}

void populateStableHLOCompositeLegalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu");
  // GeluOp doesn't currently have a flag to use the tanh approximation
  // nor is there a GeluNewOp/GeluTanhOp, otherwise we would be targetting that
  // instead.
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu_tanh");
}
} // namespace mlir::tt
