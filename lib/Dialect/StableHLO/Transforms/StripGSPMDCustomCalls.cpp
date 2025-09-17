// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {

#define GEN_PASS_DEF_STRIPGSPMDCUSTOMCALLSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

/// Pattern to remove GSPMD-related custom calls and replace their uses
/// with their input operands.
class StripGSPMDCustomCallPattern : public OpRewritePattern<mlir::stablehlo::CustomCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp customCallOp,
                                PatternRewriter &rewriter) const override {
    auto callTarget = customCallOp.getCallTargetName();

    // Check if this custom call matches one of the GSPMD patterns we want to strip
    if (callTarget != mlir::tt::gspmd_utils::kShardingCustomCallTargetName &&
        callTarget != mlir::tt::gspmd_utils::kSPMDFullToShardShapeCallTargetName &&
        callTarget != mlir::tt::gspmd_utils::kSPMDShardToFullShapeCallTargetName) {
      return failure();
    }

    // These custom calls should be unary operations
    if (customCallOp.getNumOperands() != 1) {
      return rewriter.notifyMatchFailure(customCallOp,
          "Expected unary custom call but found different number of operands");
    }

    if (customCallOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(customCallOp,
          "Expected custom call with single result");
    }

    // Replace all uses of the custom call result with the input operand
    Value inputOperand = customCallOp.getOperand(0);
    rewriter.replaceOp(customCallOp, inputOperand);

    return success();
  }
};

class StripGSPMDCustomCallsPass : public impl::StripGSPMDCustomCallsPassBase<StripGSPMDCustomCallsPass> {
public:
  using impl::StripGSPMDCustomCallsPassBase<StripGSPMDCustomCallsPass>::StripGSPMDCustomCallsPassBase;

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add the pattern to strip GSPMD custom calls
    patterns.add<StripGSPMDCustomCallPattern>(context);

    // Apply the patterns greedily
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::stablehlo