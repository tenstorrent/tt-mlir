// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_CONVERTSDYCONSTTOSTABLEHLOCONSTPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class ConvertSdyConstPattern : public OpRewritePattern<mlir::sdy::ConstantOp> {
  using OpRewritePattern<mlir::sdy::ConstantOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::ConstantOp sdyConstOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        sdyConstOp, sdyConstOp.getValueAttr());
    return success();
  }
};

class ConvertSdyConstToStableHLOConstPass
    : public impl::ConvertSdyConstToStableHLOConstPassBase<
          ConvertSdyConstToStableHLOConstPass> {
public:
  using impl::ConvertSdyConstToStableHLOConstPassBase<
      ConvertSdyConstToStableHLOConstPass>::
      ConvertSdyConstToStableHLOConstPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    // Set up rewrite patterns.
    RewritePatternSet patterns(context);
    patterns.add<ConvertSdyConstPattern>(context);

    // Apply the patterns to convert sdy.constant to stablehlo.constant.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::stablehlo
