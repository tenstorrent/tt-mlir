// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_PARTIALLYCONVERTSDYTOSTABLEHLOPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class PartiallyConvertSdyConstPattern
    : public OpRewritePattern<mlir::sdy::ConstantOp> {
  using OpRewritePattern<mlir::sdy::ConstantOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::ConstantOp sdyConstOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        sdyConstOp, sdyConstOp.getValueAttr());
    return success();
  }
};

class PartiallyConvertSdyToStableHLOPass
    : public impl::PartiallyConvertSdyToStableHLOPassBase<
          PartiallyConvertSdyToStableHLOPass> {
public:
  using impl::PartiallyConvertSdyToStableHLOPassBase<
      PartiallyConvertSdyToStableHLOPass>::
      PartiallyConvertSdyToStableHLOPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    RewritePatternSet patterns(context);
    patterns.add<PartiallyConvertSdyConstPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::stablehlo
