// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_LEGALIZESDYTOSTABLEHLOPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// Pattern to convert sdy.constant to stablehlo.constant.
// sdy.constant is identical to stablehlo.constant in terms of semantics,
// but some downstream consumers may not support the SDY dialect.
class SdyConstantToStableHLOPattern
    : public OpRewritePattern<mlir::sdy::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(mlir::sdy::ConstantOp srcOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(srcOp,
                                                             srcOp.getValue());
    return success();
  }
};

class LegalizeSdyToStableHLOPass
    : public impl::LegalizeSdyToStableHLOPassBase<LegalizeSdyToStableHLOPass> {
public:
  using impl::LegalizeSdyToStableHLOPassBase<
      LegalizeSdyToStableHLOPass>::LegalizeSdyToStableHLOPassBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Set up conversion target: SDY constant ops are illegal, StableHLO is
    // legal.
    ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addLegalDialect<mlir::sdy::SdyDialect>();
    target.addIllegalOp<mlir::sdy::ConstantOp>();

    // Populate rewrite patterns.
    RewritePatternSet patterns(ctx);
    patterns.add<SdyConstantToStableHLOPattern>(ctx);

    // Apply conversion.
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::stablehlo
