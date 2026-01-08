// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOFUSINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class ConcatenateToBroadcastFusionPattern : public OpRewritePattern<::mlir::stablehlo::ConcatenateOp> {
    using OpRewritePattern<::mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::mlir::stablehlo::ConcatenateOp concatOp, ::mlir::PatternRewriter &rewriter) const final {
        llvm::outs() << "HERE\nHERE\nHERE\n";
        return success();
    }
};

struct StablehloFusingPass : public impl::StablehloFusingPassBase<StablehloFusingPass> {
public:
    using impl::StablehloFusingPassBase<StablehloFusingPass>::StablehloFusingPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        patterns.add<ConcatenateToBroadcastFusionPattern>(&getContext());

        GreedyRewriteConfig config;
        config.setUseTopDownTraversal(true);
        (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
    }
};
} // namespace mlir::tt::stablehlo