// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DistributedRMSNormDecompositionRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDECOMPOSITION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDecomposition
    : public impl::TTNNDecompositionBase<TTNNDecomposition> {
public:
  using impl::TTNNDecompositionBase<TTNNDecomposition>::TTNNDecompositionBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<decomposition::DistributedRMSNormDecompositionRewritePattern>(
        &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    GreedyRewriteConfig config;
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::tt::ttnn
