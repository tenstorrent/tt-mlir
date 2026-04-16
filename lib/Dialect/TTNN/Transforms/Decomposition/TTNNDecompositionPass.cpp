// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DistributedRMSNormDecompositionRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"
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

    // DistributedRMSNorm decomposition always runs (shape-gated internally).
    patterns.add<decomposition::DistributedRMSNormDecompositionRewritePattern>(
        &getContext());

    // SDPA decomposition patterns: validation-gated or force-decompose.
    if (forceDecompose) {
      patterns.add<decomposition::SDPADecodeDecompositionPattern>(
          &getContext());
      patterns.add<decomposition::SDPADecompositionPattern>(&getContext());
    }

#ifdef TTMLIR_ENABLE_OPMODEL
    if (enableOpConstraints) {
      FusionValidationConfig validationConfig;
      validationConfig.maxFallbackAttempts = maxFallbackAttempts;

      patterns.add<decomposition::SDPADecodeDecompositionPattern>(
          &getContext(), validationConfig);
      patterns.add<decomposition::SDPADecompositionPattern>(&getContext(),
                                                            validationConfig);
    }
#endif

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
