// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDECOMPOSEOPSONVALIDATIONFAILURE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNDecomposeOpsOnValidationFailurePass
    : public impl::TTNNDecomposeOpsOnValidationFailureBase<
          TTNNDecomposeOpsOnValidationFailurePass> {
public:
  using impl::TTNNDecomposeOpsOnValidationFailureBase<
      TTNNDecomposeOpsOnValidationFailurePass>::
      TTNNDecomposeOpsOnValidationFailureBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

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

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::tt::ttnn
