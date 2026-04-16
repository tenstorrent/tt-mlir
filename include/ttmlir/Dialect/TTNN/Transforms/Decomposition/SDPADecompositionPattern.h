// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECOMPOSITIONPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECOMPOSITIONPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

#include "mlir/IR/PatternMatch.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

class SDPADecompositionPattern
    : public OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  explicit SDPADecompositionPattern(MLIRContext *context)
      : OpRewritePattern(context), forceDecompose(true) {}

  SDPADecompositionPattern(MLIRContext *context,
                           const FusionValidationConfig &config)
      : OpRewritePattern(context), forceDecompose(false),
        validationConfig(config) {}

  LogicalResult matchAndRewrite(ScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override;

private:
  bool forceDecompose;
  std::optional<FusionValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif
