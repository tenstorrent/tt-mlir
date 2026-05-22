// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_ROTARYEMBEDDINGDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_ROTARYEMBEDDINGDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::RotaryEmbeddingOp back into primitive TTNN ops.
// When validationConfig is provided, decomposition only occurs if op-model
// validation fails. When validationConfig is std::nullopt, decomposition
// is unconditional.
//
// Decomposition formula:
//   x1 = slice(x, [:D/2])
//   x2 = slice(x, [D/2:])
//   rotated = concat(neg(x2), x1)
//   result = add(mul(x, cos), mul(rotated, sin))
class RotaryEmbeddingDecompositionRewritePattern
    : public OpRewritePattern<ttnn::RotaryEmbeddingOp> {
public:
  RotaryEmbeddingDecompositionRewritePattern(mlir::MLIRContext *context)
      : OpRewritePattern<ttnn::RotaryEmbeddingOp>(context) {}

  RotaryEmbeddingDecompositionRewritePattern(
      mlir::MLIRContext *context, const OpValidationConfig &validationConfig)
      : OpRewritePattern<ttnn::RotaryEmbeddingOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::RotaryEmbeddingOp ropeOp,
                                PatternRewriter &rewriter) const override;

private:
  std::optional<OpValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_ROTARYEMBEDDINGDECOMPOSITIONREWRITEPATTERN_H
