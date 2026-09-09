// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECOMPOSITIONPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECOMPOSITIONPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::ScaledDotProductAttentionOp into a sequence of primitive
// TTNN ops (transpose + matmul + scale + mask + softmax + matmul, with GQA
// head expansion and optional attention sink). When validationConfig is
// provided, decomposition only occurs if op-model validation fails. When
// validationConfig is std::nullopt, decomposition is unconditional.
class SDPADecompositionPattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionOp> {
public:
  SDPADecompositionPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ttnn::ScaledDotProductAttentionOp>(context) {}

  SDPADecompositionPattern(mlir::MLIRContext *context,
                           const OpValidationConfig &validationConfig)
      : OpRewritePattern<ttnn::ScaledDotProductAttentionOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override;

private:
  std::optional<OpValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECOMPOSITIONPATTERN_H
