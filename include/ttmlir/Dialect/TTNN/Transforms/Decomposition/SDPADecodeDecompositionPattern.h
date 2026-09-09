// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECODEDECOMPOSITIONPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECODEDECOMPOSITIONPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::ScaledDotProductAttentionDecodeOp by permuting Q from
// [1, B, H, D] to [B, H, 1, D], emitting a ScaledDotProductAttentionOp (which
// the SDPADecompositionPattern can then further decompose if needed), and
// permuting the result back. When validationConfig is provided, decomposition
// only occurs if op-model validation fails. When validationConfig is
// std::nullopt, decomposition is unconditional.
class SDPADecodeDecompositionPattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionDecodeOp> {
public:
  SDPADecodeDecompositionPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ttnn::ScaledDotProductAttentionDecodeOp>(context) {}

  SDPADecodeDecompositionPattern(mlir::MLIRContext *context,
                                 const OpValidationConfig &validationConfig)
      : OpRewritePattern<ttnn::ScaledDotProductAttentionDecodeOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionDecodeOp op,
                                PatternRewriter &rewriter) const override;

private:
  std::optional<OpValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECODEDECOMPOSITIONPATTERN_H
