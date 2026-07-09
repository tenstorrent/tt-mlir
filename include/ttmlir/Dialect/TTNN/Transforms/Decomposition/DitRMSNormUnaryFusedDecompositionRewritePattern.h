// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DITRMSNORMUNARYFUSEDDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DITRMSNORMUNARYFUSEDDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::DitRMSNormUnaryFusedOp into its constituent ops:
//   x   = residual_input ? add(input, residual_input) : input
//   n   = rms_norm(x, weight, bias, epsilon, compute_config)
//   out = activation ? <activation>(n) : n
//
// This is the inverse of the rms_norm + activation fusion, and serves as the
// fallback lowering when the fused kernel is unsupported for a given shape /
// layout.
//
// When validationConfig is provided, decomposition only occurs if op-model
// validation of the fused ttnn.dit_rms_norm_unary_fused fails. When
// validationConfig is std::nullopt, decomposition is unconditional.

class DitRMSNormUnaryFusedDecompositionRewritePattern
    : public OpRewritePattern<ttnn::DitRMSNormUnaryFusedOp> {
public:
  DitRMSNormUnaryFusedDecompositionRewritePattern(mlir::MLIRContext *context)
      : OpRewritePattern<ttnn::DitRMSNormUnaryFusedOp>(context) {}

  DitRMSNormUnaryFusedDecompositionRewritePattern(
      mlir::MLIRContext *context, const OpValidationConfig &validationConfig)
      : OpRewritePattern<ttnn::DitRMSNormUnaryFusedOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::DitRMSNormUnaryFusedOp op,
                                PatternRewriter &rewriter) const override;

private:
  std::optional<OpValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DITRMSNORMUNARYFUSEDDECOMPOSITIONREWRITEPATTERN_H
