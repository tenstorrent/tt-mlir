// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_CONCATENATEHEADSDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_CONCATENATEHEADSDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::ConcatenateHeadsOp into ttnn::PermuteOp + ttnn::ReshapeOp.
//
// Input tensor shape: [batch_size, num_heads, sequence_size, head_size]
// Output tensor shape: [batch_size, sequence_size, num_heads * head_size]
//
// Rewrite strategy:
// 1. Permute: [batch_size, num_heads, sequence_size, head_size]
//             -> [batch_size, sequence_size, num_heads, head_size]
// 2. Reshape: [batch_size, sequence_size, num_heads, head_size]
//             -> [batch_size, sequence_size, num_heads * head_size]
//
// When validationConfig is provided, decomposition only occurs if op-model
// validation of the fused op fails (e.g. non-tile-aligned head_size, or
// per-core CB allocation exceeding L1 for large num_heads * head_dim).
// When validationConfig is std::nullopt, decomposition is unconditional.
class ConcatenateHeadsDecompositionRewritePattern
    : public OpRewritePattern<ttnn::ConcatenateHeadsOp> {
public:
  ConcatenateHeadsDecompositionRewritePattern(mlir::MLIRContext *context)
      : OpRewritePattern<ttnn::ConcatenateHeadsOp>(context) {}

  ConcatenateHeadsDecompositionRewritePattern(
      mlir::MLIRContext *context, const OpValidationConfig &validationConfig)
      : OpRewritePattern<ttnn::ConcatenateHeadsOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::ConcatenateHeadsOp srcOp,
                                PatternRewriter &rewriter) const override;

private:
  std::optional<OpValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_CONCATENATEHEADSDECOMPOSITIONREWRITEPATTERN_H
