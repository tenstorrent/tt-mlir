// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_GROUPNORMDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_GROUPNORMDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::GroupNormOp into a sequence of primitive TTNN ops
// (reshape / mean / subtract / multiply / add / rsqrt / multiply / reshape,
// plus optional weight*+bias broadcast tail).
//
// When validationConfig is provided, decomposition only occurs if op-model
// validation of the fused ttnn.group_norm fails (e.g. the kernel's L1 /
// circular-buffer requirements cannot be satisfied with any layout the
// validator's fallback search produces). When validationConfig is
// std::nullopt, decomposition is unconditional.

class GroupNormDecompositionRewritePattern
    : public OpRewritePattern<ttnn::GroupNormOp> {
public:
  GroupNormDecompositionRewritePattern(mlir::MLIRContext *context)
      : OpRewritePattern<ttnn::GroupNormOp>(context) {}

  GroupNormDecompositionRewritePattern(
      mlir::MLIRContext *context, const OpValidationConfig &validationConfig)
      : OpRewritePattern<ttnn::GroupNormOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::GroupNormOp op,
                                PatternRewriter &rewriter) const override;

private:
  std::optional<OpValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_GROUPNORMDECOMPOSITIONREWRITEPATTERN_H
