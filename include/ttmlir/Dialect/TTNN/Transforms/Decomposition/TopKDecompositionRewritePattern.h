// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_TOPKDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_TOPKDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::TopKOp back into ttnn::SortOp + ttnn::SliceStaticOp when
// op-model validation fails. This is the validation-gated fallback for TopK
// operations that were fused at the TTIR level but cannot be executed as a
// single TopK op on hardware.
class TopKDecompositionRewritePattern : public OpRewritePattern<ttnn::TopKOp> {
public:
  TopKDecompositionRewritePattern(
      mlir::MLIRContext *context,
      const FusionValidationConfig &validationConfig = {})
      : OpRewritePattern<ttnn::TopKOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::TopKOp topkOp,
                                PatternRewriter &rewriter) const override;

private:
  FusionValidationConfig validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_TOPKDECOMPOSITIONREWRITEPATTERN_H
