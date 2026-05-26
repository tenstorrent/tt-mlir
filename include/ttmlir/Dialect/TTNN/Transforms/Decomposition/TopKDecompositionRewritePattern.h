// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_TOPKDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_TOPKDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::tt::ttnn::decomposition {

// Decomposes ttnn::TopKOp back into ttnn::SortOp + ttnn::SliceStaticOp.
// When validationConfig is provided, decomposition only occurs if op-model
// validation fails. When validationConfig is std::nullopt (e.g. optimizer
// disabled / opt_level=0), the TopK is preserved as-is — TTNN runtime
// supports TopK natively, and regenerating sort+slice with default attributes
// (stable=false, null memory_config) can break downstream paths such as
// trace capture.
class TopKDecompositionRewritePattern : public OpRewritePattern<ttnn::TopKOp> {
public:
  TopKDecompositionRewritePattern(mlir::MLIRContext *context)
      : OpRewritePattern<ttnn::TopKOp>(context) {}

  TopKDecompositionRewritePattern(mlir::MLIRContext *context,
                                  const OpValidationConfig &validationConfig)
      : OpRewritePattern<ttnn::TopKOp>(context),
        validationConfig(validationConfig) {}

  LogicalResult matchAndRewrite(ttnn::TopKOp topkOp,
                                PatternRewriter &rewriter) const override;

private:
  std::optional<OpValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_TOPKDECOMPOSITIONREWRITEPATTERN_H
