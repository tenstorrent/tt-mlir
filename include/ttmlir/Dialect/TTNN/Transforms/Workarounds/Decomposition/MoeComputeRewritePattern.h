// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Reshards the expert-indices / expert-scores inputs to L1 HEIGHT_SHARDED on
// the tilize drain core. The kernel globally-allocates their circular buffers
// against a single backing buffer that must live on that core, and validate()
// does not check it, so the layout is fixed up here. Output layouts are
// device-derived and set separately by the TTNNDeduceMoEComputeLayouts pass.
class MoeComputeRewritePattern : public OpRewritePattern<ttnn::MoeComputeOp> {
public:
  using OpRewritePattern<ttnn::MoeComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MoeComputeOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H
