// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCATTEROPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCATTEROPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// ScatterOp in TTNN has a hardware limit on the scatter axis size.
// This workaround chunks large scatter operations into smaller ones
// when the scatter axis size exceeds the hardware limit (256).
class TTNNScatterWorkarounds : public OpRewritePattern<ttnn::ScatterOp> {
public:
  using OpRewritePattern<ttnn::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScatterOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCATTEROPREWRITEPATTERN_H
