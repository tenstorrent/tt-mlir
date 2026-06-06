// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_INTEGERPRODOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_INTEGERPRODOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Works around a tt-metal bug where `ttnn.prod` computes internally in bf16,
// silently rounding integer intermediates that lack an exact bf16
// representation (e.g. 2204 -> 2208). Decomposes a single-dim `ttnn.prod`
// on a statically-sized integer tensor into a chain of `ttnn.slice_static`
// + `ttnn.multiply` (and a trailing `ttnn.reshape` when keep_dim=false) so
// the computation stays in integer arithmetic. Remove once the metal-side
// fix lands.
// Issue: https://github.com/tenstorrent/tt-metal/issues/44942
class IntegerProdOpRewritePattern : public OpRewritePattern<ttnn::ProdOp> {
public:
  using OpRewritePattern<ttnn::ProdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ProdOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_INTEGERPRODOPREWRITEPATTERN_H
