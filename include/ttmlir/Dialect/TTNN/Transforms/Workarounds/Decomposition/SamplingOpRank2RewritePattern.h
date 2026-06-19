// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SAMPLINGOPRANK2REWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SAMPLINGOPRANK2REWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal's sampling kernel only accepts rank-4 [1, 1, batch, candidates]
// tensors and produces a rank-4 [1, 1, 1, batch] result. The TTIR-equivalent
// view carried into TTNN is rank-2 in / rank-1 out. This workaround unsqueezes
// two leading unit dims on the values and indices inputs, calls the rank-4
// sampling op, and reshapes the result back to rank-1.
//
// Without this decomposition the runtime handler used to perform the same
// reshape implicitly, which meant EmitPy / EmitC codegen — which doesn't go
// through that handler — emitted a bare rank-2 ttnn.sampling that crashed
// inside SamplingDeviceOperation::compute_output_specs (issue #8836).
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/47522
class SamplingOpRank2RewritePattern
    : public OpRewritePattern<ttnn::SamplingOp> {
public:
  using OpRewritePattern<ttnn::SamplingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SamplingOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SAMPLINGOPRANK2REWRITEPATTERN_H
