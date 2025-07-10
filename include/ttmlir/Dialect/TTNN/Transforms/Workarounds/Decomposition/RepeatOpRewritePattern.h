// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// The RepeatOp currently does not support Int32 type for the input.
// This is tracked in the following Metal issue:
// https://github.com/tenstorrent/tt-metal/issues/24749.
class TTNNRepeatFoldingWorkaround : public OpRewritePattern<ttnn::RepeatOp> {
public:
  using OpRewritePattern<ttnn::RepeatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::RepeatOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATOPREWRITEPATTERN_H
