// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Currently, remainder op does not support implicit
// broadcast on both operands.
// Should remove this workaround after the following issue is resolved :
// https://github.com/tenstorrent/tt-metal/issues/24635.
class ExplicateOperandBroadcastsRewritePattern
    : public OpInterfaceRewritePattern<
          ttnn::RequiresExplicitBroadcastInterface> {
public:
  using OpInterfaceRewritePattern<
      ttnn::RequiresExplicitBroadcastInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ttnn::RequiresExplicitBroadcastInterface srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H
