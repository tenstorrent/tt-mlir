// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Currently, not all binary eltwise ops in tt-metal support implicit
// broadcasting on their operands. For those that don't, we have to explicate
// the broadcast on each operand.
class ExplicateOperandBroadcastsRewritePattern
    : public OpInterfaceRewritePattern<
          ttnn::TTNNRequiresExplicitBroadcastInterface> {
public:
  using OpInterfaceRewritePattern<
      ttnn::TTNNRequiresExplicitBroadcastInterface>::OpInterfaceRewritePattern;

  LogicalResult
  matchAndRewrite(ttnn::TTNNRequiresExplicitBroadcastInterface srcOp,
                  PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EXPLICATEOPERANDBROADCASTSREWRITEPATTERN_H
