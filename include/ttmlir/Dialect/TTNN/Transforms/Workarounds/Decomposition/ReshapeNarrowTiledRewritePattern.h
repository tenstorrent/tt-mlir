// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_RESHAPENARROWTILEDREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_RESHAPENARROWTILEDREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// When a ReshapeOp is in TILE_LAYOUT and the output's last dimension is
// not tile-aligned (< 32), the tiled reshape kernel generates many tiny
// L1-to-L1 segment copies (e.g. 4 bytes each for INT32).  On multi-device
// meshes this can hang the device command queue.
//
// This pattern decomposes the problematic reshape into:
//   1. ttnn.to_layout  TILE -> ROW_MAJOR   (untilize)
//   2. ttnn.reshape    in ROW_MAJOR         (metadata / lightweight copy)
//   3. ttnn.to_layout  ROW_MAJOR -> TILE    (tilize)
//
// The ROW_MAJOR reshape path is far better tested and avoids the tiled
// reshape kernel entirely.
class ReshapeNarrowTiledRewritePattern
    : public mlir::OpRewritePattern<ttnn::ReshapeOp> {
public:
  using mlir::OpRewritePattern<ttnn::ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ttnn::ReshapeOp op,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_RESHAPENARROWTILEDREWRITEPATTERN_H
