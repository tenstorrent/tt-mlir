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
// This pattern decomposes the problematic reshape into a pad+reshape+slice
// sequence that stays entirely in tiled layout, avoiding the expensive
// to_layout circular buffers that a tiled->RM->tiled round-trip would require:
//   1. ttnn.reshape    flatten input to 1D [totalElements]
//   2. ttnn.pad        [totalElements] -> [outerOutput * D'] (D' tile-aligned)
//   3. ttnn.reshape    [outerOutput * D'] -> [..., D']
//   4. ttnn.slice      [..., D'] -> [..., D]  (trim padding)
class ReshapeNarrowTiledRewritePattern
    : public mlir::OpRewritePattern<ttnn::ReshapeOp> {
public:
  using mlir::OpRewritePattern<ttnn::ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ttnn::ReshapeOp op,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_RESHAPENARROWTILEDREWRITEPATTERN_H
