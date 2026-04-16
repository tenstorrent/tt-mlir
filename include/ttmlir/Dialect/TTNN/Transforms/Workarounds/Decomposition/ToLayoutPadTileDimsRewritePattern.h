// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_TOLAYOUTPADTILEDIMSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_TOLAYOUTPADTILEDIMSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround for ttnn.tilize not supporting HEIGHT_SHARDED L1 tensors with
// non-tile-aligned shard shapes (tt-metal#30541).
//
// Matches to_layout(tile) on HEIGHT_SHARDED L1 inputs where the last two
// dimensions are not multiples of 32 (tile size). Decomposes into:
//   1. Unshard to DRAM INTERLEAVED (via to_memory_config, or host round-trip
//      for UINT16 since ttnn.copy doesn't support u16 — tt-metal#41689)
//   2. Pad to tile-aligned dimensions
//   3. to_layout(TILE) on the padded DRAM tensor (handles dtype conversion
//      if the original op specifies one)
//   4. Slice back to original dimensions
//   5. Reshard to original HEIGHT_SHARDED L1 config (via to_memory_config,
//      or host round-trip for UINT16)

class ToLayoutPadTileDimsRewritePattern
    : public OpRewritePattern<ttnn::ToLayoutOp> {
public:
  using OpRewritePattern<ttnn::ToLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ToLayoutOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_TOLAYOUTPADTILEDIMSREWRITEPATTERN_H
