// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ROTARYEMBEDDINGOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ROTARYEMBEDDINGOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <utility>

namespace mlir::tt::ttnn::workarounds::decomposition {

// If seq_len (dim -2) or head_dim (dim -1) need alignment padding, creates
// padded inputs and a padded RotaryEmbeddingOp plus a SliceStaticOp to restore
// the original shape, returning them as a pair. seq_len is padded to
// TILE_HEIGHT (https://github.com/tenstorrent/tt-metal/issues/31567); head_dim
// is padded to TILE_WIDTH * 2
// (https://github.com/tenstorrent/tt-metal/issues/41313).
// Returns std::nullopt if no padding is needed.
// Caller is responsible for erasing the returned ops if they are temporary.
std::optional<std::pair<RotaryEmbeddingOp, SliceStaticOp>>
getWorkaroundedOp(RotaryEmbeddingOp ropeOp, PatternRewriter &rewriter);

class RotaryEmbeddingOpRewritePattern
    : public OpRewritePattern<RotaryEmbeddingOp> {
public:
  using OpRewritePattern<RotaryEmbeddingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RotaryEmbeddingOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ROTARYEMBEDDINGOPREWRITEPATTERN_H
