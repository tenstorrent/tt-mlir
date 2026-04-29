// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATENATEOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATENATEOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrites a ConcatOp that would overflow the circular buffer in L1.
//
// This pattern handles the case where:
//   1. concat dim is the last dim
//   2. at least one input has an unaligned last dim (logical != tile-padded),
//      which forces the untilize -> row-major -> transpose(-2,-1) -> concat
//      -> transpose(-2,-1) -> retilize path inside ttnn.concat.
//   3. after the internal transpose, the new last dim becomes dim[-2] of the
//      original tensor, making the CB page size:
//          single_page_size = element_size * padded_shape[-2]
//      which with double buffering exceeds usable L1.
//
// Fix: split all inputs along dim=0 into chunks of C rows where:
//   - element_size * C * 2 <= usable_l1  (chunk CB fits after transpose)
//   - C is a multiple of TILE_HEIGHT=32   (tile-layout alignment)
//
// Each set of input chunks is concatenated along the last dim independently,
// then all chunk results are concatenated along dim=0 to form the final output.
//
// The final dim=0 concat is safe because:
//   - concat dim=0 is not the last dim
//   - build_non_aligned_last_dim_concat does not trigger (not last dim)
//   - single_page_size = tile_size = TILE_HEIGHT * TILE_WIDTH * element_size
class ConcatOpRewritePattern : public OpRewritePattern<ttnn::ConcatOp> {
public:
  using OpRewritePattern<ttnn::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ConcatOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATENATEOPREWRITEPATTERN_H
