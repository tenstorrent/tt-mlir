// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPPADDIMREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPPADDIMREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// When a ConcatOp is in TILE_LAYOUT and any input's concat-dimension size
// is not tile-aligned (i.e. not a multiple of 32), the runtime takes the
// untilize_rm_retilize_concat path. That path calls untilize_with_unpadding
// with extremely small NOC writes (e.g. 2-4 bytes per element), which hangs
// on multi-device meshes.
//
// This narrowed pattern only applies when the final input is the only operand
// whose concat dimension is not tile-aligned. In that case, padding the tail
// input to a multiple of 32, concatenating, and trimming the trailing suffix
// preserves semantics while bypassing the problematic untilize path.
class ConcatOpPadDimRewritePattern
    : public mlir::OpRewritePattern<ttnn::ConcatOp> {
public:
  using mlir::OpRewritePattern<ttnn::ConcatOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ttnn::ConcatOp op,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPPADDIMREWRITEPATTERN_H
