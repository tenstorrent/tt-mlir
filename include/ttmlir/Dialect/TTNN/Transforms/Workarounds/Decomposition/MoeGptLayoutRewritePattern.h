// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEGPTLAYOUTREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEGPTLAYOUTREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Fixes up MoeGptOp operand and result encodings that the generic operand-
// workarounds framework cannot express:
//
//   - Weight inputs (w0_w1, w2): inserts a ToLayoutOp that converts the
//     incoming bf16 DRAM INTERLEAVED weight to bfloat4_b TILE DRAM
//     HEIGHT_SHARDED at the kernel-matching DRAM-bank coords. The bank
//     permutation is carried by the ToLayoutOp's MemoryConfigAttr via an
//     explicit CoreRangeSet; TTNNLayoutAttr/GridAttr can only express
//     affine-map-derived ranges and therefore cannot represent this
//     permutation.
//
//   - Tilize outputs (tilize_out, tilize_out_rm): L1 HEIGHT_SHARDED on a
//     {numWorkerCores, 1} grid. The runtime's insertTTNNTensorAndValidate
//     fails if the kernel's allocated memory config (HS over all worker
//     cores) doesn't match the compile-time encoding.
//
// Other encoding aspects (buffer types, memory layouts, data types) are
// handled by the generic framework via createMoeGptOpOperandsWorkarounds.
class MoeGptLayoutRewritePattern : public OpRewritePattern<ttnn::MoeGptOp> {
public:
  using OpRewritePattern<ttnn::MoeGptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MoeGptOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEGPTLAYOUTREWRITEPATTERN_H
