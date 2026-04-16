// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEGPTOUTPUTSHAPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEGPTOUTPUTSHAPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrites MoeGptOp tilize_out and tilize_out_rm output shapes to use the
// actual device worker grid size. The kernel allocates these as
// HEIGHT_SHARDED across all worker cores with shard shape [2*32, hidden_size],
// but the compiler may use a placeholder first-dim (e.g. 12 for DRAM banks).
// This pattern queries the system descriptor and corrects the first dim to
// num_worker_cores (e.g. 72 on Wormhole: 8x9 grid).
class MoeGptOutputShapeRewritePattern
    : public OpRewritePattern<ttnn::MoeGptOp> {
public:
  using OpRewritePattern<ttnn::MoeGptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MoeGptOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEGPTOUTPUTSHAPEREWRITEPATTERN_H
