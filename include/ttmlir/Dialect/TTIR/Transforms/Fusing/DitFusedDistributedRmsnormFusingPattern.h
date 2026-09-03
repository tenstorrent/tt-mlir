// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_DITFUSEDDISTRIBUTEDRMSNORMFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_DITFUSEDDISTRIBUTEDRMSNORMFUSINGPATTERN_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttir::fusing {

// Fuses:
//   permute(reshape(distributed_rms_norm(x, weight)))
//     -> ttcore.composite "dit_fused_distributed_rmsnorm"
//
// Matches Wan Q/K: RMS over the hidden shard, then split heads into metal's
// `[1, num_heads, seq, head_dim]` layout. Rank-1 γ `[H]` is reshaped to
// `[1, H]` before the composite so TTNNLayout tilizes metal's broadcast
// shape. RoPE stays outside unless transformation_mat + cos + sin are wired
// later.
//
// Anchors on DistributedRMSNormOp. Unconditional — no op-model validation.
class DitFusedDistributedRmsnormFusingPattern
    : public mlir::OpRewritePattern<DistributedRMSNormOp> {
public:
  using OpRewritePattern<DistributedRMSNormOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DistributedRMSNormOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_DITFUSEDDISTRIBUTEDRMSNORMFUSINGPATTERN_H
