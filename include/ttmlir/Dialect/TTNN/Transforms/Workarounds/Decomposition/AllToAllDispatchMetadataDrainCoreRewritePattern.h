// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ALLTOALLDISPATCHMETADATADRAINCOREREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ALLTOALLDISPATCHMETADATADRAINCOREREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which sets the drain_sync_tilizer_core attribute on
// AllToAllDispatchMetadataOp when it is not already present.
// The metal kernel requires this core coordinate to know where to L1-shard
// the output indices and scores tensors. Default: CoreCoord(0, 0).
class AllToAllDispatchMetadataDrainCoreRewritePattern
    : public OpRewritePattern<ttnn::AllToAllDispatchMetadataOp> {
public:
  using OpRewritePattern<ttnn::AllToAllDispatchMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllToAllDispatchMetadataOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ALLTOALLDISPATCHMETADATADRAINCOREREWRITEPATTERN_H
