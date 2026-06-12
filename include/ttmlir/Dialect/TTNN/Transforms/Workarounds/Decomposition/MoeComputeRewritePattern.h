// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Fixes up a ttnn.moe_compute that the tt-metal kernel relies on but validate()
// does not enforce: reshards the expert-indices/scores inputs to L1
// HEIGHT_SHARDED on the tilize drain core, and derives the compute-path output
// layouts (results 0-4) analytically, mirroring
// MoEComputeDeviceOperation::compute_output_specs. In compute_only the combine
// output (result 5) aliases matmul_output.
class MoeComputeRewritePattern : public OpRewritePattern<ttnn::MoeComputeOp> {
public:
  using OpRewritePattern<ttnn::MoeComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MoeComputeOp op,
                                PatternRewriter &rewriter) const override;

private:
  // Reshards the expert-indices/scores inputs to L1 HEIGHT_SHARDED on the
  // tilize drain core. Fails the match when they are already resharded or no
  // drain core fits the worker grid.
  LogicalResult reshardTilizeInputs(ttnn::MoeComputeOp op,
                                    PatternRewriter &rewriter) const;

  // Sets the routing and tilize output types (results 0-3) — per-expert tokens,
  // expert activation, expert-to-token, and the shared tilize buffer.
  void setRoutingAndTilizeOutputTypes(ttnn::MoeComputeOp op,
                                      PatternRewriter &rewriter) const;

  // Sets the matmul output type (result 4): tilize_output re-perceived as
  // row-major. In compute_only the combine output (result 5) aliases it.
  void setMatmulOutputType(ttnn::MoeComputeOp op,
                           PatternRewriter &rewriter) const;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H
