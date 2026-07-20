// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMWIDTHSHARDINPUTREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMWIDTHSHARDINPUTREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {

// A width-shard core-grid selection: `numCores` cores placed as a solid
// `gridW` x `gridH` rectangle anchored at (0,0).
struct WidthShardCoreGrid {
  int64_t numCores;
  int64_t gridW;
  int64_t gridH;
};

// Choose the width-shard core grid for a tensor whose last (width) dimension
// spans `numWidthTiles` tiles, on a `workerGridH` x `workerGridW` worker grid.
//
// Picks the largest core count that (a) evenly divides `numWidthTiles` and
// (b) factors into a rectangle that fits the worker grid, preferring the
// tallest rectangle so the placement mirrors the validated tt-metal decode
// config (28 tiles -> 4 wide x 7 tall). Falls back to a single core when no
// multi-core rectangle fits (e.g. a prime tile count larger than the grid).
//
// The rectangle must be *solid*: the fused DistributedRMSNorm kernel derives
// both its LayerNorm program-config grid and its cross-device semaphore core
// range from the bounding box of this shard. A ragged placement (e.g. 28 cores
// laid row-major on an 8-wide grid -> bounding box 8x4 with four empty cores)
// leaves phantom cores in the grid, which the kernel reads as garbage. Keeping
// boundingBox == shardCoreSet is what makes the fused path correct.
inline WidthShardCoreGrid chooseWidthShardCoreGrid(int64_t numWidthTiles,
                                                   int64_t workerGridH,
                                                   int64_t workerGridW) {
  int64_t maxCores = workerGridH * workerGridW;
  WidthShardCoreGrid result{/*numCores=*/1, /*gridW=*/1, /*gridH=*/1};
  for (int64_t c = std::min(maxCores, numWidthTiles); c >= 1; --c) {
    if (numWidthTiles % c != 0) {
      continue;
    }
    // Prefer the tallest rectangle (largest height first).
    for (int64_t h = std::min(workerGridH, c); h >= 1; --h) {
      if (c % h == 0 && c / h <= workerGridW) {
        return WidthShardCoreGrid{/*numCores=*/c, /*gridW=*/c / h, /*gridH=*/h};
      }
    }
  }
  return result;
}

// Workaround that prepares DistributedRMSNormOp for the tt-metal
// fused_rms_minimal kernel. Width-shards the input/residual into L1,
// reshapes the weight to ROW_MAJOR (N/32, 32), sets the output memory
// config to match the input shard spec, creates a stats scratch tensor
// (EmptyOp), sets compute_config if absent, and computes a
// LayerNormShardedMultiCoreProgramConfig from the shard spec.
// Related tt-metal issue https://github.com/tenstorrent/tt-metal/issues/37746
class DistributedRMSNormWidthShardInputRewritePattern
    : public OpRewritePattern<ttnn::DistributedRMSNormOp> {
public:
  using OpRewritePattern<ttnn::DistributedRMSNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::DistributedRMSNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMWIDTHSHARDINPUTREWRITEPATTERN_H
