// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_TOPKSHARDINGSTRATEGY_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_TOPKSHARDINGSTRATEGY_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace mlir::tt::d2m {

/// How a 2D topk is split across the worker grid.
///
/// The reduction dim (`dim`) may be banded `numShards` ways, with each band's
/// partial result merged back by a tree; the non-target dim (`1 - dim`) may be
/// sliced `ntShards` ways, which needs no merge because the slices are
/// independent. `twoDim` sets both at once.
///
/// Tile counts are per-core unless named `padded*`, which are the whole-tensor
/// extents after rounding up to a multiple of the shard count. The padding tail
/// is masked to -inf by the caller.
struct TopKShardingStrategy {
  /// Reduction tiles one output partial occupies: ceil(k / 32). k > 32 spans
  /// two tiles and folds through the large-k path.
  int64_t outputReductionTiles = 1;

  /// True when the non-target dim is sliced across cores. Triggered by the
  /// per-core tile product, not by either dim's extent alone.
  bool dataParallel = false;
  /// True when the reduction dim is banded across cores and merged by a tree.
  bool multiCore = false;
  /// True when both dims are split. The resulting virtual grid is already a
  /// physical one, so the caller leaves the fold maps null.
  bool twoDim = false;

  /// Bands the reduction dim is split into; 1 when not banded.
  int64_t numShards = 1;
  /// Reduction tiles per band.
  int64_t bandTiles = 1;
  /// numShards * bandTiles, i.e. the reduction extent including padding.
  int64_t paddedReductionTiles = 1;

  /// Slices the non-target dim is split into; 1 when not sliced.
  int64_t ntShards = 1;
  /// Non-target tiles per slice.
  int64_t ntTilesPerCore = 1;
  /// ntShards * ntTilesPerCore, i.e. the non-target extent including padding.
  int64_t paddedNonTargetTiles = 1;

  /// Bands one core can absorb in a single merge round. Consumed by
  /// pickMergeGroupCount while walking the merge tree.
  int64_t mergeCap = 0;
};

/// Picks the smallest legal group count for one merge round (divides `bands`,
/// <= `mergeCap` bands/group, valid core count); smallest groups keeps the tree
/// shallow. Returns 0 when no legal group count exists.
int64_t pickMergeGroupCount(int64_t bands, int64_t mergeCap,
                            llvm::ArrayRef<int64_t> workerGridShape);

/// Chooses how to split a 2D topk of `inputShape` reducing along `dim` across
/// `workerGridShape`, keeping every core's shard within the L1 tile budget and
/// requiring that any merge tree it implies can actually close on this grid.
///
/// On failure, writes an explanation into `failureReason` and returns failure;
/// the caller is expected to forward it to notifyMatchFailure.
///
/// Pure: depends only on its arguments, touches no IR.
mlir::FailureOr<TopKShardingStrategy> selectTopKShardingStrategy(
    int64_t k, int32_t dim, llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> workerGridShape, std::string &failureReason);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_TOPKSHARDINGSTRATEGY_H
