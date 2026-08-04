// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_TOPKSHARDINGSTRATEGY_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_TOPKSHARDINGSTRATEGY_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir::tt::d2m {

/// Which axis the reduction occupies at each granularity, for a 2D topk
/// reducing along `dim`. A device shape is `[grid..., shard...]`, so the same
/// axis has a different index depending on which half is indexed.
struct TopKGeometry {
  /// Reduction axis within one half; grid-level maps are built on this.
  std::size_t genRedDim = 0;
  /// Reduction axis in the shard half, where reduction tiles live.
  std::size_t deviceRedDim = 0;
  /// Grid axis the bands spread across.
  std::size_t deviceGridDim = 0;
  /// Axis the extract collapses to tile 0.
  std::size_t extractProjectDim = 0;
};

TopKGeometry getTopKGeometry(int32_t dim, std::size_t physicalRank);

/// How a 2D topk is split across the worker grid: the reduction dim (`dim`)
/// banded `numShards` ways and merged back by a tree, the non-target dim
/// (`1 - dim`) sliced `ntShards` independent ways, or both.
///
/// The `padded*` extents are whole-tensor tile counts rounded up to a multiple
/// of the shard count; the caller masks the padding tail to -inf.
struct TopKShardingStrategy {
  /// Reduction tiles one output partial occupies: ceil(k / 32).
  int64_t outputReductionTiles = 1;

  /// The non-target dim is sliced across cores.
  bool dataParallel = false;
  /// The reduction dim is banded across cores and merged by a tree.
  bool multiCore = false;

  /// Bands the reduction dim is split into; 1 when not banded.
  int64_t numShards = 1;
  /// Reduction tiles across all bands.
  int64_t paddedReductionTiles = 1;

  /// Slices the non-target dim is split into; 1 when not sliced.
  int64_t ntShards = 1;
  /// Non-target tiles across all slices.
  int64_t paddedNonTargetTiles = 1;

  /// Group count per merge round, outermost first, collapsing `numShards` bands
  /// to 1; empty when `multiCore` is false.
  llvm::SmallVector<int64_t> mergeSchedule;
};

/// What one topk costs in L1, measured off `leaf`'s own operands. A split is
/// legal when the leaf's per-core shard and the widest merge round fit
/// together: leafTiles * bytesPerLeafTile + mergeTiles * bytesPerMergeTile +
/// fixedBytes <= usableBytes.
struct TopKL1Budget {
  /// Chip L1 the function may allocate in.
  int64_t usableBytes = 0;
  /// Per reduction tile a leaf core holds, across its shard-sized buffers.
  int64_t bytesPerLeafTile = 0;
  /// Per partial tile a merge core gathers: the leaf's outputs, values paired
  /// with indices.
  int64_t bytesPerMergeTile = 0;
  /// Buffers a constant indexing map pins to one shard, so independent of the
  /// split; counted for both the leaf and the rebuild's own leaf.
  int64_t fixedBytes = 0;
};

/// Reads the above off `leaf`'s operand types. `numBuffers` is the CB buffering
/// factor the generic will be allocated with.
TopKL1Budget topKL1Budget(GenericOp leaf, ttcore::ChipDescAttr chipDesc,
                          int64_t numBuffers);

/// Chooses how to split a 2D topk of `inputShape` reducing along `dim` across
/// `workerGridShape`, keeping every core's shard within `maxTilesPerCore` and
/// requiring that any merge tree it implies both fits `budget` alongside the
/// leaf and can close on this grid. Touches no IR. On failure writes an
/// explanation into `failureReason`.
mlir::FailureOr<TopKShardingStrategy> selectTopKShardingStrategy(
    int64_t k, int32_t dim, llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> workerGridShape, const TopKL1Budget &budget,
    std::string &failureReason);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_TOPKSHARDINGSTRATEGY_H
