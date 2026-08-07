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

/// A band is a slice of the reduction dim: each core gets some of the elements
/// being reduced, so its top-k is partial. Data-parallel slices the non-target
/// dim instead: each core gets whole rows to reduce, so its top-k is final.
struct TopKShardingStrategy {
  /// Reduction tiles one output partial occupies: ceil(k / 32).
  int64_t outputReductionTiles = 1;

  /// Data-parallel: the non-target dim is sliced across cores.
  bool dataParallel = false;
  /// Banded: the reduction dim is split across cores and merged by a tree.
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

/// Reads TopKL1Budget's fields off `leaf`'s operand types. `numBuffers` is the
/// CB buffering factor the generic will be allocated with.
TopKL1Budget topKL1Budget(GenericOp leaf, ttcore::ChipDescAttr chipDesc,
                          int64_t numBuffers);

/// The placeholder `ttir-to-d2m` leaves behind for one topk.
struct SingleCoreTopK {
  GenericOp leaf;
  /// Pre-layout input, so a rebuild shards it rather than re-sharding a layout.
  Value logicalInput;
  /// The index element type the user asked for; every planned index buffer is
  /// i32 regardless, so the extract or a following cast converts to this.
  Type indexElementType;
};

/// Reads the placeholder's operand chain and attributes. Touches nothing
/// downstream of `leaf`, which is where the planned chain gets built.
SingleCoreTopK readSingleCoreTopK(GenericOp leaf);

/// One buffer a split topk needs.
struct TopKBufferPlan {
  mlir::RankedTensorType type;
};

/// A topk's buffers, split by the pass that materializes them.
struct TopKBufferPlans {
  /// The leaf's laid-out operand, placed onto the leaf by d2m-grid-selection.
  TopKBufferPlan input;
  /// Its padding tail's fill; a null type means the layout leaves no tail.
  TopKBufferPlan inputMask;
  /// Built by d2m-lower-topk, and all the plan attribute carries.
  llvm::SmallVector<TopKBufferPlan> lowered;
};

/// The sole definition of what buffers a topk needs. `lowered` is in the order
/// d2m-lower-topk consumes it, which is the whole contract between the two.
/// Touches no IR beyond reading `chain`'s types.
TopKBufferPlans planTopKBuffers(SingleCoreTopK chain,
                                const TopKShardingStrategy &strategy, int64_t k,
                                int32_t dim);

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
