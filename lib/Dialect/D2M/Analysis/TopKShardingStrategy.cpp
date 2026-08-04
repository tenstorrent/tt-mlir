// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/TopKShardingStrategy.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <optional>

namespace mlir::tt::d2m {

namespace {
constexpr int64_t kTileWidth = ttcore::TileType::getDefaultShape()[1];

// True when every result is a constant, i.e. the operand holds one fixed shard
// regardless of how the reduction is banded.
bool isConstantMap(AffineMap map) {
  return llvm::all_of(map.getResults(), [](AffineExpr expr) {
    return mlir::isa<AffineConstantExpr>(expr);
  });
}

// Tiles in one core's shard, the trailing half of a `[grid..., shard...]` type.
int64_t shardTileCount(mlir::RankedTensorType type) {
  return ttmlir::utils::volume<int64_t>(
      type.getShape().take_back(type.getShape().size() / 2));
}

/// Runs `accept` over core counts in [lo, hi], powers of two first, and returns
/// true at the first count it accepts. Power-of-two counts close the shallowest
/// merge chains, so they are worth trying ahead of a smaller odd count.
bool scanPowerOfTwoFirst(int64_t lo, int64_t hi,
                         llvm::function_ref<bool(int64_t)> accept) {
  for (bool powerOfTwoOnly : {true, false}) {
    for (int64_t cand = lo; cand <= hi; ++cand) {
      if (powerOfTwoOnly && !llvm::isPowerOf2_64(cand)) {
        continue;
      }
      if (accept(cand)) {
        return true;
      }
    }
  }
  return false;
}

/// Smallest legal group count for one merge round: divides `bands`, at most
/// `mergeCap` bands per group, a core count this grid can hold. 0 when none.
int64_t pickMergeGroupCount(int64_t bands, int64_t mergeCap,
                            llvm::ArrayRef<int64_t> workerGridShape) {
  for (int64_t groups = 1; groups < bands; ++groups) {
    if (bands % groups != 0 || bands / groups > mergeCap) {
      continue;
    }
    if (utils::findLegalPhysicalGridForVolume(groups, workerGridShape)
            .empty()) {
      continue;
    }
    return groups;
  }
  return 0;
}

/// Walks the merge chain from `bands` down to 1, recording each round's group
/// count. Nullopt when a level has no grid-legal divisor in [2, mergeCap].
std::optional<llvm::SmallVector<int64_t>>
buildMergeSchedule(int64_t bands, int64_t mergeCap,
                   llvm::ArrayRef<int64_t> workerGridShape) {
  llvm::SmallVector<int64_t> schedule;
  while (bands > 1) {
    int64_t groups = pickMergeGroupCount(bands, mergeCap, workerGridShape);
    if (groups == 0) {
      return std::nullopt;
    }
    schedule.push_back(groups);
    bands = groups;
  }
  return schedule;
}

struct BandSplit {
  int64_t bands;
  int64_t bandTiles;
  llvm::SmallVector<int64_t> schedule;
};
} // namespace

TopKL1Budget topKL1Budget(GenericOp leaf, ttcore::ChipDescAttr chipDesc,
                          int64_t numBuffers) {
  TopKL1Budget budget;
  budget.usableBytes = static_cast<int64_t>(chipDesc.getL1Size()) -
                       static_cast<int64_t>(chipDesc.getL1UnreservedBase());

  llvm::SmallVector<AffineMap> indexingMaps = leaf.getIndexingMapsValue();
  for (auto [operand, map] :
       llvm::zip_equal(leaf.getInputsAndOutputs(), indexingMaps)) {
    auto type = mlir::cast<mlir::RankedTensorType>(operand.getType());
    const int64_t bufferBytes =
        static_cast<int64_t>(
            ttcore::getElementSizeBytes(type.getElementType())) *
        numBuffers;
    if (isConstantMap(map)) {
      // The rebuild emits a leaf of its own, so this is paid for twice.
      budget.fixedBytes += 2 * bufferBytes * shardTileCount(type);
    } else {
      budget.bytesPerLeafTile += bufferBytes;
    }
  }
  // Merge rounds carry the leaf's outputs, never its input.
  for (Value output : leaf.getOutputs()) {
    auto type = mlir::cast<mlir::RankedTensorType>(output.getType());
    budget.bytesPerMergeTile +=
        static_cast<int64_t>(
            ttcore::getElementSizeBytes(type.getElementType())) *
        numBuffers;
  }
  TT_assertv((budget.bytesPerLeafTile > 0 && budget.bytesPerMergeTile > 0),
             "topk leaf has no shard-sized operand");
  return budget;
}

TopKGeometry getTopKGeometry(int32_t dim, std::size_t physicalRank) {
  assert((dim == 0 || dim == 1) && "topk geometry expects a 2D reduction dim");
  assert(physicalRank >= 2u &&
         "topk geometry expects a rank >= 2 device shape");

  TopKGeometry geometry;
  geometry.genRedDim = (dim == 1) ? (physicalRank - 1) : (physicalRank - 2);
  geometry.deviceRedDim = physicalRank + geometry.genRedDim;
  geometry.deviceGridDim = static_cast<std::size_t>(dim);
  geometry.extractProjectDim = (dim == 1) ? (physicalRank - 1) : 0;
  return geometry;
}

mlir::FailureOr<TopKShardingStrategy> selectTopKShardingStrategy(
    int64_t k, int32_t dim, llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> workerGridShape, const TopKL1Budget &budget,
    std::string &failureReason) {
  // What a core can hold when nothing is left for a merge round.
  const int64_t maxTilesPerCore = std::max<int64_t>(
      (budget.usableBytes - budget.fixedBytes) / budget.bytesPerLeafTile, 1);
  const int64_t nonTargetTiles =
      llvm::divideCeil(inputShape[1 - dim], kTileWidth);
  const int64_t fullReductionTiles =
      llvm::divideCeil(inputShape[dim], kTileWidth);
  // topk_block merges reduction tiles pairwise, so a core always holds two.
  const int64_t localReductionTiles = std::max<int64_t>(fullReductionTiles, 2);
  const int64_t maxGridCores = workerGridShape[0] * workerGridShape[1];

  TopKShardingStrategy strategy;
  strategy.outputReductionTiles = llvm::divideCeil(k, kTileWidth);
  // Shard counts need not divide the tile counts evenly; the tails out to
  // paddedReductionTiles and paddedNonTargetTiles are masked to -inf.
  strategy.paddedReductionTiles = fullReductionTiles;
  strategy.paddedNonTargetTiles = nonTargetTiles;

  // Bands one merge core may gather: L1 left over once its leaf shard and the
  // fixed buffers are paid for, divided by what a band's partial costs there.
  auto mergeCapFor = [&](int64_t leafTilesPerCore) {
    const int64_t remaining = budget.usableBytes - budget.fixedBytes -
                              leafTilesPerCore * budget.bytesPerLeafTile;
    return remaining /
           (budget.bytesPerMergeTile * strategy.outputReductionTiles);
  };

  // Neither one-dimensional split works once the non-target dim overflows a
  // core and the reduction dim cannot ride along whole on every core.
  if (nonTargetTiles > maxTilesPerCore &&
      localReductionTiles >= maxTilesPerCore) {
    // Bands take the reduction dim's grid axis and non-target slices the other;
    // dim==0 transposes the index grid, so its bands must fit both axes.
    const int64_t bandGridLimit =
        (dim == 1) ? workerGridShape[1]
                   : std::min(workerGridShape[0], workerGridShape[1]);
    const int64_t ntGridLimit =
        (dim == 1) ? workerGridShape[0] : workerGridShape[1];

    // Fewest bands first: each extra band level costs a merge round.
    bool found = scanPowerOfTwoFirst(2, bandGridLimit, [&](int64_t bands) {
      int64_t bandTiles = llvm::divideCeil(fullReductionTiles, bands);
      int64_t maxNtPerCore = maxTilesPerCore / bandTiles;
      if (bandTiles < 2 || maxNtPerCore < 1) {
        return false;
      }
      // Thinner non-target slices free tile budget for the merge rounds, so
      // walk up until the chain closes.
      for (int64_t ntSh = llvm::divideCeil(nonTargetTiles, maxNtPerCore);
           ntSh <= ntGridLimit; ++ntSh) {
        int64_t ntTiles = llvm::divideCeil(nonTargetTiles, ntSh);
        std::optional<llvm::SmallVector<int64_t>> schedule = buildMergeSchedule(
            bands, mergeCapFor(bandTiles * ntTiles), workerGridShape);
        if (!schedule) {
          continue;
        }
        strategy.mergeSchedule = std::move(*schedule);
        strategy.numShards = bands;
        strategy.ntShards = ntSh;
        strategy.paddedReductionTiles = bands * bandTiles;
        strategy.paddedNonTargetTiles = ntSh * ntTiles;
        return true;
      }
      return false;
    });
    if (!found) {
      failureReason = "D2M topk: no 2D split fits both dimensions within the "
                      "per-core tile budget with a merge tree this worker grid "
                      "can hold";
      return mlir::failure();
    }
    strategy.dataParallel = true;
    strategy.multiCore = true;
    return strategy;
  }

  // A core holds the product of the two dims, so the shard can overflow even
  // when neither dim alone exceeds the budget.
  const bool overBudget =
      nonTargetTiles * localReductionTiles > maxTilesPerCore;
  // Reduction tiles left for a band once the un-sliced non-target dim is paid
  // for.
  const int64_t bandBudget = maxTilesPerCore / nonTargetTiles;
  const int64_t minBands =
      llvm::divideCeil(fullReductionTiles, std::max<int64_t>(bandBudget, 1));

  // Smallest band count that fits the reduction dim in that budget and whose
  // merge tree closes on this grid.
  auto findBandSplit = [&](int64_t lo) -> std::optional<BandSplit> {
    BandSplit split;
    bool found = scanPowerOfTwoFirst(lo, maxGridCores, [&](int64_t bands) {
      int64_t bandTiles = llvm::divideCeil(fullReductionTiles, bands);
      if (bandTiles > bandBudget ||
          utils::findLegalPhysicalGridForVolume(bands, workerGridShape)
              .empty()) {
        return false;
      }
      std::optional<llvm::SmallVector<int64_t>> schedule = buildMergeSchedule(
          bands, mergeCapFor(bandTiles * nonTargetTiles), workerGridShape);
      if (!schedule) {
        return false;
      }
      split = {bands, bandTiles, std::move(*schedule)};
      return true;
    });
    if (!found) {
      return std::nullopt;
    }
    return std::move(split);
  };

  // Banding preferred over data-parallel: it shrinks what every core holds
  // rather than relying on a thin non-target slice. A large-k shape with a
  // short reduction dim has no merge budget, leaving data-parallel as the only
  // split.
  std::optional<BandSplit> bandSplit;
  if (overBudget && bandBudget >= 2) {
    bandSplit = findBandSplit(minBands);
  }

  const int64_t maxNtTilesPerCore = maxTilesPerCore / localReductionTiles;
  strategy.dataParallel = overBudget && !bandSplit && maxNtTilesPerCore >= 1;
  if (strategy.dataParallel) {
    // No merge to pay for, so take the thinnest shard: walk down from one tile
    // per core, past counts no grid can hold.
    const int64_t minNtShards =
        llvm::divideCeil(nonTargetTiles, maxNtTilesPerCore);
    std::optional<int64_t> ntShards;
    for (int64_t cand = std::min(nonTargetTiles, maxGridCores);
         cand >= minNtShards; --cand) {
      if (!utils::findLegalPhysicalGridForVolume(cand, workerGridShape)
               .empty()) {
        ntShards = cand;
        break;
      }
    }
    if (!ntShards) {
      failureReason = "D2M topk: no core count splits the non-target dimension "
                      "within the per-core tile budget on this worker grid";
      return mlir::failure();
    }
    const int64_t ntTiles = llvm::divideCeil(nonTargetTiles, *ntShards);
    strategy.ntShards = *ntShards;
    strategy.paddedNonTargetTiles = *ntShards * ntTiles;
    strategy.paddedReductionTiles = localReductionTiles;
    return strategy;
  }

  if (fullReductionTiles <= bandBudget) {
    // One core already holds the whole reduction dim.
    return strategy;
  }

  // Bands are distributed one per core and folded onto the 2D worker grid by a
  // virtual-grid map attached when the leaf topk is emitted.
  if (!bandSplit) {
    bandSplit = findBandSplit(minBands);
  }
  if (!bandSplit) {
    failureReason = "D2M topk: no band count fits the reduction dim within the "
                    "per-core tile budget with a merge tree this worker grid "
                    "can split evenly";
    return mlir::failure();
  }
  strategy.multiCore = true;
  strategy.numShards = bandSplit->bands;
  strategy.paddedReductionTiles = bandSplit->bands * bandSplit->bandTiles;
  strategy.mergeSchedule = std::move(bandSplit->schedule);
  return strategy;
}

} // namespace mlir::tt::d2m
