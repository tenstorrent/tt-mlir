// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/TopKShardingStrategy.h"

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include <algorithm>
#include <optional>

namespace mlir::tt::d2m {

namespace {
constexpr int64_t kTileWidth = 32;
// L1-safe tile budget for one core's shard of a single buffer.
constexpr int64_t kMaxTilesPerCore = 43;
// k > kTileWidth partials span two tiles and fold through the large-k path, so
// they get a fraction of the budget.
constexpr int64_t kLargeKMergeDivisor = 4;
} // namespace

int64_t pickMergeGroupCount(int64_t bands, int64_t mergeCap,
                            llvm::ArrayRef<int64_t> workerGridShape) {
  for (int64_t groups = 1; groups < bands; ++groups) {
    if (bands % groups != 0) {
      continue;
    }
    if (bands / groups > mergeCap) {
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

// True when repeatedly applying pickMergeGroupCount reduces `bands` to 1.
// Fails if any level has no divisor in [2, mergeCap] that is also grid-legal.
static bool mergeChainCloses(int64_t bands, int64_t mergeCap,
                             llvm::ArrayRef<int64_t> workerGridShape) {
  while (bands > 1) {
    int64_t groups = pickMergeGroupCount(bands, mergeCap, workerGridShape);
    if (groups == 0) {
      return false;
    }
    bands = groups;
  }
  return true;
}

mlir::FailureOr<TopKShardingStrategy> selectTopKShardingStrategy(
    int64_t k, int32_t dim, llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> workerGridShape, std::string &failureReason) {
  TopKShardingStrategy strategy;
  strategy.outputReductionTiles = (k + kTileWidth - 1) / kTileWidth;

  int64_t fullReductionElems = inputShape[dim];
  int64_t nonTargetElems = inputShape[1 - dim];
  int64_t nonTargetTiles = (nonTargetElems + kTileWidth - 1) / kTileWidth;
  int64_t fullReductionTiles =
      (fullReductionElems + kTileWidth - 1) / kTileWidth;

  int64_t localReductionTiles = std::max<int64_t>(fullReductionTiles, 2);

  int64_t maxGridCores = workerGridShape[0] * workerGridShape[1];

  strategy.ntTilesPerCore = nonTargetTiles;
  strategy.paddedNonTargetTiles = nonTargetTiles;
  // Bands need not divide fullReductionTiles evenly; the padding tail out to
  // paddedReductionTiles is masked to -inf, as is the non-target tail out to
  // paddedNonTargetTiles.
  strategy.bandTiles = fullReductionTiles;
  strategy.paddedReductionTiles = fullReductionTiles;

  auto mergeCapFor = [&](int64_t reductionTilesPerCore) -> int64_t {
    int64_t cap = reductionTilesPerCore / strategy.outputReductionTiles;
    return (k > kTileWidth) ? cap / kLargeKMergeDivisor : cap;
  };

  // A joint 2D split is the only option left once the non-target dim overflows
  // a core AND the reduction dim is too big to ride along whole on every core,
  // which is what the data-parallel split needs.
  strategy.twoDim = nonTargetTiles > kMaxTilesPerCore &&
                    localReductionTiles >= kMaxTilesPerCore;

  if (strategy.twoDim) {
    // Bands occupy the reduction dim's grid axis and non-target slices the
    // other, placed directly so every core index is a real coordinate.
    // dim==0 transposes the index grid, so its band count must fit both axes.
    int64_t bandGridLimit =
        (dim == 1) ? workerGridShape[1]
                   : std::min(workerGridShape[0], workerGridShape[1]);
    int64_t ntGridLimit = (dim == 1) ? workerGridShape[0] : workerGridShape[1];

    // Fewest bands first: non-target slices are independent, while every extra
    // band level costs a merge round. Powers of two close the shallowest
    // chains, so try those before any other count.
    for (int pass = 0; pass < 2 && !strategy.multiCore; ++pass) {
      bool powerOfTwoOnly = (pass == 0);
      for (int64_t bands = 2; bands <= bandGridLimit; ++bands) {
        if (powerOfTwoOnly && (bands & (bands - 1)) != 0) {
          continue;
        }
        int64_t candBandTiles = (fullReductionTiles + bands - 1) / bands;
        // topk_block merges reduction tiles pairwise, so a band needs two.
        if (candBandTiles < 2) {
          continue;
        }
        int64_t maxNtPerCore = kMaxTilesPerCore / candBandTiles;
        if (maxNtPerCore < 1) {
          continue;
        }
        int64_t minNtShards =
            (nonTargetTiles + maxNtPerCore - 1) / maxNtPerCore;
        if (minNtShards > ntGridLimit) {
          continue;
        }
        // More non-target slices shrink each core's slice, which frees tile
        // budget for the merge rounds, so walk up until the chain closes.
        for (int64_t ntSh = minNtShards; ntSh <= ntGridLimit; ++ntSh) {
          int64_t candNtTiles = (nonTargetTiles + ntSh - 1) / ntSh;
          int64_t candMergeCap = mergeCapFor(kMaxTilesPerCore / candNtTiles);
          if (candMergeCap < 2 ||
              !mergeChainCloses(bands, candMergeCap, workerGridShape)) {
            continue;
          }
          strategy.numShards = bands;
          strategy.bandTiles = candBandTiles;
          strategy.ntShards = ntSh;
          strategy.ntTilesPerCore = candNtTiles;
          strategy.mergeCap = candMergeCap;
          strategy.dataParallel = true;
          strategy.multiCore = true;
          break;
        }
        if (strategy.multiCore) {
          break;
        }
      }
    }
    if (!strategy.multiCore) {
      failureReason = "D2M topk: no 2D split fits both dimensions within the "
                      "per-core tile budget with a merge tree this worker grid "
                      "can hold";
      return mlir::failure();
    }
    strategy.paddedReductionTiles = strategy.numShards * strategy.bandTiles;
    strategy.paddedNonTargetTiles = strategy.ntShards * strategy.ntTilesPerCore;
    return strategy;
  }

  // A core holds nonTargetTiles * localReductionTiles tiles, so the shard can
  // overflow on the product even when neither dim alone exceeds the budget.
  // Banding the reduction dim is the cheaper fix whenever it can get the
  // product under budget on its own, because it shrinks what every core holds
  // instead of relying on a thin non-target slice; the reduction dim needs two
  // tiles per band to fold, so bands cap out at localReductionTiles / 2.
  int64_t maxNtTilesPerCore = kMaxTilesPerCore / localReductionTiles;
  bool overBudget = nonTargetTiles * localReductionTiles > kMaxTilesPerCore;

  // Searches for a band count that fits the reduction dim within the budget
  // left by an un-sliced non-target dim and whose merge tree closes on this
  // grid. Powers of two first: they always close and give the shallowest tree,
  // then any count whose chain closes, which is what reaches past 64 bands on
  // larger grids. Returns {bands, bandTiles}, or nullopt when none works.
  int64_t bandBudget = kMaxTilesPerCore / std::max<int64_t>(nonTargetTiles, 1);
  int64_t bandMergeCap = mergeCapFor(bandBudget);
  auto findBandSplit =
      [&](int64_t minShards) -> std::optional<std::pair<int64_t, int64_t>> {
    for (int pass = 0; pass < 2; ++pass) {
      bool powerOfTwoOnly = (pass == 0);
      for (int64_t cand = minShards; cand <= maxGridCores; ++cand) {
        if (powerOfTwoOnly && (cand & (cand - 1)) != 0) {
          continue;
        }
        if (utils::findLegalPhysicalGridForVolume(cand, workerGridShape)
                .empty()) {
          continue;
        }
        int64_t candBandTiles = (fullReductionTiles + cand - 1) / cand;
        if (candBandTiles > bandBudget) {
          continue;
        }
        if (!mergeChainCloses(cand, bandMergeCap, workerGridShape)) {
          continue;
        }
        return std::make_pair(cand, candBandTiles);
      }
    }
    return std::nullopt;
  };

  // Banding the reduction dim is the cheaper fix when it can get the shard
  // under budget on its own, since it shrinks what every core holds instead of
  // relying on a thin non-target slice. It only wins if a band count actually
  // exists, though: a large-k shape with a short reduction dim has no merge
  // budget to spend, and there data-parallel is the only split left.
  std::optional<std::pair<int64_t, int64_t>> bandSplit;
  if (overBudget && bandBudget >= 2) {
    bandSplit =
        findBandSplit((fullReductionTiles + bandBudget - 1) / bandBudget);
  }
  strategy.dataParallel =
      maxNtTilesPerCore >= 1 && overBudget && !bandSplit.has_value();
  if (strategy.dataParallel) {
    int64_t minNtShards =
        (nonTargetTiles + maxNtTilesPerCore - 1) / maxNtTilesPerCore;
    // Slices are independent with no merge to pay for, so take the thinnest
    // shard: walk down from one tile per core, past counts no grid can hold.
    std::optional<int64_t> legalNtShards;
    for (int64_t cand = std::min(nonTargetTiles, maxGridCores);
         cand >= minNtShards; --cand) {
      if (!utils::findLegalPhysicalGridForVolume(cand, workerGridShape)
               .empty()) {
        legalNtShards = cand;
        break;
      }
    }
    if (!legalNtShards) {
      failureReason = "D2M topk: no core count splits the non-target dimension "
                      "within the per-core tile budget on this worker grid";
      return mlir::failure();
    }
    strategy.ntShards = *legalNtShards;
    strategy.ntTilesPerCore =
        (nonTargetTiles + strategy.ntShards - 1) / strategy.ntShards;
    strategy.paddedNonTargetTiles = strategy.ntShards * strategy.ntTilesPerCore;
  }

  int64_t maxReductionTilesPerCore = strategy.dataParallel
                                         ? fullReductionTiles
                                         : kMaxTilesPerCore / nonTargetTiles;

  strategy.mergeCap = mergeCapFor(maxReductionTilesPerCore);

  int64_t minShards = (fullReductionTiles + maxReductionTilesPerCore - 1) /
                      maxReductionTilesPerCore;
  strategy.multiCore = !strategy.dataParallel && minShards > 1;

  // Bands are distributed one per core and folded onto the 2D worker grid by a
  // virtual-grid map attached when the leaf topk is emitted.
  if (strategy.multiCore) {
    // Reuse the split already found above when the budget it searched under
    // matches; otherwise the shard fits the non-target dim whole and only the
    // reduction dim needs splitting, so search now.
    if (!bandSplit) {
      bandSplit = findBandSplit(minShards);
    }
    if (!bandSplit) {
      failureReason = "D2M topk: no band count fits the reduction dim within "
                      "the per-core tile budget with a merge tree this worker "
                      "grid can split evenly";
      return mlir::failure();
    }
    strategy.numShards = bandSplit->first;
    strategy.bandTiles = bandSplit->second;
    strategy.paddedReductionTiles = strategy.numShards * strategy.bandTiles;
  }
  if (strategy.dataParallel) {
    strategy.paddedReductionTiles = localReductionTiles;
  }
  return strategy;
}

} // namespace mlir::tt::d2m
