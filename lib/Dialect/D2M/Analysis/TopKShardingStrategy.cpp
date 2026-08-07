// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/TopKShardingStrategy.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/TopKUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/STLExtras.h"
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
  // Bands take the reduction dim's grid axis and non-target slices the other;
  // dim==0 transposes the index grid, so its bands must fit both axes.
  const int64_t bandGridLimit =
      (dim == 1) ? workerGridShape[1]
                 : std::min(workerGridShape[0], workerGridShape[1]);
  const int64_t ntGridLimit =
      (dim == 1) ? workerGridShape[0] : workerGridShape[1];

  TopKShardingStrategy strategy;
  strategy.outputReductionTiles = llvm::divideCeil(k, kTileWidth);
  // Shard counts need not divide the tile counts evenly; the tails out to
  // paddedReductionTiles and paddedNonTargetTiles are masked to -inf.
  strategy.paddedReductionTiles = fullReductionTiles;
  strategy.paddedNonTargetTiles = nonTargetTiles;

  // Bands one merge core may gather: L1 left over once its leaf shard and the
  // fixed buffers are paid for, divided by what a band's partial costs there.
  // A gathered band spans the core's whole non-target extent.
  auto mergeCapFor = [&](int64_t leafTilesPerCore, int64_t ntTiles) {
    const int64_t remaining = budget.usableBytes - budget.fixedBytes -
                              leafTilesPerCore * budget.bytesPerLeafTile;
    return remaining /
           (budget.bytesPerMergeTile * strategy.outputReductionTiles * ntTiles);
  };

  auto accept = [&](int64_t bands, int64_t ntShards) {
    const int64_t bandTiles = (bands == 1)
                                  ? localReductionTiles
                                  : llvm::divideCeil(fullReductionTiles, bands);
    const int64_t ntTiles = llvm::divideCeil(nonTargetTiles, ntShards);
    // topk_block merges reduction tiles pairwise, so a band is never one tile.
    if (bands > 1 && bandTiles < 2) {
      return false;
    }
    if (bandTiles * ntTiles > maxTilesPerCore) {
      return false;
    }
    // A one-dimensional split folds its cores onto the whole grid; a 2D one
    // spends an axis per dim.
    const bool placeable =
        (bands > 1 && ntShards > 1)
            ? (bands <= bandGridLimit && ntShards <= ntGridLimit)
            : !utils::findLegalPhysicalGridForVolume(bands * ntShards,
                                                     workerGridShape)
                   .empty();
    if (!placeable) {
      return false;
    }
    if (bands > 1) {
      std::optional<llvm::SmallVector<int64_t>> schedule = buildMergeSchedule(
          bands, mergeCapFor(bandTiles * ntTiles, ntTiles), workerGridShape);
      if (!schedule) {
        return false;
      }
      strategy.mergeSchedule = std::move(*schedule);
    }
    strategy.multiCore = bands > 1;
    strategy.dataParallel = ntShards > 1;
    strategy.numShards = bands;
    strategy.ntShards = ntShards;
    strategy.paddedReductionTiles = bands * bandTiles;
    strategy.paddedNonTargetTiles = ntShards * ntTiles;
    return true;
  };

  // One core holding both dims outright, then the splits in ascending cost.
  if (accept(1, 1)) {
    return strategy;
  }

  // Banding preferred over data-parallel: it shrinks what every core holds
  // rather than relying on a thin non-target slice. Fewest bands first, since
  // each extra band level costs a merge round.
  if (scanPowerOfTwoFirst(2, maxGridCores,
                          [&](int64_t bands) { return accept(bands, 1); })) {
    return strategy;
  }

  // A large-k shape with a short reduction dim has no merge budget, leaving
  // data-parallel as the only split. No merge to pay for, so take the thinnest
  // shard: walk down from one tile per core, past counts no grid can hold.
  for (int64_t ntShards = std::min(nonTargetTiles, maxGridCores); ntShards >= 2;
       --ntShards) {
    if (accept(1, ntShards)) {
      return strategy;
    }
  }

  // Neither one-dimensional split fits when both dims are larger than L1 can
  // hold. More bands never shrinks what a merge core gathers; only a narrower
  // non-target slice does.
  if (scanPowerOfTwoFirst(2, bandGridLimit, [&](int64_t bands) {
        for (int64_t ntShards = 2; ntShards <= ntGridLimit; ++ntShards) {
          if (accept(bands, ntShards)) {
            return true;
          }
        }
        return false;
      })) {
    return strategy;
  }

  failureReason = "D2M topk: no split of the reduction and non-target "
                  "dimensions fits the per-core tile budget with a merge tree "
                  "this worker grid can hold";
  return mlir::failure();
}

namespace {

// Walks back through the to_layouts, view_layouts, masks, and transpose
// generics of ttir-to-d2m's placeholder chain to the original logical value.
Value findLogicalInput(Value value) {
  auto isDevice = [](Value v) {
    auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
    return type &&
           mlir::isa_and_nonnull<ttcore::MetalLayoutAttr>(type.getEncoding());
  };
  while (isDevice(value)) {
    Operation *def = value.getDefiningOp();
    if (auto toLayout = mlir::dyn_cast_if_present<ToLayoutOp>(def)) {
      value = toLayout.getInput();
    } else if (auto view = mlir::dyn_cast_if_present<ViewLayoutOp>(def)) {
      value = view.getInput();
    } else if (auto mask = mlir::dyn_cast_if_present<MaskOp>(def)) {
      value = mask.getInput();
    } else if (auto generic = mlir::dyn_cast_if_present<GenericOp>(def);
               generic && generic.getInputs().size() == 1) {
      value = generic.getInputs()[0];
    } else {
      return nullptr;
    }
  }
  return value;
}

} // namespace

SingleCoreTopK readSingleCoreTopK(GenericOp leaf) {
  SingleCoreTopK chain;
  chain.leaf = leaf;
  chain.logicalInput = findLogicalInput(leaf.getInputs()[0]);
  TT_assert(chain.logicalInput);
  auto indexType = leaf->getAttrOfType<TypeAttr>(utils::kTopkIndexTypeAttr);
  TT_assertv(indexType, "topk leaf carries no d2m.topk_index_type");
  chain.indexElementType = indexType.getValue();
  return chain;
}

namespace {

// Build a sharded L1 MetalLayoutAttr for `logicalShape` where dim i is padded
// out to `tilesPerDim[i]` tiles. Uses default collapsed intervals.
ttcore::MetalLayoutAttr
buildShardedTileLayout(MLIRContext *ctx, ArrayRef<int64_t> logicalShape,
                       ArrayRef<int64_t> tilesPerDim,
                       ttcore::MemorySpace memorySpace) {
  TT_assert(tilesPerDim.size() == logicalShape.size());
  const int64_t tileDim = ttcore::TileType::getDefaultShape()[0];

  SmallVector<int64_t> dimAlignments;
  dimAlignments.reserve(tilesPerDim.size());
  for (int64_t tiles : tilesPerDim) {
    dimAlignments.push_back(tiles * tileDim);
  }
  return ttcore::MetalLayoutAttr::get(
      ctx, logicalShape, memorySpace, ttcore::TensorMemoryLayout::Sharded,
      ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
          ctx, logicalShape.size()),
      dimAlignments);
}

// Rebuilds `layout`'s logical shape from a [grid..., shard...] `deviceShape`:
// narrowing a partial's device extent shrinks what it logically stands for.
ttcore::MetalLayoutAttr
rebuildLayoutForDeviceShape(ttcore::MetalLayoutAttr layout,
                            ArrayRef<int64_t> deviceShape) {
  TT_assert(deviceShape.size() % 2 == 0u);
  const std::size_t physicalRank = deviceShape.size() / 2;
  const int64_t tileDim = ttcore::TileType::getDefaultShape()[0];

  llvm::SmallVector<int64_t> logicalShape;
  logicalShape.reserve(physicalRank);
  for (std::size_t i = 0; i < physicalRank; ++i) {
    logicalShape.push_back(deviceShape[i] * deviceShape[i + physicalRank] *
                           tileDim);
  }
  return ttcore::MetalLayoutAttr::get(
      layout.getContext(), logicalShape, layout.getDimAlignments(),
      layout.getCollapsedIntervals(), layout.getMemorySpace(),
      layout.getMemoryLayout());
}

} // namespace

TopKBufferPlans planTopKBuffers(SingleCoreTopK chain,
                                const TopKShardingStrategy &strategy, int64_t k,
                                int32_t dim) {
  MLIRContext *ctx = chain.leaf.getContext();
  auto logicalInputType =
      mlir::cast<RankedTensorType>(chain.logicalInput.getType());
  const std::size_t rank = logicalInputType.getRank();
  // The reduction axis in the shard half of a [grid..., shard...] device shape.
  const std::size_t deviceRedDim = rank + dim;
  llvm::SmallVector<int64_t> tileShape =
      llvm::to_vector(ttcore::TileType::getDefaultShape());

  auto leafInputLayout = mlir::cast<ttcore::MetalLayoutAttr>(
      mlir::cast<RankedTensorType>(chain.leaf.getInputs()[0].getType())
          .getEncoding());
  const ttcore::MemorySpace memSpace = leafInputLayout.getMemorySpace();

  auto valTileType =
      ttcore::TileType::get(logicalInputType.getElementType(), tileShape);
  auto idxTileType =
      ttcore::TileType::get(IntegerType::get(ctx, 32), tileShape);

  TopKBufferPlans plans;
  auto make = [](ArrayRef<int64_t> deviceShape, ttcore::TileType tileType,
                 ttcore::MetalLayoutAttr layout) {
    return TopKBufferPlan{RankedTensorType::get(deviceShape, tileType, layout)};
  };
  auto add = [&](ArrayRef<int64_t> deviceShape, ttcore::TileType tileType,
                 ttcore::MetalLayoutAttr layout) {
    plans.lowered.push_back(make(deviceShape, tileType, layout));
  };

  // The leaf input: banded `gridCols` ways along the reduction dim, sliced
  // `ntShards` ways along the non-target dim.
  const int64_t gridCols = strategy.multiCore ? strategy.numShards : 1;
  llvm::SmallVector<int64_t> shardTiles(rank, 1);
  shardTiles[dim] = strategy.paddedReductionTiles;
  shardTiles[1 - dim] = strategy.paddedNonTargetTiles;
  llvm::SmallVector<int64_t> grid =
      (dim == 1) ? llvm::SmallVector<int64_t>{strategy.ntShards, gridCols}
                 : llvm::SmallVector<int64_t>{gridCols, strategy.ntShards};
  ttcore::MetalLayoutAttr inputLayout = buildShardedTileLayout(
      ctx, logicalInputType.getShape(), shardTiles, memSpace);
  llvm::SmallVector<int64_t> inputShape =
      inputLayout.getDeviceShape(grid, tileShape);

  plans.input = make(inputShape, valTileType, inputLayout);

  // A padding tail exists only when the grid times the shard overshoots the
  // logical extent.
  bool needsMask = false;
  for (std::size_t i = 0; i < rank; ++i) {
    needsMask |= inputShape[i] * inputShape[i + rank] * tileShape[0] >
                 logicalInputType.getShape()[i];
  }
  if (needsMask) {
    plans.inputMask = make(inputShape, valTileType, inputLayout);
  }

  // `topk_block` sorts down tile columns, so dim=1 transposes the shard first.
  if (dim == 1) {
    add(inputShape, valTileType, inputLayout);
  }

  // One scratch tile per core: the kernel derives the whole index buffer from
  // it plus its own grid coordinate, so its content doesn't depend on the
  // split.
  llvm::SmallVector<int64_t> scratchShape(grid);
  scratchShape.append({1, 1});
  add(scratchShape, idxTileType,
      ttcore::MetalLayoutAttr::get(ctx, llvm::SmallVector<int64_t>{1, 1},
                                   ttcore::MemorySpace::DeviceL1,
                                   ttcore::TensorMemoryLayout::Sharded));

  add(inputShape, valTileType, inputLayout);
  add(inputShape, idxTileType, inputLayout);

  // Only banding produces wide partials worth narrowing, to keep them from
  // staying live on every core and overflowing L1.
  llvm::SmallVector<int64_t> levelShape(inputShape);
  ttcore::MetalLayoutAttr levelLayout = inputLayout;
  if (gridCols > 1) {
    levelShape[deviceRedDim] = strategy.outputReductionTiles;
    levelLayout = rebuildLayoutForDeviceShape(inputLayout, levelShape);
    add(levelShape, valTileType, levelLayout);
    add(levelShape, idxTileType, levelLayout);
  }

  // Each round re-splits the surviving extent: `numGroups` cores each holding
  // `groupTiles` bands' worth of reduction tiles, so grid x shard is preserved.
  int64_t bands = gridCols;
  for (int64_t numGroups : strategy.mergeSchedule) {
    const int64_t groupTiles = bands / numGroups;

    llvm::SmallVector<int64_t> wideShape(levelShape);
    wideShape[dim] = numGroups;
    wideShape[deviceRedDim] = groupTiles * strategy.outputReductionTiles;
    ttcore::MetalLayoutAttr wideLayout =
        rebuildLayoutForDeviceShape(levelLayout, wideShape);

    // The gather and the merge are separate generics: the DMA-expansion pass
    // supports only one composite view per generic (#7600).
    add(wideShape, valTileType, wideLayout);
    add(wideShape, idxTileType, wideLayout);
    add(wideShape, valTileType, wideLayout);
    add(wideShape, idxTileType, wideLayout);

    levelShape = wideShape;
    levelShape[deviceRedDim] = strategy.outputReductionTiles;
    levelLayout = rebuildLayoutForDeviceShape(wideLayout, levelShape);
    add(levelShape, valTileType, levelLayout);
    add(levelShape, idxTileType, levelLayout);

    bands = numGroups;
  }

  // A data-parallel split must keep the extract's destination on the same
  // cores as its source; anything else collects onto one.
  llvm::SmallVector<int64_t> outLogicalShape(logicalInputType.getShape());
  outLogicalShape[dim] = k;
  llvm::SmallVector<int64_t> extractGrid =
      strategy.dataParallel
          ? ((dim == 1) ? llvm::SmallVector<int64_t>{strategy.ntShards, 1}
                        : llvm::SmallVector<int64_t>{1, strategy.ntShards})
          : llvm::SmallVector<int64_t>{1, 1};
  llvm::SmallVector<int64_t> extractShardTiles(rank, 1);
  extractShardTiles[dim] = strategy.outputReductionTiles;
  extractShardTiles[1 - dim] =
      strategy.dataParallel
          ? strategy.paddedNonTargetTiles
          : llvm::divideCeil(logicalInputType.getShape()[1 - dim], kTileWidth);

  ttcore::MetalLayoutAttr extractLayout =
      buildShardedTileLayout(ctx, outLogicalShape, extractShardTiles, memSpace);
  llvm::SmallVector<int64_t> extractShape =
      extractLayout.getDeviceShape(extractGrid, tileShape);

  // dim == 0's extract casts on the way out; dim == 1's transposes instead, so
  // the cast follows in a region of its own.
  const bool castsInExtract = dim == 0;
  auto extractIdxTileType = ttcore::TileType::get(
      castsInExtract ? chain.indexElementType : IntegerType::get(ctx, 32),
      tileShape);

  add(extractShape, valTileType, extractLayout);
  add(extractShape, extractIdxTileType, extractLayout);

  // The cast is elementwise, so it follows its input's layout.
  if (!castsInExtract) {
    add(extractShape, ttcore::TileType::get(chain.indexElementType, tileShape),
        extractLayout);
  }

  return plans;
}

} // namespace mlir::tt::d2m
