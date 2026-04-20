// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/DstRegisterAnalysis.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/ErrorHandling.h"

#include <numeric>

namespace mlir::tt::d2m::utils {

namespace {

enum class DstExecutionClass { FPU, SFPU };

static DstExecutionClass classifyComputeOp(Operation *op) {
  if (mlir::isa<TileMatmulOp, TileReduceMaxOp, TileReduceSumOp,
                TileReduceMeanOp>(op)) {
    return DstExecutionClass::FPU;
  }

  if (mlir::isa<TileAddOp, TileSubOp, TileMulOp>(op)) {
    TT_assertv(op->getNumOperands() == 2u,
               "expected binary op for tile add/sub/mul");
    Type lhsType = op->getOperand(0).getType();
    Type rhsType = op->getOperand(1).getType();
    if (ttcore::getDataType(lhsType) == ttcore::DataType::Float32 ||
        ttcore::getDataType(rhsType) == ttcore::DataType::Float32) {
      return DstExecutionClass::SFPU;
    }
    if (mlir::isa<ttcore::TileType>(rhsType)) {
      return DstExecutionClass::FPU;
    }
    return DstExecutionClass::SFPU;
  }

  return DstExecutionClass::SFPU;
}

static DstExecutionClass classifyLinalgExecutionClass(linalg::GenericOp op) {
  bool sawComputeOp = false;
  DstExecutionClass execClass = DstExecutionClass::FPU;

  op.getRegion().walk([&](Operation *nestedOp) {
    if (auto computeOp =
            mlir::dyn_cast<OperandLoadStoreRegisterOpInterface>(nestedOp)) {
      (void)computeOp;
      sawComputeOp = true;
      if (classifyComputeOp(nestedOp) == DstExecutionClass::SFPU) {
        execClass = DstExecutionClass::SFPU;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (!sawComputeOp) {
    return DstExecutionClass::SFPU;
  }
  return execClass;
}

static std::optional<int64_t>
getMaxDstTilesForLinalgOp(linalg::GenericOp op,
                          unsigned maxDstPhysicalSizeTiles) {
  TT_assertv(op.getOutputs().size() == 1u,
             "expected exactly one linalg.generic output");
  auto outputShapedType =
      mlir::dyn_cast<ShapedType>(op.getOutputs().front().getType());
  if (!outputShapedType) {
    return std::nullopt;
  }

  auto tileType =
      mlir::dyn_cast<ttcore::TileType>(outputShapedType.getElementType());
  if (!tileType) {
    return std::nullopt;
  }

  unsigned dstLogicalSizeTiles =
      ttcore::getOpChipDescAttr(op).getDstLogicalSizeTiles(
          tileType.getElementType());
  if (maxDstPhysicalSizeTiles > 0) {
    dstLogicalSizeTiles =
        std::min(dstLogicalSizeTiles, maxDstPhysicalSizeTiles);
  }
  int64_t maxDstTiles =
      classifyLinalgExecutionClass(op) == DstExecutionClass::FPU
          ? static_cast<int64_t>(dstLogicalSizeTiles)
          : static_cast<int64_t>(dstLogicalSizeTiles) / 2;
  return maxDstTiles;
}

static int64_t getLargestLegalChunkSize(int64_t shardSizeTiles,
                                        int64_t maxDstTiles) {
  TT_assertv(maxDstTiles > 0, "expected positive DST tile capacity");
  TT_assertv(shardSizeTiles > 0, "expected positive shard size");
  // Handle single tile shards as special case
  if (shardSizeTiles == 1) {
    return 1;
  }

  int64_t largestCandidate = std::min(maxDstTiles, shardSizeTiles / 2);
  for (int64_t numTilesPerFlip = largestCandidate; numTilesPerFlip >= 1;
       --numTilesPerFlip) {
    if ((shardSizeTiles % numTilesPerFlip) != 0) {
      continue;
    }
    return numTilesPerFlip;
  }
  llvm_unreachable("expected to find a legal chunk size");
}

static std::optional<int64_t>
getLargestCommonNumOuterLoopIters(ArrayRef<int64_t> numDstFlipsPerOp) {
  if (numDstFlipsPerOp.empty()) {
    return std::nullopt;
  }

  int64_t maxCandidate = numDstFlipsPerOp.front() / 2;
  int64_t gcdNumDstFlips = numDstFlipsPerOp.front();
  for (int64_t numDstFlips : numDstFlipsPerOp.drop_front()) {
    // Enforce num_dst_flips >= 2, i.e.
    // num_outer_loop_iters <= num_dst_flips / 2.
    maxCandidate = std::min(maxCandidate, numDstFlips / 2);
    gcdNumDstFlips = std::gcd(gcdNumDstFlips, numDstFlips);
  }
  TT_assertv(maxCandidate >= 1, "maxCandidate must be >= 1");

  // Pick the largest legal factor so the common outer loop iteration count is
  // maximized.
  for (int64_t factor :
       llvm::reverse(ttmlir::utils::getFactors(gcdNumDstFlips))) {
    if (factor <= maxCandidate) {
      return factor;
    }
  }
  llvm_unreachable("failed to find a common outer loop factor");
}

struct PendingDSTPackingResult {
  Value outputValue;
  int64_t numTilesPerFlip = 0;
  int64_t numDstFlips = 0;
};

// Fallback chunking strategy for regions whose outputs have heterogeneous
// shapes (e.g. elementwise fused with a reduction), where the standard
// numDstFlips-GCD-based factorization collapses to a single outer iteration.
//
// Strategy: find a factor R of the "row" dimension (currently defined as
// outputShape[0]) that is common to every op in the region. Each op's shard
// is split into numOuterIters = rowDim / R chunks along that row, giving
// matching Phase 1 scf.for loop counts across ops so that SpillAndScratch
// can later fuse the outer loops.
//
// TODO(ckaravasilisTT): Derive the shared parallel dim from the affine indexing
// maps.
static std::optional<DSTPackingRegionInfo>
trySharedParallelChunking(d2m::GenericOp /*generic*/,
                          ArrayRef<linalg::GenericOp> linalgOps,
                          unsigned maxDstPhysicalSizeTiles) {
  struct OpInfo {
    Value outputValue;
    int64_t rowDim = 0;
    int64_t colProduct = 0;
    int64_t shardSizeTiles = 0;
    int64_t maxDstTiles = 0;
  };
  SmallVector<OpInfo> opInfos;
  int64_t commonRowDim = -1;

  for (linalg::GenericOp linalgOp : linalgOps) {
    Value outputValue = linalgOp.getOutputs().front();
    auto outputType = mlir::dyn_cast<ShapedType>(outputValue.getType());
    if (!outputType || !outputType.hasStaticShape() ||
        outputType.getRank() < 2) {
      return std::nullopt;
    }
    ArrayRef<int64_t> shape = outputType.getShape();
    int64_t rowDim = shape[0];
    int64_t colProduct = 1;
    for (size_t i = 1; i < shape.size(); i++) {
      colProduct *= shape[i];
    }
    int64_t shardSizeTiles = rowDim * colProduct;
    if (shardSizeTiles <= 1) {
      return std::nullopt;
    }

    std::optional<int64_t> maxDst =
        getMaxDstTilesForLinalgOp(linalgOp, maxDstPhysicalSizeTiles);
    if (!maxDst) {
      return std::nullopt;
    }

    if (commonRowDim == -1) {
      commonRowDim = rowDim;
    }
    if (rowDim != commonRowDim) {
      return std::nullopt;
    }

    opInfos.push_back(
        {outputValue, rowDim, colProduct, shardSizeTiles, *maxDst});
  }

  if (opInfos.size() < 2 || commonRowDim < 2) {
    return std::nullopt;
  }

  // Try R values from largest valid factor down. R must be a proper factor
  // (< commonRowDim) and >= 2 so that per-iteration shards remain multi-tile.
  for (int64_t R : llvm::reverse(ttmlir::utils::getFactors(commonRowDim))) {
    if (R >= commonRowDim || R < 2) {
      continue;
    }

    int64_t numOuterIters = commonRowDim / R;
    DSTPackingRegionInfo results;
    bool allValid = true;

    for (const OpInfo &info : opInfos) {
      int64_t perIterShard = info.shardSizeTiles / numOuterIters;
      if (perIterShard < 2) {
        allValid = false;
        break;
      }

      // Clamp numTilesPerFlip to colProduct so Phase 2 produces row tile
      // size = 1, ensuring all ops share the same scratch_space_loop step.
      int64_t maxFlip =
          getLargestLegalChunkSize(perIterShard, info.maxDstTiles);
      int64_t numTilesPerFlip = std::min(maxFlip, info.colProduct);
      int64_t numDstFlips = perIterShard / numTilesPerFlip;

      if (numDstFlips < 2) {
        allValid = false;
        break;
      }

      results.perResult.try_emplace(
          info.outputValue,
          DSTPackingPerResultInfo{numDstFlips, numTilesPerFlip, perIterShard});
    }

    if (allValid) {
      results.numOuterLoopIters = numOuterIters;
      // Use the first op's per-iteration shard as the region-level stat.
      // Per-result numTilesPerResult is the authoritative per-op value.
      results.numTilesPerResult =
          opInfos.front().shardSizeTiles / numOuterIters;
      return results;
    }
  }

  return std::nullopt;
}

static std::optional<DSTPackingRegionInfo>
computeDSTPackingForRegion(d2m::GenericOp generic,
                           ArrayRef<linalg::GenericOp> linalgOps,
                           unsigned maxDstPhysicalSizeTiles) {
  DSTPackingRegionInfo results;
  SmallVector<PendingDSTPackingResult> pendingResults;
  SmallVector<int64_t> numDstFlipsPerOp;
  SmallVector<Value> singleTileOutputValues;
  bool sawMultiTileShard = false;

  for (linalg::GenericOp linalgOp : linalgOps) {
    if (linalgOp.getOutputs().size() != 1u) {
      linalgOp.emitOpError("expected exactly one output");
      return std::nullopt;
    }

    Value outputValue = linalgOp.getOutputs().front();
    auto outputShapedType = mlir::dyn_cast<ShapedType>(outputValue.getType());
    if (!outputShapedType || !outputShapedType.hasStaticShape()) {
      linalgOp.emitOpError(
          "expected static shaped output to compute shard size");
      return std::nullopt;
    }
    int64_t shardSizeTiles =
        ttmlir::utils::volume<int64_t>(outputShapedType.getShape());

    if (shardSizeTiles == 1) {
      singleTileOutputValues.push_back(outputValue);
      continue;
    }
    sawMultiTileShard = true;

    std::optional<int64_t> maxDstTiles =
        getMaxDstTilesForLinalgOp(linalgOp, maxDstPhysicalSizeTiles);
    if (!maxDstTiles) {
      linalgOp.emitOpError("failed to compute max DST tile capacity");
      return std::nullopt;
    }

    int64_t numTilesPerFlip =
        getLargestLegalChunkSize(shardSizeTiles, *maxDstTiles);

    int64_t numDstFlips = shardSizeTiles / numTilesPerFlip;
    TT_assertv(numDstFlips > 0, "expected positive DST flip count");
    TT_assertv(numDstFlips >= 2,
               "expected num_dst_flips >= 2 for shardSizeTiles={0} and "
               "numTilesPerFlip={1}",
               shardSizeTiles, numTilesPerFlip);

    pendingResults.push_back(
        PendingDSTPackingResult{outputValue, numTilesPerFlip, numDstFlips});
    numDstFlipsPerOp.push_back(numDstFlips);
  }

  // handle single tile shard special case early
  if (!singleTileOutputValues.empty() && !sawMultiTileShard) {
    results.numTilesPerResult = 1;
    results.numOuterLoopIters = 1;
    for (Value outputValue : singleTileOutputValues) {
      // Fused generics may have multiple linalg ops writing to the same output
      // buffer (e.g., intermediate reuse after elementwise fusion). Skip
      // duplicates since the packing info is identical for the same Value.
      results.perResult.try_emplace(
          outputValue, DSTPackingPerResultInfo{/*numDstFlips=*/1,
                                               /*numTilesPerFlip=*/1,
                                               /*numTilesPerResult=*/1});
    }
    return results;
  }

  if (!singleTileOutputValues.empty()) {
    generic.emitOpError(
        "expected all linalg.generic outputs in a region to have shard size 1");
    return std::nullopt;
  }

  if (pendingResults.empty()) {
    return std::nullopt;
  }

  std::optional<int64_t> commonNumOuterLoopIters =
      getLargestCommonNumOuterLoopIters(numDstFlipsPerOp);
  TT_assertv(commonNumOuterLoopIters.has_value(),
             "expected a common num_outer_loop_iters when pending dst packing "
             "results are non-empty");

  // When the standard factorization collapses to a single outer iteration and
  // the region has ops with heterogeneous output shard sizes (e.g. eltwise
  // fused with a reduction), fall back to chunking along a shared parallel
  // dimension so that Phase 1 produces matching outer loops across ops.
  if (*commonNumOuterLoopIters <= 1 && pendingResults.size() >= 2) {
    int64_t firstShard = pendingResults.front().numDstFlips *
                         pendingResults.front().numTilesPerFlip;
    bool heterogeneous = llvm::any_of(pendingResults, [&](const auto &pr) {
      return (pr.numDstFlips * pr.numTilesPerFlip) != firstShard;
    });
    if (heterogeneous) {
      if (auto sharedResult = trySharedParallelChunking(
              generic, linalgOps, maxDstPhysicalSizeTiles)) {
        return *sharedResult;
      }
    }
  }

  std::optional<int64_t> regionNumTilesPerResult;
  for (const PendingDSTPackingResult &pending : pendingResults) {
    int64_t numDstFlips = pending.numDstFlips / *commonNumOuterLoopIters;
    TT_assertv((pending.numDstFlips % *commonNumOuterLoopIters) == 0,
               "expected common num_outer_loop_iters={0} to evenly divide "
               "numDstFlips={1}",
               *commonNumOuterLoopIters, pending.numDstFlips);
    TT_assertv(numDstFlips >= 2,
               "expected per-result num_dst_flips >= 2 after applying "
               "common num_outer_loop_iters");

    int64_t numTilesPerResult = numDstFlips * pending.numTilesPerFlip;
    if (!regionNumTilesPerResult) {
      regionNumTilesPerResult = numTilesPerResult;
    }

    // Fused generics may have multiple linalg ops writing to the same output
    // buffer with potentially different DST constraints. Keep the most
    // restrictive (smallest numTilesPerFlip) to avoid DST overflow for ops
    // that need more DST slots per tile (e.g., clamp uses binary_max +
    // binary_min, consuming 2 DST slots per tile vs 1 for a plain SFPU op).
    auto [perResultIt, inserted] = results.perResult.try_emplace(
        pending.outputValue,
        DSTPackingPerResultInfo{numDstFlips, pending.numTilesPerFlip,
                                numTilesPerResult});
    if (!inserted &&
        pending.numTilesPerFlip < perResultIt->second.numTilesPerFlip) {
      perResultIt->second = DSTPackingPerResultInfo{
          numDstFlips, pending.numTilesPerFlip, numTilesPerResult};
    }
  }

  TT_assertv(regionNumTilesPerResult.has_value(),
             "expected num tiles per result when pending results are "
             "non-empty");
  // Region-level numTilesPerResult is kept as the first op's value for
  // back-compat with consumers that read it. Per-op phase-1 capacity is
  // driven by DSTPackingPerResultInfo::numTilesPerResult.
  results.numTilesPerResult = *regionNumTilesPerResult;
  results.numOuterLoopIters = *commonNumOuterLoopIters;

  return results;
}

} // namespace

const DSTPackingRegionInfo *DSTPackingInfo::lookup(Region *region) const {
  auto it = perRegion.find(region);
  if (it == perRegion.end()) {
    return nullptr;
  }
  return &it->second;
}

DstRegisterAnalysis::DstRegisterAnalysis(Operation *op,
                                         unsigned maxDstPhysicalSizeTiles) {
  op->walk([&](d2m::GenericOp generic) {
    if (!generic.isUnifiedForm()) {
      generic.emitOpError("expected unified form for DST packing analysis");
      return;
    }

    DSTPackingInfo packingInfo;

    llvm::DenseMap<Region *, SmallVector<linalg::GenericOp>>
        linalgOpsByParentRegion;
    generic.getRegion(0).walk([&](linalg::GenericOp op) {
      Region *parentRegion = op->getParentRegion();
      TT_assertv(parentRegion != nullptr,
                 "expected linalg.generic to have a parent region");
      linalgOpsByParentRegion[parentRegion].push_back(op);
    });

    for (const auto &[parentRegion, linalgOps] : linalgOpsByParentRegion) {
      std::optional<DSTPackingRegionInfo> regionPackingInfo =
          computeDSTPackingForRegion(generic, linalgOps,
                                     maxDstPhysicalSizeTiles);
      if (!regionPackingInfo) {
        continue;
      }
      auto [it, inserted] =
          packingInfo.perRegion.try_emplace(parentRegion, *regionPackingInfo);
      TT_assertv(inserted, "expected unique parent region "
                           "entry in dst packing analysis");
    }

    if (!packingInfo.empty()) {
      auto [it, inserted] = packingInfoMap.try_emplace(generic.getOperation(),
                                                       std::move(packingInfo));
      TT_assertv(inserted, "expected unique dst packing entry per d2m.generic");
    }
  });
}

const DSTPackingInfo *
DstRegisterAnalysis::lookup(d2m::GenericOp generic) const {
  auto it = packingInfoMap.find(generic.getOperation());
  if (it == packingInfoMap.end()) {
    return nullptr;
  }
  return &it->second;
}

} // namespace mlir::tt::d2m::utils
