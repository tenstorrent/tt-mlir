// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/DstRegisterAnalysis.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/ErrorHandling.h"

#include <numeric>

namespace mlir::tt::d2m::utils {

namespace {

enum class DstExecutionClass { FPU, SFPU };

static DstExecutionClass classifyComputeOp(Operation *op) {
  if (mlir::isa<TileMatmulOp, TileReduceMaxOp, TileReduceSumOp>(op)) {
    return DstExecutionClass::FPU;
  }

  if (mlir::isa<TileAddOp, TileSubOp, TileMulOp>(op)) {
    TT_assertv(op->getNumOperands() == 2u,
               "expected binary op for tile add/sub/mul");
    Type rhsType = op->getOperand(1).getType();
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

static std::optional<int64_t> getMaxDstTilesForLinalgOp(linalg::GenericOp op) {
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
  int64_t maxDstTiles =
      classifyLinalgExecutionClass(op) == DstExecutionClass::FPU
          ? static_cast<int64_t>(dstLogicalSizeTiles)
          : static_cast<int64_t>(dstLogicalSizeTiles) / 2;
  return maxDstTiles;
}

static std::optional<int64_t> getLargestLegalChunkSize(int64_t shardSizeTiles,
                                                       int64_t maxDstTiles) {
  TT_assertv(maxDstTiles > 0, "expected positive DST tile capacity");
  if (shardSizeTiles < 2) {
    return std::nullopt;
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

static std::optional<DSTPackingRegionInfo>
computeDSTPackingForRegion(d2m::GenericOp generic,
                           ArrayRef<linalg::GenericOp> linalgOps) {
  DSTPackingRegionInfo results;
  SmallVector<PendingDSTPackingResult> pendingResults;
  SmallVector<int64_t> numDstFlipsPerOp;

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

    std::optional<int64_t> maxDstTiles = getMaxDstTilesForLinalgOp(linalgOp);
    if (!maxDstTiles) {
      linalgOp.emitOpError("failed to compute max DST tile capacity");
      return std::nullopt;
    }

    std::optional<int64_t> numTilesPerFlip =
        getLargestLegalChunkSize(shardSizeTiles, *maxDstTiles);
    if (!numTilesPerFlip) {
      return std::nullopt;
    }

    int64_t numDstFlips = shardSizeTiles / *numTilesPerFlip;
    TT_assertv(numDstFlips > 0, "expected positive DST flip count");
    TT_assertv(numDstFlips >= 2,
               "expected num_dst_flips >= 2 for shardSizeTiles={0} and "
               "numTilesPerFlip={1}",
               shardSizeTiles, *numTilesPerFlip);

    pendingResults.push_back(
        PendingDSTPackingResult{outputValue, *numTilesPerFlip, numDstFlips});
    numDstFlipsPerOp.push_back(numDstFlips);
  }

  if (pendingResults.empty()) {
    return std::nullopt;
  }

  std::optional<int64_t> commonNumOuterLoopIters =
      getLargestCommonNumOuterLoopIters(numDstFlipsPerOp);
  TT_assertv(commonNumOuterLoopIters.has_value(),
             "expected a common num_outer_loop_iters when pending dst packing "
             "results are non-empty");

  std::optional<int64_t> commonNumTilesPerResult;
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
    if (!commonNumTilesPerResult) {
      commonNumTilesPerResult = numTilesPerResult;
    }
    if (*commonNumTilesPerResult != numTilesPerResult) {
      generic.emitOpError(
          "expected identical num tiles per result for all linalg.generic "
          "outputs");
      return std::nullopt;
    }

    if (!results.perResult
             .try_emplace(
                 pending.outputValue,
                 DSTPackingPerResultInfo{numDstFlips, pending.numTilesPerFlip})
             .second) {
      generic.emitOpError("expected unique linalg.generic output values");
      return std::nullopt;
    }
  }

  TT_assertv(commonNumTilesPerResult.has_value(),
             "expected num tiles per result when pending results are "
             "non-empty");
  results.numTilesPerResult = *commonNumTilesPerResult;
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

DstRegisterAnalysis::DstRegisterAnalysis(Operation *op) {
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
          computeDSTPackingForRegion(generic, linalgOps);
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
