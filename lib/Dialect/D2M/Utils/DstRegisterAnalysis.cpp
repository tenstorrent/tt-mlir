// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <numeric>

namespace mlir::tt::d2m::utils {

namespace {

enum class DstExecutionClass { FPU, SFPU };

static scf::ForOp getImmediateParentBlockingLoop(linalg::GenericOp op) {
  Operation *parentOp = op->getParentOp();
  if (parentOp == nullptr) {
    return nullptr;
  }

  if (auto parentScfFor = mlir::dyn_cast<scf::ForOp>(parentOp)) {
    if (parentScfFor->hasAttr("d2m.blocking_loop")) {
      return parentScfFor;
    }
  }
  return nullptr;
}

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

  Type elementType = outputShapedType.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    elementType = tileType.getElementType();
  }

  const bool isFp32 =
      ttcore::getNumberOfBits(ttcore::elementTypeToDataType(elementType)) == 32;

  int64_t maxDstTiles =
      classifyLinalgExecutionClass(op) == DstExecutionClass::FPU ? 8 : 4;
  if (isFp32) {
    maxDstTiles /= 2;
  }
  return maxDstTiles;
}

static std::optional<int64_t> getLargestLegalChunkSize(int64_t shardSizeTiles,
                                                       int64_t maxDstTiles) {
  int64_t largestCandidate = std::min(maxDstTiles, shardSizeTiles / 2);
  for (int64_t numTilesPerFlip = largestCandidate; numTilesPerFlip >= 1;
       --numTilesPerFlip) {
    if ((2 * numTilesPerFlip) > shardSizeTiles) {
      continue;
    }
    if ((shardSizeTiles % numTilesPerFlip) != 0) {
      continue;
    }
    return numTilesPerFlip;
  }
  return std::nullopt;
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
  if (maxCandidate < 1) {
    return std::nullopt;
  }

  // Pick the largest factor of gcd(num_dst_flips_per_op) that still satisfies
  // num_outer_loop_iters <= min(num_dst_flips / 2) across ops.
  for (int64_t factor :
       llvm::reverse(ttmlir::utils::getFactors(gcdNumDstFlips))) {
    if (factor <= maxCandidate) {
      return factor;
    }
  }
  return std::nullopt;
}

struct PendingDSTPackingResult {
  Value outputValue;
  int64_t numTilesPerFlip = 0;
  int64_t numDstFlips = 0;
};

} // namespace

DSTPackingResultsMap analyzeGenericForDSTPacking(d2m::GenericOp generic) {
  DSTPackingResultsMap results;
  SmallVector<PendingDSTPackingResult> pendingResults;
  SmallVector<int64_t> numDstFlipsPerOp;

  if (!generic.isUnifiedForm()) {
    generic.emitOpError("expected unified form for DST packing analysis");
    return DSTPackingResultsMap();
  }
  Region &unifiedRegion = generic.getRegion(0);

  SmallVector<linalg::GenericOp> linalgOps;
  unifiedRegion.walk([&](linalg::GenericOp op) { linalgOps.push_back(op); });

  scf::ForOp commonImmediateParentBlockingLoop = nullptr;
  for (linalg::GenericOp linalgOp : linalgOps) {
    scf::ForOp immediateParentBlockingLoop =
        getImmediateParentBlockingLoop(linalgOp);
    if (immediateParentBlockingLoop == nullptr) {
      linalgOp.emitOpError(
          "expected immediate parent to be an scf.for with d2m.blocking_loop");
      return DSTPackingResultsMap();
    }

    if (commonImmediateParentBlockingLoop == nullptr) {
      commonImmediateParentBlockingLoop = immediateParentBlockingLoop;
    } else if (commonImmediateParentBlockingLoop !=
               immediateParentBlockingLoop) {
      linalgOp.emitOpError(
          "expected all linalg.generic ops to have the same immediate parent "
          "scf.for blocking loop");
      return DSTPackingResultsMap();
    }

    if (linalgOp.getOutputs().size() != 1u) {
      linalgOp.emitOpError("expected exactly one output");
      return DSTPackingResultsMap();
    }

    Value outputValue = linalgOp.getOutputs().front();
    auto outputShapedType = mlir::dyn_cast<ShapedType>(outputValue.getType());
    if (!outputShapedType || !outputShapedType.hasStaticShape()) {
      linalgOp.emitOpError(
          "expected static shaped output to compute shard size");
      return DSTPackingResultsMap();
    }
    int64_t shardSizeTiles =
        ttmlir::utils::volume<int64_t>(outputShapedType.getShape());

    std::optional<int64_t> maxDstTiles = getMaxDstTilesForLinalgOp(linalgOp);
    if (!maxDstTiles) {
      linalgOp.emitOpError("failed to compute max DST tile capacity");
      return DSTPackingResultsMap();
    }

    std::optional<int64_t> numTilesPerFlip =
        getLargestLegalChunkSize(shardSizeTiles, *maxDstTiles);
    if (!numTilesPerFlip) {
      linalgOp.emitOpError("failed to find legal tiles per DST flip");
      return DSTPackingResultsMap();
    }

    int64_t numDstFlips = shardSizeTiles / *numTilesPerFlip;
    if (numDstFlips <= 0) {
      linalgOp.emitOpError("expected positive DST flip count");
      return DSTPackingResultsMap();
    }
    if (numDstFlips < 2) {
      linalgOp.emitOpError("failed to satisfy num_dst_flips >= 2");
      return DSTPackingResultsMap();
    }

    pendingResults.push_back(
        PendingDSTPackingResult{outputValue, *numTilesPerFlip, numDstFlips});
    numDstFlipsPerOp.push_back(numDstFlips);
  }

  if (pendingResults.empty()) {
    return DSTPackingResultsMap();
  }

  std::optional<int64_t> commonNumOuterLoopIters =
      getLargestCommonNumOuterLoopIters(numDstFlipsPerOp);
  if (!commonNumOuterLoopIters) {
    generic.emitOpError(
        "failed to infer common num_outer_loop_iters with num_dst_flips >= 2 "
        "for all "
        "linalg.generic ops");
    return DSTPackingResultsMap();
  }

  for (const PendingDSTPackingResult &pending : pendingResults) {
    int64_t numDstFlips = pending.numDstFlips / *commonNumOuterLoopIters;
    if ((pending.numDstFlips % *commonNumOuterLoopIters) != 0 ||
        numDstFlips < 2) {
      generic.emitOpError("failed to satisfy common num_outer_loop_iters and "
                          "num_dst_flips >= 2");
      return DSTPackingResultsMap();
    }
    if (!results
             .try_emplace(pending.outputValue,
                          DSTPackingInfo{pending.numTilesPerFlip, numDstFlips,
                                         *commonNumOuterLoopIters})
             .second) {
      generic.emitOpError("expected unique linalg.generic output values");
      return DSTPackingResultsMap();
    }
  }

  return results;
}

} // namespace mlir::tt::d2m::utils
