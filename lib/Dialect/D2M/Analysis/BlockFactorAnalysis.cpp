// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"

#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"

#include <functional>

namespace mlir::tt::d2m {

using namespace allocation;

namespace {

//===----------------------------------------------------------------------===//
// Block factor analysis logic
//===----------------------------------------------------------------------===//

// Shape classes used by the allocator's Automatic reblocking policy, as
// distinct from the explicit MIN and MAX policies.
enum class AutoShapeClass {
  SingleReduction,
  AllParallelEltwise,
  AllParallelLayoutConversion
};

struct AutoSearchConfig {
  AutoShapeClass shapeClass;
  SmallVector<std::size_t> candidateDims;
};

struct CandidateScore {
  SmallVector<int64_t> dimScales;
  SmallVector<int64_t> blockFactors;
  uint64_t blockingVolume = 1;
  uint64_t cbBytes = 0;
};

// Beam-search state before a full candidate legality and CB cost evaluation.
struct PartialCandidate {
  SmallVector<int64_t> dimScales;
  uint64_t blockingVolume = 1;
  uint64_t moderationPenalty = 0;
};

constexpr std::size_t kEltwiseBeamWidth = 64;
constexpr std::size_t kEltwiseScalesPerDim = 5;

static llvm::BitVector getDimMask(std::size_t rank,
                                  ArrayRef<std::size_t> dims) {
  llvm::BitVector mask(rank, false);
  for (std::size_t dim : dims) {
    mask[dim] = true;
  }
  return mask;
}

static SmallVector<std::size_t>
getReductionDims(ArrayRef<ttcore::IteratorType> iteratorTypes) {
  SmallVector<std::size_t> reductionDims;
  for (auto [dim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType == ttcore::IteratorType::Reduction) {
      reductionDims.push_back(dim);
    }
  }
  return reductionDims;
}

static SmallVector<std::size_t>
getAllParallelCandidateDims(GenericOp genericOp,
                            ArrayRef<int64_t> shardFactors) {
  SmallVector<std::size_t> candidateDims;
  const llvm::BitVector participationMask = getParticipatingDimMask(genericOp);
  for (std::size_t dim = 0; dim < shardFactors.size(); ++dim) {
    if (participationMask[dim] && shardFactors[dim] > 1) {
      candidateDims.push_back(dim);
    }
  }
  return candidateDims;
}

static bool isSingleReductionPanelDim(GenericOp genericOp, std::size_t dim) {
  unsigned numOperandsUsingDim = 0;
  AffineExpr dimExpr =
      getAffineDimExpr(static_cast<unsigned>(dim), genericOp.getContext());
  for (AffineMap indexingMap : genericOp.getIndexingMapsValue()) {
    if (llvm::is_contained(indexingMap.getResults(), dimExpr)) {
      ++numOperandsUsingDim;
    }
  }

  // Do not reblock the batch dimension.
  return numOperandsUsingDim > 0 &&
         numOperandsUsingDim < genericOp.getInputsAndOutputs().size();
}

static SmallVector<std::size_t> getSingleReductionCandidateDims(
    GenericOp genericOp, ArrayRef<std::size_t> reductionDims,
    ArrayRef<int64_t> shardFactors, bool allowMNReblocking) {
  if (!allowMNReblocking) {
    return llvm::to_vector(reductionDims);
  }

  SmallVector<std::size_t> candidateDims;
  for (std::size_t dim = 0; dim < shardFactors.size(); ++dim) {
    if (shardFactors[dim] > 1 && isSingleReductionPanelDim(genericOp, dim)) {
      candidateDims.push_back(dim);
    }
  }
  return candidateDims;
}

static bool hasTileElementType(Value operand) {
  auto shapedTy = mlir::cast<ShapedType>(operand.getType());
  return mlir::isa<ttcore::TileType>(shapedTy.getElementType());
}

static bool isScalarTileLayoutConversion(GenericOp genericOp) {
  if (genericOp.getInputs().size() != 1 || genericOp.getOutputs().size() != 1) {
    return false;
  }

  const bool inputTiled = hasTileElementType(genericOp.getInputs().front());
  const bool outputTiled = hasTileElementType(genericOp.getOutputs().front());
  if (inputTiled == outputTiled) {
    return false;
  }

  bool hasExpectedRegionOp = false;
  genericOp->walk([&](Operation *op) {
    if ((inputTiled && mlir::isa<TileUntilizeBlockOp>(op)) ||
        (!inputTiled && mlir::isa<TileTilizeBlockOp>(op))) {
      hasExpectedRegionOp = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return hasExpectedRegionOp;
}

static bool canReblockOperandToGrid(Value operand,
                                    ArrayRef<int64_t> newGridShape) {
  auto operandType = mlir::dyn_cast<ShapedType>(operand.getType());
  if (!operandType || !operandType.hasStaticShape()) {
    return false;
  }

  auto layout = ttcore::getDeviceLayout(operandType);
  if (!layout) {
    return false;
  }

  ArrayRef<int64_t> oldGridShape = layout.getGridShape(operandType);
  ArrayRef<int64_t> oldShardShape = layout.getShardShape(operandType);
  if (newGridShape.size() != oldGridShape.size() ||
      oldGridShape.size() != oldShardShape.size()) {
    return false;
  }

  for (auto [idx, gridDim] : llvm::enumerate(newGridShape)) {
    if (gridDim <= 0 ||
        (oldGridShape[idx] * oldShardShape[idx]) % gridDim != 0) {
      return false;
    }
  }

  return true;
}

/// Classify a generic op's iteration shape for auto-policy search.
/// @return the search config or nullopt if auto-blocking is not applicable.
static std::optional<AutoSearchConfig>
classifyAutoSearch(GenericOp genericOp,
                   ArrayRef<ttcore::IteratorType> iteratorTypes,
                   ArrayRef<int64_t> shardFactors, bool allowMNReblocking) {
  if (genericOp.isDMAOnlyForm() || genericOp.isExplicitDatamovementForm() ||
      genericOp.getOutputs().size() != 1) {
    return std::nullopt;
  }

  bool hasTile = false;
  bool hasNonTile = false;
  for (Value operand : genericOp.getInputsAndOutputs()) {
    if (hasTileElementType(operand)) {
      hasTile = true;
    } else {
      hasNonTile = true;
    }
  }
  const bool hasMixedElementTypeKinds = hasTile && hasNonTile;

  SmallVector<std::size_t> reductionDims = getReductionDims(iteratorTypes);
  if (reductionDims.empty()) {
    AutoShapeClass shapeClass = AutoShapeClass::AllParallelEltwise;
    if (hasMixedElementTypeKinds) {
      if (!isScalarTileLayoutConversion(genericOp)) {
        return std::nullopt;
      }
      shapeClass = AutoShapeClass::AllParallelLayoutConversion;
    }

    SmallVector<std::size_t> candidateDims =
        getAllParallelCandidateDims(genericOp, shardFactors);
    if (candidateDims.empty()) {
      return std::nullopt;
    }
    return AutoSearchConfig{shapeClass, std::move(candidateDims)};
  }

  if (reductionDims.size() != 1) {
    return std::nullopt;
  }

  if (hasMixedElementTypeKinds) {
    return std::nullopt;
  }

  SmallVector<std::size_t> candidateDims = getSingleReductionCandidateDims(
      genericOp, reductionDims, shardFactors, allowMNReblocking);
  if (candidateDims.empty()) {
    return std::nullopt;
  }
  return AutoSearchConfig{AutoShapeClass::SingleReduction,
                          std::move(candidateDims)};
}

static bool isLexicographicallyLarger(ArrayRef<int64_t> lhs,
                                      ArrayRef<int64_t> rhs) {
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    if (lhsValue != rhsValue) {
      return lhsValue > rhsValue;
    }
  }
  return false;
}

static SmallVector<int64_t> buildEltwiseScaleSearchOrder(int64_t shardFactor) {
  SmallVector<int64_t> factors = ttmlir::utils::getFactors(shardFactor);
  llvm::SmallDenseMap<int64_t, int64_t, 8> factorRanks;
  factorRanks.reserve(factors.size());
  for (auto [index, factor] : llvm::enumerate(factors)) {
    factorRanks[factor] = index;
  }
  // Visit moderate divisors first.
  const auto midpoint = static_cast<int64_t>(factors.size() - 1) / 2;
  llvm::stable_sort(factors, [&](int64_t lhs, int64_t rhs) {
    if (lhs == rhs) {
      return false;
    }
    const int64_t lhsPenalty = std::abs(factorRanks[lhs] - midpoint);
    const int64_t rhsPenalty = std::abs(factorRanks[rhs] - midpoint);
    if (lhsPenalty != rhsPenalty) {
      return lhsPenalty < rhsPenalty;
    }
    if ((lhs == 1) != (rhs == 1)) {
      return rhs == 1;
    }
    return lhs > rhs;
  });

  SmallVector<int64_t> limited;
  // Always include scale factor 1 so bounded search can leave a dim unchanged.
  limited.reserve(std::min(factors.size(), kEltwiseScalesPerDim));
  for (int64_t factor : factors) {
    if (limited.size() == kEltwiseScalesPerDim) {
      break;
    }
    limited.push_back(factor);
  }
  if (!llvm::is_contained(limited, int64_t{1})) {
    limited.pop_back();
    limited.push_back(1);
  }
  return limited;
}

static bool usesAllParallelScaleSearch(AutoShapeClass shapeClass) {
  return shapeClass == AutoShapeClass::AllParallelEltwise ||
         shapeClass == AutoShapeClass::AllParallelLayoutConversion;
}

static bool isBetterPartialCandidate(const PartialCandidate &lhs,
                                     const PartialCandidate &rhs) {
  if (lhs.moderationPenalty != rhs.moderationPenalty) {
    return lhs.moderationPenalty < rhs.moderationPenalty;
  }
  if (lhs.blockingVolume != rhs.blockingVolume) {
    return lhs.blockingVolume > rhs.blockingVolume;
  }
  return isLexicographicallyLarger(lhs.dimScales, rhs.dimScales);
}

static void trimPartialBeam(SmallVectorImpl<PartialCandidate> &beam) {
  llvm::stable_sort(
      beam, [&](const PartialCandidate &lhs, const PartialCandidate &rhs) {
        return isBetterPartialCandidate(lhs, rhs);
      });
  if (beam.size() > kEltwiseBeamWidth) {
    beam.resize(kEltwiseBeamWidth);
  }
}

// Compare two automatic-reblocking candidates for the same generic op.
//
// Each candidate represents a legal choice of reblocked factors (derived from
// per-dimension scales) together with its estimated impact on CB allocation
// pressure and parallelism. It is used to rank alternative block-factor choices
// for the allocator's Automatic policy.
static bool isBetterCandidate(AutoShapeClass shapeClass,
                              const CandidateScore &lhs,
                              const CandidateScore &rhs) {
  switch (shapeClass) {
  // Single reduction: CB cost is dominant concern, so prefer lower CB bytes.

  // All parallel eltwise: blocking volume is dominant concern because it
  // controls parallelism gain, so prefer higher blocking volume. CB cost is
  // secondary tiebreaker since eltwise operands tend to have uniform and
  // predictable buffer sizes.
  //
  // Layout conversion follows the same ordering: maximize reblocking of
  // participating movement dims while using CB bytes as a tiebreaker.

  // Final tiebreaker: prefer lexicographically larger dim scales.
  case AutoShapeClass::SingleReduction:
    if (lhs.cbBytes != rhs.cbBytes) {
      return lhs.cbBytes < rhs.cbBytes;
    }
    if (lhs.blockingVolume != rhs.blockingVolume) {
      return lhs.blockingVolume > rhs.blockingVolume;
    }
    return isLexicographicallyLarger(lhs.dimScales, rhs.dimScales);
  case AutoShapeClass::AllParallelEltwise:
  case AutoShapeClass::AllParallelLayoutConversion:
    if (lhs.blockingVolume != rhs.blockingVolume) {
      return lhs.blockingVolume > rhs.blockingVolume;
    }
    if (lhs.cbBytes != rhs.cbBytes) {
      return lhs.cbBytes < rhs.cbBytes;
    }
    return isLexicographicallyLarger(lhs.dimScales, rhs.dimScales);
  }

  llvm_unreachable("unknown auto shape class");
}

/// Check legality and estimate cost for a single candidate dim-scale vector.
/// @return the candidate score, or nullopt if the candidate is illegal.
static std::optional<CandidateScore> evaluateCandidate(
    GenericOp genericOp, ArrayRef<std::size_t> candidateDims,
    ArrayRef<AffineMap> indexingMaps, ArrayRef<int64_t> gridExtents,
    ArrayRef<int64_t> shardExtents, ArrayRef<int64_t> shardFactors,
    ArrayRef<int64_t> originalBlockFactors, ArrayRef<int64_t> dimScales,
    ttcore::DeviceAttr device, ttcore::MemorySpaceAttr l1Attr,
    uint32_t numBuffers, bool allowIdentityCandidate = false) {
  SmallVector<int64_t> candidateGridExtents(gridExtents.begin(),
                                            gridExtents.end());
  SmallVector<int64_t> candidateShardExtents(shardExtents.begin(),
                                             shardExtents.end());
  SmallVector<int64_t> candidateBlockFactors(originalBlockFactors.begin(),
                                             originalBlockFactors.end());
  uint64_t blockingVolume = 1;
  bool hasNonTrivialScale = false;

  // Apply the dim scales to the grid, shard, and block factors.
  for (std::size_t dim = 0; dim < dimScales.size(); ++dim) {
    TT_assert(shardFactors[dim] % dimScales[dim] == 0);
    candidateGridExtents[dim] *= dimScales[dim];
    candidateShardExtents[dim] /= dimScales[dim];
    candidateBlockFactors[dim] *= dimScales[dim];
    blockingVolume *= dimScales[dim];
    hasNonTrivialScale |= dimScales[dim] > 1;
  }
  if (!hasNonTrivialScale && !allowIdentityCandidate) {
    return std::nullopt;
  }

  const llvm::BitVector scaledDims =
      hasNonTrivialScale ? getBlockedDimMask(dimScales)
                         : getDimMask(dimScales.size(), candidateDims);
  bool sawAffectedBuffer = false;
  uint64_t totalCBBytes = 0;

  // Iterate over all operands and check if they are affected by the reblocked
  // dims.
  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {

    if (!isIndexingMapBlocked(indexingMaps[operandIndex], scaledDims)) {
      continue;
    }
    sawAffectedBuffer = true;

    auto [operandGridShape, operandShardShape] = getOperandGridAndShardExtents(
        genericOp, operandIndex, candidateGridExtents, candidateShardExtents);

    // Reblocking rebuilds the operand type using the derived grid shape. Reject
    // candidates that cannot evenly repartition the operand's actual layout.
    if (!canReblockOperandToGrid(operand, operandGridShape)) {
      return std::nullopt;
    }

    // Reject candidates that would result in a too small operand shard shape.
    if (ttmlir::utils::volume<int64_t>(operandShardShape) < 4) {
      return std::nullopt;
    }

    auto operandType = mlir::dyn_cast<MemRefType>(operand.getType());
    if (!operandType) {
      return std::nullopt;
    }

    // Get the size of the CB buffer for the candidate.
    const MemRefType bufferType =
        getCBBufferType(operandGridShape, operandShardShape,
                        operandType.getElementType(), l1Attr, numBuffers);
    totalCBBytes += getCBBufferSizeBytes(bufferType, device);
  }

  // Account for annotated fused intermediate allocs inside the region. Their
  // blocking relationship is stored as a d2m.blocking_map attr on the alloc.
  WalkResult walkResult = genericOp->walk([&](memref::AllocOp allocOp) {
    auto blockingMapAttr =
        allocOp->getAttrOfType<mlir::AffineMapAttr>("d2m.blocking_map");
    if (!blockingMapAttr ||
        !isIndexingMapBlocked(blockingMapAttr.getValue(), scaledDims)) {
      return WalkResult::advance();
    }

    sawAffectedBuffer = true;

    const AffineMap canonicalMap =
        canonicalizeBroadcasts(blockingMapAttr.getValue());
    SmallVector<int64_t> intermediateGridShape =
        canonicalMap.compose(candidateGridExtents);
    SmallVector<int64_t> intermediateShardShape =
        canonicalMap.compose(candidateShardExtents);

    if (ttmlir::utils::volume<int64_t>(intermediateShardShape) < 4) {
      return WalkResult::interrupt();
    }

    const MemRefType bufferType =
        getCBBufferType(intermediateGridShape, intermediateShardShape,
                        allocOp.getType().getElementType(), l1Attr, numBuffers);
    totalCBBytes += getCBBufferSizeBytes(bufferType, device);
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return std::nullopt;
  }

  if (!sawAffectedBuffer) {
    return std::nullopt;
  }

  return CandidateScore{
      SmallVector<int64_t>(dimScales.begin(), dimScales.end()),
      std::move(candidateBlockFactors), blockingVolume, totalCBBytes};
}

/// Applies the `min` policy to the block factors which shrinks all
/// non-participating dims.
static SmallVector<int64_t> applyMinPolicy(GenericOp genericOp,
                                           ArrayRef<int64_t> shardFactors) {
  SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();
  const std::size_t rank = genericOp.getNumDims();
  SmallVector<int64_t> dimScales(rank, 1);
  const llvm::BitVector participationMask = getParticipatingDimMask(genericOp);
  for (auto d = participationMask.find_first_unset(); d >= 0;
       d = participationMask.find_next_unset(d)) {
    dimScales[d] = shardFactors[d];
  }

  for (std::size_t d = 0; d < rank; ++d) {
    blockFactors[d] *= dimScales[d];
  }
  TT_ALLOC_DEBUG("applying min policy scales {}, new block factors {}",
                 asSeq(dimScales), asSeq(blockFactors));
  return blockFactors;
}

// The auto policy currently has three cases:
// 1. Single reduction: aggressively reduce the block factor of the reduction
// dim.  M/N widening for matmul is available only through the non-default
// auto-mn policy until issues #8346/#8369 model loop-carried output
// accumulation explicitly.
// 2. All parallel eltwise: aggressively reduce the block factor of all
// participating dims.
// 3. Unsupported: in all other cases, return the original block factors.
static SmallVector<int64_t>
applyAutoPolicy(GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
                ArrayRef<ttcore::IteratorType> iteratorTypes,
                ArrayRef<int64_t> gridExtents, ArrayRef<int64_t> shardExtents,
                ArrayRef<int64_t> shardFactors, ttcore::DeviceAttr device,
                ttcore::MemorySpaceAttr l1Attr, uint32_t numBuffers,
                bool useBoundedEltwiseSearch, bool allowMNReblocking) {
  const SmallVector<int64_t> originalBlockFactors =
      genericOp.getBlockFactorsValue();

  //===------------------------------------------------------------------===//
  // Classify the generic op's iteration shape.
  //===------------------------------------------------------------------===//
  auto config = classifyAutoSearch(genericOp, iteratorTypes, shardFactors,
                                   allowMNReblocking);
  if (!config) {
    return originalBlockFactors;
  }

  //===------------------------------------------------------------------===//
  // Enumerate legal per-dim scale divisors for each dimension.
  //===------------------------------------------------------------------===//
  SmallVector<SmallVector<int64_t>> legalScales;
  legalScales.reserve(config->candidateDims.size());
  uint64_t boundedSearchSpace = 1;
  for (std::size_t dim : config->candidateDims) {
    SmallVector<int64_t> dimScales =
        usesAllParallelScaleSearch(config->shapeClass)
            ? buildEltwiseScaleSearchOrder(shardFactors[dim])
            : ttmlir::utils::getFactors(shardFactors[dim]);
    boundedSearchSpace *= dimScales.size();
    legalScales.push_back(std::move(dimScales));
  }

  //===------------------------------------------------------------------===//
  // Evaluate candidates.
  //===------------------------------------------------------------------===//
  SmallVector<int64_t> currentDimScales(genericOp.getNumDims(), 1);
  std::optional<CandidateScore> bestCandidate;
  if (config->shapeClass == AutoShapeClass::SingleReduction) {
    bestCandidate = evaluateCandidate(
        genericOp, config->candidateDims, indexingMaps, gridExtents,
        shardExtents, shardFactors, originalBlockFactors, currentDimScales,
        device, l1Attr, numBuffers, /*allowIdentityCandidate=*/true);
  }

  // Restrict the large eltwise search space to the top kEltwiseBeamWidth
  // candidates unless the caller explicitly requests exhaustive search.
  if (useBoundedEltwiseSearch &&
      usesAllParallelScaleSearch(config->shapeClass) &&
      boundedSearchSpace > kEltwiseBeamWidth) {
    SmallVector<PartialCandidate> beam;
    beam.push_back(
        PartialCandidate{SmallVector<int64_t>(currentDimScales), 1, 0});

    for (auto [dimIndex, dim] : llvm::enumerate(config->candidateDims)) {
      SmallVector<PartialCandidate> expanded;
      expanded.reserve(beam.size() * legalScales[dimIndex].size());
      for (const PartialCandidate &partial : beam) {
        for (auto [scaleRank, scale] : llvm::enumerate(legalScales[dimIndex])) {
          PartialCandidate candidate = partial;
          candidate.dimScales[dim] = scale;
          candidate.blockingVolume *= scale;
          candidate.moderationPenalty += scaleRank;
          expanded.push_back(std::move(candidate));
        }
      }
      trimPartialBeam(expanded);
      beam = std::move(expanded);
    }

    for (const PartialCandidate &candidate : beam) {
      std::optional<CandidateScore> score = evaluateCandidate(
          genericOp, config->candidateDims, indexingMaps, gridExtents,
          shardExtents, shardFactors, originalBlockFactors, candidate.dimScales,
          device, l1Attr, numBuffers);
      if (score &&
          (!bestCandidate ||
           isBetterCandidate(config->shapeClass, *score, *bestCandidate))) {
        bestCandidate = std::move(score);
      }
    }

    if (!bestCandidate) {
      return originalBlockFactors;
    }

    TT_ALLOC_DEBUG("applying auto policy scales {}, new block factors {}, CB "
                   "bytes {}",
                   asSeq(bestCandidate->dimScales),
                   asSeq(bestCandidate->blockFactors), bestCandidate->cbBytes);
    return bestCandidate->blockFactors;
  }

  std::function<void(std::size_t)> enumerateCandidates =
      [&](std::size_t dimIndex) {
        if (dimIndex == config->candidateDims.size()) {
          // All dimensions have been reblocked, so evaluate the candidate.
          auto candidate = evaluateCandidate(
              genericOp, config->candidateDims, indexingMaps, gridExtents,
              shardExtents, shardFactors, originalBlockFactors,
              currentDimScales, device, l1Attr, numBuffers);
          if (candidate && (!bestCandidate ||
                            isBetterCandidate(config->shapeClass, *candidate,
                                              *bestCandidate))) {
            bestCandidate = std::move(candidate);
          }
          return;
        }

        const std::size_t dim = config->candidateDims[dimIndex];
        // Try all legal scales for the current dimension.
        for (int64_t scale : legalScales[dimIndex]) {
          currentDimScales[dim] = scale;
          // Recurse to the next dimension.
          enumerateCandidates(dimIndex + 1);
        }
        // Backtrack by resetting the current dimension to 1.
        currentDimScales[dim] = 1;
      };

  // Kick off the recursive candidate enumeration.
  enumerateCandidates(0);

  if (!bestCandidate) {
    return originalBlockFactors;
  }

  TT_ALLOC_DEBUG("applying auto policy scales {}, new block factors {}, CB "
                 "bytes {}",
                 asSeq(bestCandidate->dimScales),
                 asSeq(bestCandidate->blockFactors), bestCandidate->cbBytes);
  return bestCandidate->blockFactors;
}

static SmallVector<int64_t> chooseReblockedFactors(
    GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<ttcore::IteratorType> iteratorTypes, ArrayRef<int64_t> gridExtents,
    ArrayRef<int64_t> shardExtents, ttcore::DeviceAttr device,
    BlockFactorAnalysis::BufferSizePolicy policy,
    ttcore::MemorySpaceAttr l1Attr, uint32_t numBuffers) {
  const SmallVector<int64_t> shardFactors = getShardBlockFactors(genericOp);
  switch (policy) {
  case BlockFactorAnalysis::BufferSizePolicy::Max:
    return genericOp.getBlockFactorsValue();
  case BlockFactorAnalysis::BufferSizePolicy::Min:
    return applyMinPolicy(genericOp, shardFactors);
  case BlockFactorAnalysis::BufferSizePolicy::Auto:
    return applyAutoPolicy(genericOp, indexingMaps, iteratorTypes, gridExtents,
                           shardExtents, shardFactors, device, l1Attr,
                           numBuffers, /*useBoundedEltwiseSearch=*/false,
                           /*allowMNReblocking=*/false);
  case BlockFactorAnalysis::BufferSizePolicy::Bounded:
    return applyAutoPolicy(genericOp, indexingMaps, iteratorTypes, gridExtents,
                           shardExtents, shardFactors, device, l1Attr,
                           numBuffers, /*useBoundedEltwiseSearch=*/true,
                           /*allowMNReblocking=*/false);
  case BlockFactorAnalysis::BufferSizePolicy::AutoMN:
    return applyAutoPolicy(genericOp, indexingMaps, iteratorTypes, gridExtents,
                           shardExtents, shardFactors, device, l1Attr,
                           numBuffers, /*useBoundedEltwiseSearch=*/false,
                           /*allowMNReblocking=*/true);
  }

  llvm_unreachable("unknown buffer size policy");
}

} // namespace

//===----------------------------------------------------------------------===//
// BlockFactorAnalysis public interface.
//===----------------------------------------------------------------------===//

BlockFactorAnalysis::BlockFactorAnalysis(Operation *op, const Options &opts) {
  auto *ctx = op->getContext();
  auto l1Attr =
      ttcore::MemorySpaceAttr::get(ctx, ttcore::MemorySpace::DeviceL1);

  // Walk over all generic ops and compute the reblocked factors.
  op->walk([&](GenericOp genericOp) {
    if (genericOp.isExplicitDatamovementForm()) {
      return;
    }

    ttcore::DeviceAttr device = ttcore::lookupDevice(genericOp);

    auto indexingMaps = genericOp.getIndexingMapsValue();
    auto iteratorTypes = genericOp.getIteratorTypesValue();

    auto [gridExtents, shardExtents] = getGridAndShardExtents(genericOp);

    SmallVector<int64_t> reblockedFactors = chooseReblockedFactors(
        genericOp, indexingMaps, iteratorTypes, gridExtents, shardExtents,
        device, opts.policy, l1Attr, opts.numBuffers);

    results[genericOp.getOperation()] = Result{std::move(reblockedFactors)};
  });
}

const BlockFactorAnalysis::Result *
BlockFactorAnalysis::lookup(GenericOp genericOp) const {
  auto it = results.find(genericOp.getOperation());
  if (it == results.end()) {
    return nullptr;
  }
  return &it->second;
}

} // namespace mlir::tt::d2m
