// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"

#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <functional>

namespace mlir::tt::d2m {

using namespace allocation;

namespace {

//===----------------------------------------------------------------------===//
// Block factor analysis logic
//===----------------------------------------------------------------------===//

// Shape classes used by the allocator's Automatic reblocking policy, as
// distinct from the explicit MIN and MAX policies.
enum class AutoShapeClass { SingleReduction, AllParallelEltwise };

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

static llvm::BitVector getDimMask(std::size_t rank,
                                  ArrayRef<std::size_t> dims) {
  llvm::BitVector mask(rank, false);
  for (std::size_t dim : dims) {
    mask[dim] = true;
  }
  return mask;
}

// @return true if the operand needs a dedicated CB by existing rules.
static bool operandNeedsDedicatedCBByExistingRules(GenericOp genericOp,
                                                   uint32_t operandIndex) {
  Value operand = genericOp.getInputsAndOutputs()[operandIndex];

  if (hasNonTrivialView(operand)) {
    return true;
  }

  auto operandType = mlir::dyn_cast<MemRefType>(operand.getType());
  if (!operandType) {
    return false;
  }
  if (ttcore::getMemorySpace(operandType) == ttcore::MemorySpace::DeviceDRAM) {
    return true;
  }

  const AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);
  const auto broadcastDims = indexingMap.getBroadcastDims();
  const auto iteratorTypes = genericOp.getIteratorTypesValue();
  for (std::size_t resultIndex = 0; resultIndex < indexingMap.getNumResults();
       ++resultIndex) {
    if (llvm::is_contained(broadcastDims, resultIndex)) {
      return true;
    }
    if (iteratorTypes[indexingMap.getDimPosition(resultIndex)] ==
        ttcore::IteratorType::Reduction) {
      return true;
    }
  }

  return false;
}

/// Classify a generic op's iteration shape for auto-policy search.
/// @return the search config or nullopt if auto-blocking is not applicable.
static std::optional<AutoSearchConfig>
classifyAutoSearch(GenericOp genericOp,
                   ArrayRef<ttcore::IteratorType> iteratorTypes,
                   ArrayRef<int64_t> shardFactors) {
  if (genericOp.isDMAOnlyForm() || genericOp.isExplicitDatamovementForm() ||
      genericOp.getOutputs().size() != 1) {
    return std::nullopt;
  }

  // Reject generics with mixed element type kinds (e.g. tilize/untilize which
  // have one scalar operand and one tile operand).  These are layout-conversion
  // ops, not true eltwise compute.
  {
    bool hasTile = false, hasNonTile = false;
    for (Value operand : genericOp.getInputsAndOutputs()) {
      auto shapedTy = mlir::cast<ShapedType>(operand.getType());
      if (mlir::isa<ttcore::TileType>(shapedTy.getElementType())) {
        hasTile = true;
      } else {
        hasNonTile = true;
      }
    }
    if (hasTile && hasNonTile) {
      return std::nullopt;
    }
  }

  SmallVector<std::size_t> reductionDims;
  for (auto [dim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType == ttcore::IteratorType::Reduction) {
      reductionDims.push_back(dim);
    }
  }

  if (reductionDims.size() == 1) {
    return AutoSearchConfig{AutoShapeClass::SingleReduction,
                            std::move(reductionDims)};
  }
  if (!reductionDims.empty()) {
    return std::nullopt;
  }
  // At this point, the generic op is an all-parallel operation.

  // Candidate dim filter: all participating dims with a shard factor > 1.
  SmallVector<std::size_t> candidateDims;
  const llvm::BitVector participationMask = getParticipatingDimMask(genericOp);
  for (std::size_t dim = 0; dim < shardFactors.size(); ++dim) {
    if (participationMask[dim] && shardFactors[dim] > 1) {
      candidateDims.push_back(dim);
    }
  }
  if (candidateDims.empty()) {
    return std::nullopt;
  }

  return AutoSearchConfig{AutoShapeClass::AllParallelEltwise,
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
    uint32_t numBuffers, bool allowAliasedEltwiseBlocking,
    bool allowIdentityCandidate = false) {
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

    // Reject candidates that would result in a too small operand shard shape.
    if (ttmlir::utils::volume<int64_t>(operandShardShape) < 4) {
      return std::nullopt;
    }

    // Reject candidates that would result in a dedicated CB being needed for
    // output operands if aliased eltwise blocking is not allowed.
    if (genericOp.isOutputOperandIdx(operandIndex) &&
        !operandNeedsDedicatedCBByExistingRules(genericOp, operandIndex) &&
        !allowAliasedEltwiseBlocking) {
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

  // Walk over annotated allocs inside the region with similar checks.
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
// dim.
// 2. All parallel eltwise: aggressively reduce the block factor of all
// participating dims.
// 3. Unsupported: in all other cases, return the original block factors.
static SmallVector<int64_t>
applyAutoPolicy(GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
                ArrayRef<ttcore::IteratorType> iteratorTypes,
                ArrayRef<int64_t> gridExtents, ArrayRef<int64_t> shardExtents,
                ArrayRef<int64_t> shardFactors, ttcore::DeviceAttr device,
                ttcore::MemorySpaceAttr l1Attr, uint32_t numBuffers,
                bool allowAliasedEltwiseBlocking) {
  const SmallVector<int64_t> originalBlockFactors =
      genericOp.getBlockFactorsValue();

  //===------------------------------------------------------------------===//
  // Classify the generic op's iteration shape.
  //===------------------------------------------------------------------===//
  auto config = classifyAutoSearch(genericOp, iteratorTypes, shardFactors);
  if (!config) {
    return originalBlockFactors;
  }

  //===------------------------------------------------------------------===//
  // Enumerate legal per-dim scale divisors for each dimension.
  //===------------------------------------------------------------------===//
  SmallVector<SmallVector<int64_t>> legalScales;
  legalScales.reserve(config->candidateDims.size());
  for (std::size_t dim : config->candidateDims) {
    legalScales.push_back(ttmlir::utils::getFactors(shardFactors[dim]));
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
        device, l1Attr, numBuffers, allowAliasedEltwiseBlocking,
        /*allowIdentityCandidate=*/true);
  }

  std::function<void(std::size_t)> enumerateCandidates =
      [&](std::size_t dimIndex) {
        if (dimIndex == config->candidateDims.size()) {
          // All dimensions have been reblocked, so evaluate the candidate.
          auto candidate = evaluateCandidate(
              genericOp, config->candidateDims, indexingMaps, gridExtents,
              shardExtents, shardFactors, originalBlockFactors,
              currentDimScales, device, l1Attr, numBuffers,
              allowAliasedEltwiseBlocking);
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
    ttcore::MemorySpaceAttr l1Attr, uint32_t numBuffers,
    bool allowAliasedEltwiseBlocking) {
  const SmallVector<int64_t> shardFactors = getShardBlockFactors(genericOp);
  switch (policy) {
  case BlockFactorAnalysis::BufferSizePolicy::Max:
    return genericOp.getBlockFactorsValue();
  case BlockFactorAnalysis::BufferSizePolicy::Min:
    return applyMinPolicy(genericOp, shardFactors);
  case BlockFactorAnalysis::BufferSizePolicy::Auto:
    return applyAutoPolicy(genericOp, indexingMaps, iteratorTypes, gridExtents,
                           shardExtents, shardFactors, device, l1Attr,
                           numBuffers, allowAliasedEltwiseBlocking);
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
        device, opts.policy, l1Attr, opts.numBuffers,
        opts.allowAliasedEltwiseBlocking);

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
