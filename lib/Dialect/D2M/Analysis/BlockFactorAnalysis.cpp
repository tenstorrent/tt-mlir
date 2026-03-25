// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"

#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::d2m {

using namespace allocation;

namespace {

//===----------------------------------------------------------------------===//
// Block factor analysis logic
//===----------------------------------------------------------------------===//

/// Returns the index of the single reduction dimension, if any.
static std::optional<std::size_t>
getSingleTuningDim(ArrayRef<ttcore::IteratorType> iteratorTypes) {
  std::optional<std::size_t> tuningDim;
  for (auto [dim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != ttcore::IteratorType::Reduction) {
      continue;
    }
    if (tuningDim.has_value()) {
      return std::nullopt;
    }
    tuningDim = dim;
  }
  return tuningDim;
}

/// Estimates the total stream buffer bytes for a candidate grid and shard
/// extents.
static std::optional<uint64_t> estimateCBBytesForCandidate(
    GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<int64_t> candidateGridExtents,
    ArrayRef<int64_t> candidateShardExtents, std::size_t tuningDim,
    ttcore::DeviceAttr device, ttcore::MemorySpaceAttr l1Attr,
    uint32_t numBuffers) {
  uint64_t totalBytes = 0;
  bool sawTunedInput = false;

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    if (genericOp.isOutputOperandIdx(operandIndex) ||
        genericOp.isScratchInput(operandIndex)) {
      continue;
    }

    const AffineMap indexingMap = indexingMaps[operandIndex];
    if (!indexingMap.isFunctionOfDim(tuningDim)) {
      continue;
    }

    auto operandType = mlir::dyn_cast<MemRefType>(operand.getType());
    if (!operandType ||
        !mlir::isa<ttcore::TileType>(operandType.getElementType())) {
      return std::nullopt;
    }

    sawTunedInput = true;
    const AffineMap canonicalMap = canonicalizeBroadcasts(indexingMap);
    const SmallVector<int64_t> gridShapeRescaled =
        canonicalMap.compose(candidateGridExtents);
    const SmallVector<int64_t> shardShapeRescaled =
        canonicalMap.compose(candidateShardExtents);

    if (ttmlir::utils::volume<int64_t>(shardShapeRescaled) < 4) {
      return std::nullopt;
    }

    const MemRefType bufferType =
        getStreamBufferType(gridShapeRescaled, shardShapeRescaled,
                            operandType.getElementType(), l1Attr, numBuffers);
    totalBytes += getStreamBufferSizeBytes(bufferType, device);
  }

  if (!sawTunedInput) {
    return std::nullopt;
  }

  return totalBytes;
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

static SmallVector<int64_t>
applyAutoPolicy(GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
                ArrayRef<ttcore::IteratorType> iteratorTypes,
                ArrayRef<int64_t> gridExtents, ArrayRef<int64_t> shardExtents,
                ArrayRef<int64_t> shardFactors, ttcore::DeviceAttr device,
                ttcore::MemorySpaceAttr l1Attr, uint32_t numBuffers) {
  SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();
  const std::optional<std::size_t> tuningDim =
      getSingleTuningDim(iteratorTypes);
  if (!tuningDim.has_value()) {
    return blockFactors;
  }

  if (shardFactors[*tuningDim] <= 1) {
    return blockFactors;
  }

  const auto baseBytes = estimateCBBytesForCandidate(
      genericOp, indexingMaps, gridExtents, shardExtents, *tuningDim, device,
      l1Attr, numBuffers);
  if (!baseBytes.has_value()) {
    return blockFactors;
  }

  int64_t bestScale = 1;
  uint64_t bestBytes = *baseBytes;
  for (int64_t scale = shardFactors[*tuningDim]; scale >= 2; --scale) {
    if (shardFactors[*tuningDim] % scale != 0) {
      continue;
    }

    SmallVector<int64_t> candidateGridExtents(gridExtents.begin(),
                                              gridExtents.end());
    SmallVector<int64_t> candidateShardExtents(shardExtents.begin(),
                                               shardExtents.end());
    candidateGridExtents[*tuningDim] *= scale;
    candidateShardExtents[*tuningDim] /= scale;

    const auto candidateBytes = estimateCBBytesForCandidate(
        genericOp, indexingMaps, candidateGridExtents, candidateShardExtents,
        *tuningDim, device, l1Attr, numBuffers);
    if (!candidateBytes.has_value()) {
      continue;
    }

    // Break ties toward the larger scale so we maximize blocking at the
    // same estimated CB buffer cost.
    if (*candidateBytes < bestBytes ||
        (*candidateBytes == bestBytes && scale > bestScale)) {
      bestBytes = *candidateBytes;
      bestScale = scale;
    }
  }

  blockFactors[*tuningDim] *= bestScale;
  TT_ALLOC_DEBUG("applying auto policy scale {} on dim {}, new block "
                 "factors {}, estimated CB bytes {}",
                 bestScale, *tuningDim, asSeq(blockFactors), bestBytes);
  return blockFactors;
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
                           numBuffers);
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
