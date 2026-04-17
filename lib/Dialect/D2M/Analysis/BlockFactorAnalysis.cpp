// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"

#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include <limits>

namespace mlir::tt::d2m {

using namespace allocation;

namespace {

//===----------------------------------------------------------------------===//
// Block factor analysis logic
//===----------------------------------------------------------------------===//

/// Computes the total circular buffer bytes for a candidate grid and shard
/// extents.
static std::optional<uint64_t> computeCBBytesForCandidate(
    GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<int64_t> candidateGridExtents,
    ArrayRef<int64_t> candidateShardExtents, std::size_t scalableDim,
    ttcore::DeviceAttr device, ttcore::MemorySpaceAttr l1Attr,
    uint32_t numBuffers) {
  uint64_t totalBytes = 0;
  bool sawScaledInput = false;

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    if (genericOp.isOutputOperandIdx(operandIndex)) {
      continue;
    }

    const AffineMap indexingMap = indexingMaps[operandIndex];
    if (!indexingMap.isFunctionOfDim(scalableDim)) {
      continue;
    }

    auto operandType = mlir::dyn_cast<MemRefType>(operand.getType());
    if (!operandType ||
        !mlir::isa<ttcore::TileType>(operandType.getElementType())) {
      return std::nullopt;
    }

    sawScaledInput = true;
    const AffineMap canonicalMap = canonicalizeBroadcasts(indexingMap);
    const SmallVector<int64_t> gridShapeRescaled =
        canonicalMap.compose(candidateGridExtents);
    const SmallVector<int64_t> shardShapeRescaled =
        canonicalMap.compose(candidateShardExtents);

    if (ttmlir::utils::volume<int64_t>(shardShapeRescaled) < 4) {
      return std::nullopt;
    }

    const MemRefType bufferType =
        getCBBufferType(gridShapeRescaled, shardShapeRescaled,
                        operandType.getElementType(), l1Attr, numBuffers);
    totalBytes += getCBBufferSizeBytes(bufferType, device);
  }

  if (!sawScaledInput) {
    return std::nullopt;
  }

  return totalBytes;
}

/// Computes the total circular buffer bytes across all operands (inputs AND
/// outputs) for a candidate grid/shard. Used by the parallel-only auto policy
/// where every operand's CB shrinks as we split parallel dims.
static std::optional<uint64_t>
computeTotalCBBytes(GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
                    ArrayRef<int64_t> candidateGridExtents,
                    ArrayRef<int64_t> candidateShardExtents,
                    ttcore::DeviceAttr device,
                    ttcore::MemorySpaceAttr l1Attr, uint32_t numBuffers) {
  uint64_t totalBytes = 0;
  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    const AffineMap indexingMap = indexingMaps[operandIndex];

    auto operandType = mlir::dyn_cast<MemRefType>(operand.getType());
    if (!operandType ||
        !mlir::isa<ttcore::TileType>(operandType.getElementType())) {
      return std::nullopt;
    }

    const AffineMap canonicalMap = canonicalizeBroadcasts(indexingMap);
    const SmallVector<int64_t> gridShapeRescaled =
        canonicalMap.compose(candidateGridExtents);
    const SmallVector<int64_t> shardShapeRescaled =
        canonicalMap.compose(candidateShardExtents);

    if (shardShapeRescaled.empty() ||
        ttmlir::utils::volume<int64_t>(shardShapeRescaled) < 1) {
      return std::nullopt;
    }

    const MemRefType bufferType =
        getCBBufferType(gridShapeRescaled, shardShapeRescaled,
                        operandType.getElementType(), l1Attr, numBuffers);
    totalBytes += getCBBufferSizeBytes(bufferType, device);
  }
  return totalBytes;
}

/// Enumerate divisors of `n` in ascending order (always includes 1 and n).
static SmallVector<int64_t> divisorsAscending(int64_t n) {
  SmallVector<int64_t> divs;
  if (n < 1) {
    divs.push_back(1);
    return divs;
  }
  for (int64_t d = 1; d <= n; ++d) {
    if (n % d == 0) {
      divs.push_back(d);
    }
  }
  return divs;
}

/// Parallel-only auto policy: if the baseline circular-buffer footprint
/// (across all operands) exceeds the L1 budget, search over per-dim divisor
/// scales and pick one that fits. Tiebreakers (in order): fewer non-trivial
/// split dims, smaller product of scales, smaller max scale.
///
/// Typical outcome for a large-shard elementwise op: pick a single parallel
/// dim and split it enough to bring the per-core CB under budget (e.g.
/// `[1, 15]` or `[17, 1]`).
static SmallVector<int64_t> applyAutoPolicyParallelOnly(
    GenericOp genericOp, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<int64_t> gridExtents, ArrayRef<int64_t> shardExtents,
    ArrayRef<int64_t> shardFactors, ttcore::DeviceAttr device,
    ttcore::MemorySpaceAttr l1Attr, uint32_t numBuffers) {
  SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();

  ttcore::ChipDescAttr chipDesc = ttcore::getOpChipDescAttr(genericOp);
  if (!chipDesc) {
    return blockFactors;
  }
  const uint64_t l1Budget =
      static_cast<uint64_t>(chipDesc.getUsableL1Size());

  const auto baseBytes =
      computeTotalCBBytes(genericOp, indexingMaps, gridExtents, shardExtents,
                          device, l1Attr, numBuffers);
  if (!baseBytes.has_value()) {
    return blockFactors;
  }
  if (*baseBytes <= l1Budget) {
    return blockFactors;
  }

  const std::size_t rank = blockFactors.size();
  if (rank == 0) {
    return blockFactors;
  }

  SmallVector<SmallVector<int64_t>> perDimScales(rank);
  uint64_t totalCombinations = 1;
  for (std::size_t d = 0; d < rank; ++d) {
    perDimScales[d] =
        divisorsAscending(std::max<int64_t>(shardFactors[d], 1));
    totalCombinations *= perDimScales[d].size();
  }

  // Track the best fitting scale using a (non-trivial dims, product, max)
  // lexicographic key (smaller is better).
  struct Key {
    std::size_t nonOneCount;
    uint64_t product;
    int64_t maxScale;
    bool operator<(const Key &rhs) const {
      if (nonOneCount != rhs.nonOneCount) {
        return nonOneCount < rhs.nonOneCount;
      }
      if (product != rhs.product) {
        return product < rhs.product;
      }
      return maxScale < rhs.maxScale;
    }
  };

  SmallVector<int64_t> best;
  Key bestKey{std::numeric_limits<std::size_t>::max(),
              std::numeric_limits<uint64_t>::max(),
              std::numeric_limits<int64_t>::max()};

  SmallVector<std::size_t> idx(rank, 0);
  SmallVector<int64_t> current(rank, 1);
  auto advance = [&]() {
    for (std::size_t d = 0; d < rank; ++d) {
      if (++idx[d] < perDimScales[d].size()) {
        return true;
      }
      idx[d] = 0;
    }
    return false;
  };

  for (uint64_t i = 0; i < totalCombinations; ++i) {
    for (std::size_t d = 0; d < rank; ++d) {
      current[d] = perDimScales[d][idx[d]];
    }

    SmallVector<int64_t> candidateGridExtents(gridExtents.begin(),
                                              gridExtents.end());
    SmallVector<int64_t> candidateShardExtents(shardExtents.begin(),
                                               shardExtents.end());
    bool valid = true;
    for (std::size_t d = 0; d < rank; ++d) {
      if (current[d] <= 0 || shardExtents[d] % current[d] != 0) {
        valid = false;
        break;
      }
      candidateGridExtents[d] *= current[d];
      candidateShardExtents[d] /= current[d];
    }
    if (valid) {
      const auto cbBytes = computeTotalCBBytes(
          genericOp, indexingMaps, candidateGridExtents, candidateShardExtents,
          device, l1Attr, numBuffers);
      if (cbBytes.has_value() && *cbBytes <= l1Budget) {
        Key key{0, 1, 1};
        for (int64_t s : current) {
          if (s != 1) {
            ++key.nonOneCount;
          }
          key.product *= static_cast<uint64_t>(s);
          key.maxScale = std::max(key.maxScale, s);
        }
        if (key < bestKey) {
          bestKey = key;
          best.assign(current.begin(), current.end());
        }
      }
    }

    if (!advance()) {
      break;
    }
  }

  if (best.empty()) {
    TT_ALLOC_DEBUG(
        "parallel-only auto policy: no divisor combination fits L1 budget {}, "
        "falling back to per-tile factors (shardFactors {})",
        l1Budget, asSeq(shardFactors));
    for (std::size_t d = 0; d < rank; ++d) {
      blockFactors[d] *= std::max<int64_t>(shardFactors[d], 1);
    }
    return blockFactors;
  }

  for (std::size_t d = 0; d < rank; ++d) {
    blockFactors[d] *= best[d];
  }
  TT_ALLOC_DEBUG("applying parallel-only auto policy, scales {}, new block "
                 "factors {}, base CB bytes {}, L1 budget {}",
                 asSeq(best), asSeq(blockFactors), *baseBytes, l1Budget);
  return blockFactors;
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
  const std::optional<std::size_t> scalableDim =
      getSingleReductionDim(iteratorTypes);

  // Parallel-only (no reduction dim) path: if the baseline CB footprint
  // exceeds L1, search divisor scales over parallel dims and pick the fewest
  // inner iterations that fit. For elementwise ops this typically yields
  // factors like [1, 15] or [17, 1] (single-dim splits), falling back to a
  // per-tile layout if nothing else fits.
  if (!scalableDim.has_value()) {
    return applyAutoPolicyParallelOnly(genericOp, indexingMaps, gridExtents,
                                       shardExtents, shardFactors, device,
                                       l1Attr, numBuffers);
  }

  if (shardFactors[*scalableDim] <= 1) {
    return blockFactors;
  }

  const auto baseBytes = computeCBBytesForCandidate(
      genericOp, indexingMaps, gridExtents, shardExtents, *scalableDim, device,
      l1Attr, numBuffers);
  if (!baseBytes.has_value()) {
    return blockFactors;
  }

  int64_t bestScale = 1;
  uint64_t bestBytes = *baseBytes;
  for (int64_t scale = shardFactors[*scalableDim]; scale >= 2; --scale) {
    if (shardFactors[*scalableDim] % scale != 0) {
      continue;
    }

    SmallVector<int64_t> candidateGridExtents(gridExtents.begin(),
                                              gridExtents.end());
    SmallVector<int64_t> candidateShardExtents(shardExtents.begin(),
                                               shardExtents.end());
    candidateGridExtents[*scalableDim] *= scale;
    candidateShardExtents[*scalableDim] /= scale;

    const auto candidateBytes = computeCBBytesForCandidate(
        genericOp, indexingMaps, candidateGridExtents, candidateShardExtents,
        *scalableDim, device, l1Attr, numBuffers);
    if (!candidateBytes.has_value()) {
      continue;
    }

    // Break ties toward the larger scale so we maximize blocking at the
    // same CB buffer cost.
    if (*candidateBytes < bestBytes ||
        (*candidateBytes == bestBytes && scale > bestScale)) {
      bestBytes = *candidateBytes;
      bestScale = scale;
    }
  }

  blockFactors[*scalableDim] *= bestScale;
  TT_ALLOC_DEBUG("applying auto policy scale {} on dim {}, new block "
                 "factors {}, CB bytes {}",
                 bestScale, *scalableDim, asSeq(blockFactors), bestBytes);
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
