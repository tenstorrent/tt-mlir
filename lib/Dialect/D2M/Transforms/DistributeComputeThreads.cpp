// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDISTRIBUTECOMPUTETHREADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct SplitCandidate {
  unsigned loopDim;
  int64_t tripCount;
};

struct SplitDim {
  unsigned loopDim;
  int64_t factor;
};

struct SplitPlan {
  SmallVector<SplitDim, 2> dims;
  int64_t totalFactor = 1;
};

static bool isEligibleForFactor(const SplitCandidate &candidate,
                                int64_t factor) {
  return ShapedType::isDynamic(candidate.tripCount) ||
         candidate.tripCount >= factor;
}

static SmallVector<int64_t> getCurrentDims(ArrayRef<int64_t> splitDims,
                                           ArrayRef<int64_t> interchange) {
  if (interchange.empty()) {
    return llvm::to_vector(splitDims);
  }

  SmallVector<int64_t> inverseInterchange(interchange.size(), -1);
  for (auto [newDim, oldDim] : llvm::enumerate(interchange)) {
    if (oldDim >= 0 && oldDim < static_cast<int64_t>(interchange.size())) {
      inverseInterchange[oldDim] = static_cast<int64_t>(newDim);
    }
  }

  SmallVector<int64_t> currentDims;
  for (int64_t splitDim : splitDims) {
    if (splitDim < 0 ||
        splitDim >= static_cast<int64_t>(inverseInterchange.size()) ||
        inverseInterchange[splitDim] < 0) {
      currentDims.push_back(splitDim);
      continue;
    }
    currentDims.push_back(inverseInterchange[splitDim]);
  }
  return currentDims;
}

static std::optional<SplitCandidate>
getCandidateForLoopDim(linalg::GenericOp op, unsigned loopDim) {
  auto iterTypes = op.getIteratorTypesArray();
  if (loopDim >= iterTypes.size() ||
      iterTypes[loopDim] != mlir::utils::IteratorType::parallel) {
    return std::nullopt;
  }

  AffineMap outMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  auto outputType =
      mlir::dyn_cast<ShapedType>(op.getDpsInitOperand(0)->get().getType());
  if (!outputType) {
    return std::nullopt;
  }

  for (auto [resultIdx, expr] : llvm::enumerate(outMap.getResults())) {
    auto dim = mlir::dyn_cast<AffineDimExpr>(expr);
    if (!dim || dim.getPosition() != loopDim ||
        resultIdx >= static_cast<size_t>(outputType.getRank())) {
      continue;
    }
    return SplitCandidate{loopDim, outputType.getDimSize(resultIdx)};
  }
  return std::nullopt;
}

static SmallVector<SplitCandidate, 2>
getOrderedCandidates(linalg::GenericOp op, ArrayRef<int64_t> splitDims,
                     ArrayRef<int64_t> interchange) {
  SmallVector<SplitCandidate, 2> candidates;
  llvm::SmallDenseSet<unsigned, 4> seen;

  if (!splitDims.empty()) {
    for (int64_t splitDim : getCurrentDims(splitDims, interchange)) {
      if (splitDim < 0 || splitDim >= static_cast<int64_t>(op.getNumLoops()) ||
          !seen.insert(static_cast<unsigned>(splitDim)).second) {
        continue;
      }
      if (std::optional<SplitCandidate> candidate =
              getCandidateForLoopDim(op, splitDim)) {
        candidates.push_back(*candidate);
      }
    }
    return candidates;
  }

  for (unsigned loopDim = 0; loopDim < op.getNumLoops(); ++loopDim) {
    if (std::optional<SplitCandidate> candidate =
            getCandidateForLoopDim(op, loopDim)) {
      candidates.push_back(*candidate);
    }
  }
  return candidates;
}

static std::optional<SplitPlan>
chooseSplitPlan(ArrayRef<SplitCandidate> candidates,
                int64_t numComputeThreads) {
  if (numComputeThreads >= 4) {
    for (const SplitCandidate &candidate : candidates) {
      if (isEligibleForFactor(candidate, 4)) {
        return SplitPlan{{SplitDim{candidate.loopDim, 4}}, 4};
      }
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
      if (!isEligibleForFactor(candidates[i], 2)) {
        continue;
      }
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (isEligibleForFactor(candidates[j], 2)) {
          return SplitPlan{{SplitDim{candidates[i].loopDim, 2},
                            SplitDim{candidates[j].loopDim, 2}},
                           4};
        }
      }
    }
  }

  if (numComputeThreads >= 3) {
    for (const SplitCandidate &candidate : candidates) {
      if (isEligibleForFactor(candidate, 3)) {
        return SplitPlan{{SplitDim{candidate.loopDim, 3}}, 3};
      }
    }
  }

  if (numComputeThreads >= 2) {
    for (const SplitCandidate &candidate : candidates) {
      if (isEligibleForFactor(candidate, 2)) {
        return SplitPlan{{SplitDim{candidate.loopDim, 2}}, 2};
      }
    }
  }

  return std::nullopt;
}

static bool mayContainComputeThreadWork(GenericOp genericOp,
                                        unsigned regionIndex) {
  auto threads = genericOp.getThreadsAttr().getValue();
  if (regionIndex >= threads.size()) {
    return false;
  }
  auto thread = mlir::cast<ThreadAttr>(threads[regionIndex]);
  return thread.getThreadType() != ThreadType::Datamovement;
}

static void collectLinalgOps(d2m::GenericOp genericOp,
                             SmallVectorImpl<linalg::GenericOp> &targets) {
  for (auto [regionIndex, region] : llvm::enumerate(genericOp->getRegions())) {
    if (!mayContainComputeThreadWork(genericOp, regionIndex)) {
      continue;
    }
    region.walk(
        [&](linalg::GenericOp linalgOp) { targets.push_back(linalgOp); });
  }
}

class D2MDistributeComputeThreads
    : public impl::D2MDistributeComputeThreadsBase<
          D2MDistributeComputeThreads> {
public:
  using impl::D2MDistributeComputeThreadsBase<
      D2MDistributeComputeThreads>::D2MDistributeComputeThreadsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Validate options.
    if (numComputeThreads < 1) {
      moduleOp->emitError() << "num-compute-threads must be at least 1, got "
                            << numComputeThreads;
      return signalPassFailure();
    }

    // Collect targets first; tiling mutates the IR.
    SmallVector<linalg::GenericOp> targets;
    moduleOp->walk([&](d2m::GenericOp genericOp) {
      collectLinalgOps(genericOp, targets);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp linalgOp : targets) {
      if (failed(distribute(rewriter, linalgOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult distribute(IRRewriter &rewriter, linalg::GenericOp linalgOp) {
    SmallVector<SplitCandidate, 2> candidates =
        getOrderedCandidates(linalgOp, splitDims, matmulInterchange);
    std::optional<SplitPlan> plan =
        chooseSplitPlan(candidates, numComputeThreads);
    if (!plan) {
      return success();
    }

    SmallVector<OpFoldResult> numThreads(linalgOp.getNumLoops(),
                                         rewriter.getIndexAttr(0));
    for (SplitDim dim : plan->dims) {
      numThreads[dim.loopDim] = rewriter.getIndexAttr(dim.factor);
    }

    SmallVector<Attribute> mapping;
    for (SplitDim dim : plan->dims) {
      mapping.push_back(rewriter.getAttr<ComputeThreadMappingAttr>(dim.factor));
    }

    scf::SCFTilingOptions opts;
    opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    opts.setNumThreads(numThreads);
    opts.setMapping(mapping);

    auto tilingInterface = cast<TilingInterface>(linalgOp.getOperation());
    rewriter.setInsertionPoint(linalgOp);
    FailureOr<scf::SCFTilingResult> result =
        scf::tileUsingSCF(rewriter, tilingInterface, opts);
    if (failed(result)) {
      return linalgOp->emitOpError("failed to tile linalg op to scf.forall");
    }

    rewriter.replaceOp(linalgOp, result->replacements);
    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
