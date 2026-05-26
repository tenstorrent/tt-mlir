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

constexpr int64_t kNumComputeThreadsPerNeo = 4;
constexpr unsigned kSplitDimSetInlineCapacity =
    static_cast<unsigned>(kNumComputeThreadsPerNeo);

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

static FailureOr<SmallVector<int64_t>>
getCurrentDimsForExplicitSplitDims(linalg::GenericOp op,
                                   ArrayRef<int64_t> splitDims,
                                   ArrayRef<int64_t> interchange) {
  SmallVector<int64_t> currentDims;

  if (interchange.empty()) {
    for (int64_t splitDim : splitDims) {
      if (splitDim < 0) {
        op.emitOpError()
            << "split-dims original dim index must be non-negative, got "
            << splitDim;
        return failure();
      }
      currentDims.push_back(splitDim);
    }
    return currentDims;
  }

  SmallVector<int64_t> inverseInterchange(interchange.size(), -1);
  for (auto [newDim, oldDim] : llvm::enumerate(interchange)) {
    if (oldDim >= 0 && oldDim < static_cast<int64_t>(interchange.size())) {
      inverseInterchange[oldDim] = static_cast<int64_t>(newDim);
    }
  }

  for (int64_t splitDim : splitDims) {
    if (splitDim < 0) {
      op.emitOpError()
          << "split-dims original dim index must be non-negative, got "
          << splitDim;
      return failure();
    }
    if (splitDim >= static_cast<int64_t>(inverseInterchange.size()) ||
        inverseInterchange[splitDim] < 0) {
      op.emitOpError() << "split-dims original dim " << splitDim
                       << " cannot be mapped through matmul-interchange";
      return failure();
    }
    currentDims.push_back(inverseInterchange[splitDim]);
  }
  return currentDims;
}

static FailureOr<SplitCandidate>
getExplicitCandidateForLoopDim(linalg::GenericOp op, int64_t currentDim) {
  if (currentDim >= static_cast<int64_t>(op.getNumLoops())) {
    op.emitOpError() << "split-dims current dim " << currentDim
                     << " is outside linalg loop rank " << op.getNumLoops();
    return failure();
  }

  unsigned loopDim = static_cast<unsigned>(currentDim);
  auto iterTypes = op.getIteratorTypesArray();
  if (iterTypes[loopDim] != mlir::utils::IteratorType::parallel) {
    op.emitOpError() << "requested split dim " << loopDim
                     << " is not a parallel iterator";
    return failure();
  }

  auto outputType =
      mlir::dyn_cast<ShapedType>(op.getDpsInitOperand(0)->get().getType());
  if (!outputType) {
    op.emitOpError() << "expected shaped output for explicit split-dims";
    return failure();
  }

  AffineMap outMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  std::optional<int64_t> outputExtent;
  for (auto [resultIdx, expr] : llvm::enumerate(outMap.getResults())) {
    auto dim = mlir::dyn_cast<AffineDimExpr>(expr);
    if (!dim || dim.getPosition() != loopDim ||
        resultIdx >= static_cast<size_t>(outputType.getRank())) {
      continue;
    }
    outputExtent = outputType.getDimSize(resultIdx);
    break;
  }

  if (!outputExtent) {
    op.emitOpError() << "requested split dim " << loopDim
                     << " is not a plain output affine dim";
    return failure();
  }

  if (!ShapedType::isDynamic(*outputExtent) && *outputExtent < 2) {
    op.emitOpError() << "requested split dim " << loopDim
                     << " has static output extent " << *outputExtent
                     << ", smaller than any supported split factor";
    return failure();
  }

  return SplitCandidate{loopDim, *outputExtent};
}

static FailureOr<SmallVector<SplitCandidate, 2>>
getOrderedCandidates(linalg::GenericOp op, ArrayRef<int64_t> splitDims,
                     ArrayRef<int64_t> interchange) {
  SmallVector<SplitCandidate, 2> candidates;
  llvm::SmallDenseSet<unsigned, kSplitDimSetInlineCapacity> seen;

  if (!splitDims.empty()) {
    FailureOr<SmallVector<int64_t>> currentDims =
        getCurrentDimsForExplicitSplitDims(op, splitDims, interchange);
    if (failed(currentDims)) {
      return failure();
    }

    for (int64_t currentDim : *currentDims) {
      if (currentDim >= static_cast<int64_t>(op.getNumLoops())) {
        op.emitOpError() << "split-dims current dim " << currentDim
                         << " is outside linalg loop rank " << op.getNumLoops();
        return failure();
      }
      if (!seen.insert(static_cast<unsigned>(currentDim)).second) {
        op.emitOpError() << "split-dims maps multiple requests to current dim "
                         << currentDim;
        return failure();
      }
      FailureOr<SplitCandidate> candidate =
          getExplicitCandidateForLoopDim(op, currentDim);
      if (failed(candidate)) {
        return failure();
      }
      candidates.push_back(*candidate);
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
chooseSplitPlan(ArrayRef<SplitCandidate> candidates) {
  for (const SplitCandidate &candidate : candidates) {
    if (isEligibleForFactor(candidate, kNumComputeThreadsPerNeo)) {
      return SplitPlan{{SplitDim{candidate.loopDim, kNumComputeThreadsPerNeo}},
                       kNumComputeThreadsPerNeo};
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
                         kNumComputeThreadsPerNeo};
      }
    }
  }

  for (const SplitCandidate &candidate : candidates) {
    if (isEligibleForFactor(candidate, 3)) {
      return SplitPlan{{SplitDim{candidate.loopDim, 3}}, 3};
    }
  }

  for (const SplitCandidate &candidate : candidates) {
    if (isEligibleForFactor(candidate, 2)) {
      return SplitPlan{{SplitDim{candidate.loopDim, 2}}, 2};
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
    FailureOr<SmallVector<SplitCandidate, 2>> candidates =
        getOrderedCandidates(linalgOp, splitDims, matmulInterchange);
    if (failed(candidates)) {
      return failure();
    }

    std::optional<SplitPlan> plan = chooseSplitPlan(*candidates);
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
