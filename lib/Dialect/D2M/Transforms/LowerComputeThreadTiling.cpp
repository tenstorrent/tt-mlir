// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERCOMPUTETHREADTILING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Returns the ComputeThreadMappingAttrs on a forall's mapping array, nullopt if
// the forall is not a compute-thread distribution.
static FailureOr<std::optional<SmallVector<ComputeThreadMappingAttr>>>
getComputeThreadMappings(scf::ForallOp forall) {
  std::optional<ArrayAttr> mapping = forall.getMapping();
  if (!mapping) {
    return std::optional<SmallVector<ComputeThreadMappingAttr>>();
  }
  SmallVector<ComputeThreadMappingAttr> computeThreadMappings;
  bool sawOtherMapping = false;
  for (Attribute attr : mapping->getValue()) {
    if (auto computeThread = mlir::dyn_cast<ComputeThreadMappingAttr>(attr)) {
      computeThreadMappings.push_back(computeThread);
      continue;
    }
    sawOtherMapping = true;
  }
  if (!computeThreadMappings.empty() && sawOtherMapping) {
    return forall->emitOpError("compute_thread forall mapping must not mix "
                               "#d2m.compute_thread with other mapping attrs");
  }
  if (computeThreadMappings.empty()) {
    return std::optional<SmallVector<ComputeThreadMappingAttr>>();
  }
  return std::optional<SmallVector<ComputeThreadMappingAttr>>(
      std::move(computeThreadMappings));
}

static FailureOr<bool> isComputeThreadForall(scf::ForallOp forall) {
  FailureOr<std::optional<SmallVector<ComputeThreadMappingAttr>>> mappings =
      getComputeThreadMappings(forall);
  if (failed(mappings)) {
    return failure();
  }
  return mappings->has_value();
}

// Find the d2m.generic enclosing this forall and the index of the region
// within that generic that (transitively) contains the forall. Returns
// std::nullopt if the forall is not inside a d2m.generic region. The
// region index corresponds to the position in `generic.getThreads()`
static std::optional<std::pair<GenericOp, unsigned>>
findEnclosingGenericRegion(scf::ForallOp forall) {
  Operation *parent = forall->getParentOp();
  Region *childRegion = forall->getParentRegion();
  while (parent) {
    if (auto generic = mlir::dyn_cast<GenericOp>(parent)) {
      // childRegion is a top-level region of `generic`.
      assert(childRegion->getParentOp() == generic.getOperation() &&
             "childRegion should be a top-level region of the enclosing "
             "d2m.generic");
      return std::make_pair(generic, childRegion->getRegionNumber());
    }
    childRegion = parent->getParentRegion();
    parent = parent->getParentOp();
  }
  return std::nullopt;
}

static FailureOr<SmallVector<int64_t, 2>>
getStaticUpperBounds(scf::ForallOp forall) {
  SmallVector<int64_t, 2> upperBounds;
  for (OpFoldResult bound : forall.getMixedUpperBound()) {
    std::optional<int64_t> staticBound = getConstantIntValue(bound);
    if (!staticBound) {
      return failure();
    }
    upperBounds.push_back(*staticBound);
  }
  return upperBounds;
}

struct ComputeThreadShape {
  SmallVector<int64_t, 2> upperBounds;
  int64_t numComputeThreads = 1;
};

static LogicalResult verifySupportedForall(scf::ForallOp forall) {
  if (forall.getRank() < 1 || forall.getRank() > 2) {
    return forall->emitOpError(
               "compute_thread forall must be rank 1 or 2; got rank ")
           << forall.getRank();
  }

  // Only support scf.forall with memref inputs and no outputs.
  if (!forall.getOutputs().empty()) {
    return forall->emitOpError(
        "compute_thread forall must not have shared_outs (memref-only)");
  }
  scf::InParallelOp terminator = forall.getTerminator();
  if (!terminator.getYieldingOps().empty()) {
    return forall->emitOpError(
        "compute_thread forall must have empty scf.in_parallel terminator");
  }

  return success();
}

static FailureOr<ComputeThreadShape>
verifyComputeThreadShape(scf::ForallOp forall) {
  FailureOr<std::optional<SmallVector<ComputeThreadMappingAttr>>> mappings =
      getComputeThreadMappings(forall);
  if (failed(mappings)) {
    return failure();
  }
  assert(*mappings && "lower() invoked on a non-compute-thread forall");
  if ((*mappings)->size() != static_cast<size_t>(forall.getRank())) {
    forall->emitOpError(
        "compute_thread forall mapping count must match rank, got ")
        << (*mappings)->size() << " mappings for rank " << forall.getRank();
    return failure();
  }

  FailureOr<SmallVector<int64_t, 2>> upperBounds = getStaticUpperBounds(forall);
  if (failed(upperBounds)) {
    forall->emitOpError("compute_thread forall must have static upper bounds");
    return failure();
  }
  for (int64_t upperBound : *upperBounds) {
    if (upperBound <= 0) {
      forall->emitOpError(
          "compute_thread forall upper bounds must be positive, got ")
          << upperBound;
      return failure();
    }
  }
  int64_t numComputeThreads = 1;
  for (auto [mapping, upperBound] : llvm::zip(**mappings, *upperBounds)) {
    if (mapping.getNum() != upperBound) {
      forall->emitOpError("compute_thread mapping factor ")
          << mapping.getNum()
          << " does not match corresponding forall upper bound " << upperBound;
      return failure();
    }
    numComputeThreads *= mapping.getNum();
  }

  return ComputeThreadShape{*upperBounds, numComputeThreads};
}

static LogicalResult stampComputeThreadCount(scf::ForallOp forall,
                                             int64_t numComputeThreads) {
  auto enclosing = findEnclosingGenericRegion(forall);
  if (!enclosing) {
    return forall->emitOpError(
        "compute_thread forall must be inside a d2m.generic region");
  }
  GenericOp generic = enclosing->first;
  unsigned regionIdx = enclosing->second;
  ArrayAttr threadsAttr = generic.getThreadsAttr();
  SmallVector<Attribute> rebuiltThreads(threadsAttr.getValue().begin(),
                                        threadsAttr.getValue().end());
  auto origThread = mlir::cast<ThreadAttr>(rebuiltThreads[regionIdx]);
  if (origThread.getThreadType() != ThreadType::Compute) {
    return forall->emitOpError("compute_thread forall must live inside the "
                               "d2m.generic's Compute "
                               "thread region (got threadType ")
           << stringifyEnum(origThread.getThreadType()) << ")";
  }
  if (origThread.getNumComputeThreads() != -1 &&
      origThread.getNumComputeThreads() != numComputeThreads) {
    return forall->emitOpError("compute thread region already carries "
                               "num_compute_threads = ")
           << origThread.getNumComputeThreads()
           << " which conflicts with this forall's #d2m.compute_thread<num="
           << numComputeThreads << ">";
  }
  rebuiltThreads[regionIdx] =
      ThreadAttr::get(generic.getContext(), origThread.getThreadType(),
                      origThread.getKernelSymbol(),
                      origThread.getProcessorIndex(), numComputeThreads);
  generic.setThreadsAttr(ArrayAttr::get(generic.getContext(), rebuiltThreads));

  return success();
}

// Lower a single compute-thread forall to SPMD form: insert d2m.my_thread_id,
// substitute the forall induction variables with coordinates derived from that
// op, inline the body into the parent block, and erase the forall.
static LogicalResult lower(scf::ForallOp forall) {
  if (failed(verifySupportedForall(forall))) {
    return failure();
  }

  FailureOr<ComputeThreadShape> shape = verifyComputeThreadShape(forall);
  if (failed(shape)) {
    return failure();
  }

  if (failed(stampComputeThreadCount(forall, shape->numComputeThreads))) {
    return failure();
  }

  OpBuilder builder(forall);
  Value tid = builder.create<d2m::MyThreadIdOp>(forall.getLoc());
  SmallVector<Value, 2> replacements;
  if (forall.getRank() == 1) {
    replacements.push_back(tid);
  } else {
    int64_t innerBound = shape->upperBounds[1];
    AffineExpr tidExpr = builder.getAffineDimExpr(0);
    Location loc = forall.getLoc();
    replacements.push_back(builder.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, tidExpr.floorDiv(innerBound)), tid));
    replacements.push_back(builder.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, tidExpr % innerBound), tid));
  }

  // Substitute the induction variables, drop the terminator, and inline the
  // body.
  Block *body = forall.getBody();
  for (auto [arg, replacement] : llvm::zip(
           body->getArguments().take_front(forall.getRank()), replacements)) {
    arg.replaceAllUsesWith(replacement);
  }
  scf::InParallelOp terminator = forall.getTerminator();
  terminator.erase();
  Block *parent = forall->getBlock();
  parent->getOperations().splice(forall->getIterator(), body->getOperations());
  forall.erase();
  return success();
}

class D2MLowerComputeThreadTiling
    : public impl::D2MLowerComputeThreadTilingBase<
          D2MLowerComputeThreadTiling> {
public:
  using impl::D2MLowerComputeThreadTilingBase<
      D2MLowerComputeThreadTiling>::D2MLowerComputeThreadTilingBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    SmallVector<scf::ForallOp> targets;
    auto collectResult = moduleOp->walk([&](scf::ForallOp forall) {
      FailureOr<bool> isComputeThread = isComputeThreadForall(forall);
      if (failed(isComputeThread)) {
        return WalkResult::interrupt();
      }
      if (*isComputeThread) {
        targets.push_back(forall);
      }
      return WalkResult::advance();
    });
    if (collectResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    for (scf::ForallOp forall : targets) {
      if (failed(lower(forall))) {
        signalPassFailure();
        return;
      }
    }

    // Verifier: no compute-thread forall should remain.
    auto remaining = moduleOp->walk([&](scf::ForallOp forall) {
      FailureOr<bool> isComputeThread = isComputeThreadForall(forall);
      if (failed(isComputeThread) || *isComputeThread) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (remaining.wasInterrupted()) {
      moduleOp->emitError(
          "D2MLowerComputeThreadTiling left a #d2m.compute_thread "
          "forall behind");
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
