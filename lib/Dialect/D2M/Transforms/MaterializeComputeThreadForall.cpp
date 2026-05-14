// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMATERIALIZECOMPUTETHREADFORALL
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Returns the ComputeThreadMappingAttr on a forall's mapping array, or
// nullopt if the forall is not a compute-thread distribution.
static std::optional<ComputeThreadMappingAttr>
getComputeThreadMapping(scf::ForallOp forall) {
  std::optional<ArrayAttr> mapping = forall.getMapping();
  if (!mapping) {
    return std::nullopt;
  }
  for (Attribute attr : mapping->getValue()) {
    if (auto computeThread = mlir::dyn_cast<ComputeThreadMappingAttr>(attr)) {
      return computeThread;
    }
  }
  return std::nullopt;
}

static bool isComputeThreadForall(scf::ForallOp forall) {
  return getComputeThreadMapping(forall).has_value();
}

// Find the d2m.generic enclosing this forall and the index of the region
// within that generic that (transitively) contains the forall. Returns
// std::nullopt if the forall is not inside a d2m.generic region. The
// region index corresponds to the position in `generic.getThreads()` —
// the thread region (Compute) whose ThreadAttr we need to stamp the
// per-generic N onto.
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

// Lower a single compute-thread forall to SPMD form: insert d2m.my_thread_id
// before the forall, substitute the forall IV with that op's result, inline
// the body into the parent block, and erase the forall.
static LogicalResult materialize(scf::ForallOp forall) {
  // v1 only handles 1-D compute-thread distribution. Higher-rank forall would
  // require multiple my_thread_id ops or a more elaborate mapping scheme; not
  // in scope.
  if (forall.getRank() != 1) {
    return forall->emitOpError(
        "compute_thread forall must be 1-D (rank=1); got rank ")
           << forall.getRank();
  }

  // For memref operands the forall has no shared_outs and no
  // tensor.parallel_insert_slice in the terminator. Refuse to lower a forall
  // that does — Approach B is bufferized-memref only.
  if (!forall.getOutputs().empty()) {
    return forall->emitOpError(
        "compute_thread forall must not have shared_outs (memref-only)");
  }
  scf::InParallelOp terminator = forall.getTerminator();
  if (!terminator.getYieldingOps().empty()) {
    return forall->emitOpError(
        "compute_thread forall must have empty scf.in_parallel terminator");
  }

  // Extract N from the #d2m.compute_thread<num=N> mapping and stamp it
  // onto the enclosing d2m.generic's matching thread region. After this
  // pass erases the forall, ThreadAttr.numThreadsPerCluster is the only
  // surviving record of the per-generic compute-thread fan-out for
  // downstream DFB cardinality.
  auto mapping = getComputeThreadMapping(forall);
  assert(mapping && "materialize() invoked on a non-compute-thread forall");
  int64_t numThreadsPerCluster = mapping->getNum();
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
    return forall->emitOpError(
        "compute_thread forall must live inside the d2m.generic's Compute "
        "thread region (got threadType ")
           << stringifyEnum(origThread.getThreadType()) << ")";
  }
  if (origThread.getNumThreadsPerCluster() != 1 &&
      origThread.getNumThreadsPerCluster() != numThreadsPerCluster) {
    return forall->emitOpError(
        "compute thread region already carries num_threads_per_cluster = ")
           << origThread.getNumThreadsPerCluster()
           << " which conflicts with this forall's #d2m.compute_thread<num="
           << numThreadsPerCluster << ">";
  }
  rebuiltThreads[regionIdx] = ThreadAttr::get(
      generic.getContext(), origThread.getThreadType(),
      origThread.getKernelSymbol(), origThread.getNocIndex(),
      origThread.getProcessorIndex(), numThreadsPerCluster);
  generic.setThreadsAttr(
      ArrayAttr::get(generic.getContext(), rebuiltThreads));

  OpBuilder builder(forall);
  Value tid = builder.create<d2m::MyThreadIdOp>(forall.getLoc());

  // Substitute the IV with %tid, drop the terminator, and inline the body.
  Block *body = forall.getBody();
  body->getArgument(0).replaceAllUsesWith(tid);
  terminator.erase();
  Block *parent = forall->getBlock();
  parent->getOperations().splice(forall->getIterator(),
                                 body->getOperations());
  forall.erase();
  return success();
}

class D2MMaterializeComputeThreadForall
    : public impl::D2MMaterializeComputeThreadForallBase<
          D2MMaterializeComputeThreadForall> {
public:
  using impl::D2MMaterializeComputeThreadForallBase<
      D2MMaterializeComputeThreadForall>::D2MMaterializeComputeThreadForallBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Collect first, then transform — modifying ops during a walk that visits
    // them is brittle.
    SmallVector<scf::ForallOp> targets;
    moduleOp->walk([&](scf::ForallOp forall) {
      if (isComputeThreadForall(forall)) {
        targets.push_back(forall);
      }
    });

    for (scf::ForallOp forall : targets) {
      if (failed(materialize(forall))) {
        signalPassFailure();
        return;
      }
    }

    // Verifier: no compute-thread forall should remain. This is the single
    // materialization boundary.
    auto remaining = moduleOp->walk([&](scf::ForallOp forall) {
      return isComputeThreadForall(forall) ? WalkResult::interrupt()
                                           : WalkResult::advance();
    });
    if (remaining.wasInterrupted()) {
      moduleOp->emitError(
          "D2MMaterializeComputeThreadForall left a #d2m.compute_thread "
          "forall behind");
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
