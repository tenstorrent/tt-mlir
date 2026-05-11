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

// Returns true if the forall's mapping carries a #d2m.compute_thread attr.
static bool isComputeThreadForall(scf::ForallOp forall) {
  std::optional<ArrayAttr> mapping = forall.getMapping();
  if (!mapping) {
    return false;
  }
  for (Attribute attr : mapping->getValue()) {
    if (mlir::isa<ComputeThreadMappingAttr>(attr)) {
      return true;
    }
  }
  return false;
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
