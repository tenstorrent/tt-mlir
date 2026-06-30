// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/ADT/DenseSet.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELCONTROLDSTSECTION
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {

template <typename ConcreteOp>
static Block *findBlockContaining(Operation *op) {
  Block *block = op->getBlock();
  while (block->getOps<ConcreteOp>().empty()) {
    Operation *parentOp = block->getParentOp();
    assert(parentOp && "expected enclosing op before acquire block");
    block = parentOp->getBlock();
  }
  return block;
}

static Operation *parentOpAtBlock(Operation *child, Block *atBlock) {
  Operation *parent = child;
  while (parent->getBlock() != atBlock) {
    parent = parent->getParentOp();
    assert(parent);
  }
  return parent;
}

// Returns true if a TileRegsCommitOp exists between the most recent
// TileRegsAcquireOp before `op` (in `op`'s block) and `op` itself. Used to
// prevent double-insertion on re-application.
static bool hasPrecedingCommit(Operation *op) {
  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    if (isa<ttkernel::TileRegsCommitOp>(it)) {
      return true;
    }
    if (isa<ttkernel::TileRegsAcquireOp>(it)) {
      return false;
    }
  }
  return false;
}

} // namespace

namespace {
class TTKernelControlDstSection
    : public impl::TTKernelControlDstSectionBase<TTKernelControlDstSection> {
public:
  using impl::TTKernelControlDstSectionBase<
      TTKernelControlDstSection>::TTKernelControlDstSectionBase;

  void runOnOperation() final {
    SmallVector<std::pair<Operation *, Location>> insertionPoints;
    llvm::DenseSet<Operation *> visitedParents;

    getOperation()->walk([&](Operation *op) {
      if (!isa<ttkernel::PackTileOp, ttkernel::PackTileBlockOp>(op)) {
        return;
      }

      Block *acquireBlock =
          findBlockContaining<ttkernel::TileRegsAcquireOp>(op);
      Operation *parent = parentOpAtBlock(op, acquireBlock);

      if (!visitedParents.insert(parent).second || hasPrecedingCommit(parent)) {
        return;
      }

      insertionPoints.push_back({parent, op->getLoc()});
    });

    OpBuilder builder(&getContext());
    for (auto [parent, loc] : insertionPoints) {
      builder.setInsertionPoint(parent);
      builder.create<ttkernel::TileRegsCommitOp>(loc);
      builder.create<ttkernel::TileRegsWaitOp>(loc);
      builder.setInsertionPointAfter(parent);
      builder.create<ttkernel::TileRegsReleaseOp>(loc);
    }
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
