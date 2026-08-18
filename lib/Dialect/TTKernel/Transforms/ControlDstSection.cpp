// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

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

// Returns true if `op` is a pack_tile that already lives inside a
// self-managed DST section: a TileRegsAcquireOp precedes it and a
// TileRegsReleaseOp follows it in the same block, with no intervening
// acquire/release breaking that pairing. Some group ops emit a complete
// commit/wait/pack/release sequence in the same block, so this pass must
// not wrap them again.
static bool isSelfManagedPackTile(Operation *op) {
  bool precededByAcquire = false;
  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    if (isa<ttkernel::TileRegsAcquireOp>(it)) {
      precededByAcquire = true;
      break;
    }
    if (isa<ttkernel::TileRegsReleaseOp>(it)) {
      break;
    }
  }
  if (!precededByAcquire) {
    return false;
  }

  for (Operation *it = op->getNextNode(); it != nullptr;
       it = it->getNextNode()) {
    if (isa<ttkernel::TileRegsReleaseOp>(it)) {
      return true;
    }
    if (isa<ttkernel::TileRegsAcquireOp>(it)) {
      return false;
    }
  }
  return false;
}

// The TileRegsAcquireOp that opens `op`'s DST section: the closest one
// preceding `parent` in `parent`'s own block. Sections are keyed on this op
// rather than on the enclosing block, because a single block may open several
// independent sections in sequence.
static Operation *findOpeningAcquire(Operation *parent) {
  for (Operation *it = parent->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    if (isa<ttkernel::TileRegsAcquireOp>(it)) {
      return it;
    }
  }
  return nullptr;
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
    // A DST section is `acquire ... commit/wait ... pack* ... release`, so
    // every pack under one acquire shares a section and gets one handshake.
    // Keying on each pack's parent instead would emit unbalanced handshakes for
    // a multi-result op like d2m.tile_argmax. Keyed on (acquire, pack op kind)
    // rather than the block, since a block may open several sections in
    // sequence and pack_tile vs pack_tile_block need different packer
    // configurations.
    struct DstSectionSpan {
      Operation *first = nullptr;
      Operation *last = nullptr;
      std::optional<Location> loc;
    };
    llvm::MapVector<std::pair<Operation *, mlir::TypeID>, DstSectionSpan> spans;

    getOperation()->walk([&](Operation *op) {
      if (!isa<ttkernel::PackTileOp, ttkernel::PackTileBlockOp>(op)) {
        return;
      }

      // When group ops self-manage their DST section, don't wrap them.
      if (isSelfManagedPackTile(op)) {
        return;
      }

      Block *acquireBlock =
          findBlockContaining<ttkernel::TileRegsAcquireOp>(op);
      Operation *parent = parentOpAtBlock(op, acquireBlock);

      if (hasPrecedingCommit(parent)) {
        return;
      }

      Operation *acquire = findOpeningAcquire(parent);
      if (!acquire) {
        return;
      }

      auto [it, inserted] =
          spans.try_emplace({acquire, op->getName().getTypeID()});
      DstSectionSpan &span = it->second;
      if (inserted) {
        span.first = parent;
        span.last = parent;
        span.loc = op->getLoc();
        return;
      }
      // `walk` visits in program order within a block, so a later candidate
      // only ever extends the span forward.
      if (span.last->isBeforeInBlock(parent)) {
        span.last = parent;
      }
      if (parent->isBeforeInBlock(span.first)) {
        span.first = parent;
      }
    });

    OpBuilder builder(&getContext());
    for (auto &[key, span] : spans) {
      Location loc = *span.loc;
      builder.setInsertionPoint(span.first);
      builder.create<ttkernel::TileRegsCommitOp>(loc);
      builder.create<ttkernel::TileRegsWaitOp>(loc);
      builder.setInsertionPointAfter(span.last);
      builder.create<ttkernel::TileRegsReleaseOp>(loc);
    }
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
