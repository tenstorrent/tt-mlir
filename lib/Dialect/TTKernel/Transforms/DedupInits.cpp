// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelTraits.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELDEDUPINITS
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {

// Two init ops describe the same hardware configuration when they are the same
// op with identical operands (same SSA values) and identical attributes. Init
// ops produce no results, so re-applying an identical configuration with no
// reconfiguration in between is a no-op.
static bool sameInit(Operation *a, Operation *b) {
  return a->getName() == b->getName() &&
         a->getNumOperands() == b->getNumOperands() &&
         std::equal(a->operand_begin(), a->operand_end(), b->operand_begin()) &&
         a->getAttrDictionary() == b->getAttrDictionary();
}

class TTKernelDedupInits
    : public impl::TTKernelDedupInitsBase<TTKernelDedupInits> {
public:
  using impl::TTKernelDedupInitsBase<
      TTKernelDedupInits>::TTKernelDedupInitsBase;

  void runOnOperation() final {
    llvm::SmallPtrSet<Operation *, 16> toErase;

    // R1: an init op identical to the op immediately preceding it (in the same
    // block) is redundant. getPrevNode() is block-local, so this never reasons
    // across block boundaries. For a run of N identical adjacent inits, each
    // but the first matches its predecessor and is collected, leaving exactly
    // one.
    getOperation()->walk([&](Operation *op) {
      if (!op->hasTrait<ttkernel::TTKernelInitOpTrait>()) {
        return;
      }
      Operation *prev = op->getPrevNode();
      if (prev && prev->hasTrait<ttkernel::TTKernelInitOpTrait>() &&
          sameInit(prev, op)) {
        toErase.insert(op);
      }
    });

    // R2: collapse same-config reduce runs to init-once / reduce-many /
    // uninit-once. The lowering emits, per reduced tile,
    //   reduce_init(X); reduce_tile; reduce_uninit;
    // with index arithmetic (arith dialect) interspersed for the tile offsets.
    // Within a block, track the open reduce config; a reduce_uninit followed
    // (through only transparent index math) by a same-config reduce_init is a
    // redundant round-trip, so that uninit and that reinit are erased. The
    // opening reduce_init and the final reduce_uninit (the one not followed by
    // a matching reinit) are kept. Index math does not touch the reduce
    // hardware config, so it is transparent; any other op conservatively closes
    // the run.
    auto isTransparent = [](Operation *op) {
      Dialect *d = op->getDialect();
      return d && d->getNamespace() == "arith";
    };
    getOperation()->walk([&](Block *block) {
      ReduceInitOp open = nullptr;
      ReduceUninitOp pendingUninit = nullptr;
      for (Operation &o : *block) {
        Operation *op = &o;
        if (auto init = dyn_cast<ReduceInitOp>(op)) {
          if (open && pendingUninit && sameInit(open, init)) {
            toErase.insert(pendingUninit);
            toErase.insert(init);
            pendingUninit = nullptr; // the open config carries through
          } else {
            open = init;
            pendingUninit = nullptr;
          }
        } else if (auto uninit = dyn_cast<ReduceUninitOp>(op)) {
          // Candidate for removal, but only if a matching reinit follows.
          pendingUninit = uninit;
        } else if (isa<ReduceTileOp>(op)) {
          // A reduce_tile that runs while an uninit is pending executed under
          // the post-uninit (reset) config, so that uninit is load-bearing and
          // must not be coalesced away. (Index math is different - see below.)
          pendingUninit = nullptr;
        } else if (isTransparent(op)) {
          // Index math does not touch the reduce config; a pending uninit
          // survives it (the real lowering computes the next tile offset here).
        } else {
          // Anything else (e.g. tile_regs_commit / pack_tile) ends the run; the
          // pending uninit is the final one and must stay.
          open = nullptr;
          pendingUninit = nullptr;
        }
      }
    });

    for (Operation *op : toErase) {
      op->erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttkernel
