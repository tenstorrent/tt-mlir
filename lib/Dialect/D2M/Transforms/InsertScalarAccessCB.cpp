// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/MapVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTSCALARACCESSCB
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// What one CB needs on the datamovement thread: the transfers that fill it and
// the scalar reads that consume it.
struct ScalarReadSync {
  SmallVector<Operation *> transfers;
  SmallVector<memref::LoadOp> reads;
  // The operands naming the buffer, rewired onto the wait result.
  SmallVector<OpOperand *> uses;
};

// The ancestor of `op` that sits directly in `block`, or null if `block` is not
// on `op`'s parent chain. This is the op whose iteration cadence a wait/pop
// pair placed in `block` would see.
static Operation *ancestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block) {
    op = op->getParentOp();
  }
  return op;
}

class D2MInsertScalarAccessCB
    : public impl::D2MInsertScalarAccessCBBase<D2MInsertScalarAccessCB> {
public:
  using impl::D2MInsertScalarAccessCBBase<
      D2MInsertScalarAccessCB>::D2MInsertScalarAccessCBBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    SmallVector<GenericOp> generics;
    getOperation().walk(
        [&](GenericOp generic) { generics.push_back(generic); });

    for (GenericOp generic : generics) {
      for (unsigned regionIdx = 0; regionIdx < generic.getNumRegions();
           ++regionIdx) {
        if (generic.getRegionThreadType(regionIdx) !=
            ThreadType::Datamovement) {
          continue;
        }
        Region &region = generic.getRegion(regionIdx);
        if (region.empty()) {
          continue;
        }
        if (failed(insertCBOpsForScalarReads(rewriter, generic,
                                             &region.front()))) {
          signalPassFailure();
          return;
        }
      }
    }
  }

private:
  // Collect, per CB port, the transfers filling it and the scalar reads of it.
  static llvm::MapVector<unsigned, ScalarReadSync>
  collectSyncsByPort(Block *dmBlock) {
    llvm::MapVector<unsigned, ScalarReadSync> byPort;
    dmBlock->walk([&](Operation *op) {
      if (auto dma = mlir::dyn_cast<ShardDMAOpInterface>(op)) {
        byPort[dma.getCBPort()].transfers.push_back(op);
        return;
      }
      auto load = mlir::dyn_cast<memref::LoadOp>(op);
      if (!load || !utils::getScalarL1AccessMemref(load)) {
        return;
      }
      // Already bracketed: the read names an acquired buffer, not the operand.
      // Keeps a second run of this pass from stacking another wait/pop pair.
      if (load.getMemRef().getDefiningOp<WaitOp>()) {
        return;
      }
      if (std::optional<unsigned> port = getScalarL1AccessPort(load)) {
        ScalarReadSync &sync = byPort[*port];
        sync.reads.push_back(load);
        sync.uses.push_back(&load->getOpOperand(0));
      }
    });
    return byPort;
  }

  static LogicalResult insertCBOpsForScalarReads(RewriterBase &rewriter,
                                                 GenericOp generic,
                                                 Block *dmBlock) {
    llvm::MapVector<unsigned, ScalarReadSync> byPort =
        collectSyncsByPort(dmBlock);

    for (auto &[port, sync] : byPort) {
      if (sync.reads.empty()) {
        continue;
      }
      if (sync.transfers.empty()) {
        // Nothing on this thread fills the CB, so there is no push for a wait
        // to observe and no way to synchronize the read here. Reading anyway
        // would race against whatever does fill it, so reject rather than emit
        // a silently unsynchronized dereference.
        return generic.emitOpError()
               << "circular buffer (port " << port
               << ") is scalar-read on a datamovement thread that does not fill"
                  " it; a cross-thread producer for a scalar read is not yet"
                  " supported";
      }

      // The wait/pop pair must fire once per transfer, so it belongs in the
      // transfer's block: if the transfer sits outside a loop that reads the
      // buffer, a pair inside the loop would wait a second time on a CB that
      // was pushed once.
      Block *syncBlock = sync.transfers.front()->getBlock();
      if (llvm::any_of(sync.transfers, [&](Operation *transfer) {
            return transfer->getBlock() != syncBlock;
          })) {
        return generic.emitOpError()
               << "circular buffer (port " << port
               << ") is filled from distinct blocks and scalar-read; a single "
                  "wait/pop pair cannot match both cadences";
      }

      // Anchor each read at its ancestor in the transfer's block.
      SmallVector<Operation *> anchors;
      for (memref::LoadOp read : sync.reads) {
        Operation *anchor = ancestorInBlock(read, syncBlock);
        if (!anchor) {
          return generic.emitOpError()
                 << "circular buffer (port " << port
                 << ") is scalar-read outside the block that fills it; the "
                    "read would outlive its wait/pop pair";
        }
        anchors.push_back(anchor);
      }
      llvm::sort(anchors, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });

      Operation *first = anchors.front();
      Operation *last = anchors.back();
      // The wait has to observe the transfer's push, so every read must follow
      // the transfer. Data dependence normally guarantees this.
      if (first->isBeforeInBlock(sync.transfers.front())) {
        return generic.emitOpError()
               << "circular buffer (port " << port
               << ") is scalar-read before the transfer that fills it";
      }

      rewriter.setInsertionPoint(first);
      Value cbHandle = d2m::getOrCreateCB(rewriter, generic, dmBlock, port);
      auto waitOp = rewriter.create<WaitOp>(first->getLoc(), cbHandle);
      rewriter.setInsertionPointAfter(last);
      rewriter.create<PopOp>(last->getLoc(), cbHandle);

      // Rewiring the reads onto the wait result makes the dependency explicit:
      // the read can no longer be scheduled above the acquisition, and the
      // access lowers straight off the CB handle instead of a buffer address.
      for (OpOperand *use : sync.uses) {
        use->set(waitOp.getResult());
      }
    }

    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
