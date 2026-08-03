// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTCOMPUTECB
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Aliased remote ops carry no DMA transfer (the buffer is already in L1 via
// operand_alias); they live on the datamovement thread but impose a CB
// obligation on the compute thread (which must produce/consume the CB the DMA
// thread does not). Null-safe so they may be called on explicit-CB-form ops.
static bool isAliasedStore(RemoteStoreOp storeOp) {
  if (!storeOp.getLocalBuffer()) {
    return false;
  }
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(storeOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == storeOp.getMemref();
}

static bool isAliasedLoad(RemoteLoadOp loadOp) {
  if (!loadOp.getLocalBuffer()) {
    return false;
  }
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(loadOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == loadOp.getMemref();
}

// The CB port a shard DMA op synchronizes on, form-agnostic. Aliased
// remote_load/store ops are left in implicit form (they carry no transfer), and
// getCBPort() asserts on implicit form, so for those we derive the port from
// the CB the op references: the local buffer (for remote ops) / dst (for
// local_copy) operand is itself a d2m.generic operand, and its operand index is
// the CB port
// -- matching the operand index the compute side keys its CBs on.
// NOTE: duplicated in SplitThreads.cpp; hoist to a shared header later.
static unsigned getDMACBPort(GenericOp generic, Operation *op) {
  return llvm::TypeSwitch<Operation *, unsigned>(op)
      .Case<RemoteLoadOp, RemoteStoreOp>([&](auto dma) -> unsigned {
        if (dma.isExplicitCBForm()) {
          return dma.getCBPort();
        }
        return generic.getOperandIndex(dma.getLocalBuffer());
      })
      .Case<LocalCopyOp>([&](LocalCopyOp copy) -> unsigned {
        if (copy.isExplicitCBForm()) {
          return copy.getCBPort();
        }
        return generic.getOperandIndex(copy.getDst());
      })
      .Default([](Operation *) -> unsigned {
        llvm_unreachable("unexpected ShardDMAOpInterface op");
      });
}

// Trace `use` (an operand reading/writing a memref) up through
// collapse_shape/subview to a CB additional-arg of `generic`. Returns the CB
// value together with the operand that directly references it (so the caller
// can rewrite that operand to a wait/reserve result). Scratch and
// reduction-scaler buffers are not CBs.
static std::optional<std::pair<Value, OpOperand *>>
traceCBUse(OpOperand &startUse, GenericOp generic) {
  OpOperand *cbUse = &startUse;
  Value value = startUse.get();
  while (value) {
    if (mlir::isa<MemRefType>(value.getType()) &&
        llvm::is_contained(generic.getAdditionalArgs(), value)) {
      Operation *definingOp = value.getDefiningOp();
      if (definingOp && definingOp->getAttr("d2m.scratch_buffer")) {
        return std::nullopt;
      }
      if (utils::isReductionScalerBuffer(definingOp)) {
        return std::nullopt;
      }
      return std::make_pair(value, cbUse);
    }
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !generic->isProperAncestor(definingOp)) {
      return std::nullopt;
    }
    if (mlir::isa<memref::CollapseShapeOp, memref::SubViewOp>(definingOp)) {
      cbUse = &definingOp->getOpOperand(0);
      value = definingOp->getOperand(0);
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// Scratch and reduction-scaler buffers are compute-local L1 allocations, not
// CBs: no datamovement thread ever fills or drains them, so they take part in
// no wait/push handshake. `traceCBUse` already refuses them on the memref
// access path; synchronizable compute ops must refuse them on theirs too, or a
// consumer that names one as an operand (d2m.tile_topk_* naming its index
// buffer, say) gets a wait that nothing can ever satisfy.
static bool isComputeLocalBuffer(Value cb) {
  Operation *definingOp = cb.getDefiningOp();
  return definingOp && (definingOp->getAttr("d2m.scratch_buffer") ||
                        utils::isReductionScalerBuffer(definingOp));
}

// collapse_shape / subview compute an address into a CB; they do not wait
// for or produce tiles. They are often hoisted above the loop that uses them.
static bool isCBViewOp(Operation *op) {
  return mlir::isa<memref::CollapseShapeOp, memref::SubViewOp>(op);
}

// Find the "raw" compute spans in a compute block: the outermost ancestor of
// each compute op that is not itself a synchronizable op (i.e. a lowered
// scf.for nest or a bare TileMatmulBlock containing memref.load/store + tile
// ops). Interface compute ops (linalg.generic, tile_*_block) are excluded --
// they are synchronizable and handled directly via isConsumer/isProducer.
static SmallVector<Operation *> collectRawSpanAnchors(Block *computeBlock,
                                                      GenericOp generic) {
  DenseSet<Operation *> opsWithSynchronizableOps;
  computeBlock->walk([&](Operation *op) {
    if (dyn_cast<SynchronizableOpInterface>(op)) {
      opsWithSynchronizableOps.insert(op->getParentOp());
    }
  });

  SmallVector<Operation *> spans;
  DenseSet<Operation *> seen;
  computeBlock->walk([&](Operation *op) {
    if (!op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      return;
    }
    Operation *outermostOp = op;
    while (outermostOp->getParentOp() != generic.getOperation() &&
           !opsWithSynchronizableOps.contains(outermostOp->getParentOp())) {
      outermostOp = outermostOp->getParentOp();
    }
    if (!dyn_cast<SynchronizableOpInterface>(outermostOp) &&
        seen.insert(outermostOp).second) {
      spans.push_back(outermostOp);
    }
  });
  return spans;
}

// Number of scf.for ops between `op` and its enclosing generic region.
static unsigned forDepth(Operation *op) {
  unsigned d = 0;
  for (Operation *p = op->getParentOp(); p && !isa<GenericOp>(p);
       p = p->getParentOp()) {
    if (isa<scf::ForOp>(p)) {
      ++d;
    }
  }
  return d;
}

// Climb `useOwner` to the scf.for body at loop-depth `depth` in the compute
// region (depth 0 == the generic op compute block itself).
static Block *scopeBlockAtDepth(Operation *useOwner, unsigned depth,
                                Block *computeBlock) {
  SmallVector<Block *> forBodies; // outermost first
  for (Operation *p = useOwner; p && p->getBlock() != computeBlock;
       p = p->getParentOp()) {
    if (auto f = dyn_cast<scf::ForOp>(p->getParentOp())) {
      forBodies.push_back(f.getBody());
    }
  }
  std::reverse(forBodies.begin(), forBodies.end());
  return depth == 0 ? computeBlock : forBodies[depth - 1];
}

struct CBSync {
  SmallVector<Operation *> anchors;
  SmallVector<OpOperand *> uses;
};

// Compute-region block at loop-depth `depth` enclosing this CB's accesses.
// Prefer any anchor/use op nested at >= depth; hoisted loop-invariant view ops
// sit shallower and are skipped. Returns null if nothing reaches transfer depth
// (malformed CB — caller diagnoses).
static Block *computeScopeBlock(const CBSync &sync, unsigned depth,
                                Block *computeBlock) {
  if (depth == 0) {
    return computeBlock;
  }
  auto tryOp = [&](Operation *op) -> Block * {
    return op && forDepth(op) >= depth
               ? scopeBlockAtDepth(op, depth, computeBlock)
               : nullptr;
  };
  for (Operation *anchor : sync.anchors) {
    if (Block *b = tryOp(anchor)) {
      return b;
    }
  }
  for (OpOperand *use : sync.uses) {
    if (Block *b = tryOp(use->getOwner())) {
      return b;
    }
  }
  return nullptr;
}

// Insert the compute-side CB synchronization ops for a single (already split)
// compute block. Builds the ordered list of compute units (anchors) that
// consume/produce each CB from two sources -- interface compute ops
// (linalg.generic, tile_*_block) and raw lowered compute spans (traced via
// memref.load/store) -- and emits one wait/pop pair per consumed CB and one
// reserve/push pair per produced CB, threading the CB uses to the wait/reserve
// result. `aliasedLoadCBs`/`aliasedStoreCBs` are the CBs whose DMA-side
// producer/consumer is aliased (no transfer); compute supplies the missing
// half.
static LogicalResult insertCBOpsForCompute(
    Block *computeBlock, RewriterBase &rewriter,
    const llvm::SetVector<Value> &aliasedLoadCBs,
    const llvm::SetVector<Value> &aliasedStoreCBs,
    const llvm::DenseMap<unsigned, unsigned> &cbTransferDepth) {
  auto generic = cast<GenericOp>(computeBlock->getParentOp());

  llvm::MapVector<Value, CBSync> consumers, producers;
  auto add = [](llvm::MapVector<Value, CBSync> &map, Value cb,
                Operation *anchor, OpOperand *use) {
    CBSync &s = map[cb];
    if (s.anchors.empty() || s.anchors.back() != anchor) {
      s.anchors.push_back(anchor);
    }
    s.uses.push_back(use);
  };

  // (1) Raw compute spans: trace internal loads/stores back to CBs.
  SmallVector<Operation *> rawSpans =
      collectRawSpanAnchors(computeBlock, generic);
  for (Operation *span : rawSpans) {
    // Per CB: (access op, traced CB-view operand) pairs. The access op is the
    // in-loop memref.load/store or tile op -- it carries the true loop depth.
    // The coarse `span` may be the whole enclosing loop, which would hide the
    // per-access depth that computeScopeBlock needs.
    using Access = std::pair<Operation *, OpOperand *>;
    llvm::MapVector<Value, SmallVector<Access>> spanConsumed, spanProduced;
    span->walk([&](memref::LoadOp ld) {
      if (auto t = traceCBUse(ld->getOpOperand(0), generic)) {
        spanConsumed[t->first].push_back({ld, t->second});
      }
    });
    span->walk([&](memref::StoreOp st) {
      // memref.store operands: (value, memref, indices...).
      if (auto t = traceCBUse(st->getOpOperand(1), generic)) {
        spanProduced[t->first].push_back({st, t->second});
      }
    });
    span->walk([&](d2m::TileMatmulBlockOp mm) {
      for (OpOperand &operand : mm->getOpOperands()) {
        auto t = traceCBUse(operand, generic);
        if (!t) {
          continue;
        }
        if (operand.get() == mm.getOutput()) {
          spanProduced[t->first].push_back({mm, t->second});
        } else {
          spanConsumed[t->first].push_back({mm, t->second});
        }
      }
    });
    // Output reuse: a CB written by the span is not also treated as an input.
    for (auto &[cb, accesses] : spanProduced) {
      spanConsumed.erase(cb);
    }
    for (auto &[cb, accesses] : spanConsumed) {
      for (auto &[access, use] : accesses) {
        add(consumers, cb, access, use);
      }
    }
    for (auto &[cb, accesses] : spanProduced) {
      for (auto &[access, use] : accesses) {
        add(producers, cb, access, use);
      }
    }
  }

  // (1b) Accesses that no raw span covers. When the compute region is lowered
  // to a *flat* sequence -- memref.load/store and tile ops sitting directly in
  // the compute block, with no enclosing scf.for -- every span degenerates to a
  // single tile op, which holds no memref accesses of its own. The CB reads and
  // writes around those tile ops would then go unattributed, and the CB would
  // get no wait/pop: the datamovement thread keeps pushing into a CB nobody
  // drains and blocks forever on reserve. Attribute them here, anchored at the
  // access itself so the bracket still lands at the right loop depth.
  DenseSet<Operation *> coveredBySpan;
  for (Operation *span : rawSpans) {
    span->walk([&](Operation *op) { coveredBySpan.insert(op); });
  }
  {
    using Access = std::pair<Operation *, OpOperand *>;
    llvm::MapVector<Value, SmallVector<Access>> blockConsumed, blockProduced;
    computeBlock->walk([&](Operation *op) {
      if (coveredBySpan.contains(op)) {
        return;
      }
      if (auto ld = mlir::dyn_cast<memref::LoadOp>(op)) {
        if (auto t = traceCBUse(ld->getOpOperand(0), generic)) {
          blockConsumed[t->first].push_back({ld, t->second});
        }
      } else if (auto st = mlir::dyn_cast<memref::StoreOp>(op)) {
        // memref.store operands: (value, memref, indices...).
        if (auto t = traceCBUse(st->getOpOperand(1), generic)) {
          blockProduced[t->first].push_back({st, t->second});
        }
      }
    });
    // Output reuse: a CB written here is not also treated as an input.
    for (auto &[cb, accesses] : blockProduced) {
      blockConsumed.erase(cb);
    }
    for (auto &[cb, accesses] : blockConsumed) {
      for (auto &[access, use] : accesses) {
        add(consumers, cb, access, use);
      }
    }
    for (auto &[cb, accesses] : blockProduced) {
      for (auto &[access, use] : accesses) {
        add(producers, cb, access, use);
      }
    }
  }

  // (2) Interface compute ops not nested inside a raw span.
  DenseSet<Operation *> rawSpanSet(rawSpans.begin(), rawSpans.end());
  computeBlock->walk([&](Operation *op) {
    auto sync = dyn_cast<SynchronizableOpInterface>(op);
    if (!sync || op->hasTrait<D2MGenericRegionDatamovementOpTrait>()) {
      return;
    }
    for (Operation *p = op; p; p = p->getParentOp()) {
      if (rawSpanSet.contains(p)) {
        return;
      }
    }
    for (OpOperand &operand : op->getOpOperands()) {
      if (isComputeLocalBuffer(operand.get())) {
        continue;
      }
      if (sync.isConsumer(operand)) {
        add(consumers, operand.get(), op, &operand);
      }
      if (sync.isProducer(operand)) {
        add(producers, operand.get(), op, &operand);
      }
    }
  });

  // (3) Ops that reference a CB memref directly but are neither memref
  // accesses nor synchronizable compute ops -- e.g. d2m.fill_arange_tile, a
  // plain generic-region op whose `output` operand *is* the CB. They do not
  // decide whether the CB is produced or consumed, but their operand must still
  // be threaded to the CB handle (the wait/reserve result) and the bracket must
  // cover them. Left alone they address the raw allocation instead of the CB's
  // current slot, which is the wrong buffer as soon as the CB rotates.
  llvm::MapVector<Value, SmallVector<std::pair<Operation *, OpOperand *>>>
      directCBUses;
  computeBlock->walk([&](Operation *op) {
    if (mlir::isa<memref::LoadOp, memref::StoreOp, memref::CollapseShapeOp,
                  memref::SubViewOp, d2m::TileMatmulBlockOp>(op) ||
        mlir::isa<SynchronizableOpInterface>(op) ||
        mlir::isa<ShardDMAOpInterface>(op)) {
      return;
    }
    for (OpOperand &operand : op->getOpOperands()) {
      if (!mlir::isa<MemRefType>(operand.get().getType())) {
        continue;
      }
      if (auto t = traceCBUse(operand, generic)) {
        directCBUses[t->first].push_back({op, t->second});
      }
    }
  });
  for (auto &[cb, accesses] : directCBUses) {
    // Join whichever side already owns this CB; a CB only touched here is
    // written by compute and so belongs to the producers.
    llvm::MapVector<Value, CBSync> &map =
        consumers.count(cb) ? consumers : producers;
    for (auto &[access, use] : accesses) {
      add(map, cb, access, use);
    }
  }

  // Program-order index for sorting anchors (which may be discovered out of
  // order across the two sources above).
  DenseMap<Operation *, unsigned> order;
  {
    unsigned idx = 0;
    computeBlock->walk([&](Operation *op) { order[op] = idx++; });
  }
  auto byProgramOrder = [&](Operation *a, Operation *b) {
    return order[a] < order[b];
  };

  // The wait/reserve must dominate both the compute reads/writes (inside the
  // anchor) and the CB-view ops being rewritten to its result (e.g. a top-level
  // collapse_shape feeding a lowered loop nest). So bracket anchors *and* the
  // use-owners, lifted into the anchor's block.
  // Lift every anchor and use-owner into `block` (its direct-child ancestor).
  // Do NOT seed with the raw anchors: a raw span anchor may be the whole
  // enclosing loop (which lives *outside* `block`), and including it unlifted
  // would drag the bracket out of the transfer-depth block.
  //
  // includeViewOwners: producers need views in the bracket so they can be
  // rewritten next to the reserve. Self-produced consumers must not -- a
  // hoisted view would pull the wait before this region's own push.
  auto bracket =
      [&](CBSync &sync, Block *block,
          bool includeViewOwners) -> std::pair<Operation *, Operation *> {
    SmallVector<Operation *> boundary;
    auto lift = [&](Operation *op) {
      if (Operation *a = block->findAncestorOpInBlock(*op)) {
        boundary.push_back(a);
      }
    };
    for (Operation *anchor : sync.anchors) {
      lift(anchor);
      // A CB read into an SSA value (e.g. a hoisted, loop-invariant reduction
      // scaler load) stays live until that value's consumers finish, which may
      // sit deeper than the load itself. Extend the bracket over the
      // load-result users so the pop lands after the compute that reads the
      // value, not right after the load. Only memref.load reads produce such a
      // value; store/tile anchors are themselves the final CB access.
      if (isa<memref::LoadOp>(anchor)) {
        for (Operation *user : anchor->getUsers()) {
          lift(user);
        }
      }
    }
    for (OpOperand *use : sync.uses) {
      if (!includeViewOwners && isCBViewOp(use->getOwner())) {
        continue;
      }
      lift(use->getOwner());
    }

    if (boundary.empty()) {
      return {nullptr, nullptr};
    }
    llvm::sort(boundary, byProgramOrder);
    return {boundary.front(), boundary.back()};
  };

  // Consumers: wait once before the first consumer, pop once after the last.
  for (auto &[cb, sync] : consumers) {
    llvm::sort(sync.anchors, byProgramOrder);

    unsigned cbOperandIdx = generic.getOperandIndex(cb);
    unsigned depth =
        cbTransferDepth.lookup(cbOperandIdx); // 0 if aliased/no marker
    Block *anchorBlock = computeScopeBlock(sync, depth, computeBlock);
    if (!anchorBlock) {
      return generic.emitOpError()
             << "CB has consumers across distinct loop nests; cross-nest "
                "fan-out is not yet supported (would deadlock on a wait/pop "
                "cadence mismatch)";
    }

    // If this region also produces `cb`, ignore hoisted views so the wait
    // lands at the real read (after the push), not at the shared view.
    auto [first, last] = bracket(sync, anchorBlock,
                                 /*includeViewOwners=*/!producers.count(cb));
    if (!first) {
      std::tie(first, last) = bracket(sync, anchorBlock, true);
    }

    rewriter.setInsertionPoint(first);
    auto cbHandle =
        d2m::getOrCreateCB(rewriter, generic, computeBlock, cbOperandIdx);

    // Aliased remote_load producer has no DMA, so compute reserves+pushes.
    if (aliasedLoadCBs.contains(cb)) {
      rewriter.create<ReserveOp>(first->getLoc(), cbHandle);
      rewriter.create<PushOp>(first->getLoc(), cbHandle);
    }
    WaitOp waitOp = rewriter.create<WaitOp>(first->getLoc(), cbHandle);
    rewriter.setInsertionPointAfter(last);
    rewriter.create<PopOp>(last->getLoc(), cbHandle);

    auto afterWait = [&](Operation *op) {
      Operation *inBlock = waitOp->getBlock()->findAncestorOpInBlock(*op);
      return inBlock && waitOp->isBeforeInBlock(inBlock);
    };

    for (OpOperand *use : sync.uses) {
      Operation *owner = use->getOwner();
      if (isCBViewOp(owner) && !afterWait(owner)) {
        // The view sits before the wait (typically shared with the producer).
        // Rewriting it would break dominance; moving it would break the
        // producer. Clone after the wait and retarget only later uses.
        rewriter.setInsertionPointAfter(waitOp);
        Operation *clone = rewriter.clone(*owner);
        clone->setOperand(use->getOperandNumber(), waitOp.getResult());
        for (OpOperand &viewUse :
             llvm::make_early_inc_range(owner->getResult(0).getUses())) {
          if (viewUse.getOwner() != clone && afterWait(viewUse.getOwner())) {
            viewUse.set(clone->getResult(0));
          }
        }
        continue;
      }
      use->set(waitOp.getResult());
    }
  }

  // Producers: reserve once before the first producer, push once after the
  // last.
  for (auto &[cb, sync] : producers) {
    llvm::sort(sync.anchors, byProgramOrder);
    unsigned cbOperandIdx = generic.getOperandIndex(cb);
    unsigned depth =
        cbTransferDepth.lookup(cbOperandIdx); // 0 if aliased/no marker
    Block *anchorBlock = computeScopeBlock(sync, depth, computeBlock);
    if (!anchorBlock) {
      return generic.emitOpError()
             << "CB has producers across distinct loop nests; cross-nest "
                "fan-out is not yet supported (would deadlock on a "
                "reserve/push cadence mismatch)";
    }
    auto [first, last] = bracket(sync, anchorBlock, /*includeViewOwners=*/true);

    rewriter.setInsertionPoint(first);
    auto cbHandle =
        d2m::getOrCreateCB(rewriter, generic, computeBlock, cbOperandIdx);
    auto reserveOp = rewriter.create<ReserveOp>(first->getLoc(), cbHandle);
    rewriter.setInsertionPointAfter(last);
    rewriter.create<PushOp>(last->getLoc(), cbHandle);

    // Aliased remote_store consumer has no DMA, so compute waits+pops.
    if (aliasedStoreCBs.contains(cb)) {
      rewriter.create<WaitOp>(last->getLoc(), cbHandle);
      rewriter.create<PopOp>(last->getLoc(), cbHandle);
    }

    for (OpOperand *use : sync.uses) {
      Operation *owner = use->getOwner();
      if (owner->getBlock() != anchorBlock &&
          !anchorBlock->getParent()->isAncestor(owner->getParentRegion())) {
        owner->moveAfter(
            reserveOp); // sink the hoisted view next to the reserve
      }
      use->set(reserveOp.getResult());
    }
  }

  // (3) Standalone aliased CBs: an aliased CB with no compute-side peer is
  // synchronized against the DMA thread, which fills/drains it. Place the
  // bookkeeping pair at the end of the compute block.
  auto setEndInsertionPoint = [&]() {
    if (computeBlock->mightHaveTerminator()) {
      rewriter.setInsertionPoint(computeBlock->getTerminator());
    } else {
      rewriter.setInsertionPointToEnd(computeBlock);
    }
  };
  for (Value cb : aliasedLoadCBs) {
    if (consumers.count(cb)) {
      continue;
    }
    setEndInsertionPoint();
    auto cbHandle = d2m::getOrCreateCB(rewriter, generic, computeBlock,
                                       generic.getOperandIndex(cb));
    rewriter.create<ReserveOp>(generic.getLoc(), cbHandle);
    rewriter.create<PushOp>(generic.getLoc(), cbHandle);
  }
  for (Value cb : aliasedStoreCBs) {
    if (producers.count(cb)) {
      continue;
    }
    setEndInsertionPoint();
    auto cbHandle = d2m::getOrCreateCB(rewriter, generic, computeBlock,
                                       generic.getOperandIndex(cb));
    rewriter.create<WaitOp>(generic.getLoc(), cbHandle);
    rewriter.create<PopOp>(generic.getLoc(), cbHandle);
  }

  return success();
}

// Erase the aliased remote markers from the DMA thread (their CB obligations
// are now satisfied on the compute side) and drop any trivially dead leftovers.
static void eraseAliasedMarkers(RewriterBase &rewriter, Block *dmBlock) {
  SmallVector<Operation *> markers;
  dmBlock->walk([&](Operation *op) {
    if (auto load = mlir::dyn_cast<RemoteLoadOp>(op);
        load && isAliasedLoad(load)) {
      markers.push_back(op);
    } else if (auto store = mlir::dyn_cast<RemoteStoreOp>(op);
               store && isAliasedStore(store)) {
      markers.push_back(op);
    }
  });
  for (Operation *op : markers) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  // Drop trivially-dead leftovers (e.g. the now-unused core_index feeders and
  // empty loop nests). Erase one at a time so erasing a dead parent never
  // dangles a dead child collected earlier.
  bool changed = true;
  while (changed) {
    changed = false;
    Operation *dead = nullptr;
    dmBlock->walk([&](Operation *op) {
      if (isOpTriviallyDead(op)) {
        dead = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (dead) {
      rewriter.eraseOp(dead);
      changed = true;
    }
  }
}

class D2MInsertComputeCB
    : public impl::D2MInsertComputeCBBase<D2MInsertComputeCB> {
public:
  using impl::D2MInsertComputeCBBase<
      D2MInsertComputeCB>::D2MInsertComputeCBBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    SmallVector<GenericOp> generics;
    getOperation().walk([&](GenericOp generic) {
      // Only the split form (datamovement + compute) needs compute-side CB
      // synchronization inserted.
      if (generic.getNumRegions() == 2 &&
          generic.getRegionThreadType(0) == ThreadType::Datamovement &&
          generic.getRegionThreadType(1) == ThreadType::Compute) {
        generics.push_back(generic);
      }
    });

    for (GenericOp generic : generics) {
      Block *dmBlock = &generic.getRegion(0).front();
      Block *computeBlock = &generic.getRegion(1).front();

      // The CBs whose DMA-side half is aliased (no transfer); compute supplies
      // the missing producer/consumer. The aliased markers live on the DMA
      // thread and reference the same CB values the compute region uses.
      llvm::SetVector<Value> aliasedLoadCBs, aliasedStoreCBs;
      dmBlock->walk([&](Operation *op) {
        if (auto load = mlir::dyn_cast<RemoteLoadOp>(op);
            load && isAliasedLoad(load)) {
          aliasedLoadCBs.insert(load.getLocalBuffer());
        } else if (auto store = mlir::dyn_cast<RemoteStoreOp>(op);
                   store && isAliasedStore(store)) {
          aliasedStoreCBs.insert(store.getLocalBuffer());
        }
      });

      llvm::DenseMap<unsigned, unsigned>
          cbTransferDepth; // generic operand index --> loop nest depth of its
                           // DMA marker
      dmBlock->walk([&](ShardDMAOpInterface dma) {
        cbTransferDepth[getDMACBPort(generic, dma.getOperation())] =
            forDepth(dma.getOperation());
      });
      if (failed(insertCBOpsForCompute(computeBlock, rewriter, aliasedLoadCBs,
                                       aliasedStoreCBs, cbTransferDepth))) {
        signalPassFailure();
        return;
      }
      eraseAliasedMarkers(rewriter, dmBlock);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
