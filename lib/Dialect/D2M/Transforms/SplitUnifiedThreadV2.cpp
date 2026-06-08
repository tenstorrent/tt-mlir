// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// d2m-split-unified-thread-v2: experimental rewrite of
// d2m-split-unified-thread, selectable via the `use-split-unified-thread-v2`
// option on d2m-be-pipeline for A/B comparison.
//
// This file is initially a verbatim fork of SplitUnifiedThread.cpp (symbols
// renamed to *V2) so the toggle infrastructure can be exercised with
// behavior identical to the legacy pass. The dataflow-model rewrite -- which
// classifies buffer accesses by thread and synchronizes only cross-thread
// (compute<->DM) edges instead of assuming a single producer/consumer per CB
// -- will replace the forked internals incrementally.

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREADV2
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool isAliasedStore(RemoteStoreOp storeOp) {
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(storeOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == storeOp.getMemref();
}

bool isAliasedLoad(RemoteLoadOp loadOp) {
  auto operandAliasOp =
      mlir::dyn_cast<OperandAliasOp>(loadOp.getLocalBuffer().getDefiningOp());
  return operandAliasOp && operandAliasOp.getMemref() == loadOp.getMemref();
}

Value traceComputeMemrefToCB(Value value, GenericOp genericOp) {
  while (value) {
    // Check if its a cb (hoisted generic arg with cb layout attr).
    if (auto memrefType = mlir::dyn_cast<MemRefType>(value.getType())) {
      if (llvm::find(genericOp.getAdditionalArgs(), value) !=
          genericOp.getAdditionalArgs().end()) {
        // Skip scratch buffers.
        Operation *definingOp = value.getDefiningOp();
        if (definingOp && definingOp->getAttr("d2m.scratch_buffer")) {
          return nullptr;
        }
        return value;
      }
    }

    // If we are no longer inside the generic or have reached the root, stop
    // tracing and return nullptr.
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !genericOp->isProperAncestor(definingOp)) {
      return nullptr;
    }

    // Otherwise keep tracing up the chain, if we reach an op we don't support,
    // stop tracing and return nullptr.
    if (auto collapseOp = mlir::dyn_cast<memref::CollapseShapeOp>(definingOp)) {
      value = collapseOp.getSrc();
      continue;
    }
    if (auto subviewOp = mlir::dyn_cast<memref::SubViewOp>(definingOp)) {
      value = subviewOp.getSource();
      continue;
    }
    if (auto castOp = mlir::dyn_cast<memref::CastOp>(definingOp)) {
      value = castOp.getSource();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

[[maybe_unused]] LogicalResult wrapComputeInSynchronizedRegion(GenericOp genericOp,
                                              PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Collect ops that directly contain a SynchronizableOpInterface op. Each
  // such parent delimits an independent synchronized scope at its own nesting
  // level (e.g. remote_loads in an accumulator loop bound the matmul compute,
  // while a remote_store in the enclosing loop bounds the epilogue compute).
  DenseSet<Operation *> opsWithSynchronizableOps;
  genericOp.getRegion(0).walk([&](Operation *op) {
    if (dyn_cast<SynchronizableOpInterface>(op)) {
      opsWithSynchronizableOps.insert(op->getParentOp());
    }
  });

  // Early exit if there are no synchronizable ops.
  if (opsWithSynchronizableOps.empty()) {
    return success();
  }

  DenseSet<Operation *> outermostOps;
  bool walkFailed = false;
  genericOp.getRegion(0).walk([&](Operation *op) {
    if (!op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      return WalkResult::advance();
    }

    // Walk up loops until we reach the generic op as a parent, or a op
    // that directly contains a SynchronizableOpInterface op.
    Operation *outermostOp = op;
    while (outermostOp->getParentOp() != genericOp.getOperation() &&
           !opsWithSynchronizableOps.contains(outermostOp->getParentOp())) {
      outermostOp = outermostOp->getParentOp();
      if (!mlir::isa<scf::ForOp>(outermostOp) &&
          !mlir::isa<linalg::GenericOp>(outermostOp)) {
        outermostOp->emitOpError(
            "Parent ops containing compute ops must be scf.for or "
            "linalg.generic");
        walkFailed = true;
        return WalkResult::interrupt();
      }
    }

    // Skip synchronizable ops (e.g. TileTilize/UntilizeBlockOp) not yet
    // lowered to non-synchronized form.
    if (!dyn_cast<SynchronizableOpInterface>(outermostOp)) {
      outermostOps.insert(outermostOp);
    }

    return WalkResult::advance();
  });

  if (walkFailed) {
    return failure();
  }

  // Expand each compute region outward until it hits a boundary. Three stop
  // conditions: (1) a SynchronizableOpInterface op, which bounds the wrap;
  // (2) a sibling in `opsWithSynchronizableOps`, whose own deeper scope must
  // not be nested inside this one; (3) an op whose result is used past the
  // wrap — wrapInSynchronizedRegion erases non-pure ops in [start, end), so an
  // outside user would dangle. Check users unconditionally: a Pure op like
  // arith.index_cast is treated as non-pure once a faulting ancestor (e.g.
  // arith.divsi) enters its operand chain.
  auto hasUserOutsideRange = [](Operation *op, Block::iterator s,
                                Block::iterator e) {
    Block *block = op->getBlock();
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        Operation *userInBlock = user;
        while (userInBlock && userInBlock->getBlock() != block) {
          userInBlock = userInBlock->getParentOp();
        }
        if (!userInBlock) {
          return true;
        }
        bool inRange = false;
        for (auto it = s; it != e; ++it) {
          if (&*it == userInBlock) {
            inRange = true;
            break;
          }
        }
        if (!inRange) {
          return true;
        }
      }
    }
    return false;
  };

  SmallVector<std::pair<Block::iterator, Block::iterator>> computeRegions;
  while (!outermostOps.empty()) {
    Operation *outermostOp = *outermostOps.begin();
    outermostOps.erase(outermostOp);
    Block::iterator start = outermostOp->getIterator();
    Block::iterator end = outermostOp->getIterator();

    // Expand above.
    while (start != outermostOp->getBlock()->begin()) {
      Operation *prev = &*std::prev(start);
      if (dyn_cast<SynchronizableOpInterface>(prev)) {
        break;
      }
      if (opsWithSynchronizableOps.contains(prev)) {
        break;
      }
      if (hasUserOutsideRange(prev, std::prev(start), std::next(end))) {
        break;
      }
      start--;
      if (outermostOps.contains(&*start)) {
        outermostOps.erase(&*start);
      }
    }

    // Expand below.
    while (std::next(end) != outermostOp->getBlock()->end()) {
      Operation *next = &*std::next(end);
      if (next->hasTrait<OpTrait::IsTerminator>()) {
        break;
      }
      if (dyn_cast<SynchronizableOpInterface>(next)) {
        break;
      }
      if (opsWithSynchronizableOps.contains(next)) {
        break;
      }
      if (hasUserOutsideRange(next, start, std::next(end, 2))) {
        break;
      }
      end++;
      if (outermostOps.contains(&*end)) {
        outermostOps.erase(&*end);
      }
    }

    computeRegions.push_back({start, std::next(end)});
  }

  // Merge overlapping compute regions and fixpoint-expand each so
  // wrapInSynchronizedRegion's precondition holds: no op it erases may have
  // result users outside [start, end).
  //
  // Overlaps arise when hasUserOutsideRange splits an interdependent value
  // chain across per-outermostOp expansions. E.g. acquire_dst and tile_add in
  // a flat block yield overlapping [A, B) and [B-1, C); wrapping the first
  // erases the shared op B-1, leaving the second's start iterator dangling
  // (ilist sentinel crash).
  if (!computeRegions.empty()) {
    // Purity cache mirroring isPurelyDerivedOp in wrapInSynchronizedRegion: an
    // op is purely derived if it is pure and all operand-defining ops are too.
    // Non-purely-derived ops get erased there, so their results must not escape
    // the region. Block-agnostic; shared across all per-block passes below.
    DenseMap<Operation *, bool> pureCache;
    auto isPurelyDerived = [&](auto &self, Operation *op) -> bool {
      auto it = pureCache.find(op);
      if (it != pureCache.end()) {
        return it->second;
      }
      bool result = mlir::isPure(op);
      if (result) {
        for (Value operand : op->getOperands()) {
          if (Operation *defOp = operand.getDefiningOp()) {
            if (!self(self, defOp)) {
              result = false;
              break;
            }
          }
        }
      }
      return (pureCache[op] = result);
    };

    // Group regions by block: multi-level scopes can produce regions in
    // different blocks (e.g. inner K-loop body vs. enclosing block). The
    // insertion-order vector keeps processing deterministic.
    SmallVector<Block *> blockOrder;
    DenseMap<Block *, SmallVector<std::pair<Block::iterator, Block::iterator>>>
        regionsByBlock;
    for (auto [s, e] : computeRegions) {
      Block *blk = s->getBlock();
      if (!regionsByBlock.count(blk)) {
        blockOrder.push_back(blk);
      }
      regionsByBlock[blk].push_back({s, e});
    }

    computeRegions.clear();
    for (Block *block : blockOrder) {
      auto &regions = regionsByBlock[block];

      // Position map: op → sequential index within the block.
      DenseMap<Operation *, unsigned> posMap;
      SmallVector<Block::iterator> posToIter;
      {
        unsigned pos = 0;
        for (auto it = block->begin(); it != block->end(); ++it, ++pos) {
          posMap[&*it] = pos;
          posToIter.push_back(it);
        }
      }

      // Convert regions to (startPos, endPosInclusive) pairs and sort.
      SmallVector<std::pair<unsigned, unsigned>> posRegions;
      posRegions.reserve(regions.size());
      for (auto [s, e] : regions) {
        posRegions.push_back({posMap[&*s], posMap[&*std::prev(e)]});
      }
      llvm::sort(posRegions, [](const auto &a, const auto &b) {
        return a.first < b.first;
      });

      // Merge overlapping intervals.
      SmallVector<std::pair<unsigned, unsigned>> merged;
      for (auto [s, e] : posRegions) {
        if (!merged.empty() && s <= merged.back().second) {
          merged.back().second = std::max(merged.back().second, e);
        } else {
          merged.push_back({s, e});
        }
      }

      // Fixpoint-expand each merged region in both directions, stopping at
      // SynchronizableOpInterface boundaries.
      //
      // Downward: a non-pure op in range has result users past endPos that
      // would dangle once wrapInSynchronizedRegion erases it — pull them in.
      // Upward: a non-pure op has an operand defined by a non-pure op before
      // startPos — pull it in so CB-load detection (which walks [start, end))
      // can see the CB operand feeding the region.
      for (auto &[startPos, endPos] : merged) {
        bool changed = true;
        while (changed) {
          changed = false;
          for (unsigned p = startPos; p <= endPos; ++p) {
            Operation *op = &*posToIter[p];
            if (isPurelyDerived(isPurelyDerived, op)) {
              continue;
            }
            // Downward: result users outside range.
            for (Value result : op->getResults()) {
              for (Operation *user : result.getUsers()) {
                Operation *userInBlock = user;
                while (userInBlock && userInBlock->getBlock() != block) {
                  userInBlock = userInBlock->getParentOp();
                }
                if (!userInBlock) {
                  continue;
                }
                auto posIt = posMap.find(userInBlock);
                if (posIt == posMap.end()) {
                  continue;
                }
                unsigned userPos = posIt->second;
                if (userPos <= endPos) {
                  continue;
                }
                // Don't expand past a SynchronizableOpInterface boundary.
                bool blocked = false;
                for (unsigned q = endPos + 1; q <= userPos; ++q) {
                  if (isa<SynchronizableOpInterface>(&*posToIter[q])) {
                    blocked = true;
                    break;
                  }
                }
                if (!blocked) {
                  endPos = userPos;
                  changed = true;
                }
              }
            }
            // Upward: operands defined by non-pure ops before startPos.
            for (Value operand : op->getOperands()) {
              Operation *defOp = operand.getDefiningOp();
              if (!defOp) {
                continue;
              }
              auto posIt = posMap.find(defOp);
              if (posIt == posMap.end()) {
                continue;
              }
              unsigned defPos = posIt->second;
              if (defPos >= startPos) {
                continue; // already in range
              }
              if (isPurelyDerived(isPurelyDerived, defOp)) {
                continue; // pure ops are not erased; no need to pull them in
              }
              // Don't expand past a SynchronizableOpInterface boundary.
              bool blocked = false;
              for (unsigned q = defPos; q < startPos; ++q) {
                if (isa<SynchronizableOpInterface>(&*posToIter[q])) {
                  blocked = true;
                  break;
                }
              }
              if (!blocked) {
                startPos = defPos;
                changed = true;
              }
            }
          }
        }
      }

      // Append this block's merged+expanded regions to computeRegions.
      for (auto [s, e] : merged) {
        computeRegions.push_back({posToIter[s], std::next(posToIter[e])});
      }
    }
  }

  for (auto [start, end] : computeRegions) {

    DenseSet<Value> loadedCBOperands;
    DenseSet<Value> storedCBOperands;

    // For memref load and stores, trace to cb operand to get producers and
    // consumers for syncrhonized region.
    for (Operation &op : llvm::make_range(start, end)) {
      // For load trace src memref up to defining op and check if its a cb (as
      // opposed to dst).
      op.walk([&](memref::LoadOp loadOp) {
        Value cb = traceComputeMemrefToCB(loadOp.getMemref(), genericOp);
        if (cb) {
          loadedCBOperands.insert(cb);
        }
        return WalkResult::advance();
      });

      // For store trace dst memref up to defining op and check if its a cb (as
      // opposed to dst)
      op.walk([&](memref::StoreOp storeOp) {
        Value cb = traceComputeMemrefToCB(storeOp.getMemref(), genericOp);
        if (cb) {
          storedCBOperands.insert(cb);
        }
        return WalkResult::advance();
      });

      // TileMatmulBlockOp uses CBs directly without load/store.
      op.walk([&](d2m::TileMatmulBlockOp tileMatmulBlockOp) {
        Value cbA = traceComputeMemrefToCB(tileMatmulBlockOp.getA(), genericOp);
        Value cbB = traceComputeMemrefToCB(tileMatmulBlockOp.getB(), genericOp);
        Value cbOutput =
            traceComputeMemrefToCB(tileMatmulBlockOp.getOutput(), genericOp);
        if (cbA) {
          loadedCBOperands.insert(cbA);
        }
        if (cbB) {
          loadedCBOperands.insert(cbB);
        }
        if (cbOutput) {
          storedCBOperands.insert(cbOutput);
        }
        return WalkResult::advance();
      });
    }

    // Remove allocs in load that are also in store since this is output cb
    // reuse and not an actual input.
    for (Value storedCBOperand : storedCBOperands) {
      if (loadedCBOperands.contains(storedCBOperand)) {
        loadedCBOperands.erase(storedCBOperand);
      }
    }

    utils::wrapInSynchronizedRegion(
        rewriter, start, end,
        SmallVector<Value>(loadedCBOperands.begin(), loadedCBOperands.end()),
        SmallVector<Value>(storedCBOperands.begin(), storedCBOperands.end()));
  }

  return success();
}

// From cb usage info, check for load-store pairs and insert aliased cb ops for
// alias side.
[[maybe_unused]] static LogicalResult processSharedBufferPairs(
    Block *computeBlock, PatternRewriter &rewriter,
    llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  for (auto [localBuffer, usageInfo] : cbUsageInfo) {
    assert(usageInfo.producers.size() == 1 && usageInfo.consumers.size() == 1 &&
           "Expected exactly one producer and one consumer for CB");
    auto *producer = usageInfo.producers.front();
    auto *consumer = usageInfo.consumers.front();

    // Insert compute-side CB ops for the aliased half of the pair.
    // The streaming half stays as a remote_load/store for DMA.
    if (mlir::isa<RemoteLoadOp>(producer) &&
        mlir::isa<RemoteStoreOp>(consumer) &&
        isAliasedStore(mlir::cast<RemoteStoreOp>(consumer))) {
      Location loc = producer->getLoc();
      unsigned cbOperandIdx =
          producer->getParentOfType<GenericOp>().getOperandIndex(localBuffer);
      // Set insertion point before consumer so GetCBOp dominates WaitOp/PopOp.
      rewriter.setInsertionPoint(consumer);
      auto cb =
          d2m::getOrCreateCB(rewriter, producer->getParentOfType<GenericOp>(),
                             computeBlock, cbOperandIdx);
      rewriter.create<WaitOp>(loc, cb);
      rewriter.create<PopOp>(loc, cb);
    } else if (mlir::isa<RemoteLoadOp>(producer) &&
               mlir::isa<RemoteStoreOp>(consumer) &&
               isAliasedLoad(mlir::cast<RemoteLoadOp>(producer))) {
      Location loc = consumer->getLoc();
      unsigned cbOperandIdx =
          consumer->getParentOfType<GenericOp>().getOperandIndex(localBuffer);
      // Set insertion point before producer so GetCBOp dominates
      // ReserveOp/PushOp.
      rewriter.setInsertionPoint(producer);
      auto cb =
          d2m::getOrCreateCB(rewriter, consumer->getParentOfType<GenericOp>(),
                             computeBlock, cbOperandIdx);
      rewriter.create<ReserveOp>(loc, cb);
      rewriter.create<PushOp>(loc, cb);
    }
    // Else if both sides are streaming/need DMA, let the DM thread handle
    // everything.
  }
  return success();
}

[[maybe_unused]] static LogicalResult
insertCBOpsForCompute(Block *computeBlock, PatternRewriter &rewriter,
                      llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  SmallVector<RemoteLoadOp> loads;

  computeBlock->walk([&](Operation *op) {
    if (dyn_cast<SynchronizableOpInterface>(op) &&
        !op->hasTrait<D2MGenericRegionDatamovementOpTrait>()) {
      auto synchronizedOp = mlir::cast<SynchronizableOpInterface>(op);

      // CB ops must fire once per DMA partner activation, not once per compute
      // iteration. When the partner sits in an outer block (e.g. a matmul
      // accumulator: remote_loads in the K loop, remote_store outside it),
      // wrapping `synchronizedOp` would emit N pairs against one DMA op. Climb
      // synchronizedOp's ancestors until one shares the partner's block; that
      // ancestor anchors the CB ops.
      auto anchorForPartner = [&](Operation *partner) -> Operation * {
        Block *targetBlock = partner->getBlock();
        Operation *anchor = synchronizedOp;
        while (anchor->getBlock() != targetBlock) {
          Operation *parent = anchor->getParentOp();
          assert(parent &&
                 "Unexpected ancestor-less op while looking for anchor");
          anchor = parent;
        }
        return anchor;
      };

      // get consumers and insert wait+pop
      for (auto &operand : synchronizedOp->getOpOperands()) {
        if (synchronizedOp.isConsumer(operand)) {
          Location loc = synchronizedOp.getLoc();
          Value localBuffer = operand.get();
          unsigned cbOperandIdx =
              synchronizedOp->getParentOfType<GenericOp>().getOperandIndex(
                  localBuffer);

          // get the associated producer for this operand
          // Assumes only one producer for this local buffer
          assert(cbUsageInfo[localBuffer].producers.size() == 1 &&
                 cbUsageInfo[localBuffer].consumers.size() == 1 &&
                 "Expected exactly one producer and one consumer for CB");
          auto *associatedProducer = cbUsageInfo[localBuffer].producers.front();
          Operation *anchor = anchorForPartner(associatedProducer);
          rewriter.setInsertionPoint(anchor);
          auto cb = d2m::getOrCreateCB(
              rewriter, synchronizedOp->getParentOfType<GenericOp>(),
              computeBlock, cbOperandIdx);

          if (mlir::isa<RemoteLoadOp>(associatedProducer) &&
              isAliasedLoad(mlir::cast<RemoteLoadOp>(associatedProducer))) {
            rewriter.create<ReserveOp>(loc, cb);
            rewriter.create<PushOp>(loc, cb);
          }
          WaitOp waitOp = rewriter.create<WaitOp>(loc, cb);
          rewriter.setInsertionPointAfter(anchor);
          rewriter.create<PopOp>(loc, cb);

          // Replace uses of the local buffer in compute consumer
          localBuffer.replaceUsesWithIf(
              waitOp.getResult(),
              [&](OpOperand &use) { return use.getOwner() == synchronizedOp; });
        }
      }

      // Get producers and insert reserve+push.
      for (auto &operand : synchronizedOp->getOpOperands()) {
        if (synchronizedOp.isProducer(operand)) {
          Location loc = synchronizedOp.getLoc();
          Value localBuffer = operand.get();
          unsigned cbOperandIdx =
              synchronizedOp->getParentOfType<GenericOp>().getOperandIndex(
                  localBuffer);

          // Get the associated consumer for this operand.
          // Assumes only one consumer for this local buffer
          assert(cbUsageInfo[localBuffer].consumers.size() == 1 &&
                 cbUsageInfo[localBuffer].producers.size() == 1 &&
                 "Expected exactly one producer and one consumer for CB");
          auto *associatedConsumer = cbUsageInfo[localBuffer].consumers.front();
          Operation *anchor = anchorForPartner(associatedConsumer);
          rewriter.setInsertionPoint(anchor);
          auto cb = d2m::getOrCreateCB(
              rewriter, synchronizedOp->getParentOfType<GenericOp>(),
              computeBlock, cbOperandIdx);
          auto reserveOp = rewriter.create<ReserveOp>(loc, cb);
          rewriter.setInsertionPointAfter(anchor);
          rewriter.create<PushOp>(loc, cb);
          if (mlir::isa<RemoteStoreOp>(associatedConsumer) &&
              isAliasedStore(mlir::cast<RemoteStoreOp>(associatedConsumer))) {
            rewriter.create<WaitOp>(loc, cb);
            rewriter.create<PopOp>(loc, cb);
          }

          // Replace uses of the local buffer in compute consumer
          localBuffer.replaceUsesWithIf(
              reserveOp.getResult(),
              [&](OpOperand &use) { return use.getOwner() == synchronizedOp; });
        }
      }
    }

    return WalkResult::advance();
  });

  return success();
}

// ---------------------------------------------------------------------------
// DMA thread: convert implicit-form ops to explicit CB form
// ---------------------------------------------------------------------------

// Erase aliased load and store ops (no DMA needed).
[[maybe_unused]] static LogicalResult eraseAliasedLoadStoreOps(
    PatternRewriter &rewriter,
    llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  for (auto [localBuffer, usageInfo] : cbUsageInfo) {
    assert(usageInfo.producers.size() == 1 && usageInfo.consumers.size() == 1 &&
           "Expected exactly one producer and one consumer for CB");
    auto *producer = usageInfo.producers.front();
    auto *consumer = usageInfo.consumers.front();

    if (mlir::isa<RemoteStoreOp>(consumer) &&
        isAliasedStore(mlir::cast<RemoteStoreOp>(consumer))) {
      rewriter.eraseOp(consumer);
    } else if (mlir::isa<RemoteLoadOp>(producer) &&
               isAliasedLoad(mlir::cast<RemoteLoadOp>(producer))) {
      rewriter.eraseOp(producer);
    }
  }
  return success();
}

// Convert remote_load/store to explicit CB form in the DMA thread.
// Aliased ops are collected for deferred erasure (no DMA needed). Shared
// buffer pairs use the output operand's CB for both ops.
static LogicalResult convertDMAToExplicitCBForm(Block *dmBlock,
                                                PatternRewriter &rewriter) {
  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  SmallVector<LocalCopyOp> localCopies;
  dmBlock->walk([&](RemoteLoadOp op) { loads.push_back(op); });
  dmBlock->walk([&](RemoteStoreOp op) { stores.push_back(op); });
  dmBlock->walk([&](LocalCopyOp op) { localCopies.push_back(op); });

  for (RemoteLoadOp loadOp : loads) {
    if (loadOp.isExplicitCBForm()) {
      continue;
    }

    Value localBuffer = loadOp.getLocalBuffer();
    unsigned cbOperandIdx =
        loadOp->getParentOfType<GenericOp>().getOperandIndex(localBuffer);

    rewriter.setInsertionPoint(loadOp);
    auto cb = d2m::getOrCreateCB(rewriter, loadOp->getParentOfType<GenericOp>(),
                                 dmBlock, cbOperandIdx);
    auto newLoad = rewriter.create<RemoteLoadOp>(
        loadOp.getLoc(), loadOp.getMemref(), loadOp.getIndices(), cb,
        loadOp.getMcastStartIndex(), loadOp.getMcastShape());
    // Preserve preallocated semaphore indices set by
    // D2MPreallocateMcastSemaphores (needed by LowerLoadStoreOpsToDMA).
    if (auto semAttr = loadOp->getAttr("preallocated_semaphores")) {
      newLoad->setAttr("preallocated_semaphores", semAttr);
    }

    loadOp->dropAllUses();
    rewriter.eraseOp(loadOp);
  }

  for (RemoteStoreOp storeOp : stores) {
    if (storeOp.isExplicitCBForm()) {
      continue;
    }

    Value localBuffer = storeOp.getLocalBuffer();
    assert(localBuffer && "could not find associated local buffer for store");
    unsigned cbOperandIdx =
        storeOp->getParentOfType<GenericOp>().getOperandIndex(localBuffer);

    rewriter.setInsertionPoint(storeOp);
    auto cb = d2m::getOrCreateCB(
        rewriter, storeOp->getParentOfType<GenericOp>(), dmBlock, cbOperandIdx);
    rewriter.create<RemoteStoreOp>(
        storeOp.getLoc(), storeOp.getMemref(), storeOp.getIndices(), cb,
        storeOp.getStartDevice(), storeOp.getDeviceMcastShape(),
        storeOp.getSemaphore(), storeOp.getSemaphoreIndices());
    storeOp->dropAllUses();
    rewriter.eraseOp(storeOp);
  }

  // Convert implicit-form local_copy ops to explicit CB form.
  for (LocalCopyOp copyOp : localCopies) {
    if (copyOp.isExplicitCBForm()) {
      continue;
    }

    Location loc = copyOp.getLoc();

    unsigned srcCbOperandIdx =
        copyOp->getParentOfType<GenericOp>().getOperandIndex(copyOp.getSrc());
    auto srcCb =
        d2m::getOrCreateCB(rewriter, copyOp->getParentOfType<GenericOp>(),
                           dmBlock, srcCbOperandIdx);
    unsigned dstCbOperandIdx =
        copyOp->getParentOfType<GenericOp>().getOperandIndex(copyOp.getDst());
    auto dstCb =
        d2m::getOrCreateCB(rewriter, copyOp->getParentOfType<GenericOp>(),
                           dmBlock, dstCbOperandIdx);

    // Create explicit CB form: local_copy %srcCb into %dstCb.
    rewriter.setInsertionPoint(copyOp);
    rewriter.create<LocalCopyOp>(loc, TypeRange{}, /*src=*/Value{},
                                 /*dst=*/Value{}, srcCb, dstCb,
                                 copyOp.getIndexingMaps());
    copyOp->dropAllUses();
    rewriter.eraseOp(copyOp);
  }

  return success();
}

// ---------------------------------------------------------------------------
// Dead-op cleanup
// ---------------------------------------------------------------------------

// Recursively collect ops to erase from a block based on thread type.
static void collectOpsToErase(Block *block, DenseSet<Operation *> &eraseSet,
                              bool isDatamovementThread) {
  for (Operation &op : block->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      collectOpsToErase(forOp.getBody(), eraseSet, isDatamovementThread);
      continue;
    }

    bool isDMAOp = isa<ShardDMAOpInterface, DeviceSynchronizeOp>(&op);
    bool isReplicated = isa<SemaphoreWaitOp>(&op);

    if (isDatamovementThread && !isDMAOp && !isReplicated) {
      eraseSet.insert(&op);
    } else if (!isDatamovementThread && isDMAOp) {
      eraseSet.insert(&op);
    }
  }
}

// Iteratively erase unused ops from a block until fixpoint.
static void eraseDeadOps(PatternRewriter &rewriter, Block *block,
                         bool isDatamovementThread) {
  bool changed = true;
  while (changed) {
    changed = false;
    DenseSet<Operation *> eraseSet;
    collectOpsToErase(block, eraseSet, isDatamovementThread);

    SmallVector<Operation *> toErase;
    for (Operation *op : eraseSet) {
      if (op->use_empty()) {
        toErase.push_back(op);
        changed = true;
      }
    }
    for (Operation *op : llvm::reverse(toErase)) {
      rewriter.eraseOp(op);
    }
  }
}

// Erase collected ops. All legitimate uses must have been replaced or dropped
// before adding ops to this set, so we just drop any stale uses and erase.
static void eraseDMAOpsInComputeBlock(PatternRewriter &rewriter,
                                      Block *computeBlock) {
  DenseSet<Operation *> ops;
  computeBlock->walk([&](Operation *op) {
    if (isa<RemoteLoadOp, RemoteStoreOp, LocalCopyOp>(op)) {
      ops.insert(op);
    }
    return WalkResult::advance();
  });
  for (Operation *op : ops) {
    op->dropAllUses();
  }
  for (Operation *op : ops) {
    rewriter.eraseOp(op);
  }
  ops.clear();
}

// ---------------------------------------------------------------------------
// V2 compute-side CB insertion (dataflow model)
// ---------------------------------------------------------------------------
//
// Instead of wrapping compute in a SynchronizedRegionOp and assuming a single
// producer/consumer per CB, V2 drives CB op placement directly off the
// remote_load/remote_store boundaries (per @nsmith's observation that those
// program points are exactly where push/pop semantics belong) and tolerates a
// buffer with multiple intra-thread producers/consumers (e.g. a loop-carried
// accumulator) by emitting a single handshake per cross-thread edge:
//
//   input  CB (remote_load -> compute reads):  d2m.wait at the load site,
//          d2m.pop after the last compute consumer.
//   output CB (compute writes -> remote_store): d2m.reserve before the first
//          compute producer, d2m.push at the store site.
//
// Scope: handles the lowered memref/tile form the real pipeline produces --
// CB access via memref.load/store (through view ops) into the dst register,
// tile_*_block direct-CB operands, and aliased (operand_alias) load/store.

// True if `op` lies within `region` (possibly nested).
static bool isInRegion(Region &region, Operation *op) {
  for (Region *r = op->getParentRegion(); r;) {
    if (r == &region) {
      return true;
    }
    Operation *parent = r->getParentOp();
    r = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}

// Climb `op`'s ancestors until reaching the op whose parent block is `block`,
// so CB ops anchor outside any loop nest wrapping the access.
[[maybe_unused]] static Operation *topLevelInBlock(Operation *op,
                                                   Block *block) {
  Operation *cur = op;
  while (cur && cur->getBlock() != block) {
    cur = cur->getParentOp();
  }
  return cur;
}

namespace {
// Per-CB compute-thread access summary used to decide which handshake to emit.
struct CBComputeInfo {
  bool consumed = false; // compute reads it (memref.load / tile_*_block input)
  bool produced = false; // compute writes it (memref.store / tile_*_block output)
  RemoteLoadOp dmLoad;    // DM partner that loads into it, if any
  RemoteStoreOp dmStore;  // DM partner that stores from it, if any
};
} // namespace

// Compute-side CB synchronization (dataflow model, no SynchronizedRegionOp).
//
// For each CB touched by the compute thread, emit the handshake bracketing the
// compute access(es), anchored outside any wrapping loop:
//   - non-aliased input  (DM loads):   wait ... pop
//   - non-aliased output (DM stores):  reserve ... push
//   - aliased (operand_alias) buffer:  full reserve/push/wait/pop cycle, since
//     the alias makes the compute thread own the buffer's whole lifecycle.
// CB uses (including those reached through memref view ops) are rewired to the
// acquired buffer (wait/reserve result).
static LogicalResult insertComputeCBOpsV2(GenericOp generic, Block *computeBlock,
                                          PatternRewriter &rewriter) {
  Region &computeRegion = *computeBlock->getParent();

  // Pre-order index for program-order comparisons across nested blocks.
  DenseMap<Operation *, unsigned> order;
  {
    unsigned i = 0;
    computeBlock->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { order[op] = i++; });
  }

  llvm::MapVector<Value, CBComputeInfo> cbs;

  // DM partners (identify which CBs cross the thread boundary, and aliasing).
  computeBlock->walk([&](RemoteLoadOp op) { cbs[op.getLocalBuffer()].dmLoad = op; });
  computeBlock->walk(
      [&](RemoteStoreOp op) { cbs[op.getLocalBuffer()].dmStore = op; });

  // Compute consume/produce via dst-register memref accesses (through views).
  computeBlock->walk([&](memref::LoadOp ld) {
    if (Value cb = traceComputeMemrefToCB(ld.getMemref(), generic)) {
      cbs[cb].consumed = true;
    }
  });
  computeBlock->walk([&](memref::StoreOp st) {
    if (Value cb = traceComputeMemrefToCB(st.getMemref(), generic)) {
      cbs[cb].produced = true;
    }
  });

  // Compute consume/produce via synchronizable compute ops (tile_*_block etc.).
  for (auto &[cb, usage] : utils::getCBUsageInfo(computeRegion)) {
    for (Operation *p : usage.producers) {
      if (!isa<RemoteLoadOp, RemoteStoreOp>(p)) {
        cbs[cb].produced = true;
      }
    }
    for (Operation *c : usage.consumers) {
      if (!isa<RemoteLoadOp, RemoteStoreOp>(c)) {
        cbs[cb].consumed = true;
      }
    }
  }

  for (auto &[cb, info] : cbs) {
    // A buffer only needs CB synchronization if it crosses the thread
    // boundary. An output is paired with the DM store; otherwise an input is
    // paired with the DM load. Buffers with no DM partner are compute-local
    // (e.g. dst-register intermediates) and need no handshake.
    bool outputCB = info.produced && info.dmStore;
    bool inputCB = !outputCB && info.consumed && info.dmLoad;
    if (!outputCB && !inputCB) {
      continue;
    }

    // Anchor CB ops at the DM partner's block (loop level), so a streaming
    // access inside a loop gets a per-iteration handshake matching the
    // per-iteration DMA, while an accumulator whose store sits outside the
    // reduction loop gets a single handshake around it.
    Block *partnerBlock =
        outputCB ? info.dmStore->getBlock() : info.dmLoad->getBlock();
    auto climbToPartner = [&](Operation *op) -> Operation * {
      Operation *cur = op;
      while (cur && cur->getBlock() != partnerBlock) {
        cur = cur->getParentOp();
      }
      return cur;
    };

    // Direct compute-region uses of `cb` to rewire to the acquired buffer
    // (view ops and tile_*_block; the remote ops are erased later). The
    // acquire is anchored before the earliest of these.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : cb.getUses()) {
      Operation *owner = use.getOwner();
      if (isInRegion(computeRegion, owner) &&
          !isa<RemoteLoadOp, RemoteStoreOp>(owner)) {
        uses.push_back(&use);
      }
    }
    if (uses.empty()) {
      continue;
    }

    // Transitive accesses: follow memref view ops forward to the actual
    // load/store/tile accesses. The release (pop/push) must come after the
    // last of these, which may sit inside a loop nest below the view op.
    SmallVector<Operation *> accesses;
    {
      DenseSet<Operation *> visited;
      SmallVector<Value> worklist{cb};
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        for (Operation *user : v.getUsers()) {
          if (!isInRegion(computeRegion, user) ||
              isa<RemoteLoadOp, RemoteStoreOp>(user) ||
              !visited.insert(user).second) {
            continue;
          }
          if (isa<memref::CollapseShapeOp, memref::SubViewOp, memref::CastOp>(
                  user)) {
            worklist.push_back(user->getResult(0));
          } else {
            accesses.push_back(user);
          }
        }
      }
    }

    // Acquire anchor: earliest direct use (climbed to the partner block).
    // Release anchor: latest transitive access (climbed to the partner block).
    Operation *earliest = nullptr, *latest = nullptr;
    for (OpOperand *use : uses) {
      Operation *top = climbToPartner(use->getOwner());
      if (top && (!earliest || order.lookup(top) < order.lookup(earliest))) {
        earliest = top;
      }
    }
    for (Operation *access : accesses) {
      Operation *top = climbToPartner(access);
      if (top && (!latest || order.lookup(top) > order.lookup(latest))) {
        latest = top;
      }
    }
    if (!earliest) {
      continue;
    }
    if (!latest) {
      latest = earliest;
    }

    Location loc = generic.getLoc();
    unsigned cbIdx = generic.getOperandIndex(cb);
    bool aliasedLoad = info.dmLoad && isAliasedLoad(info.dmLoad);
    bool aliasedStore = info.dmStore && isAliasedStore(info.dmStore);

    rewriter.setInsertionPoint(earliest);
    Value cbVal = d2m::getOrCreateCB(rewriter, generic, computeBlock, cbIdx);

    Value acquired;
    if (outputCB) {
      // Output buffer (compute writes; DM stores). Compute reserves before
      // producing, pushes after; an aliased store adds the consumer half.
      acquired = rewriter.create<ReserveOp>(loc, cbVal).getResult();
      rewriter.setInsertionPointAfter(latest);
      rewriter.create<PushOp>(loc, cbVal);
      if (aliasedStore) {
        rewriter.create<WaitOp>(loc, cbVal);
        rewriter.create<PopOp>(loc, cbVal);
      }
    } else {
      // Input buffer (compute reads; DM loads). An aliased load adds the
      // producer half before the wait.
      if (aliasedLoad) {
        rewriter.create<ReserveOp>(loc, cbVal);
        rewriter.create<PushOp>(loc, cbVal);
      }
      acquired = rewriter.create<WaitOp>(loc, cbVal).getResult();
      rewriter.setInsertionPointAfter(latest);
      rewriter.create<PopOp>(loc, cbVal);
    }

    for (OpOperand *use : uses) {
      use->set(acquired);
    }
  }

  return success();
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class D2MSplitUnifiedThreadV2Rewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getNumRegions() != 1) {
      return failure();
    }
    if (generic.getRegionThreadType(0) != ThreadType::Unified) {
      return failure();
    }

    // V2: no SynchronizedRegionOp wrapping -- insertComputeCBOpsV2 anchors CB
    // ops directly at the remote_load/remote_store boundaries.

    Region &originalRegion = generic.getRegion(0);
    if (originalRegion.empty()) {
      return failure();
    }
    Block *originalBlock = &originalRegion.front();

    if (failed(utils::checkForIllegalSemaphoreOps(originalBlock))) {
      return failure();
    }

    // Create new 2-region GenericOp: datamovement + compute.
    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(),
        rewriter.getArrayAttr(
            {rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement),
             rewriter.getAttr<ThreadAttr>(ThreadType::Compute)}),
        generic.getFabricConnectionConfigAttr(),
        /*numRegions*/ 2);

    Block *dmBlock = &newGeneric.getRegion(0).emplaceBlock();
    Block *computeBlock = &newGeneric.getRegion(1).emplaceBlock();

    // Map semaphore block arguments to both new blocks.
    IRMapping dmMapping, computeMapping;
    for (unsigned i = 0; i < originalBlock->getNumArguments(); ++i) {
      BlockArgument arg = originalBlock->getArgument(i);
      assert(mlir::isa<d2m::LocalSemaphoreType>(arg.getType()) &&
             "region block arguments must be of local semaphore type");
      dmMapping.map(arg, dmBlock->addArgument(arg.getType(), generic.getLoc()));
      computeMapping.map(
          arg, computeBlock->addArgument(arg.getType(), generic.getLoc()));
    }

    // Clone all ops into both regions.
    rewriter.setInsertionPointToStart(dmBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, dmMapping);
    }
    rewriter.setInsertionPointToStart(computeBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, computeMapping);
    }
    if (originalBlock->mightHaveTerminator()) {
      Operation *term = originalBlock->getTerminator();
      rewriter.setInsertionPointToEnd(dmBlock);
      rewriter.clone(*term, dmMapping);
      rewriter.setInsertionPointToEnd(computeBlock);
      rewriter.clone(*term, computeMapping);
    }

    // Compute thread: insert CB sync ops anchored at the remote_load/store
    // boundaries (dataflow model; no SynchronizedRegionOp).
    if (failed(insertComputeCBOpsV2(newGeneric, computeBlock, rewriter))) {
      return failure();
    }

    // DMA thread: convert datamovement ops to explicit CB form.
    if (failed(convertDMAToExplicitCBForm(dmBlock, rewriter))) {
      return failure();
    }

    eraseDMAOpsInComputeBlock(rewriter, computeBlock);
    eraseDeadOps(rewriter, dmBlock, /*isDatamovementThread=*/true);
    eraseDeadOps(rewriter, computeBlock, /*isDatamovementThread=*/false);

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MSplitUnifiedThreadV2
    : public impl::D2MSplitUnifiedThreadV2Base<D2MSplitUnifiedThreadV2> {
public:
  using impl::D2MSplitUnifiedThreadV2Base<
      D2MSplitUnifiedThreadV2>::D2MSplitUnifiedThreadV2Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MSplitUnifiedThreadV2Rewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
