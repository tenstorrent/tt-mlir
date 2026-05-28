// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREAD
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

LogicalResult wrapComputeInSynchronizedRegion(GenericOp genericOp,
                                              PatternRewriter &rewriter) {
  // Look for a D2M_GenericRegionComputeOp, and collect the outermost ops that
  // contain them in the generic op.
  // Skip ops that have the SynchronizableOpInterface,
  // such as TileTilizeBlockOp and TileUntilizeBlockOp ops since
  // they haven't been lowered yet into non-synchronized ops
  OpBuilder::InsertionGuard guard(rewriter);

  // Collect the ops that directly contain SynchronizableOpInterface ops.
  // These delimit the scope where compute is synchronized: the outermost
  // compute ancestor is a direct child of such an op, alongside the
  // synchronizable ops that bound it. Multiple parents are allowed: each
  // identifies an independent synchronized scope at its own nesting level
  // (e.g. remote_loads in an accumulator loop bounding the matmul compute,
  // while a remote_store in the enclosing loop bounds a separate compute
  // region around the epilogue).
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

    // Skip ops that have the SynchronizableOpInterface,
    // such as TileTilizeBlockOp and TileUntilizeBlockOp ops since
    // they haven't been lowered yet into non-synchronized ops.
    if (!dyn_cast<SynchronizableOpInterface>(outermostOp)) {
      outermostOps.insert(outermostOp);
    }

    return WalkResult::advance();
  });

  if (walkFailed) {
    return failure();
  }

  // Expand and merge compute regions until we hit a synchronizable op on both
  // ends.
  //
  // Beyond hitting a SynchronizableOpInterface op (which bounds the wrap),
  // two additional stop conditions matter:
  // (1) A sibling op already in `opsWithSynchronizableOps` hosts its own
  //     synchronized scope at a deeper level; folding it in would nest
  //     sync regions (e.g. an inner accumulator loop sitting next to a
  //     remote_store in the outer loop's body).
  // (2) Including an op whose result is consumed past the wrap would
  //     violate `wrapInSynchronizedRegion`'s precondition: any op in
  //     [start, end) whose operand chain reaches a non-pure op gets
  //     erased after the wrap is created, so leaving an outside user
  //     dangles. Check users unconditionally — an op like
  //     `arith.index_cast` is itself Pure but is treated as non-pure
  //     by the wrap utility once an ancestor like `arith.divsi` (which
  //     can fault) enters the operand chain.
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

  // Merge overlapping compute regions and fixpoint-expand each merged region
  // so wrapInSynchronizedRegion's precondition holds: no op that will be
  // erased by that utility may have result users outside [start, end).
  //
  // Overlaps arise when hasUserOutsideRange causes independent per-outermostOp
  // expansions to produce regions that share ops.  For example, acquire_dst
  // and tile_add form an interdependent value chain in a flat block: the
  // acquire_dst expansion stops early because tile_add's result escapes, and
  // tile_add's expansion starts one op after acquire_dst for the same reason,
  // yielding overlapping [A, B) and [B-1, C).  Wrapping the first region then
  // erases the shared op (B-1), leaving the second region's start iterator
  // dangling and triggering an ilist sentinel crash.
  if (!computeRegions.empty()) {
    // Purity cache mirroring isPurelyDerivedOp in wrapInSynchronizedRegion:
    // an op is purely derived if it is pure and every operand-defining op is
    // also purely derived.  Ops that are not purely derived will be erased by
    // wrapInSynchronizedRegion; their results must not escape the region.
    // The cache is block-agnostic and shared across all per-block passes below.
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
    // different blocks (e.g. inner K-loop body vs. enclosing block).
    // A stable insertion-order vector preserves deterministic processing.
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

      // Fixpoint expansion: extend each merged region in both directions.
      //
      // Downward (end grows): a non-pure op in [startPos, endPos) has result
      // users outside the range that would dangle after
      // wrapInSynchronizedRegion erases the op.  Expand endPos to include those
      // users.
      //
      // Upward (start shrinks): a non-pure op in the range has an operand
      // defined by another non-pure op just before startPos.  Expand startPos
      // to include that defining op so that CB-load detection (which walks ops
      // inside [start, end)) can find the CB operand feeding the region.
      //
      // Both directions stop at SynchronizableOpInterface boundaries.
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
static LogicalResult processSharedBufferPairs(
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

static LogicalResult
insertCBOpsForCompute(Block *computeBlock, PatternRewriter &rewriter,
                      llvm::DenseMap<Value, utils::CBUsageInfo> &cbUsageInfo) {
  SmallVector<RemoteLoadOp> loads;

  computeBlock->walk([&](Operation *op) {
    if (dyn_cast<SynchronizableOpInterface>(op) &&
        !op->hasTrait<D2MGenericRegionDatamovementOpTrait>()) {
      auto synchronizedOp = mlir::cast<SynchronizableOpInterface>(op);

      // The CB ops we emit must fire once per DMA partner activation, not
      // once per compute iteration. When the DMA partner sits in an outer
      // block (e.g. a matmul accumulator: remote_loads inside the K loop,
      // remote_store outside it), placing wait/pop or reserve/push around
      // `synchronizedOp` would emit N pairs against a single DMA op. Climb
      // synchronizedOp's ancestors until we reach one whose parent block
      // matches the partner's; that ancestor is the anchor for the CB ops.
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
static LogicalResult eraseAliasedLoadStoreOps(
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
// Main rewriter
// ---------------------------------------------------------------------------

class D2MSplitUnifiedThreadRewriter : public OpRewritePattern<GenericOp> {
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

    if (failed(wrapComputeInSynchronizedRegion(generic, rewriter))) {
      return failure();
    }

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

    // Compute thread: insert CB sync ops for implicit-form remote ops.
    auto cbUsageInfoCompute = utils::getCBUsageInfo(newGeneric.getRegion(1));
    auto cbUsageInfoDm = utils::getCBUsageInfo(newGeneric.getRegion(0));
    if (failed(processSharedBufferPairs(computeBlock, rewriter,
                                        cbUsageInfoCompute)) ||
        failed(insertCBOpsForCompute(computeBlock, rewriter,
                                     cbUsageInfoCompute))) {
      return failure();
    }

    if (failed(eraseAliasedLoadStoreOps(rewriter, cbUsageInfoDm))) {
      return failure();
    }

    // DMA thread: convert datamovement ops to explicit CB form.
    if (failed(convertDMAToExplicitCBForm(dmBlock, rewriter))) {
      return failure();
    }

    eraseDMAOpsInComputeBlock(rewriter, computeBlock);
    eraseDeadOps(rewriter, dmBlock, /*isDatamovementThread=*/true);
    eraseDeadOps(rewriter, computeBlock, /*isDatamovementThread=*/false);

    // Remove synchronized region ops, and move its ops to the parent level
    computeBlock->walk([&](SynchronizedRegionOp synchronizedOp) {
      if (failed(utils::unwrapSynchronizedRegion(rewriter, synchronizedOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MSplitUnifiedThread
    : public impl::D2MSplitUnifiedThreadBase<D2MSplitUnifiedThread> {
public:
  using impl::D2MSplitUnifiedThreadBase<
      D2MSplitUnifiedThread>::D2MSplitUnifiedThreadBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MSplitUnifiedThreadRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
