// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTSPILLANDSCRATCH
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

#include <algorithm>

namespace {

/// Information about one scratch_space_loop "region of compute".
/// Records enough structural information to:
///   1. decide whether one loop nest produces a value used by another,
///   2. size the compiler-inserted scratch buffer, and
///   3. build the affine indexing used to spill into scratch and reload later.
struct ScratchLoopInfo {
  affine::AffineForOp scratchLoop; // The loop with d2m.scratch_space_loop attr
  affine::AffineForOp linalgRoot;  // The nested loop with d2m.linalg_root attr
  // All affine loops nested inside scratch_space_loop (linalg_root and inner
  // loops), in top-down order. Does NOT include scratch_space_loop itself.
  SmallVector<affine::AffineForOp> allLoops;
  SmallVector<int64_t> scratchShape; // Shape for scratch buffer
};

/// A single consumer loop and the loads it contains for a given intermediate.
/// Keep the whole list of loads because the same intermediate allocation may
/// be read multiple times inside a later scratch_space_loop and all of those
/// loads must be redirected to the same scratch slot once spilling is inserted.
struct ConsumerLoopLoads {
  ScratchLoopInfo *loop;
  SmallVector<affine::AffineLoadOp> loads;
};

/// Information about one intermediate allocation that should be materialized
/// through compiler-managed scratch instead of a local memref.alloc.
///
/// "producer" here is intentionally the loop nest that first defines the value
/// that must survive into a later scratch_space_loop. A later loop may both
/// read and then update the same allocation in place, but that later write does
/// not replace the original producer for cross-loop lifetime purposes.
struct IntermediateAllocInfo {
  memref::AllocOp allocOp;
  ScratchLoopInfo *producer; // Loop nest that writes to this alloc
  // Loads are only rewritten while the producer's definition still reaches.
  // If a later loop writes the same allocation, that loop is the last consumer
  // we can safely redirect to this producer's scratch slot.
  size_t lastConsumerIndex;
  SmallVector<ConsumerLoopLoads>
      consumers; // all consumer loop nests that read from it
  SmallVector<affine::AffineStoreOp> stores; // Stores to this alloc
};

/// Find the innermost affine.for loop with d2m.linalg_root attribute.
///
/// This identifies the loop where the actual elementwise/tilewise linalg-style
/// compute begins. Scratch indexing is built from the enclosing
/// scratch_space_loop plus this linalg_root subtree.
static affine::AffineForOp findLinalgRootLoop(affine::AffineForOp scratchLoop) {
  affine::AffineForOp result = nullptr;
  scratchLoop.getBody()->walk([&](affine::AffineForOp forOp) {
    if (forOp->hasAttr("d2m.linalg_root")) {
      result = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Collect ALL affine.for loops nested inside scratch_space_loop (excluding
/// scratch_space_loop itself), including linalg_root and any inner loops, in
/// top-down (pre-order) order. Pre-order is critical to ensure the shape
/// dimensions match the loop nesting: outermost loop -> first dimension.
static void collectAllInnerLoops(affine::AffineForOp scratchLoop,
                                 SmallVector<affine::AffineForOp> &allLoops) {
  scratchLoop.walk<WalkOrder::PreOrder>([&](affine::AffineForOp innerFor) {
    if (innerFor != scratchLoop) {
      allLoops.push_back(innerFor);
    }
  });
}

/// Compute the scratch shape from loop bounds.
/// Returns: [scratch_iters, linalg_root_iters, inner_iters...]
/// Each loop dimension contributes ceildiv(upper - lower, step) iterations.
static bool computeScratchShape(ScratchLoopInfo &info) {
  // First dimension: scratch_space_loop iterations.
  if (!info.scratchLoop.hasConstantBounds()) {
    return false;
  }
  int64_t lower = info.scratchLoop.getConstantLowerBound();
  int64_t upper = info.scratchLoop.getConstantUpperBound();
  int64_t step = info.scratchLoop.getStepAsInt();
  info.scratchShape.push_back(llvm::divideCeil(upper - lower, step));

  // Remaining dimensions: linalg_root and inner loops.
  for (auto loopOp : info.allLoops) {
    if (!loopOp.hasConstantBounds()) {
      return false;
    }
    int64_t loopLower = loopOp.getConstantLowerBound();
    int64_t loopUpper = loopOp.getConstantUpperBound();
    int64_t loopStep = loopOp.getStepAsInt();
    info.scratchShape.push_back(
        llvm::divideCeil(loopUpper - loopLower, loopStep));
  }

  return true;
}

/// Gather all unprocessed scratch_space_loop ops in a generic.
static SmallVector<affine::AffineForOp>
collectScratchSpaceLoops(GenericOp genericOp) {
  SmallVector<affine::AffineForOp> scratchSpaceLoops;
  genericOp.walk([&](affine::AffineForOp forOp) {
    if (forOp->hasAttr("d2m.scratch_space_loop") &&
        !forOp->hasAttr("d2m.scratch_inserted")) {
      scratchSpaceLoops.push_back(forOp);
    }
  });
  return scratchSpaceLoops;
}

/// Check whether an allocation is a compiler-local intermediate of this
/// d2m.generic.
/// Inputs/outputs of the generic already have externally visible storage and
/// should never be replaced by scratch slots.
static bool isIntermediateAlloc(memref::AllocOp allocOp, GenericOp genericOp) {
  // Check if alloc is inside the generic op's region
  if (!genericOp->isProperAncestor(allocOp)) {
    return false;
  }

  // Check that it's not one of the generic's operands
  Value allocResult = allocOp.getResult();
  for (Value operand : genericOp->getOperands()) {
    if (operand == allocResult) {
      return false;
    }
  }

  return true;
}

/// Check whether a memref SSA value ultimately aliases the target allocation.
///
/// Loads/stores often use subviews rather than the alloc result directly.
/// Spill insertion cares about the underlying storage object, so walk
/// through subviews and treat them as accesses to the same intermediate.
static bool tracesToAlloc(Value memref, Value targetAlloc) {
  Value current = memref;
  while (current) {
    if (current == targetAlloc) {
      return true;
    }
    if (auto subview = current.getDefiningOp<memref::SubViewOp>()) {
      current = subview.getSource();
    } else {
      break;
    }
  }
  return false;
}

/// Collect all affine stores to a given intermediate anywhere inside one
/// scratch_space_loop nest.
static void collectStoresInLoop(Value memref, ScratchLoopInfo &loopInfo,
                                SmallVector<affine::AffineStoreOp> &stores) {
  loopInfo.scratchLoop.walk([&](affine::AffineStoreOp storeOp) {
    if (tracesToAlloc(storeOp.getMemref(), memref)) {
      stores.push_back(storeOp);
    }
  });
}

/// Collect all affine loads from a given intermediate anywhere inside one
/// scratch_space_loop nest.
static void collectLoadsInLoop(Value memref, ScratchLoopInfo &loopInfo,
                               SmallVector<affine::AffineLoadOp> &loads) {
  loopInfo.scratchLoop.walk([&](affine::AffineLoadOp loadOp) {
    if (tracesToAlloc(loadOp.getMemref(), memref)) {
      loads.push_back(loadOp);
    }
  });
}

/// Check whether two affine loops have identical static iteration spaces.
static bool haveMatchingConstantBounds(affine::AffineForOp lhs,
                                       affine::AffineForOp rhs) {
  if (!lhs.hasConstantBounds() || !rhs.hasConstantBounds()) {
    return false;
  }
  return lhs.getConstantLowerBound() == rhs.getConstantLowerBound() &&
         lhs.getConstantUpperBound() == rhs.getConstantUpperBound() &&
         lhs.getStepAsInt() == rhs.getStepAsInt();
}

/// Return the single top-level affine child in a loop body, if one exists.
static affine::AffineForOp
getSingleTopLevelAffineChild(affine::AffineForOp loop) {
  affine::AffineForOp child = nullptr;
  for (Operation &op : *loop.getBody()) {
    if (isa<affine::AffineYieldOp>(op)) {
      continue;
    }
    auto innerFor = dyn_cast<affine::AffineForOp>(op);
    if (!innerFor || child) {
      return nullptr;
    }
    child = innerFor;
  }
  return child;
}

/// Return true if an op uses any value from the provided set.
static bool usesAnyValue(Operation *op, ArrayRef<Value> values) {
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return llvm::is_contained(values, operand);
  });
}

static UnpackStallOnPackOp
getUnpackStallImmediatelyBefore(Operation *consumerLoop) {
  Operation *previous = consumerLoop->getPrevNode();
  if (!previous) {
    return nullptr;
  }

  return dyn_cast<UnpackStallOnPackOp>(previous);
}

static void ensureUnpackStallBeforeConsumer(
    ScratchLoopInfo &producer, ScratchLoopInfo &consumer, IRRewriter &rewriter,
    llvm::SmallPtrSetImpl<Operation *> &guardedConsumers) {
  if (consumer.scratchLoop == producer.scratchLoop) {
    return;
  }

  Operation *producerLoop = producer.scratchLoop.getOperation();
  Operation *consumerLoop = consumer.scratchLoop.getOperation();
  if (producerLoop->getBlock() != consumerLoop->getBlock() ||
      !producerLoop->isBeforeInBlock(consumerLoop)) {
    return;
  }

  if (!guardedConsumers.insert(consumerLoop).second) {
    return;
  }

  if (auto previousStall = getUnpackStallImmediatelyBefore(consumerLoop)) {
    rewriter.eraseOp(previousStall);
  }

  rewriter.setInsertionPoint(consumerLoop);
  rewriter.create<UnpackStallOnPackOp>(consumerLoop->getLoc());
}

/// Fuse a dependence-related group of sibling scratch loops under one shared
/// affine prefix.
static bool manuallyFuseScratchLoopGroup(ArrayRef<affine::AffineForOp> loops,
                                         IRRewriter &rewriter) {
  if (loops.size() < 2) {
    return false;
  }

  Block *parentBlock = loops.front()->getBlock();
  SmallVector<Value> oldIvs;
  oldIvs.reserve(loops.size());
  for (auto loop : loops) {
    if (loop->getBlock() != parentBlock ||
        !haveMatchingConstantBounds(loops.front(), loop)) {
      return false;
    }

    affine::AffineForOp inner = getSingleTopLevelAffineChild(loop);
    // Hoist exactly one common affine prefix. The inner loop becomes the new
    // scratch boundary and must still contain a nested linalg root.
    if (!inner || inner->hasAttr("d2m.linalg_root")) {
      return false;
    }
    oldIvs.push_back(loop.getInductionVar());
  }

  for (size_t i = 1; i < loops.size(); ++i) {
    if (!loops[i - 1]->isBeforeInBlock(loops[i])) {
      return false;
    }
  }

  SmallVector<SmallVector<Operation *>> setupOpsBeforeLoop(loops.size());
  SmallVector<Operation *> staleStalls;
  for (size_t i = 1; i < loops.size(); ++i) {
    for (auto it = std::next(loops[i - 1]->getIterator());
         it != loops[i]->getIterator(); ++it) {
      Operation *op = &*it;
      if (isa<UnpackStallOnPackOp>(op)) {
        staleStalls.push_back(op);
        continue;
      }

      if (usesAnyValue(op, oldIvs)) {
        return false;
      }
      setupOpsBeforeLoop[i].push_back(op);
    }
  }

  rewriter.setInsertionPoint(loops.front());
  affine::AffineForOp firstLoop = loops.front();
  auto sharedLoop = rewriter.create<affine::AffineForOp>(
      firstLoop.getLoc(), firstLoop.getConstantLowerBound(),
      firstLoop.getConstantUpperBound(), firstLoop.getStepAsInt());

  for (auto loop : loops) {
    loop.getInductionVar().replaceAllUsesWith(sharedLoop.getInductionVar());
  }

  Operation *sharedYield = sharedLoop.getBody()->getTerminator();
  for (auto &ops : setupOpsBeforeLoop) {
    for (Operation *op : ops) {
      op->moveBefore(sharedLoop);
    }
  }

  for (auto indexedLoop : llvm::enumerate(loops)) {
    affine::AffineForOp loop = indexedLoop.value();
    auto &ops = loop.getBody()->getOperations();
    sharedLoop.getBody()->getOperations().splice(
        sharedYield->getIterator(), ops, ops.begin(), std::prev(ops.end()));
  }

  for (Operation *stallOp : staleStalls) {
    rewriter.eraseOp(stallOp);
  }

  for (Operation &op : *sharedLoop.getBody()) {
    auto innerFor = dyn_cast<affine::AffineForOp>(&op);
    if (!innerFor) {
      continue;
    }
    innerFor->setAttr("d2m.scratch_space_loop",
                      UnitAttr::get(rewriter.getContext()));
    innerFor->removeAttr("d2m.scratch_inserted");
  }

  for (auto loop : loops) {
    rewriter.eraseOp(loop);
  }

  return true;
}

/// Build an affine map and operands for accessing the scratch buffer.
/// For loops with step > 1, the IV is normalized to give contiguous indices.
/// E.g., affine.for %i = 0 to 8 step 2 gives IV values 0,2,4,6 which are
/// normalized to 0,1,2,3 via (d) -> (d floordiv 2).
static std::pair<AffineMap, SmallVector<Value>>
getScratchAccessMapAndOperands(ScratchLoopInfo &info, MLIRContext *ctx) {
  SmallVector<AffineExpr> exprs;
  SmallVector<Value> operands;
  unsigned dimIdx = 0;

  auto addLoop = [&](affine::AffineForOp loop) {
    operands.push_back(loop.getInductionVar());
    AffineExpr dim = getAffineDimExpr(dimIdx++, ctx);
    int64_t lb = loop.getConstantLowerBound();
    int64_t step = loop.getStepAsInt();
    if (lb != 0) {
      dim = dim - lb;
    }
    if (step != 1) {
      dim = dim.floorDiv(step);
    }
    exprs.push_back(dim);
  };

  // First: scratch_space_loop.
  addLoop(info.scratchLoop);

  // Then: linalg_root and inner loops.
  for (auto loop : info.allLoops) {
    addLoop(loop);
  }

  return {AffineMap::get(dimIdx, 0, exprs, ctx), operands};
}

static SmallVector<affine::AffineForOp>
getMissingConsumerPrefixLoops(ScratchLoopInfo &producer,
                              ScratchLoopInfo &consumer) {
  // Find the shared outer loops the consumer needs to line up with producer scratch.
  size_t consumerRank = 1 + consumer.allLoops.size();
  if (producer.scratchShape.size() <= consumerRank) {
    return {};
  }

  size_t missingRank = producer.scratchShape.size() - consumerRank;
  SmallVector<affine::AffineForOp> prefixes;
  Operation *parent = consumer.scratchLoop->getParentOp();
  while (parent && prefixes.size() < missingRank) {
    if (auto parentLoop = dyn_cast<affine::AffineForOp>(parent)) {
      prefixes.push_back(parentLoop);
    }
    parent = parent->getParentOp();
  }

  std::reverse(prefixes.begin(), prefixes.end());
  return prefixes;
}

/// Build an access map for a consumer loading from a scratch buffer that was
/// shaped by a producer with a potentially different inner loop step.
///
/// The scratch buffer layout is determined by the producer's allLoops[0] step
/// (the outer column-batching loop inside scratch_space_loop). When a consumer
/// has a different allLoops[0] step, the absolute tile column index
/// (allLoops[0]_IV + allLoops[-1]_IV) must be decomposed using the producer's
/// step to correctly address the scratch buffer.
///
/// Example: producer allLoops[0] step=4 shapes scratch as [1,3,1,4].
///   Producer stores at [arg10, arg11/4, arg12, arg13] for arg11 in {0,4,8}.
/// Consumer allLoops[0] step=6 must access as:
///   [arg10, (arg11+arg13)/4, arg12, (arg11+arg13)%4]
/// instead of the wrong: [arg10, arg11/6, arg12, arg13].
static std::pair<AffineMap, SmallVector<Value>>
getConsumerScratchAccessMap(ScratchLoopInfo &producer,
                            ScratchLoopInfo &consumer, MLIRContext *ctx) {
  // Cross-step handling is needed when the first inner loop (column-batching
  // loop, allLoops[0]) has a different step between producer and consumer.
  // The scratchLoop itself typically has step=1 for both, so we must compare
  // allLoops[0] steps rather than scratchLoop steps.
  bool needsCrossStep = consumer.allLoops.size() >= 2 &&
                        !producer.allLoops.empty() &&
                        producer.allLoops.front().getStepAsInt() !=
                            consumer.allLoops.front().getStepAsInt();

  if (!needsCrossStep) {
    SmallVector<AffineExpr> exprs;
    SmallVector<Value> operands;
    unsigned dimIdx = 0;

    // Build the straightforward consumer index including any shared prefix loops.
    auto addLoop = [&](affine::AffineForOp loop) {
      operands.push_back(loop.getInductionVar());
      AffineExpr dim = getAffineDimExpr(dimIdx++, ctx);
      int64_t lb = loop.getConstantLowerBound();
      int64_t step = loop.getStepAsInt();
      if (lb != 0) {
        dim = dim - lb;
      }
      if (step != 1) {
        dim = dim.floorDiv(step);
      }
      exprs.push_back(dim);
    };

    for (auto prefixLoop : getMissingConsumerPrefixLoops(producer, consumer)) {
      addLoop(prefixLoop);
    }
    addLoop(consumer.scratchLoop);
    for (auto loop : consumer.allLoops) {
      addLoop(loop);
    }

    return {AffineMap::get(dimIdx, 0, exprs, ctx), operands};
  }

  // Cross-step case: consumer batches tile columns in different sizes than the
  // producer. Compute the absolute column tile index and decompose it into the
  // producer's scratch layout.
  //
  // absolute_col = allLoops[0]_IV_raw + allLoops[-1]_IV_raw
  //   (allLoops[0] gives the batch start, allLoops[-1] gives offset in batch)
  //
  // Map to producer scratch layout:
  //   [scratchDim, abs_col / prodColStep, middle..., abs_col % prodColStep]
  int64_t prodColStep = producer.allLoops.front().getStepAsInt();

  SmallVector<AffineExpr> exprs;
  SmallVector<Value> operands;
  unsigned dimIdx = 0;

  // Helper: add loop with normalization, push to exprs.
  auto addLoop = [&](affine::AffineForOp loop) -> AffineExpr {
    operands.push_back(loop.getInductionVar());
    AffineExpr dim = getAffineDimExpr(dimIdx++, ctx);
    int64_t lb = loop.getConstantLowerBound();
    int64_t step = loop.getStepAsInt();
    if (lb != 0) {
      dim = dim - lb;
    }
    if (step != 1) {
      dim = dim.floorDiv(step);
    }
    return dim;
  };

  for (auto prefixLoop : getMissingConsumerPrefixLoops(producer, consumer)) {
    exprs.push_back(addLoop(prefixLoop));
  }

  // Dimension 0: scratchLoop (normalized pass-through, same step for both).
  exprs.push_back(addLoop(consumer.scratchLoop));

  // Outer col loop (allLoops[0]): raw IV without step normalization.
  // The raw value (e.g., 0 or 6 for step-6) is used in the absolute tile index.
  operands.push_back(consumer.allLoops.front().getInductionVar());
  AffineExpr d_outerCol = getAffineDimExpr(dimIdx++, ctx);
  {
    int64_t lb = consumer.allLoops.front().getConstantLowerBound();
    if (lb != 0) {
      d_outerCol = d_outerCol - lb;
    }
    // Do NOT normalize by step — we need the raw batch-start value.
  }

  // Middle dimensions (allLoops[1..-2]): normalized pass-through (row loops).
  SmallVector<AffineExpr> middleExprs;
  for (size_t i = 1; i + 1 < consumer.allLoops.size(); ++i) {
    middleExprs.push_back(addLoop(consumer.allLoops[i]));
  }

  // Inner col loop (allLoops[-1]): raw IV (offset within the col batch).
  operands.push_back(consumer.allLoops.back().getInductionVar());
  AffineExpr d_innerCol = getAffineDimExpr(dimIdx++, ctx);
  {
    int64_t lb = consumer.allLoops.back().getConstantLowerBound();
    if (lb != 0) {
      d_innerCol = d_innerCol - lb;
    }
    // Do NOT normalize — inner col loop should have step=1.
  }

  // Absolute column tile index (invariant across producer/consumer steps).
  AffineExpr absCol = d_outerCol + d_innerCol;

  // Map to producer scratch layout using producer's column batch size.
  exprs.push_back(absCol.floorDiv(prodColStep));
  for (auto me : middleExprs) {
    exprs.push_back(me);
  }
  exprs.push_back(absCol % prodColStep);

  return {AffineMap::get(dimIdx, 0, exprs, ctx), operands};
}

/// Fuse outer scf.for loops that enclose scratch_space_loop nests.
///
/// When multiple scratch_space_loop nests are siblings with identical outer
/// scf.for structures (same bounds/steps), this function merges them into a
/// single shared outer loop. This is critical for scratch buffer correctness:
/// without fusion, each nest has its own outer loop, and each iteration
/// overwrites the same scratch positions. The consumer (in a separate nest)
/// only sees the last iteration's data.
///
/// After fusion, all computations share the outer loop, so within each
/// iteration the producer writes to scratch and the consumer reads from it
/// before the next iteration overwrites.
///
/// Before:
///   scf.for { scratch_space_loop_1 { producer } }
///   scf.for { scratch_space_loop_2 { consumer } }
///
/// After:
///   scf.for {
///     scratch_space_loop_1 { producer }
///     scratch_space_loop_2 { consumer }
///   }
static bool fuseOuterScfLoops(SmallVector<affine::AffineForOp> &scratchLoops,
                              IRRewriter &rewriter) {
  if (scratchLoops.size() < 2) {
    return true;
  }

  struct ChainInfo {
    // The enclosing scf.for chain, ordered from outermost to innermost.
    SmallVector<scf::ForOp> chain;
  };

  SmallVector<ChainInfo> chains;
  for (auto sl : scratchLoops) {
    ChainInfo ci;
    Operation *parent = sl->getParentOp();
    while (auto scfFor = dyn_cast<scf::ForOp>(parent)) {
      ci.chain.push_back(scfFor);
      parent = scfFor->getParentOp();
    }
    std::reverse(ci.chain.begin(), ci.chain.end());
    chains.push_back(ci);
  }

  // Every candidate must have the same number of enclosing scf.for loops.
  size_t chainLen = chains[0].chain.size();
  if (chainLen == 0) {
    return true;
  }
  for (auto &ci : chains) {
    if (ci.chain.size() != chainLen) {
      return false;
    }
  }

  // Find the first level where the nests stop sharing the exact same loop op.
  size_t firstDivergentLevel = chainLen;
  for (size_t level = 0; level < chainLen; ++level) {
    scf::ForOp ref = chains[0].chain[level];
    bool allMatch = true;
    for (size_t i = 1; i < chains.size(); ++i) {
      if (chains[i].chain[level] != ref) {
        allMatch = false;
        break;
      }
    }
    if (allMatch) {
      continue;
    }
    firstDivergentLevel = level;
    break;
  }

  if (firstDivergentLevel == chainLen) {
    return true;
  }

  // Walk linearly through the block and reorder the operations between
  // consecutive loop nests.
  Block *outerBlock = chains[0].chain[firstDivergentLevel]->getBlock();
  for (auto &ci : chains) {
    if (ci.chain[firstDivergentLevel]->getBlock() != outerBlock) {
      return false;
    }
  }

  // Put chains in source order so "chain i then chain i+1" matches the original
  // execution order in the parent block.
  std::sort(chains.begin(), chains.end(),
            [](const ChainInfo &lhs, const ChainInfo &rhs) -> bool {
              size_t level = 0;
              while (level < lhs.chain.size() &&
                     lhs.chain[level] == rhs.chain[level]) {
                ++level;
              }
              if (level == lhs.chain.size()) {
                return false;
              }
              return lhs.chain[level]->isBeforeInBlock(rhs.chain[level]);
            });

  // Verify that the divergent loops are structurally identical. Fusion here is
  // intentionally conservative: only merge loops when they iterate over the
  // same space with the same step so the shared loop preserves semantics.
  for (size_t level = firstDivergentLevel; level < chainLen; ++level) {
    scf::ForOp ref = chains[0].chain[level];
    auto refLb = getConstantIntValue(ref.getLowerBound());
    auto refUb = getConstantIntValue(ref.getUpperBound());
    auto refStep = getConstantIntValue(ref.getStep());
    if (!refLb || !refUb || !refStep) {
      return false;
    }
    for (size_t i = 1; i < chains.size(); ++i) {
      scf::ForOp other = chains[i].chain[level];
      auto lb = getConstantIntValue(other.getLowerBound());
      auto ub = getConstantIntValue(other.getUpperBound());
      auto step = getConstantIntValue(other.getStep());
      if (!lb || !ub || !step) {
        return false;
      }
      if (*refLb != *lb || *refUb != *ub || *refStep != *step) {
        return false;
      }
    }
  }

  // Move operations between consecutive loop nests to before the first.
  //
  // Source shape before fusion looks roughly like:
  //   scf.for A
  //   setup ops for B
  //   scf.for B
  //   setup ops for C
  //   scf.for C
  //
  // After we create one shared outer loop, hoist the setup ops in front of the
  // first fused chain and then splice the old loop bodies into the new shared
  // chain.
  //
  // These are setup ops (allocs, remote_loads, etc.) that don't depend on
  // loop IVs and need to dominate the shared loop body.
  //
  // UnpackStallOnPackOp is special: it must remain ordered relative to the
  // compute it guards and must end up inside the fused loop body (after
  // chain[i]'s content, before chain[i+1]'s content). Collect these
  // separately and splice them in at the right position below.
  SmallVector<SmallVector<Operation *>> stallsAfterChain(chains.size());
  Operation *firstOuter = chains[0].chain[firstDivergentLevel];
  for (size_t i = 1; i < chains.size(); ++i) {
    Operation *thisOuter = chains[i].chain[firstDivergentLevel];
    SmallVector<Operation *> toMove;
    auto it =
        std::next(chains[i - 1].chain[firstDivergentLevel]->getIterator());
    auto end = outerBlock->end();
    auto thisOuterIt = thisOuter->getIterator();
    for (; it != end && it != thisOuterIt; ++it) {
      Operation *op = &*it;
      if (isa<UnpackStallOnPackOp>(op)) {
        stallsAfterChain[i - 1].push_back(op);
      } else {
        toMove.push_back(op);
      }
    }
    if (it == end) {
      return false;
    }
    for (auto *op : toMove) {
      op->moveBefore(firstOuter);
    }
  }

  // Materialize a fresh scf.for chain with the same iteration space as the
  // first original chain.
  rewriter.setInsertionPoint(firstOuter);
  SmallVector<scf::ForOp> shared;
  for (size_t level = firstDivergentLevel; level < chainLen; ++level) {
    scf::ForOp ref = chains[0].chain[level];
    auto newLoop = rewriter.create<scf::ForOp>(
        ref.getLoc(), ref.getLowerBound(), ref.getUpperBound(), ref.getStep());
    shared.push_back(newLoop);
    rewriter.setInsertionPointToStart(newLoop.getBody());
  }

  // Move body operations from each old chain into the shared chain.
  //
  // Conceptually we are turning:
  //   outerA { bodyA }
  //   outerB { bodyB }
  //
  // into:
  //   sharedOuter { bodyA; bodyB; }
  //
  // but we must do so level-by-level to keep nested setup/teardown ordering
  // valid at each depth.
  for (size_t chainIdx = 0; chainIdx < chains.size(); ++chainIdx) {
    auto &ci = chains[chainIdx];
    // Rewrite uses of the old induction vars to the new shared IVs before
    // moving operations.
    for (size_t level = firstDivergentLevel; level < chainLen; ++level) {
      ci.chain[level].getInductionVar().replaceAllUsesWith(
          shared[level - firstDivergentLevel].getInductionVar());
    }

    for (size_t level = firstDivergentLevel; level < chainLen; ++level) {
      size_t sharedLevel = level - firstDivergentLevel;
      Block *oldBody = ci.chain[level].getBody();
      Block *sharedBody = shared[sharedLevel].getBody();
      Operation *sharedYield = sharedBody->getTerminator();

      if (level + 1 < chainLen) {
        // Non-innermost level: move everything except the next-level scf.for
        // and the yield terminator.
        //
        // Ops are split by their position relative to chain[level+1] in the
        // original body to preserve memory operation ordering:
        //   - Pre-ops (before the inner loop): allocs, loads, setup. These
        //     must appear before shared[level+1] in the shared body so they
        //     dominate uses in deeply nested shared loops.
        //   - Post-ops (after the inner loop): stores, teardown. These depend
        //     on the inner compute completing and must stay after
        //     shared[level+1], before the shared yield.
        SmallVector<Operation *> preOps;
        SmallVector<Operation *> postOps;
        bool pastInnerLoop = false;
        for (auto &op : *oldBody) {
          if (&op == ci.chain[level + 1].getOperation()) {
            pastInnerLoop = true;
            continue;
          }
          if (isa<scf::YieldOp>(&op)) {
            continue;
          }
          if (pastInnerLoop) {
            postOps.push_back(&op);
          } else {
            preOps.push_back(&op);
          }
        }
        Operation *innerSharedLoop = shared[sharedLevel + 1].getOperation();
        for (auto *op : preOps) {
          op->moveBefore(innerSharedLoop);
        }
        for (auto *op : postOps) {
          op->moveBefore(sharedYield);
        }
      } else {
        // Innermost level: splice everything except the yield.
        auto &srcOps = oldBody->getOperations();
        sharedBody->getOperations().splice(sharedYield->getIterator(), srcOps,
                                           srcOps.begin(),
                                           std::prev(srcOps.end()));

        // After this chain's body content, splice in any stall ops that must
        // execute after chain[chainIdx] and before chain[chainIdx+1].
        for (auto *stallOp : stallsAfterChain[chainIdx]) {
          stallOp->moveBefore(sharedYield);
        }
      }
    }
  }

  // Erase the old divergent scf.for chains.
  for (auto &ci : chains) {
    rewriter.eraseOp(ci.chain[firstDivergentLevel]);
  }

  return true;
}

static SmallVector<IntermediateAllocInfo>
findIntermediates(GenericOp genericOp,
                  SmallVectorImpl<ScratchLoopInfo> &scratchLoops) {
  SmallVector<IntermediateAllocInfo> intermediates;
  if (scratchLoops.size() < 2) {
    return intermediates;
  }

  genericOp.walk([&](memref::AllocOp allocOp) {
    if (!isIntermediateAlloc(allocOp, genericOp)) {
      return;
    }
    IntermediateAllocInfo allocInfo;
    allocInfo.allocOp = allocOp;
    allocInfo.producer = nullptr;
    allocInfo.lastConsumerIndex = 0;

    Value allocResult = allocOp.getResult();
    SmallVector<SmallVector<affine::AffineStoreOp>> storesByLoop(
        scratchLoops.size());

    // Find which loop nests write to and read from this allocation.
    // Keep per-loop stores in a side table so we can later answer: "which
    // loop is the earliest writer whose value is consumed by a later loop?"
    for (auto indexedLoop : llvm::enumerate(scratchLoops)) {
      size_t loopIdx = indexedLoop.index();
      auto &loopInfo = indexedLoop.value();
      SmallVector<affine::AffineStoreOp> stores;
      collectStoresInLoop(allocResult, loopInfo, stores);
      if (!stores.empty()) {
        storesByLoop[loopIdx] = std::move(stores);
      }

      SmallVector<affine::AffineLoadOp> loads;
      collectLoadsInLoop(allocResult, loopInfo, loads);
      if (!loads.empty()) {
        allocInfo.consumers.push_back(
            ConsumerLoopLoads{&loopInfo, std::move(loads)});
      }
    }

    // Only include if we found a producer and at least one consumer in a
    // *subsequent* scratch_space_loop (i.e., consumer index > producer
    // index). A consumer that appears before or at the same loop as the
    // producer does not require cross-loop scratch spilling.
    //
    // Example:
    //   loop 0: alloc = sin(...)
    //   loop 1: tmp = load alloc; alloc = add(tmp, ...)
    //
    // loop 1 both reads and writes the same alloc, but loop 0 is still the
    // true producer of the value that must survive into loop 1. If we chose
    // loop 1 as the producer just because it is a later writer, we would lose
    // the real producer->consumer dependency and either spill the wrong thing
    // or fail to spill at all.
    bool hasSubsequentExternalConsumer = false;
    for (auto indexedStores : llvm::enumerate(storesByLoop)) {
      size_t loopIdx = indexedStores.index();
      auto &stores = indexedStores.value();
      if (stores.empty()) {
        continue;
      }

      // Pick the earliest writer with a later read so in-place consumer
      // updates do not replace the true producer for this intermediate.
      size_t firstLaterWriterIdx = scratchLoops.size();
      for (size_t laterIdx = loopIdx + 1; laterIdx < storesByLoop.size();
           ++laterIdx) {
        if (!storesByLoop[laterIdx].empty()) {
          firstLaterWriterIdx = laterIdx;
          break;
        }
      }

      bool hasLaterConsumer =
          llvm::any_of(allocInfo.consumers, [&](const ConsumerLoopLoads &c) {
            size_t consIdx = static_cast<size_t>(c.loop - scratchLoops.data());
            return consIdx > loopIdx && consIdx <= firstLaterWriterIdx;
          });
      if (!hasLaterConsumer) {
        continue;
      }

      allocInfo.producer = &scratchLoops[loopIdx];
      allocInfo.lastConsumerIndex = firstLaterWriterIdx;
      allocInfo.stores = std::move(stores);
      hasSubsequentExternalConsumer = true;
      break;
    }
    if (hasSubsequentExternalConsumer) {
      intermediates.push_back(allocInfo);
    }
  });

  return intermediates;
}

/// Repeatedly fuse dependence-related sibling scratch loops.
static bool
fuseScratchLoopSiblings(GenericOp genericOp,
                        SmallVector<affine::AffineForOp> &scratchSpaceLoops,
                        IRRewriter &rewriter) {
  bool changed = false;
  bool madeProgress = true;

  while (madeProgress) {
    madeProgress = false;

    SmallVector<ScratchLoopInfo> loopShells;
    loopShells.reserve(scratchSpaceLoops.size());
    for (auto scratchLoop : scratchSpaceLoops) {
      ScratchLoopInfo info;
      info.scratchLoop = scratchLoop;
      loopShells.push_back(info);
    }

    SmallVector<IntermediateAllocInfo> intermediates =
        findIntermediates(genericOp, loopShells);
    if (intermediates.empty()) {
      break;
    }

    size_t bestBegin = scratchSpaceLoops.size();
    size_t bestEnd = 0;
    // Pick the widest producer->consumer span so one fusion covers the whole chain.
    for (auto &intermediate : intermediates) {
      size_t producerIdx =
          static_cast<size_t>(intermediate.producer - loopShells.data());
      for (auto &consumer : intermediate.consumers) {
        if (consumer.loop == intermediate.producer) {
          continue;
        }
        size_t consumerIdx =
            static_cast<size_t>(consumer.loop - loopShells.data());
        if (consumerIdx <= producerIdx ||
            consumerIdx > intermediate.lastConsumerIndex) {
          continue;
        }
        if (bestBegin == scratchSpaceLoops.size() ||
            consumerIdx - producerIdx > bestEnd - bestBegin) {
          bestBegin = producerIdx;
          bestEnd = consumerIdx;
        }
      }
    }

    if (bestBegin == scratchSpaceLoops.size()) {
      break;
    }

    SmallVector<affine::AffineForOp> loopsToFuse;
    for (size_t i = bestBegin; i <= bestEnd; ++i) {
      loopsToFuse.push_back(scratchSpaceLoops[i]);
    }

    if (manuallyFuseScratchLoopGroup(loopsToFuse, rewriter)) {
      scratchSpaceLoops = collectScratchSpaceLoops(genericOp);
      changed = madeProgress = true;
    }
  }

  return changed;
}

/// Main pass implementation
class D2MInsertSpillAndScratch
    : public impl::D2MInsertSpillAndScratchBase<D2MInsertSpillAndScratch> {
public:
  using impl::D2MInsertSpillAndScratchBase<
      D2MInsertSpillAndScratch>::D2MInsertSpillAndScratchBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](GenericOp genericOp) {
      if (failed(processGenericOp(genericOp))) {
        // Continue processing other generics even if one fails
      }
    });
  }

private:
  LogicalResult processGenericOp(GenericOp genericOp) {
    IRRewriter rewriter(&getContext());

    // Find all scratch_space_loops first (before any modifications).
    //
    // Gather the candidate loops up front because later rewriting may erase or
    // re-parent surrounding structure. Skip loops already marked as processed
    // so the pass is idempotent if it is run more than once.
    SmallVector<affine::AffineForOp> scratchSpaceLoops =
        collectScratchSpaceLoops(genericOp);

    if (scratchSpaceLoops.empty()) {
      return success();
    }

    // Fuse outer scf.for loops across scratch_space_loop nests.
    // This ensures producer/consumer computations share the outer iteration
    // space, allowing the scratch buffer to be reused per outer iteration
    // rather than needing full-shard capacity.
    if (!fuseOuterScfLoops(scratchSpaceLoops, rewriter)) {
      return success();
    }

    // Regather, fuse sibling loops and gather again.
    scratchSpaceLoops = collectScratchSpaceLoops(genericOp);
    fuseScratchLoopSiblings(genericOp, scratchSpaceLoops, rewriter);
    scratchSpaceLoops = collectScratchSpaceLoops(genericOp);

    // Collect structural information for each scratch_space_loop.
    //
    // Only keep loops where we can identify a linalg_root and where every
    // relevant loop bound/step is static enough to build a concrete scratch
    // shape and affine access map.
    SmallVector<ScratchLoopInfo> scratchLoops;
    for (auto scratchLoop : scratchSpaceLoops) {
      ScratchLoopInfo info;
      info.scratchLoop = scratchLoop;
      info.linalgRoot = findLinalgRootLoop(scratchLoop);

      if (info.linalgRoot) {
        collectAllInnerLoops(scratchLoop, info.allLoops);
        if (computeScratchShape(info)) {
          scratchLoops.push_back(info);
        }
      }
    }

    if (scratchLoops.size() < 2) {
      // Need at least 2 loop nests to have producer/consumer relationship.
      return success();
    }

    // Find intermediate allocations that connect loop nests.
    //
    // Looking for the pattern:
    //   scratch_space_loop P writes alloc A
    //   scratch_space_loop C (later in program order) reads alloc A
    SmallVector<IntermediateAllocInfo> intermediates =
        findIntermediates(genericOp, scratchLoops);

    if (intermediates.empty()) {
      // Mark loops as processed even if no intermediates found.
      for (auto &info : scratchLoops) {
        info.scratchLoop->setAttr("d2m.scratch_inserted",
                                  UnitAttr::get(&getContext()));
      }
      return success();
    }

    // Replace intermediate allocations with scratch buffers.
    auto l1Attr = ttcore::MemorySpaceAttr::get(&getContext(),
                                               ttcore::MemorySpace::DeviceL1);

    int64_t slotIndex = 0;
    llvm::SmallPtrSet<Operation *, 8> guardedScratchConsumers;
    for (auto &allocInfo : intermediates) {
      ScratchLoopInfo &producer = *allocInfo.producer;

      // Get element type from the original allocation.
      MemRefType origType = allocInfo.allocOp.getType();
      Type elementType = origType.getElementType();

      // Create scratch buffer type based on loop structure.
      // Shape: [scratch_iters, gap_iters..., root_iters, inner_iters...]
      auto scratchMemRefType = MemRefType::get(
          producer.scratchShape, elementType, AffineMap(), l1Attr);

      // Insert scratch allocation at the same location as the original alloc.
      // This ensures proper dominance for both producer and consumer.
      rewriter.setInsertionPoint(allocInfo.allocOp);
      Location loc = allocInfo.allocOp.getLoc();

      // Each intermediate gets a unique slot index. Sizes are inferred from
      // the memref type.
      auto scratchOp = rewriter.create<ScratchAllocateOp>(
          loc, scratchMemRefType, slotIndex++);
      Value scratchBuf = scratchOp.getResult();

      // Update stores in the producer loop.
      // Use an affine map to normalize loop IVs (handling step > 1).
      auto [producerMap, producerOperands] =
          getScratchAccessMapAndOperands(producer, &getContext());

      for (auto storeOp : allocInfo.stores) {
        rewriter.setInsertionPoint(storeOp);
        Value storedValue = storeOp.getValue();

        rewriter.create<affine::AffineStoreOp>(storeOp.getLoc(), storedValue,
                                               scratchBuf, producerMap,
                                               producerOperands);

        rewriter.eraseOp(storeOp);
      }

      // Update loads in all consumer loops.
      for (auto &consumerEntry : allocInfo.consumers) {
        if (consumerEntry.loop == allocInfo.producer) {
          continue;
        }
        size_t consumerIdx =
            static_cast<size_t>(consumerEntry.loop - scratchLoops.data());
        if (consumerIdx > allocInfo.lastConsumerIndex) {
          continue;
        }

        ScratchLoopInfo &consumer = *consumerEntry.loop;
        ensureUnpackStallBeforeConsumer(producer, consumer, rewriter,
                                        guardedScratchConsumers);

        // Use producer-aware map so that consumers with a different linalgRoot
        // step size correctly address the scratch buffer (which is laid out
        // according to the producer's step).
        auto [consumerMap, consumerOperands] =
            getConsumerScratchAccessMap(producer, consumer, &getContext());

        for (auto loadOp : consumerEntry.loads) {
          rewriter.setInsertionPoint(loadOp);

          Value reloaded = rewriter.create<affine::AffineLoadOp>(
              loadOp.getLoc(), scratchBuf, consumerMap, consumerOperands);

          rewriter.replaceOp(loadOp, reloaded);
        }
      }

      // Clean up dead pieces of the original local-allocation path.
      SmallVector<memref::SubViewOp> subviews;
      for (Operation *user : allocInfo.allocOp.getResult().getUsers()) {
        if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
          subviews.push_back(subview);
        }
      }

      for (auto subview : subviews) {
        if (subview.getResult().use_empty()) {
          rewriter.eraseOp(subview);
        }
      }

      if (allocInfo.allocOp.getResult().use_empty()) {
        rewriter.eraseOp(allocInfo.allocOp);
      }
    }

    // Mark all scratch loops as processed so a repeated run of this pass does
    // not try to insert a second layer of scratch for the same loop nests.
    for (auto &info : scratchLoops) {
      info.scratchLoop->setAttr("d2m.scratch_inserted",
                                UnitAttr::get(&getContext()));
    }

    return success();
  }
};

} // namespace

} // namespace mlir::tt::d2m
