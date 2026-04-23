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

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTSPILLANDSCRATCH
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Information about a loop nest with d2m.scratch_space_loop
struct ScratchLoopInfo {
  affine::AffineForOp scratchLoop; // The loop with d2m.scratch_space_loop attr
  affine::AffineForOp linalgRoot;  // The nested loop with d2m.linalg_root attr
  // All affine loops nested inside scratch_space_loop (linalg_root and inner
  // loops), in top-down order. Does NOT include scratch_space_loop itself.
  SmallVector<affine::AffineForOp> allLoops;
  SmallVector<int64_t> scratchShape; // Shape for scratch buffer
};

/// A single consumer loop and the loads it contains for a given intermediate.
struct ConsumerLoopLoads {
  ScratchLoopInfo *loop;
  SmallVector<affine::AffineLoadOp> loads;
};

/// Information about an intermediate allocation that needs to become scratch
struct IntermediateAllocInfo {
  memref::AllocOp allocOp;
  ScratchLoopInfo *producer; // Loop nest that writes to this alloc
  SmallVector<ConsumerLoopLoads>
      consumers; // all consumer loop nests that read from it
  SmallVector<affine::AffineStoreOp> stores; // Stores to this alloc
};

/// Find the innermost affine.for loop with d2m.linalg_root attribute.
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

/// Check if a memref is an intermediate allocation (not an input/output of the
/// generic op, but created inside the generic region)
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

/// Check if a memref value traces back to the target allocation through
/// subviews
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

/// Collect all affine stores to a given memref within a loop nest
static void collectStoresInLoop(Value memref, ScratchLoopInfo &loopInfo,
                                SmallVector<affine::AffineStoreOp> &stores) {
  loopInfo.scratchLoop.walk([&](affine::AffineStoreOp storeOp) {
    if (tracesToAlloc(storeOp.getMemref(), memref)) {
      stores.push_back(storeOp);
    }
  });
}

/// Collect all affine loads from a given memref within a loop nest
static void collectLoadsInLoop(Value memref, ScratchLoopInfo &loopInfo,
                               SmallVector<affine::AffineLoadOp> &loads) {
  loopInfo.scratchLoop.walk([&](affine::AffineLoadOp loadOp) {
    if (tracesToAlloc(loadOp.getMemref(), memref)) {
      loads.push_back(loadOp);
    }
  });
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

/// Build an access map for a consumer loading from a scratch buffer that was
/// shaped by a producer with a potentially different loop step.
///
/// Two cross-step cases are handled:
///
/// Column cross-step: the outer column loop (allLoops[0]) has a different
/// step between producer and consumer. The absolute tile column index
/// (allLoops[0]_IV + allLoops[-1]_IV) is decomposed using the producer's step.
///
/// Row cross-step: the outer row loop (scratchLoop) has a different step
/// between producer and consumer (e.g. eltwise step=1 fused with reduction
/// step=2). The absolute row index (scratchLoop_IV + inner_row_IVs) is
/// decomposed using the producer's scratchLoop step.
///
/// Column cross-step example: producer allLoops[0] step=4 shapes scratch as
/// [1,3,1,4]. Producer stores at [arg10, arg11/4, arg12, arg13] for arg11 in
/// {0,4,8}. Consumer allLoops[0] step=6 must access as:
///   [arg10, (arg11+arg13)/4, arg12, (arg11+arg13)%4]
/// instead of the wrong: [arg10, arg11/6, arg12, arg13].
static std::pair<AffineMap, SmallVector<Value>>
getConsumerScratchAccessMap(ScratchLoopInfo &producer,
                            ScratchLoopInfo &consumer, MLIRContext *ctx) {
  // Column cross-step: allLoops[0] (outer column loop) has a different step.
  bool needsColCrossStep = consumer.allLoops.size() >= 2 &&
                           !producer.allLoops.empty() &&
                           producer.allLoops.front().getStepAsInt() !=
                               consumer.allLoops.front().getStepAsInt();

  // Row cross-step: scratchLoop (outer row loop) has a different step.
  // Requires middle loops (inner row) to compute the absolute row index.
  bool needsRowCrossStep =
      consumer.allLoops.size() >= 3 && producer.scratchLoop.getStepAsInt() !=
                                           consumer.scratchLoop.getStepAsInt();

  if (!needsColCrossStep && !needsRowCrossStep) {
    return getScratchAccessMapAndOperands(consumer, ctx);
  }

  // Row cross-step (without column cross-step): consumer's scratchLoop spans
  // multiple producer scratch rows per iteration. Combine scratchLoop_IV and
  // inner-row IVs into an absolute row index, then decompose with the
  // producer's scratchLoop step.
  //
  // Example: producer step=1 shapes scratch as [4,1,1,4]. Consumer step=2 with
  // an inner row loop 0..2 accesses
  //   [scratchIV + innerRowIV, outerCol/step, 0, innerCol].
  if (needsRowCrossStep && !needsColCrossStep) {
    int64_t prodRowStep = producer.scratchLoop.getStepAsInt();

    SmallVector<AffineExpr> exprs;
    SmallVector<Value> operands;
    unsigned dimIdx = 0;

    // Scratch loop: raw IV (not normalized by step).
    operands.push_back(consumer.scratchLoop.getInductionVar());
    AffineExpr dScratchRow = getAffineDimExpr(dimIdx++, ctx);
    {
      int64_t lb = consumer.scratchLoop.getConstantLowerBound();
      if (lb != 0) {
        dScratchRow = dScratchRow - lb;
      }
    }

    auto addNormalizedLoop = [&](affine::AffineForOp loop) -> AffineExpr {
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

    // Outer col loop (allLoops[0]): normalized pass-through.
    AffineExpr outerColExpr = addNormalizedLoop(consumer.allLoops.front());

    // Middle loops (inner row): collect raw IVs for absolute row computation.
    AffineExpr innerRowSum = getAffineConstantExpr(0, ctx);
    for (size_t i = 1; i + 1 < consumer.allLoops.size(); ++i) {
      operands.push_back(consumer.allLoops[i].getInductionVar());
      AffineExpr d = getAffineDimExpr(dimIdx++, ctx);
      int64_t lb = consumer.allLoops[i].getConstantLowerBound();
      if (lb != 0) {
        d = d - lb;
      }
      innerRowSum = innerRowSum + d;
    }

    // Inner col loop (allLoops[-1]): normalized pass-through.
    AffineExpr innerColExpr = addNormalizedLoop(consumer.allLoops.back());

    // Absolute row = scratchLoop_IV_raw + inner_row_IVs_raw.
    AffineExpr absRow = dScratchRow + innerRowSum;

    // Build expressions matching producer's scratch shape:
    //   [absRow / prodRowStep, outerCol, absRow % prodRowStep, innerCol]
    exprs.push_back(absRow.floorDiv(prodRowStep));
    exprs.push_back(outerColExpr);
    for (size_t i = 1; i + 1 < consumer.allLoops.size(); ++i) {
      exprs.push_back(absRow % prodRowStep);
    }
    exprs.push_back(innerColExpr);

    return {AffineMap::get(dimIdx, 0, exprs, ctx), operands};
  }

  // Column cross-step case: consumer batches tile columns in different sizes
  // than the producer. Compute the absolute column tile index and decompose
  // it into the producer's scratch layout.
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
static void fuseOuterScfLoops(SmallVector<affine::AffineForOp> &scratchLoops,
                              IRRewriter &rewriter) {
  if (scratchLoops.size() < 2) {
    return;
  }

  struct ChainInfo {
    affine::AffineForOp scratchLoop;
    SmallVector<scf::ForOp> chain; // outermost first
  };

  SmallVector<ChainInfo> chains;
  for (auto sl : scratchLoops) {
    ChainInfo ci;
    ci.scratchLoop = sl;
    Operation *parent = sl->getParentOp();
    while (auto scfFor = dyn_cast<scf::ForOp>(parent)) {
      ci.chain.push_back(scfFor);
      parent = scfFor->getParentOp();
    }
    std::reverse(ci.chain.begin(), ci.chain.end());
    chains.push_back(ci);
  }

  size_t chainLen = chains[0].chain.size();
  if (chainLen == 0) {
    return;
  }
  for (auto &ci : chains) {
    if (ci.chain.size() != chainLen) {
      return;
    }
  }

  // Verify matching bounds/steps at each nesting level.
  for (size_t level = 0; level < chainLen; ++level) {
    auto ref = chains[0].chain[level];
    auto refLb = getConstantIntValue(ref.getLowerBound());
    auto refUb = getConstantIntValue(ref.getUpperBound());
    auto refStep = getConstantIntValue(ref.getStep());
    if (!refLb || !refUb || !refStep) {
      return;
    }
    for (size_t i = 1; i < chains.size(); ++i) {
      auto other = chains[i].chain[level];
      auto lb = getConstantIntValue(other.getLowerBound());
      auto ub = getConstantIntValue(other.getUpperBound());
      auto step = getConstantIntValue(other.getStep());
      if (!lb || !ub || !step) {
        return;
      }
      if (*refLb != *lb || *refUb != *ub || *refStep != *step) {
        return;
      }
    }
  }

  // Move operations between consecutive loop nests to before the first.
  // These are setup ops (allocs, remote_loads, etc.) that don't depend on
  // loop IVs and need to dominate the shared loop body.
  //
  // UnpackStallOnPackOp is special: it must remain ordered relative to the
  // compute it guards and must end up inside the fused loop body (after
  // chain[i]'s content, before chain[i+1]'s content). Collect these
  // separately and splice them in at the right position below.
  SmallVector<SmallVector<Operation *>> stallsAfterChain(chains.size());
  Operation *firstOuter = chains[0].chain[0];
  for (size_t i = 1; i < chains.size(); ++i) {
    Operation *thisOuter = chains[i].chain[0];
    SmallVector<Operation *> toMove;
    for (auto it = std::next(chains[i - 1].chain[0]->getIterator());
         &*it != thisOuter; ++it) {
      if (isa<UnpackStallOnPackOp>(&*it)) {
        stallsAfterChain[i - 1].push_back(&*it);
      } else {
        toMove.push_back(&*it);
      }
    }
    for (auto *op : toMove) {
      op->moveBefore(firstOuter);
    }
  }

  // Create the shared scf.for chain with matching bounds.
  rewriter.setInsertionPoint(firstOuter);
  SmallVector<scf::ForOp> shared;
  for (size_t level = 0; level < chainLen; ++level) {
    auto &ref = chains[0].chain[level];
    auto newLoop = rewriter.create<scf::ForOp>(
        ref.getLoc(), ref.getLowerBound(), ref.getUpperBound(), ref.getStep());
    shared.push_back(newLoop);
    rewriter.setInsertionPointToStart(newLoop.getBody());
  }

  // Move body operations from each old chain into the shared chain.
  for (size_t chainIdx = 0; chainIdx < chains.size(); ++chainIdx) {
    auto &ci = chains[chainIdx];
    // Replace old IVs with shared IVs at every nesting level.
    for (size_t level = 0; level < chainLen; ++level) {
      ci.chain[level].getInductionVar().replaceAllUsesWith(
          shared[level].getInductionVar());
    }

    for (size_t level = 0; level < chainLen; ++level) {
      Block *oldBody = ci.chain[level].getBody();
      Block *sharedBody = shared[level].getBody();
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
        Operation *innerSharedLoop = shared[level + 1].getOperation();
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

  // Erase old scf.for chains (outermost erases the entire chain).
  for (auto &ci : chains) {
    rewriter.eraseOp(ci.chain[0]);
  }
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

    // Step 0: Find all scratch_space_loops first (before any modifications).
    SmallVector<affine::AffineForOp> scratchSpaceLoops;
    genericOp.walk([&](affine::AffineForOp forOp) {
      if (forOp->hasAttr("d2m.scratch_space_loop") &&
          !forOp->hasAttr("d2m.scratch_inserted")) {
        scratchSpaceLoops.push_back(forOp);
      }
    });

    if (scratchSpaceLoops.empty()) {
      return success();
    }

    // Step 0.5: Fuse outer scf.for loops across scratch_space_loop nests.
    // This ensures producer/consumer computations share the outer iteration
    // space, allowing the scratch buffer to be reused per outer iteration
    // rather than needing full-shard capacity.
    fuseOuterScfLoops(scratchSpaceLoops, rewriter);

    // Step 1: Collect full loop info.
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

    // Step 3: Find intermediate allocations that connect the loop nests.
    SmallVector<IntermediateAllocInfo> intermediates;

    genericOp.walk([&](memref::AllocOp allocOp) {
      if (!isIntermediateAlloc(allocOp, genericOp)) {
        return;
      }

      IntermediateAllocInfo allocInfo;
      allocInfo.allocOp = allocOp;
      allocInfo.producer = nullptr;

      Value allocResult = allocOp.getResult();

      // Find which loop nests write to and read from this allocation.
      for (auto &loopInfo : scratchLoops) {
        SmallVector<affine::AffineStoreOp> stores;
        collectStoresInLoop(allocResult, loopInfo, stores);
        if (!stores.empty()) {
          allocInfo.producer = &loopInfo;
          allocInfo.stores = stores;
        }

        SmallVector<affine::AffineLoadOp> loads;
        collectLoadsInLoop(allocResult, loopInfo, loads);
        if (!loads.empty()) {
          allocInfo.consumers.push_back({&loopInfo, std::move(loads)});
        }
      }

      // Only include if we found a producer and at least one consumer in a
      // *subsequent* scratch_space_loop (i.e., consumer index > producer
      // index). A consumer that appears before or at the same loop as the
      // producer does not require cross-loop scratch spilling.
      bool hasSubsequentExternalConsumer = false;
      if (allocInfo.producer) {
        size_t producerIdx =
            static_cast<size_t>(allocInfo.producer - scratchLoops.data());
        hasSubsequentExternalConsumer =
            llvm::any_of(allocInfo.consumers, [&](const ConsumerLoopLoads &c) {
              if (c.loop == allocInfo.producer) {
                return false;
              }
              size_t consIdx =
                  static_cast<size_t>(c.loop - scratchLoops.data());
              return consIdx > producerIdx;
            });
      }
      if (hasSubsequentExternalConsumer) {
        intermediates.push_back(allocInfo);
      }
    });

    if (intermediates.empty()) {
      // Mark loops as processed even if no intermediates found.
      for (auto &info : scratchLoops) {
        info.scratchLoop->setAttr("d2m.scratch_inserted",
                                  UnitAttr::get(&getContext()));
      }
      return success();
    }

    // Step 4: Replace intermediate allocations with scratch buffers.
    auto l1Attr = ttcore::MemorySpaceAttr::get(&getContext(),
                                               ttcore::MemorySpace::DeviceL1);

    int64_t slotIndex = 0;
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

      // Step 5: Update stores in the producer loop.
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

      // Step 6: Rewrite loads in all consumer loops. Loads within the
      // producer's own loop (e.g. a reduction reading its own accumulator) use
      // the producer's map; loads in external loops use the cross-step-aware
      // consumer map.
      for (auto &consumerEntry : allocInfo.consumers) {
        bool isProducerSelfRead = (consumerEntry.loop == allocInfo.producer);

        auto [loadMap, loadOperands] =
            isProducerSelfRead
                ? getScratchAccessMapAndOperands(producer, &getContext())
                : getConsumerScratchAccessMap(producer, *consumerEntry.loop,
                                              &getContext());

        for (auto loadOp : consumerEntry.loads) {
          rewriter.setInsertionPoint(loadOp);

          Value reloaded = rewriter.create<affine::AffineLoadOp>(
              loadOp.getLoc(), scratchBuf, loadMap, loadOperands);

          rewriter.replaceOp(loadOp, reloaded);
        }
      }

      // Step 7: Clean up - remove subviews and old allocation if unused.
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

    // Mark all scratch loops as processed.
    for (auto &info : scratchLoops) {
      info.scratchLoop->setAttr("d2m.scratch_inserted",
                                UnitAttr::get(&getContext()));
    }

    return success();
  }
};

} // namespace

} // namespace mlir::tt::d2m
