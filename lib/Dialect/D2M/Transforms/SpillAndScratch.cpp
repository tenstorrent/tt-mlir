// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPILLANDSCRATCH
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Determine if an operation is a D2M tile compute op that could be a "leaf"
/// in the compute DAG (i.e., its result is consumed by another tile op).
static bool isTileComputeOp(Operation *op) {
  return isa<TileAddOp, TileSubOp, TileMulOp, TileDivOp>(op);
}

/// Find the innermost affine.for loop with d2m.linalg_root attribute
static affine::AffineForOp findLinalgRootLoop(scf::ForOp scratchLoop) {
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

/// Analyze the compute DAG to find leaf ops (ops whose results are used by
/// other tile compute ops) and build a mapping from leaf results to scratch
/// slots.
static void analyzeComputeDAG(affine::AffineForOp rootLoop,
                              SmallVector<Operation *> &leafOps,
                              DenseMap<Value, unsigned> &leafResultToSlot) {
  // Find the innermost loop body
  affine::AffineForOp innermostLoop = rootLoop;
  rootLoop.getBody()->walk(
      [&](affine::AffineForOp forOp) { innermostLoop = forOp; });

  unsigned slotIdx = 0;
  innermostLoop.getBody()->walk([&](Operation *op) {
    if (!isTileComputeOp(op)) {
      return;
    }

    // Check if this op's result is used by another tile compute op
    bool isLeaf = false;
    for (Operation *user : op->getResult(0).getUsers()) {
      if (isTileComputeOp(user)) {
        isLeaf = true;
        break;
      }
    }

    if (isLeaf) {
      leafOps.push_back(op);
      leafResultToSlot[op->getResult(0)] = slotIdx++;
    }
  });
}

/// Pattern to match scf.for loops with d2m.scratch_space_loop attribute
/// and insert scratch allocations with spill/reload stores.
struct InsertSpillAndScratchPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    // Only match loops tagged with d2m.scratch_space_loop
    if (!forOp->hasAttr("d2m.scratch_space_loop")) {
      return failure();
    }

    // Check if we've already processed this loop
    if (forOp->hasAttr("d2m.scratch_inserted")) {
      return failure();
    }

    // Find the innermost affine loop with d2m.linalg_root
    affine::AffineForOp rootLoop = findLinalgRootLoop(forOp);
    if (!rootLoop) {
      return failure();
    }

    // Find the parent d2m.generic to get the element type
    auto genericOp = forOp->getParentOfType<GenericOp>();
    if (!genericOp) {
      return failure();
    }

    // Get tile element type from the first output operand of the generic
    Type tileType;
    if (!genericOp.getOutputs().empty()) {
      auto outType = genericOp.getOutputs().front().getType();
      if (auto memrefType = dyn_cast<MemRefType>(outType)) {
        tileType = memrefType.getElementType();
      }
    }

    if (!tileType || !isa<ttcore::TileType>(tileType)) {
      return failure();
    }

    // Analyze the compute DAG
    SmallVector<Operation *> leafOps;
    DenseMap<Value, unsigned> leafResultToSlot;
    analyzeComputeDAG(rootLoop, leafOps, leafResultToSlot);

    // If no leaf ops found, nothing to spill
    if (leafOps.empty()) {
      rewriter.modifyOpInPlace(forOp, [&]() {
        forOp->setAttr("d2m.scratch_inserted", rewriter.getUnitAttr());
      });
      return success();
    }

    // Create L1 memory space attribute
    auto l1Attr = ttcore::MemorySpaceAttr::get(rewriter.getContext(),
                                               ttcore::MemorySpace::DeviceL1);

    // Collect the affine loop nest to determine scratch shape and indices
    SmallVector<affine::AffineForOp> affineLoopNest;
    affineLoopNest.push_back(rootLoop);
    rootLoop.getBody()->walk([&](affine::AffineForOp innerFor) {
      affineLoopNest.push_back(innerFor);
    });

    // Determine scratch buffer shape from the affine loop bounds
    // This creates a 2D shape matching the CB subview shapes (e.g., [1, 8])
    SmallVector<int64_t> scratchShape;
    SmallVector<Value> loopIndices;
    for (auto loopOp : affineLoopNest) {
      // Get the constant loop bounds
      if (!loopOp.hasConstantBounds()) {
        return failure();
      }
      int64_t upperBound = loopOp.getConstantUpperBound();
      int64_t lowerBound = loopOp.getConstantLowerBound();
      scratchShape.push_back(upperBound - lowerBound);
      loopIndices.push_back(loopOp.getInductionVar());
    }

    int64_t totalTiles = 1;
    for (int64_t dim : scratchShape) {
      totalTiles *= dim;
    }

    // Create scratch memref type with shape matching CB subviews
    auto scratchMemRefType =
        MemRefType::get(scratchShape, tileType, AffineMap(), l1Attr);

    // Insert scratch allocations before the scratch_space_loop
    rewriter.setInsertionPoint(forOp);
    Location loc = forOp.getLoc();

    // Allocate one scratch buffer per leaf op
    SmallVector<Value> scratchBuffers;
    for (size_t i = 0; i < leafOps.size(); ++i) {
      auto scratch = rewriter.create<ScratchAllocateOp>(loc, scratchMemRefType,
                                                        totalTiles);
      scratchBuffers.push_back(scratch.getResult());
    }

    // Insert spill stores after each leaf op
    for (auto [leafOp, slot] :
         llvm::zip(leafOps, llvm::seq<unsigned>(0, leafOps.size()))) {
      Value leafResult = leafOp->getResult(0);
      Value scratchBuf = scratchBuffers[slot];

      rewriter.setInsertionPointAfter(leafOp);
      rewriter.create<affine::AffineStoreOp>(leafOp->getLoc(), leafResult,
                                             scratchBuf, loopIndices);
    }

    // Find parent ops (ops that consume leaf results) and insert reloads
    SmallVector<Operation *> parentOps;
    for (Operation *leafOp : leafOps) {
      for (Operation *user : leafOp->getResult(0).getUsers()) {
        if (isTileComputeOp(user) && !llvm::is_contained(parentOps, user)) {
          parentOps.push_back(user);
        }
      }
    }

    // For each parent op, reload the operands that came from leaf ops
    for (Operation *parentOp : parentOps) {
      rewriter.setInsertionPoint(parentOp);

      for (unsigned i = 0; i < parentOp->getNumOperands(); ++i) {
        Value operand = parentOp->getOperand(i);
        auto it = leafResultToSlot.find(operand);
        if (it != leafResultToSlot.end()) {
          unsigned slot = it->second;
          Value scratchBuf = scratchBuffers[slot];

          // Insert reload from scratch using all loop indices
          Value reloaded = rewriter.create<affine::AffineLoadOp>(
              parentOp->getLoc(), scratchBuf, loopIndices);

          // Replace the operand with the reloaded value
          rewriter.modifyOpInPlace(
              parentOp, [&]() { parentOp->setOperand(i, reloaded); });
        }
      }
    }

    // Mark the loop as having scratch inserted
    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp->setAttr("d2m.scratch_inserted", rewriter.getUnitAttr());
      forOp->setAttr("d2m.num_scratch_buffers",
                     rewriter.getI64IntegerAttr(leafOps.size()));
    });

    return success();
  }
};

class D2MSpillAndScratch
    : public impl::D2MSpillAndScratchBase<D2MSpillAndScratch> {
public:
  using impl::D2MSpillAndScratchBase<
      D2MSpillAndScratch>::D2MSpillAndScratchBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<InsertSpillAndScratchPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
