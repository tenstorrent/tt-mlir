// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"

#include <numeric>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTTILEMATMULBLOCK
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Higher rank matmuls are incompatible with tile matmul block, but there
// is a special case if all outer ranks are 1's.
// e.g.
//   2x1x2x4 -> Not compatible
//   1x2x2x4 -> Not Compatible
//   1x1x2x4 -> Compatible (special case)
//
// Technically the rank > 2 special case seems feasible, but the indexing math
// for matmul block gets tripped up during ttkernel lowering. Filed follow
// on issue to support special case below:
//   https://github.com/tenstorrent/tt-mlir/issues/6955
// Once linked issue is fixed, the rank > 2 early out can be removed.
static bool canReplaceWithMatmulBlock(ShapedType shapedType) {
  if (shapedType.getRank() < 2) {
    return false;
  }

  if (shapedType.getRank() > 2) {
    return false;
  }

  auto outerShape = shapedType.getShape().drop_back(2);
  int64_t outerVolume = std::accumulate(outerShape.begin(), outerShape.end(), 1,
                                        std::multiplies<int64_t>());
  return outerVolume == 1;
}

static d2m::TileMatmulOp findTileMatmul(Operation *op) {
  d2m::TileMatmulOp result = nullptr;
  op->walk([&](d2m::TileMatmulOp matmulOp) {
    result = matmulOp;
    return WalkResult::interrupt();
  });
  return result;
}

// Trace a tile_matmul operand back through its defining affine.load to find
// the CB memref in L1 memory space.
static Value findL1MemrefFromOperand(Value operand) {
  auto loadOp = operand.getDefiningOp<affine::AffineLoadOp>();
  if (!loadOp) {
    return nullptr;
  }
  Value memref = loadOp.getMemRef();
  if (ttcore::getMemorySpace(memref) == ttcore::MemorySpace::DeviceL1) {
    return memref;
  }
  return nullptr;
}

// Find the output C memref by scanning sibling operations after the compute
// loop for the store copy loop pattern: affine.load from DST followed by
// affine.store to L1.
static Value findOutputMemrefFromStoreCopyLoop(Operation *computeLoop,
                                               Value dstValue) {
  Block *parentBlock = computeLoop->getBlock();
  for (auto it = std::next(Block::iterator(computeLoop));
       it != parentBlock->end(); ++it) {
    Value outputMemref = nullptr;
    it->walk([&](affine::AffineStoreOp storeOp) {
      Value target = storeOp.getMemRef();
      if (target == dstValue) {
        return WalkResult::advance();
      }
      if (ttcore::getMemorySpace(target) == ttcore::MemorySpace::DeviceL1) {
        outputMemref = target;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (outputMemref) {
      return outputMemref;
    }
  }
  return nullptr;
}

// When the store copy loop's affine.store folds a memref.cast (via
// canonicalization), we get the raw CB instead of the cast result. Look for a
// memref.cast or memref.subview of the raw CB defined before the compute loop
// and prefer that, since it preserves the strided type annotation expected by
// downstream passes.
static Value preferCastOrSubview(Value rawMemref, Operation *computeLoop) {
  Block *parentBlock = computeLoop->getBlock();
  for (auto it = parentBlock->begin(); it != Block::iterator(computeLoop);
       ++it) {
    if (auto castOp = dyn_cast<memref::CastOp>(&*it)) {
      if (castOp.getSource() == rawMemref) {
        return castOp.getResult();
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(&*it)) {
      if (subviewOp.getSource() == rawMemref) {
        return subviewOp.getResult();
      }
    }
  }
  return rawMemref;
}

static bool blockTransitivelyContainsAcquireDst(Block *block) {
  for (Operation &op : *block) {
    if (isa<d2m::AcquireDstOp>(op)) {
      return true;
    }
    for (Region &region : op.getRegions()) {
      for (Block &inner : region.getBlocks()) {
        if (blockTransitivelyContainsAcquireDst(&inner)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Root of the affine.for nest that implements tile_matmul after DST insertion.
// Outer loops (e.g. scratch_space / subview iteration) also use affine.for;
// stop climbing when the enclosing loop's body already contains acquire_dst,
// since that loop wraps load + compute + store rather than compute alone.
static affine::AffineForOp findComputeNestRoot(d2m::TileMatmulOp matmulOp) {
  affine::AffineForOp innermost = nullptr;
  for (Operation *walk = matmulOp->getParentOp(); walk;
       walk = walk->getParentOp()) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(walk)) {
      innermost = forOp;
      break;
    }
  }
  if (!innermost) {
    return nullptr;
  }

  affine::AffineForOp root = innermost;
  while (Operation *parentOp = root->getParentOp()) {
    auto parentFor = dyn_cast<affine::AffineForOp>(parentOp);
    if (!parentFor) {
      break;
    }
    if (blockTransitivelyContainsAcquireDst(parentFor.getBody())) {
      break;
    }
    root = parentFor;
  }
  return root;
}

class D2MInsertTileMatmulBlock
    : public impl::D2MInsertTileMatmulBlockBase<D2MInsertTileMatmulBlock> {
public:
  using impl::D2MInsertTileMatmulBlockBase<
      D2MInsertTileMatmulBlock>::D2MInsertTileMatmulBlockBase;

  void runOnOperation() final {
    if (useTileMatmul) {
      return;
    }

    ModuleOp moduleOp = getOperation();

    // Find all tile_matmul ops and determine their outermost compute loops.
    SmallVector<affine::AffineForOp> candidates;
    DenseSet<Operation *> seen;

    moduleOp->walk([&](d2m::TileMatmulOp matmulOp) {
      affine::AffineForOp outerLoop = findComputeNestRoot(matmulOp);
      if (outerLoop && !seen.contains(outerLoop.getOperation())) {
        seen.insert(outerLoop.getOperation());
        candidates.push_back(outerLoop);
      }
    });

    for (affine::AffineForOp candidate : candidates) {
      if (failed(processCandidate(candidate))) {
        candidate->emitOpError()
            << "failed to replace matmul compute loops with "
               "tile_matmul_block";
        return signalPassFailure();
      }
    }
  }

private:
  LogicalResult processCandidate(affine::AffineForOp computeLoop) {
    d2m::TileMatmulOp matmulOp = findTileMatmul(computeLoop);
    if (!matmulOp) {
      return computeLoop->emitOpError()
             << "compute loop does not contain a tile_matmul op";
    }

    // Find A and B memrefs from the tile_matmul's first two operands.
    Value inputA = findL1MemrefFromOperand(matmulOp->getOperand(0));
    Value inputB = findL1MemrefFromOperand(matmulOp->getOperand(1));
    if (!inputA || !inputB) {
      return matmulOp->emitOpError()
             << "could not find L1 input memrefs for tile_matmul operands";
    }

    // Find the DST value from the accumulator operand.
    auto accLoad =
        matmulOp->getOperand(2).getDefiningOp<affine::AffineLoadOp>();
    if (!accLoad) {
      return matmulOp->emitOpError()
             << "tile_matmul accumulator operand is not an affine.load";
    }
    Value dstValue = accLoad.getMemRef();

    // Find the output C memref from the store copy loop.
    Value outputC = findOutputMemrefFromStoreCopyLoop(computeLoop, dstValue);
    if (!outputC) {
      return computeLoop->emitOpError()
             << "could not find output L1 memref in store copy loop";
    }

    // Prefer a cast/subview of the raw CB if one exists before the loop.
    outputC = preferCastOrSubview(outputC, computeLoop);

    // Verify that the output shape is eligible for matmul block replacement.
    auto shapedType = mlir::cast<ShapedType>(outputC.getType());
    if (!canReplaceWithMatmulBlock(shapedType)) {
      return success();
    }

    // Insert tile_matmul_block right after the compute loop, then erase it.
    OpBuilder builder(computeLoop->getContext());
    builder.setInsertionPointAfter(computeLoop);
    builder.create<d2m::TileMatmulBlockOp>(computeLoop->getLoc(), inputA,
                                           inputB, outputC);
    computeLoop->erase();

    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
