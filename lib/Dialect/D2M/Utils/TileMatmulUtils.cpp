// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/TileMatmulUtils.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m::utils {

namespace {

// Helper to check if a memref is in DST memory space.
bool isDstMemspace(MemRefType memrefType) {
  auto memSpaceAttr = mlir::dyn_cast_or_null<ttcore::MemorySpaceAttr>(
      memrefType.getMemorySpace());
  return memSpaceAttr &&
         memSpaceAttr.getValue() == ttcore::MemorySpace::RegisterDst;
}

// Look through subview operations to find the original block argument.
BlockArgument lookThroughSubView(Value memref) {
  while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
             memref.getDefiningOp())) {
    memref = subView.getSource();
  }
  if (auto *definingOp = memref.getDefiningOp();
      mlir::isa_and_nonnull<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
    memref = definingOp->getOperand(0);
  }
  return mlir::dyn_cast<BlockArgument>(memref);
}

// Collect DST access information from the region.
// Returns the CB element type and maximum DST slice index needed.
std::pair<MemRefType, int64_t> collectMatmulDstInfo(Region &region,
                                                    Operation *outermostLoop) {
  MemRefType cbType = nullptr;
  int64_t maxDstSlice = 0; // Matmul uses single DST slice for accumulator

  // Walk the region to find affine loads/stores from compute ops.
  region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
    // For matmul, we need the CB type from the output operand.
    for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
      if (computeOp.isScalarOperand(operandIdx)) {
        continue;
      }

      Value operand = computeOp->getOperand(operandIdx);
      if (auto loadOp = operand.getDefiningOp<affine::AffineLoadOp>()) {
        if (!isDstMemspace(loadOp.getMemRefType())) {
          cbType = loadOp.getMemRefType();
        }
      }
    }
  });

  return {cbType, maxDstSlice};
}

// Create acquire_dst operation with appropriate shape.
AcquireDstOp createAcquireDst(RewriterBase &rewriter, Location loc,
                              Region &region, Operation *outermostLoop,
                              MemRefType cbType, unsigned totalDstTiles) {
  if (outermostLoop) {
    rewriter.setInsertionPoint(outermostLoop);
  } else {
    rewriter.setInsertionPointToStart(&region.front());
  }

  // Calculate DST shape: [numSlices, cbShape...]
  const int64_t volume = ttmlir::utils::volume(cbType.getShape());
  TT_assert(volume <= totalDstTiles);
  const int64_t numDstSlices = totalDstTiles / volume;

  SmallVector<int64_t> dstShape({numDstSlices});
  dstShape.append(cbType.getShape().begin(), cbType.getShape().end());

  MemRefType dstType =
      MemRefType::get(dstShape, cbType.getElementType(),
                      mlir::AffineMap::getMultiDimIdentityMap(
                          dstShape.size(), rewriter.getContext()),
                      rewriter.getAttr<ttcore::MemorySpaceAttr>(
                          ttcore::MemorySpace::RegisterDst));

  return rewriter.create<AcquireDstOp>(loc, dstType);
}

// Insert a guard for prologue loops that should only execute on non-first
// iterations (for matmul accumulation). The guard checks if any of the
// reduction loop indices are non-zero using d2m.iter_index ops.
//
// Returns the created scf::IfOp, or nullptr if no guard is needed.
scf::IfOp insertGuardForReductionIterations(RewriterBase &rewriter, Location loc,
                                            ArrayRef<int64_t> guardIndices) {
  if (guardIndices.empty()) {
    return nullptr;
  }
  auto zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  auto cmp = rewriter
                 .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                            rewriter.getBoolAttr(false))
                 .getResult();
  for (int64_t index : guardIndices) {
    auto iterIndex = rewriter.create<IterIndexOp>(loc, index);
    auto ne = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                             iterIndex, zero);
    cmp = rewriter.create<arith::OrIOp>(loc, cmp, ne).getResult();
  }
  return rewriter.create<scf::IfOp>(loc, cmp);
}

// Clone a loop nest and erase all non-loop operations from the clone.
// Returns the cloned outermost loop and the IRMapping that maps original
// values (including induction variables) to cloned values.
std::pair<affine::AffineForOp, IRMapping>
cloneLoopNestStructure(RewriterBase &rewriter, affine::AffineForOp forLoop) {
  IRMapping mapper;
  auto *cloned = rewriter.clone(*forLoop.getOperation(), mapper);

  // Erase all non-loop operations from the cloned loop nest.
  cloned->walk([&](Operation *op) {
    if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp,
                   affine::AffineApplyOp>(op)) {
      op->dropAllUses();
      rewriter.eraseOp(op);
    }
  });

  return {cast<affine::AffineForOp>(cloned), std::move(mapper)};
}

// Generate prologue loop to copy from L1 CB to DST.
// Uses the current insertion point - caller is responsible for setting it.
void generatePrologueLoop(RewriterBase &rewriter, Location loc,
                          affine::AffineForOp forLoop, Value dst,
                          affine::AffineLoadOp srcLoad, int64_t dstSlice) {
  // Clone the loop nest structure (erases non-loop ops from clone).
  std::pair<affine::AffineForOp, IRMapping> cloneResult =
      cloneLoopNestStructure(rewriter, forLoop);
  IRMapping &mapper = cloneResult.second;

  // Find the cloned block corresponding to where the original load was.
  Block *fromScope = srcLoad->getBlock();
  Block *toScope = mapper.lookupOrNull(fromScope);
  if (toScope) {
    Operation *terminator = toScope->getTerminator();
    if (terminator) {
      rewriter.setInsertionPoint(terminator);
    } else {
      rewriter.setInsertionPointToEnd(toScope);
    }
  }

  // Map indices from original load to cloned loops.
  SmallVector<Value> mappedIndices = llvm::map_to_vector(
      srcLoad.getIndices(),
      [&mapper](Value idx) { return mapper.lookupOrDefault(idx); });

  // Load from L1 using original map with mapped indices.
  auto l1Value = rewriter.create<affine::AffineLoadOp>(
      loc, srcLoad.getMemRef(), srcLoad.getMap(), mappedIndices);

  // Store to DST with slice index inserted at position 0 of the map.
  AffineMap dstMap = srcLoad.getMap().insertResult(
      getAffineConstantExpr(dstSlice, rewriter.getContext()), 0);
  rewriter.create<affine::AffineStoreOp>(loc, l1Value.getResult(), dst, dstMap,
                                         mappedIndices);
}

// Generate epilogue loop to copy from DST to L1 CB (pack).
void generateEpilogueLoop(RewriterBase &rewriter, Location loc,
                          affine::AffineForOp forLoop, Value dst,
                          affine::AffineStoreOp dstStore, int64_t dstSlice) {
  rewriter.setInsertionPointAfter(forLoop);

  // Clone the loop nest structure (erases non-loop ops from clone).
  std::pair<affine::AffineForOp, IRMapping> cloneResult =
      cloneLoopNestStructure(rewriter, forLoop);
  IRMapping &mapper = cloneResult.second;

  // Find the cloned block corresponding to where the original store was.
  Block *fromScope = dstStore->getBlock();
  Block *toScope = mapper.lookupOrNull(fromScope);
  if (toScope) {
    Operation *terminator = toScope->getTerminator();
    if (terminator) {
      rewriter.setInsertionPoint(terminator);
    } else {
      rewriter.setInsertionPointToEnd(toScope);
    }
  }

  // Map indices from original store to cloned loops.
  SmallVector<Value> mappedIndices = llvm::map_to_vector(
      dstStore.getIndices(),
      [&mapper](Value idx) { return mapper.lookupOrDefault(idx); });

  // Load from DST with slice index inserted at position 0 of the map.
  AffineMap dstMap = dstStore.getMap().insertResult(
      getAffineConstantExpr(dstSlice, rewriter.getContext()), 0);
  auto dstValue =
      rewriter.create<affine::AffineLoadOp>(loc, dst, dstMap, mappedIndices);

  // Store to L1 using original map with mapped indices.
  rewriter.create<affine::AffineStoreOp>(loc, dstValue.getResult(),
                                         dstStore.getMemRef(), dstStore.getMap(),
                                         mappedIndices);
}

} // namespace

bool hasTileMatmulOp(linalg::GenericOp linalgOp) {
  return linalgOp
      ->walk([](d2m::TileMatmulOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

FailureOr<TileMatmulBlockOp> convertTileMatmulLinalgToBlock(
    RewriterBase &rewriter, GenericOp genericOp, linalg::GenericOp linalgOp,
    Region &region,
    llvm::function_ref<LogicalResult(RewriterBase &, Region &, Operation *)>
        dstInsertionCallback) {

  // Verify preconditions
  if (linalgOp.getInputs().size() != 2) {
    return linalgOp.emitError("Expected exactly 2 inputs for tile_matmul");
  }
  if (linalgOp.getOutputs().size() != 1) {
    return linalgOp.emitError("Expected exactly 1 output for tile_matmul");
  }

  // Extract operands before converting
  Value inputAMemref = linalgOp.getInputs()[0];
  Value inputBMemref = linalgOp.getInputs()[1];
  Value outputCMemref = linalgOp.getOutputs()[0];
  Location loc = linalgOp.getLoc();

  // Convert linalg.generic to affine loops
  rewriter.setInsertionPoint(linalgOp);
  auto linalgLoops = linalg::linalgOpToAffineLoops(rewriter, linalgOp);
  if (failed(linalgLoops)) {
    return failure();
  }

  // Erase the original linalg operation
  rewriter.eraseOp(linalgOp);

  // Get the outermost loop (or nullptr if no loops created)
  Operation *outermostLoop =
      !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr;

  // Invoke callback to insert DST operations in the affine loop region
  if (failed(dstInsertionCallback(rewriter, region, outermostLoop))) {
    return failure();
  }

  // Create the tile_matmul_block operation BEFORE deleting the loops
  // (while the operands are still accessible), but position it where
  // the loops currently are so it remains valid after loop deletion.
  TileMatmulBlockOp blockOp;
  if (!linalgLoops.value().empty()) {
    // Position before the outermost loop
    rewriter.setInsertionPoint(linalgLoops.value().front());
  } else {
    // No loops were created, insert at end of region
    rewriter.setInsertionPointToEnd(&region.front());
  }
  blockOp = rewriter.create<d2m::TileMatmulBlockOp>(
      loc, inputAMemref, inputBMemref, outputCMemref);

  // Delete the temporary affine loops (they were only needed for DST
  // insertion)
  if (!linalgLoops.value().empty()) {
    for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
      rewriter.eraseOp(loopOp);
    }
  }

  return blockOp;
}

LogicalResult processTileMatmulLinalgOps(
    RewriterBase &rewriter, GenericOp genericOp, Region &region,
    llvm::function_ref<LogicalResult(RewriterBase &, Region &, Operation *)>
        dstInsertionCallback) {
  // Process linalg.generic ops that contain tile_matmul operations.
  // These are left unconverted by LinalgToAffine when useTileMatmul=false.
  Block &block = region.front();
  WalkResult walkResult = block.walk([&](linalg::GenericOp linalgOp) {
    if (hasTileMatmulOp(linalgOp)) {
      // Only handle tile_matmul when not in explicit datamovement form.
      // Explicit datamovement form should be converted by LinalgToAffine pass.
      if (!genericOp.isExplicitDatamovementForm()) {
        // Convert linalg to tile_matmul_block with DST allocation callback.
        auto result = convertTileMatmulLinalgToBlock(
            rewriter, genericOp, linalgOp, region, dstInsertionCallback);
        if (failed(result)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
    }

    // All other linalg ops should have been converted by LinalgToAffine pass.
    // If we encounter one in non-datamovement form, that's an error.
    if (!genericOp.isExplicitDatamovementForm()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return genericOp.emitError(
        "linalg.generic operations were not converted to affine loops");
  }
  return success();
}

LogicalResult insertMatmulDstAllocation(RewriterBase &rewriter,
                                        GenericOp genericOp, Region &region,
                                        Operation *outermostLoop,
                                        unsigned totalDstTiles) {
  // Collect DST access information from the temporary affine loops.
  auto [cbType, maxDstSlice] = collectMatmulDstInfo(region, outermostLoop);

  if (!cbType) {
    // No DST accesses found - nothing to do.
    return success();
  }

  Location loc = genericOp.getLoc();

  // Guard indices will be computed from the accumulator load's memref.
  SmallVector<int64_t> guardIndices;

  // 1. Create acquire_dst before the outermost loop.
  AcquireDstOp acquireDst = createAcquireDst(
      rewriter, loc, region, outermostLoop, cbType, totalDstTiles);
  Value dst = acquireDst.getResult();

  // 2. Find the affine store to the output CB (the matmul result store).
  //    This will be used to generate the epilogue pack operation.
  affine::AffineStoreOp outputStore;
  affine::AffineLoadOp accumulatorLoad;

  region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
    // Find the accumulator load (operand 2 for matmul).
    for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
      if (computeOp.isScalarOperand(operandIdx)) {
        continue;
      }
      Value operand = computeOp->getOperand(operandIdx);
      if (auto loadOp = operand.getDefiningOp<affine::AffineLoadOp>()) {
        if (!isDstMemspace(loadOp.getMemRefType())) {
          accumulatorLoad = loadOp;
          // Compute guard indices from the accumulator's block argument.
          // This follows the same pattern as the legacy pass.
          BlockArgument blockArg = lookThroughSubView(loadOp.getMemRef());
          if (blockArg && !genericOp.isExplicitDatamovementForm()) {
            guardIndices =
                genericOp.getNonParticipatingLoopDims(blockArg.getArgNumber());
          }
        }
      }
    }

    // Find the output store.
    for (auto *user : computeOp->getUsers()) {
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(user)) {
        if (!isDstMemspace(storeOp.getMemRefType())) {
          outputStore = storeOp;
        }
      }
    }
  });

  // 3. Generate prologue loop to copy accumulator from L1 to DST.
  //    This is wrapped in a guard to skip on the first iteration of the
  //    reduction loop (when the accumulator hasn't been initialized yet).
  if (accumulatorLoad && outermostLoop) {
    auto forOp = cast<affine::AffineForOp>(outermostLoop);
    rewriter.setInsertionPoint(forOp);
    scf::IfOp guard =
        insertGuardForReductionIterations(rewriter, loc, guardIndices);
    if (guard) {
      rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
    }
    // generatePrologueLoop uses current insertion point (either before forOp
    // or inside guard's then block).
    generatePrologueLoop(rewriter, loc, forOp, dst, accumulatorLoad, 0);
  }

  // 4. Generate epilogue to pack from DST back to output CB.
  if (outputStore && outermostLoop) {
    auto forOp = cast<affine::AffineForOp>(outermostLoop);
    generateEpilogueLoop(rewriter, loc, forOp, dst, outputStore, 0);
  }

  // Note: We don't need to rewrite the original load/store operations
  // because the affine loops are temporary and will be deleted by
  // convertTileMatmulLinalgToBlock after this callback returns. The
  // important pieces are the prologue/epilogue loops and acquire_dst.

  return success();
}

} // namespace mlir::tt::d2m::utils
