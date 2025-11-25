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
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m::utils {

namespace {

// Helper to check if a memref is in DST memory space.
bool isDstMemspace(MemRefType memrefType) {
  auto memSpaceAttr =
      mlir::dyn_cast_or_null<ttcore::MemorySpaceAttr>(memrefType.getMemorySpace());
  return memSpaceAttr &&
         memSpaceAttr.getValue() == ttcore::MemorySpace::RegisterDst;
}

// Collect DST access information from the region.
// Returns the CB element type and maximum DST slice index needed.
std::pair<MemRefType, int64_t>
collectMatmulDstInfo(Region &region, Operation *outermostLoop) {
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

// Create release_dst operation at the end of the region.
void createReleaseDst(RewriterBase &rewriter, Location loc, Region &region,
                      Value dst) {
  Block &block = region.front();
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
    rewriter.setInsertionPoint(&block.back());
  } else {
    rewriter.setInsertionPointToEnd(&block);
  }
  rewriter.create<ReleaseDstOp>(loc, dst);
}

// Generate prologue loop to copy from L1 CB to DST.
void generatePrologueLoop(RewriterBase &rewriter, Location loc,
                          affine::AffineForOp forLoop, Value dst,
                          affine::AffineLoadOp srcLoad, int64_t dstSlice) {
  // Clone the loop structure for prologue.
  IRMapping mapper;
  rewriter.setInsertionPoint(forLoop);
  Operation *clonedLoop = rewriter.clone(*forLoop.getOperation(), mapper);
  auto prologueLoop = cast<affine::AffineForOp>(clonedLoop);

  // Replace the body with a copy from L1 to DST.
  Block &prologueBody = prologueLoop.getRegion().front();
  rewriter.setInsertionPointToStart(&prologueBody);

  // Find the innermost loop and its induction variable.
  affine::AffineForOp innermostLoop = prologueLoop;
  while (true) {
    auto nestedFor = innermostLoop.getRegion()
                         .front()
                         .getOps<affine::AffineForOp>();
    auto it = nestedFor.begin();
    if (it == nestedFor.end()) {
      break;
    }
    innermostLoop = *it;
  }

  // Clear the innermost loop body and add the copy.
  Block &innerBody = innermostLoop.getRegion().front();
  rewriter.setInsertionPointToStart(&innerBody);

  // Map indices from original load to cloned loop.
  SmallVector<Value> mappedIndices;
  for (Value idx : srcLoad.getIndices()) {
    mappedIndices.push_back(mapper.lookupOrDefault(idx));
  }

  // Load from L1.
  auto l1Value = rewriter.create<affine::AffineLoadOp>(
      loc, srcLoad.getMemRef(), srcLoad.getMap(), mappedIndices);

  // Store to DST with slice index prepended.
  SmallVector<Value> dstIndices;
  dstIndices.push_back(
      rewriter.create<arith::ConstantIndexOp>(loc, dstSlice));
  dstIndices.append(mappedIndices.begin(), mappedIndices.end());

  rewriter.create<affine::AffineStoreOp>(
      loc, l1Value.getResult(), dst,
      AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                        rewriter.getContext()),
      dstIndices);

  // Erase operations from original body that were cloned but not needed.
  // Keep only the yield.
  SmallVector<Operation *> toErase;
  for (Operation &op : innerBody) {
    if (!isa<affine::AffineYieldOp>(&op) &&
        &op != l1Value.getOperation() &&
        !isa<affine::AffineStoreOp>(&op)) {
      toErase.push_back(&op);
    }
  }
  for (Operation *op : llvm::reverse(toErase)) {
    rewriter.eraseOp(op);
  }
}

// Generate epilogue loop to copy from DST to L1 CB (pack).
void generateEpilogueLoop(RewriterBase &rewriter, Location loc,
                          affine::AffineForOp forLoop, Value dst,
                          affine::AffineStoreOp dstStore, int64_t dstSlice) {
  // Clone the loop structure for epilogue.
  IRMapping mapper;
  rewriter.setInsertionPointAfter(forLoop);
  Operation *clonedLoop = rewriter.clone(*forLoop.getOperation(), mapper);
  auto epilogueLoop = cast<affine::AffineForOp>(clonedLoop);

  // Find the innermost loop.
  affine::AffineForOp innermostLoop = epilogueLoop;
  while (true) {
    auto nestedFor = innermostLoop.getRegion()
                         .front()
                         .getOps<affine::AffineForOp>();
    auto it = nestedFor.begin();
    if (it == nestedFor.end()) {
      break;
    }
    innermostLoop = *it;
  }

  // Clear the innermost loop body and add the copy.
  Block &innerBody = innermostLoop.getRegion().front();
  rewriter.setInsertionPointToStart(&innerBody);

  // Map indices from original store to cloned loop.
  SmallVector<Value> mappedIndices;
  for (Value idx : dstStore.getIndices()) {
    mappedIndices.push_back(mapper.lookupOrDefault(idx));
  }

  // Load from DST with slice index prepended.
  SmallVector<Value> dstIndices;
  dstIndices.push_back(
      rewriter.create<arith::ConstantIndexOp>(loc, dstSlice));
  dstIndices.append(mappedIndices.begin(), mappedIndices.end());

  auto dstValue = rewriter.create<affine::AffineLoadOp>(
      loc, dst,
      AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                        rewriter.getContext()),
      dstIndices);

  // Store to L1.
  rewriter.create<affine::AffineStoreOp>(
      loc, dstValue.getResult(), dstStore.getMemRef(), dstStore.getMap(),
      mappedIndices);

  // Erase operations from original body that were cloned but not needed.
  SmallVector<Operation *> toErase;
  for (Operation &op : innerBody) {
    if (!isa<affine::AffineYieldOp>(&op) &&
        &op != dstValue.getOperation() &&
        !isa<affine::AffineStoreOp>(&op)) {
      toErase.push_back(&op);
    }
  }
  for (Operation *op : llvm::reverse(toErase)) {
    rewriter.eraseOp(op);
  }
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

  // 1. Create acquire_dst before the outermost loop.
  AcquireDstOp acquireDst =
      createAcquireDst(rewriter, loc, region, outermostLoop, cbType, totalDstTiles);
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

  // 3. Rewrite the accumulator load to load from DST instead of L1.
  //    The matmul accumulates in DST, so we need to initialize it from CB
  //    and write it back after.
  if (accumulatorLoad && outermostLoop) {
    auto forOp = cast<affine::AffineForOp>(outermostLoop);
    generatePrologueLoop(rewriter, loc, forOp, dst, accumulatorLoad, 0);
  }

  // 4. Generate epilogue to pack from DST back to output CB.
  if (outputStore && outermostLoop) {
    auto forOp = cast<affine::AffineForOp>(outermostLoop);
    generateEpilogueLoop(rewriter, loc, forOp, dst, outputStore, 0);
  }

  // 5. Rewrite the original stores to store to DST.
  //    The matmul stores its accumulator to DST slot 0.
  if (outputStore) {
    rewriter.setInsertionPoint(outputStore);
    SmallVector<Value> dstIndices;
    dstIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    for (Value idx : outputStore.getIndices()) {
      dstIndices.push_back(idx);
    }
    rewriter.create<affine::AffineStoreOp>(
        loc, outputStore.getValue(), dst,
        AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                          rewriter.getContext()),
        dstIndices);
    rewriter.eraseOp(outputStore);
  }

  // Similarly rewrite the accumulator load.
  if (accumulatorLoad) {
    rewriter.setInsertionPoint(accumulatorLoad);
    SmallVector<Value> dstIndices;
    dstIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    for (Value idx : accumulatorLoad.getIndices()) {
      dstIndices.push_back(idx);
    }
    rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
        accumulatorLoad, dst,
        AffineMap::getMultiDimIdentityMap(dstIndices.size(),
                                          rewriter.getContext()),
        dstIndices);
  }

  // 6. Create release_dst at the end of the region.
  createReleaseDst(rewriter, loc, region, dst);

  return success();
}

} // namespace mlir::tt::d2m::utils
