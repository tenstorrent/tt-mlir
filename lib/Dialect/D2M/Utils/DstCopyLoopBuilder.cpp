// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/DstCopyLoopBuilder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m::utils {

DstCopyLoopBuilder::DstCopyLoopBuilder(RewriterBase &rewriter,
                                       const DstCopyConfig &config)
    : rewriter(rewriter), config(config), semanticsAnalyzer(config.genericOp) {}

scf::IfOp
DstCopyLoopBuilder::createGuard(Location loc,
                                const SmallVector<unsigned> &guardDims) {
  if (guardDims.empty()) {
    return nullptr;
  }

  // Create constant zero for comparison
  auto zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));

  // Start with false
  auto cmp = rewriter
                 .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                            rewriter.getBoolAttr(false))
                 .getResult();

  // OR together checks for each guard dimension: (dim != 0)
  for (unsigned dim : guardDims) {
    auto iterIndex =
        rewriter.create<IterIndexOp>(loc, static_cast<int64_t>(dim));
    auto ne = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                             iterIndex, zero);
    cmp = rewriter.create<arith::OrIOp>(loc, cmp, ne).getResult();
  }

  return rewriter.create<scf::IfOp>(loc, cmp);
}

AffineMap DstCopyLoopBuilder::buildDstAccessMap(AffineMap originalMap) const {
  // Insert slice index as constant at position 0
  return originalMap.insertResult(
      getAffineConstantExpr(config.dstSlice, rewriter.getContext()), 0);
}

SmallVector<Value>
DstCopyLoopBuilder::remapIndices(ArrayRef<Value> originalIndices,
                                 const IRMapping &mapping) const {
  return llvm::map_to_vector(originalIndices, [&mapping](Value idx) {
    return mapping.lookupOrDefault(idx);
  });
}

FailureOr<DstCopyLoopResult>
DstCopyLoopBuilder::generatePrologue(Location loc,
                                     affine::AffineLoadOp srcLoad) {
  // Get which dimensions this output uses
  SmallVector<unsigned> prologueDims =
      semanticsAnalyzer.getPrologueEpilogueDims(config.outputOperandIndex);

  if (prologueDims.empty()) {
    return failure();
  }

  // Build filtered loop nest
  LoopBuildConfig loopConfig{.sourceLoop = config.sourceLoop,
                             .dimensionsToInclude = prologueDims,
                             .clearBodies = true,
                             .preserveAffineApply = true};

  SelectiveLoopBuilder loopBuilder(rewriter, loopConfig);
  auto loopResult = loopBuilder.build();

  if (failed(loopResult)) {
    return failure();
  }

  // Find where to insert the copy operations (innermost block)
  Block *insertBlock = loopResult->innermostBlock;
  if (!insertBlock) {
    return failure();
  }

  // Set insertion point before terminator
  Operation *terminator = insertBlock->getTerminator();
  if (terminator) {
    rewriter.setInsertionPoint(terminator);
  } else {
    rewriter.setInsertionPointToEnd(insertBlock);
  }

  // Get guard dimensions (non-participating dimensions)
  unsigned operandIndex =
      semanticsAnalyzer.getDimensionInfo().numDimensions > 0
          ? config.outputOperandIndex + config.genericOp.getInputs().size()
          : 0;
  SmallVector<unsigned> guardDims =
      semanticsAnalyzer.getGuardDims(operandIndex);

  // Create guard if needed
  scf::IfOp guardOp = createGuard(loc, guardDims);

  // Set insertion point inside guard (if exists) or at current point
  if (guardOp) {
    rewriter.setInsertionPointToStart(&guardOp.getThenRegion().front());
  }

  // Remap indices from original load
  SmallVector<Value> srcIndices(srcLoad.getIndices().begin(),
                                srcLoad.getIndices().end());
  SmallVector<Value> mappedIndices =
      remapIndices(srcIndices, loopResult->irMapping);

  // Load from L1 using original map with mapped indices
  auto l1Value = rewriter.create<affine::AffineLoadOp>(
      loc, srcLoad.getMemRef(), srcLoad.getMap(), mappedIndices);

  // Store to DST with slice index inserted
  AffineMap dstMap = buildDstAccessMap(srcLoad.getMap());
  rewriter.create<affine::AffineStoreOp>(
      loc, l1Value.getResult(), config.dstBuffer, dstMap, mappedIndices);

  // Build result
  DstCopyLoopResult result;
  result.loopNest = loopResult->outermostLoop;
  result.guardOp = guardOp;
  result.irMapping = std::move(loopResult->irMapping);

  return result;
}

FailureOr<DstCopyLoopResult>
DstCopyLoopBuilder::generateEpilogue(Location loc,
                                     affine::AffineStoreOp dstStore) {
  // Set insertion point after source loop
  rewriter.setInsertionPointAfter(config.sourceLoop);

  // Get which dimensions this output uses
  SmallVector<unsigned> epilogueDims =
      semanticsAnalyzer.getPrologueEpilogueDims(config.outputOperandIndex);

  if (epilogueDims.empty()) {
    return failure();
  }

  // Build filtered loop nest
  LoopBuildConfig loopConfig{.sourceLoop = config.sourceLoop,
                             .dimensionsToInclude = epilogueDims,
                             .clearBodies = true,
                             .preserveAffineApply = true};

  SelectiveLoopBuilder loopBuilder(rewriter, loopConfig);
  auto loopResult = loopBuilder.build();

  if (failed(loopResult)) {
    return failure();
  }

  // Find where to insert the copy operations (innermost block)
  Block *insertBlock = loopResult->innermostBlock;
  if (!insertBlock) {
    return failure();
  }

  // Set insertion point before terminator
  Operation *terminator = insertBlock->getTerminator();
  if (terminator) {
    rewriter.setInsertionPoint(terminator);
  } else {
    rewriter.setInsertionPointToEnd(insertBlock);
  }

  // Get guard dimensions (non-participating dimensions)
  unsigned operandIndex =
      semanticsAnalyzer.getDimensionInfo().numDimensions > 0
          ? config.outputOperandIndex + config.genericOp.getInputs().size()
          : 0;
  SmallVector<unsigned> guardDims =
      semanticsAnalyzer.getGuardDims(operandIndex);

  // Create guard if needed
  scf::IfOp guardOp = createGuard(loc, guardDims);

  // Set insertion point inside guard (if exists) or at current point
  if (guardOp) {
    rewriter.setInsertionPointToStart(&guardOp.getThenRegion().front());
  }

  // Remap indices from original store
  SmallVector<Value> storeIndices(dstStore.getIndices().begin(),
                                  dstStore.getIndices().end());
  SmallVector<Value> mappedIndices =
      remapIndices(storeIndices, loopResult->irMapping);

  // Load from DST with slice index inserted
  AffineMap dstMap = buildDstAccessMap(dstStore.getMap());
  auto dstValue = rewriter.create<affine::AffineLoadOp>(loc, config.dstBuffer,
                                                        dstMap, mappedIndices);

  // Store to L1 using original map with mapped indices
  rewriter.create<affine::AffineStoreOp>(loc, dstValue.getResult(),
                                         dstStore.getMemRef(),
                                         dstStore.getMap(), mappedIndices);

  // Build result
  DstCopyLoopResult result;
  result.loopNest = loopResult->outermostLoop;
  result.guardOp = guardOp;
  result.irMapping = std::move(loopResult->irMapping);

  return result;
}

} // namespace mlir::tt::d2m::utils
