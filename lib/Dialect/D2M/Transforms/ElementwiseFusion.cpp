// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include <llvm/Support/Casting.h>
#include <tuple>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MELEMENTWISEFUSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

static int countBinaryOps(Operation *op) {
  int count = 0;

  for (Region &region : op->getRegions()) {
    for (Operation &opInner : region.getOps()) {
      if (opInner.hasTrait<D2MGenericRegionComputeOpTrait>()) {
        if (opInner.getNumOperands() == 2) {
          ++count;
        }
      }
      count += countBinaryOps(&opInner);
    }
  }
  return count;
}

static bool fitsInDstPostFusion(GenericOp producer, GenericOp consumer,
                                OpOperand *use, unsigned dstCapacity) {
  auto tensorType = mlir::cast<RankedTensorType>(use->get().getType());

  auto shape = tensorType.getShape();
  int blockSize = shape.back();
  shape = shape.drop_back(1);
  blockSize *= shape.back();

  if (blockSize > static_cast<int>(dstCapacity)) {
    blockSize = dstCapacity;
  }

  int dstTilesRemaining = dstCapacity;

  // Account for number of tiles needed to store input operands after fusion.
  // -2 accounts for removal of producer init/output operand and consumer
  // input operand since we're fusing over them.
  // -1 accounts for the final output tile which is only needed for binary
  // ops, unaries do everything in place. Will be added back in next part of
  // check.
  int numConsOperands = static_cast<int>(consumer.getNumOperands());
  int numProdOperands = static_cast<int>(producer.getNumOperands());

  dstTilesRemaining -= blockSize * (numConsOperands + numProdOperands - 2 - 1);

  if (dstTilesRemaining < 0) {
    return false;
  }

  // Subtract # of intermediate tiles needed
  dstTilesRemaining -=
      blockSize * (countBinaryOps(consumer) + countBinaryOps(producer));

  if (dstTilesRemaining < 0) {
    return false;
  }

  return true;
}

static bool isValidElementwiseFusionTarget(GenericOp gOp) {
  if (!gOp.isComputeOnlyForm()) {
    return false;
  }

  if (!gOp.hasPureTensorSemantics()) {
    return false;
  }

  if (!gOp.isAllParallel()) {
    return false;
  }

  if (gOp.hasSkipOpEltwiseFusionTrait()) {
    return false;
  }

  if (gOp.hasMultiUseInputOperand()) {
    return false;
  }

  return true;
}

static bool isElementwiseFusable(OpOperand *fusionTargetOperand,
                                 unsigned dstCapacity,
                                 bool checkConsumer = true,
                                 bool checkProducer = true) {
  if (!fusionTargetOperand) {
    return false;
  }

  auto producer = fusionTargetOperand->get().getDefiningOp<GenericOp>();
  auto consumer = dyn_cast<GenericOp>(fusionTargetOperand->getOwner());

  if (!producer || !consumer) {
    return false;
  }

  if (checkConsumer && !isValidElementwiseFusionTarget(consumer)) {
    return false;
  }

  if (checkProducer && !isValidElementwiseFusionTarget(producer)) {
    return false;
  }

  if (!producer.hasCompatibleBlocking(consumer)) {
    return false;
  }

  if (!fitsInDstPostFusion(producer, consumer, fusionTargetOperand,
                           dstCapacity)) {
    return false;
  }

  // Rank/perm checks
  AffineMap consMap =
      consumer.getIndexingMap(fusionTargetOperand->getOperandNumber());
  if (consMap.getNumResults() != producer.getNumDims()) {
    return false;
  }

  // Producer result map is that of first init (assume single init/result
  // layout)
  AffineMap prodResMap = producer.getIndexingMap(
      producer.getDpsInitOperand(0)->getOperandNumber());
  if (!prodResMap.isPermutation()) {
    return false;
  }

  return true;
}

// Compute producer operand map in fused coordinates:
// argMap ∘ inv(resMap) ∘ consMap
static AffineMap computeFusedArgMap(GenericOp producer, OpOperand *prodOpnd,
                                    AffineMap prodResMap,
                                    AffineMap consMapForFused) {
  AffineMap inv = inversePermutation(prodResMap);
  AffineMap arg = producer.getIndexingMap(prodOpnd->getOperandNumber());
  return arg.compose(inv).compose(consMapForFused);
}

static std::tuple<SmallVector<Value>, SmallVector<Value>,
                  SmallVector<AffineMap>>
getFusedOperands(OpOperand *fusedOperand, GenericOp producer,
                 GenericOp consumer) {
  SmallVector<Value> fusedInputs;
  SmallVector<Value> fusedOutputs;
  SmallVector<Type> fusedResultTypes;
  SmallVector<AffineMap> fusedMaps;

  AffineMap prodResMap = producer.getIndexingMap(
      producer.getDpsInitOperand(0)->getOperandNumber());
  AffineMap consMap = consumer.getIndexingMap(fusedOperand->getOperandNumber());

  // consumer inputs before fused
  auto inputs = consumer.getInputs();
  size_t fusedIdx = fusedOperand->getOperandNumber();
  for (size_t i = 0; i < fusedIdx; ++i) {
    fusedInputs.push_back(inputs[i]);
    fusedMaps.push_back(consumer.getIndexingMap(i));
  }
  // producer inputs (remapped)
  for (OpOperand *pi : producer.getDpsInputOperands()) {
    fusedInputs.push_back(pi->get());
    fusedMaps.push_back(computeFusedArgMap(producer, pi, prodResMap, consMap));
  }
  // remaining consumer inputs after fused
  for (size_t i = fusedIdx + 1; i < inputs.size(); ++i) {
    fusedInputs.push_back(inputs[i]);
    fusedMaps.push_back(consumer.getIndexingMap(i));
  }

  for (OpOperand &co : consumer.getDpsInitsMutable()) {
    fusedOutputs.push_back(co.get());
    fusedMaps.push_back(consumer.getIndexingMap(co.getOperandNumber()));
    if (!isa<MemRefType>(co.get().getType())) {
      fusedResultTypes.push_back(co.get().getType());
    }
  }

  return std::make_tuple(std::move(fusedInputs), std::move(fusedOutputs),
                         std::move(fusedMaps));
}

static GenericOp createFusedGeneric(OpOperand *fusedOperand, GenericOp producer,
                                    GenericOp consumer,
                                    SmallVector<Value> &fusedInputs,
                                    SmallVector<Value> &fusedOutputs,
                                    SmallVector<AffineMap> &fusedMaps,
                                    PatternRewriter &rewriter) {
  /////////////////////////////////////////////////////////////////////////////
  // Create fused op
  /////////////////////////////////////////////////////////////////////////////
  auto fusedResultTypes = TypeRange(fusedOutputs);

  auto fusedOp = rewriter.create<GenericOp>(
      consumer.getLoc(), fusedResultTypes, fusedInputs, fusedOutputs,
      consumer.getGrid(), consumer.getBlockFactors(),
      rewriter.getAffineMapArrayAttr(fusedMaps), consumer.getIteratorTypes(),
      consumer.getThreads(), /*regions=*/1);

  /////////////////////////////////////////////////////////////////////////////
  // Map the block arguments of the fusedOp
  /////////////////////////////////////////////////////////////////////////////
  IRMapping irMap;
  Block &fusedBlock = fusedOp.getRegion(0).emplaceBlock();
  Block &pb = producer.getRegion(0).front();
  Block &cb = consumer.getRegion(0).front();

  // For each fused operand, pick the corresponding region block-argument
  // type from the originating op (producer/consumer) to satisfy verifier.
  SmallVector<Type> fusedBlockArgTypes;
  SmallVector<Location> fusedBlockArgLocs;
  fusedBlockArgTypes.reserve(fusedOp->getNumOperands());
  fusedBlockArgLocs.reserve(fusedOp->getNumOperands());

  // Helper to append a source (op, operandNumber)
  SmallVector<std::pair<Operation *, unsigned>> argSources;
  auto appendSource = [&](Operation *op, unsigned operandNumber) {
    argSources.emplace_back(op, operandNumber);
    Block *srcBlock = op == producer.getOperation() ? &pb : &cb;
    BlockArgument srcArg = srcBlock->getArgument(operandNumber);
    fusedBlockArgTypes.push_back(srcArg.getType());
    fusedBlockArgLocs.push_back(srcArg.getLoc());
  };

  auto inputs = consumer.getInputs();
  size_t fusedIdx = fusedOperand->getOperandNumber();

  // Reconstruct argSources in the same order as fused operands.
  // consumer inputs before fused
  for (size_t i = 0; i < fusedIdx; ++i) {
    appendSource(consumer.getOperation(), /*operandNumber=*/i);
  }
  // producer inputs
  for (OpOperand *pi : producer.getDpsInputOperands()) {
    appendSource(producer.getOperation(), pi->getOperandNumber());
  }
  // remaining consumer inputs after fused
  for (size_t i = fusedIdx + 1; i < inputs.size(); ++i) {
    appendSource(consumer.getOperation(), /*operandNumber=*/i);
  }
  // consumer outputs
  for (OpOperand &co : consumer.getDpsInitsMutable()) {
    appendSource(consumer.getOperation(), co.getOperandNumber());
  }

  (void)fusedBlock.addArguments(fusedBlockArgTypes, fusedBlockArgLocs);

  // Map original region block arguments to fused region block arguments.
  DenseMap<std::pair<Operation *, unsigned>, unsigned> sourceToFusedIdx;
  for (auto it : llvm::enumerate(argSources)) {
    sourceToFusedIdx[it.value()] = static_cast<unsigned>(it.index());
  }
  auto mapConsRegionArgs = [&](Operation *op, Block &orig) {
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      unsigned fusedIndex = sourceToFusedIdx[{op, i}];
      irMap.map(orig.getArgument(i), fusedBlock.getArgument(fusedIndex));
    }
  };
  auto mapProdRegionArgs = [&](Operation *op, Block &orig) {
    GenericOp prodGeneric = llvm::dyn_cast<GenericOp>(op);
    unsigned prodInitArgNum = prodGeneric.getDpsInitOperand(0)->getOperandNumber();
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      if (i != prodInitArgNum) {
        unsigned fusedIndex = sourceToFusedIdx[{op, i}];
        irMap.map(orig.getArgument(i), fusedBlock.getArgument(fusedIndex));
      }
    }
    // The producer's init block argument (output) is not part of the fused
    // operands, but we need to ensure it's mapped so that any operations in the
    // producer that reference it can be cloned properly. Map it to the consumer's
    // first output block argument, since that's where the fused result will go.
    unsigned consumerFirstOutputArgNum = 
        consumer.getDpsInitOperand(0)->getOperandNumber();
    unsigned consumerOutputFusedIdx = sourceToFusedIdx[{consumer.getOperation(), consumerFirstOutputArgNum}];
    irMap.map(orig.getArgument(prodInitArgNum), fusedBlock.getArgument(consumerOutputFusedIdx));
  };
  mapProdRegionArgs(producer.getOperation(), pb);
  mapConsRegionArgs(consumer.getOperation(), cb);

  /////////////////////////////////////////////////////////////////////////////
  // Build fused region: clone producer then consumer, map fused operand to
  // producer yield value. Ensure fused block argument types match original
  // region operand argument types (shard types), not the top-level types.
  /////////////////////////////////////////////////////////////////////////////

  // Insert all cloned ops into the fused block, in order: producer then
  // consumer.
  rewriter.setInsertionPointToEnd(&fusedBlock);

  // Clone producer body (skip explicit d2m.yield if present).
  for (Operation &op : pb) {
    if (isa<YieldOp>(op)) {
      continue;
    }
    rewriter.clone(op, irMap);
  }

  YieldOp prodYield = nullptr;
  for (Operation &op : pb) {
    if (auto y = dyn_cast<YieldOp>(op)) {
      prodYield = y;
    }
  }
  assert(prodYield);

  int prodResultNumber = cast<OpResult>(fusedOperand->get()).getResultNumber();
  Value prodYieldOperand = prodYield.getOperand(prodResultNumber);
  // Now that all producer block arguments (including init) are properly mapped,
  // we can safely look up the yielded value. It should either be a mapped
  // operation result or a mapped block argument.
  Value repl = irMap.lookupOrDefault(prodYieldOperand);

  // Map consumer arg corresponding to fused operand number to repl.
  // If `repl` is a tensor, and consumer block arg is tensor, direct map.
  // If consumer expects a tensor but repl is produced by inner ops,
  // ensure we cloned those inner ops already (we did by cloning pb body),
  // so `repl` dominates this point.
  irMap.map(cb.getArgument(fusedOperand->getOperandNumber()), repl);

  // Clone remaining consumer body ops (except terminator). Treat nested
  // operations opaquely; cloning preserves dominance as long as operands
  // are mapped.
  // Track which output buffers have been reserved to avoid multiple reserves
  // on the same output buffer. We track by the mapped block argument (in the
  // fused block) since that's what both producer and consumer reserves map to.
  DenseMap<Value, Value> reservedOutputs;
  
  // First, populate reservedOutputs with any reserves from the producer.
  // This prevents duplicate reserves when the consumer also reserves the same output.
  for (Operation &op : pb) {
    if (auto reserveOp = dyn_cast<d2m::ReserveOp>(&op)) {
      Value originalCB = reserveOp.getCb();
      if (auto blockArg = dyn_cast<BlockArgument>(originalCB)) {
        // This producer reserve was already cloned, find the cloned result
        Value clonedResult = irMap.lookupOrDefault(reserveOp.getResult());
        if (clonedResult) {
          // The producer's CB block arg has been mapped to a fused block arg.
          // Track using the mapped (fused) block arg as the key.
          Value mappedCB = irMap.lookupOrDefault(originalCB);
          if (mappedCB) {
            reservedOutputs[mappedCB] = clonedResult;
          }
        }
      }
    }
  }
  for (Operation &op : cb.without_terminator()) {
    // Handle reserve operations specially to deduplicate reserves on the same output buffer
    if (auto reserveOp = dyn_cast<d2m::ReserveOp>(&op)) {
      Value originalCB = reserveOp.getCb();
      // Check if this is a reserve on an output buffer (block argument)
      if (auto blockArg = dyn_cast<BlockArgument>(originalCB)) {
        // Look up what this CB is mapped to in the fused block
        Value mappedCB = irMap.lookupOrDefault(originalCB);
        // Check if we've already reserved this output buffer (from producer or earlier in consumer)
        if (reservedOutputs.count(mappedCB)) {
          // Reuse the previously reserved buffer
          irMap.map(reserveOp.getResult(), reservedOutputs[mappedCB]);
          continue;
        }
        // First reserve on this output buffer - clone it and track it
        Operation *cloned = rewriter.clone(*reserveOp, irMap);
        Value clonedResult = cloned->getResult(0);
        reservedOutputs[mappedCB] = clonedResult;
        // Explicitly map the original reserve's result to the cloned result
        irMap.map(reserveOp.getResult(), clonedResult);
        continue;
      }
      // If reserve is on a non-block-argument CB (intermediate), check if it's mapped
      Value cbOperand = irMap.lookupOrDefault(originalCB);
      if (cbOperand && !mlir::isa<BlockArgument>(cbOperand)) {
        irMap.map(reserveOp.getResult(), cbOperand);
        continue;
      }
      // Otherwise clone it normally
      rewriter.clone(op, irMap);
      continue;
    }
    
    // Handle wait operations - skip if waiting on intermediate values
    if (auto waitOp = dyn_cast<d2m::WaitOp>(&op)) {
      Value cbOperand = irMap.lookupOrDefault(waitOp.getCb());
      if (cbOperand && !mlir::isa<BlockArgument>(cbOperand)) {
        irMap.map(waitOp.getResult(), cbOperand);
        continue;
      }
    }
    
    // Clone all other operations
    rewriter.clone(op, irMap);
  }

  // Build fused yield: kept producer yields then consumer yields
  SmallVector<Value> fusedYields;
  YieldOp consYield = nullptr;
  for (Operation &op : cb) {
    if (auto y = dyn_cast<YieldOp>(op)) {
      consYield = y;
    }
  }
  for (Value y : consYield.getOperands()) {
    fusedYields.push_back(irMap.lookupOrDefault(y));
  }
  rewriter.setInsertionPointToEnd(&fusedBlock);
  rewriter.create<YieldOp>(fusedOp.getLoc(), fusedYields);

  return fusedOp;
}

namespace {
struct FuseD2MElementwiseOpsPattern : public OpRewritePattern<GenericOp> {
  FuseD2MElementwiseOpsPattern(MLIRContext *context,
                               unsigned maxDstPhysicalSizeTiles)
      : OpRewritePattern<GenericOp>(context),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles) {}

  LogicalResult matchAndRewrite(GenericOp consumer,
                                PatternRewriter &rewriter) const final {
    if (!isValidElementwiseFusionTarget(consumer)) {
      return failure();
    }

    assert(consumer.getNumRegions() == 1u);

    Type largestDstType =
        utils::getRegionLargestDstElemType(consumer.getRegion(0));
    const unsigned dstCapacity =
        ttcore::getOpChipDescAttr(consumer).getDstLogicalSizeTiles(
            largestDstType, false, maxDstPhysicalSizeTiles);

    auto findFusableOperand = [&]() -> OpOperand * {
      for (OpOperand *use : consumer.getDpsInputOperands()) {
        if (isElementwiseFusable(use, dstCapacity,
                                 // already checked consumer outside
                                 false, true)) {
          return use;
        }
      }
      return nullptr;
    };

    OpOperand *fusedOperand = findFusableOperand();

    if (!fusedOperand) {
      return failure();
    }

    auto producer = fusedOperand->get().getDefiningOp<GenericOp>();
    // we already check that producer is a valid GenericOp,
    // use assert instead of if()
    assert(producer);

    // Build fused op operands, maps, results
    auto [fusedInputs, fusedOutputs, fusedMaps] =
        getFusedOperands(fusedOperand, producer, consumer);

    auto fusedOp =
        createFusedGeneric(fusedOperand, producer, consumer, fusedInputs,
                           fusedOutputs, fusedMaps, rewriter);

    // Replace uses: from producer and consumer results to fused results.
    int resIdx = 0;
    // Consumer results
    for (auto r : consumer->getResults()) {
      r.replaceAllUsesWith(fusedOp->getResult(resIdx++));
    }

    rewriter.eraseOp(consumer);
    rewriter.eraseOp(producer);

    return success();
  }

  unsigned maxDstPhysicalSizeTiles = 0;
};
} // namespace

namespace {
class D2MElementwiseFusion
    : public tt::d2m::impl::D2MElementwiseFusionBase<D2MElementwiseFusion> {
  using D2MElementwiseFusionBase::D2MElementwiseFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseD2MElementwiseOpsPattern>(
        ctx, maxDstPhysicalSizeTiles.getValue());
    GreedyRewriteConfig cfg;

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), cfg))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m

// Factory is generated by TableGen.
