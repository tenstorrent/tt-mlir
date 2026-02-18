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
#include "llvm/Support/Casting.h"

#include <tuple>

#define DEBUG_TYPE "D2MElementwiseFusion"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MELEMENTWISEFUSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Check that we're not exceeding physical limit of 32 CBs per d2m.generic
static bool fitsInL1PostFusion(GenericOp producer, GenericOp consumer) {

  int cbLimit = 32;

  int cbRemaining = cbLimit;

  // Account for number of CBs needed to store input/output operands after
  // fusion. -2 accounts for removal of producer init (output) operand and
  // consumer input operand since we're fusing over them.
  int numConsOperands = static_cast<int>(consumer.getNumOperands());
  int numProdOperands = static_cast<int>(producer.getNumOperands());

  cbRemaining -= numConsOperands + numProdOperands - 2;

  if (cbRemaining < 0) {
    return false;
  }

  return true;
}

// A fused op with 31 inputs and 1 output (all 16b) can be fully dst fused over
// sub-blocks of size 1xTile. Hence, if a 16b fused op meets the CB limit, it
// will automatically meet the dst limit as well (at least for now, when we're
// forcing sub-blocks of size 1xTile). In the case where one or more operands
// are >16b, we must be conservative for now and set a limit of 7 inputs (8
// total operands, including output), as that is the largest number of inputs
// that we can reduce down with a maximum of 4 DST tiles available. A lot of
// ways to increase this limit or use the resources more intelligently but we'll
// save that for later revisions.
static bool fitsInDstPostFusion(GenericOp producer, GenericOp consumer) {
  bool has32bit = false;

  Type largestDstType =
      utils::getRegionLargestDstElemType(producer.getRegion(0));
  if (largestDstType.getIntOrFloatBitWidth() > 16) {
    has32bit = true;
  }

  largestDstType = utils::getRegionLargestDstElemType(consumer.getRegion(0));
  if (largestDstType.getIntOrFloatBitWidth() > 16) {
    has32bit = true;
  }

  if (!has32bit) {
    return true;
  }

  int operandLimit32b = 8;

  int operandsRemaining = operandLimit32b;

  // Account for number of CBs needed to store input/output operands after
  // fusion. -2 accounts for removal of producer init (output) operand and
  // consumer input operand since we're fusing over them.
  int numConsOperands = static_cast<int>(consumer.getNumOperands());
  int numProdOperands = static_cast<int>(producer.getNumOperands());

  operandsRemaining -= numConsOperands + numProdOperands - 2;

  if (operandsRemaining < 0) {
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

  // Check that the producer's result is only used by the consumer
  // Count external users (users outside producer's own regions and outside
  // consumer's regions).
  for (auto result : producer->getResults()) {
    unsigned numExternalUsers = 0;
    for (auto *user : result.getUsers()) {
      // Skip users inside the producer's own regions.
      if (producer.getOperation()->isProperAncestor(user)) {
        continue;
      }
      // Skip users inside the consumer's regions (e.g., remote_load
      // operations).
      if (consumer.getOperation()->isProperAncestor(user)) {
        continue;
      }
      numExternalUsers++;
    }
    // Producer result should only be used by the consumer.
    if (numExternalUsers != 1) {
      return false;
    }
    // Verify the single external user is indeed the consumer operation itself.
    bool foundConsumerAsUser = false;
    for (auto *user : result.getUsers()) {
      if (!producer.getOperation()->isProperAncestor(user) &&
          !consumer.getOperation()->isProperAncestor(user)) {
        if (user == consumer.getOperation()) {
          foundConsumerAsUser = true;
        } else {
          return false;
        }
      }
    }
    if (!foundConsumerAsUser) {
      return false;
    }
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

  if (!fitsInL1PostFusion(producer, consumer)) {
    return false;
  }

  if (!fitsInDstPostFusion(producer, consumer)) {
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
                                    SmallVector<Value> &mergedCaptures,
                                    SmallVector<AffineMap> &fusedMaps,
                                    PatternRewriter &rewriter) {
  /////////////////////////////////////////////////////////////////////////////
  // Create fused op
  /////////////////////////////////////////////////////////////////////////////
  auto fusedResultTypes = TypeRange(fusedOutputs);

  auto fusedOp = rewriter.create<GenericOp>(
      consumer.getLoc(), fusedResultTypes, fusedInputs, fusedOutputs,
      mergedCaptures, consumer.getGrid(), consumer.getBlockFactors(),
      rewriter.getAffineMapArrayAttr(fusedMaps), consumer.getIteratorTypes(),
      consumer.getThreads(), consumer.getScratchInputsAttr(), /*regions=*/1);

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

  // Map original region block arguments to their indices in the fused region
  // block arguments.
  DenseMap<std::pair<Operation *, unsigned>, unsigned> sourceToFusedIdx;
  for (auto it : llvm::enumerate(argSources)) {
    sourceToFusedIdx[it.value()] = static_cast<unsigned>(it.index());
  }

  // Map consumer args to fused args
  auto mapConsRegionArgs = [&](Operation *op, Block &orig) {
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      unsigned fusedIndex = sourceToFusedIdx[{op, i}];
      irMap.map(orig.getArgument(i), fusedBlock.getArgument(fusedIndex));
    }
  };

  // Map all of producer's args to fused args
  auto mapProdRegionArgs = [&](Operation *op, Block &orig) {
    // Map all input args and skip mapping the output arg for now
    GenericOp prodGeneric = llvm::dyn_cast<GenericOp>(op);
    unsigned prodInitArgNum =
        prodGeneric.getDpsInitOperand(0)->getOperandNumber();
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      if (i != prodInitArgNum) {
        unsigned fusedIndex = sourceToFusedIdx[{op, i}];
        irMap.map(orig.getArgument(i), fusedBlock.getArgument(fusedIndex));
      }
    }

    // The producer's init block argument (output) is not part of the fused
    // operands, but we need to ensure it's mapped so that any operations in the
    // producer that reference it can be cloned properly. Map it to the
    // consumer's first output block argument, since that's where the fused
    // result/intermediate will go.
    unsigned consumerFirstOutputArgNum =
        consumer.getDpsInitOperand(0)->getOperandNumber();
    unsigned consumerOutputFusedIdx =
        sourceToFusedIdx[{consumer.getOperation(), consumerFirstOutputArgNum}];
    irMap.map(orig.getArgument(prodInitArgNum),
              fusedBlock.getArgument(consumerOutputFusedIdx));
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

  // Clone producer body (skip explicit d2m.yield and remote_store to output).
  for (Operation &op : pb) {
    if (isa<YieldOp>(op)) {
      continue;
    }
    // Skip remote_store operations that store to the producer's output operand
    // The producer's output is now an intermediate value, not a real output.
    if (auto storeOp = dyn_cast<d2m::RemoteStoreOp>(&op)) {
      Value storeMemref = storeOp.getMemref();
      // Check if this is storing to the producer's output/init operand.
      if (storeMemref == producer.getDpsInitOperand(0)->get()) {
        // The remote_store result is the updated output tensor. In fusion,
        // we don't actually store the intermediate result. Map the store result
        // to the tensor value being stored (the computation result).
        Value tensorBeingStored = storeOp.getLocalBuffer();
        Value mappedTensor = irMap.lookupOrDefault(tensorBeingStored);
        irMap.map(storeOp.getResult(), mappedTensor);
        continue;
      }
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

  // NOTE: We do NOT map the consumer's CB block argument here. The CB block arg
  // is only used as a parameter to remote_load operations. We will handle the
  // remote_load specially below (skip it and map its result to repl).
  // Storing repl for later use when we skip the consumer's remote_load:
  Value producerYieldedValue = repl;

  // Clone remaining consumer body ops (except terminator). Treat nested
  // operations opaquely; cloning preserves dominance as long as operands
  // are mapped.
  // Track which output buffers have been reserved to avoid multiple reserves
  // on the same output buffer. We track by the mapped block argument (in the
  // fused block) since that's what both producer and consumer reserves map to.
  DenseMap<Value, Value> reservedOutputs;

  // First, populate reservedOutputs with any reserves from the producer.
  // This prevents duplicate reserves when the consumer also reserves the same
  // output.
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
    // Handle reserve operations specially to deduplicate reserves on the same
    // output buffer
    if (auto reserveOp = dyn_cast<d2m::ReserveOp>(&op)) {
      Value originalCB = reserveOp.getCb();
      // Check if this is a reserve on an output buffer (block argument)
      if (auto blockArg = dyn_cast<BlockArgument>(originalCB)) {
        // Look up what this CB is mapped to in the fused block
        Value mappedCB = irMap.lookupOrDefault(originalCB);
        // Check if we've already reserved this output buffer (from producer or
        // earlier in consumer)
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

      // If reserve is on a non-block-argument CB (intermediate), check if it's
      // mapped
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

    // Handle remote_load operations - skip if loading from the fused operand
    // (producer's result), since the producer's computation is now inline.
    if (auto loadOp = dyn_cast<d2m::RemoteLoadOp>(&op)) {
      // Check if this remote_load is loading from the producer's result
      // (the fusedOperand value).
      if (loadOp.getMemref() == fusedOperand->get()) {
        // Map the remote_load result directly to the producer's yielded value.
        irMap.map(loadOp.getResult(), producerYieldedValue);
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
    auto mergedCaptures = llvm::to_vector(
        llvm::concat<Value>(consumer.getCaptures(), producer.getCaptures()));

    auto fusedOp =
        createFusedGeneric(fusedOperand, producer, consumer, fusedInputs,
                           fusedOutputs, mergedCaptures, fusedMaps, rewriter);

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
