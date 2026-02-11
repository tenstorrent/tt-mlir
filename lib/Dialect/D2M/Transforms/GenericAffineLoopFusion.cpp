// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "D2MGenericAffineLoopFusion"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICAFFINELOOPFUSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Peel through optional stream_layout to get the underlying memref.
static Value peelStreamLayout(Value val) {
  if (auto streamOp = val.getDefiningOp<StreamLayoutOp>()) {
    return streamOp.getInput();
  }
  return val;
}

// Find a preceding GenericOp that writes to `sharedMemref` as a DPS init.
static GenericOp findProducerGeneric(Value sharedMemref, GenericOp consumer) {
  for (Operation *user : sharedMemref.getUsers()) {
    auto producer = dyn_cast<GenericOp>(user);
    if (!producer || producer == consumer) {
      continue;
    }
    if (!producer->isBeforeInBlock(consumer)) {
      continue;
    }
    for (OpOperand &init : producer.getDpsInitsMutable()) {
      if (init.get() == sharedMemref) {
        return producer;
      }
    }
  }
  return nullptr;
}

// Check that the consumer is the sole reader of `sharedMemref` among
// GenericOps. Also check through stream_layout wrappers.
static bool isSoleReader(Value sharedMemref, GenericOp consumer) {
  for (Operation *user : sharedMemref.getUsers()) {
    if (user == consumer.getOperation()) {
      continue;
    }
    if (auto streamOp = dyn_cast<StreamLayoutOp>(user)) {
      for (Operation *streamUser : streamOp.getResult().getUsers()) {
        if (streamUser == consumer.getOperation()) {
          continue;
        }
        if (isa<GenericOp>(streamUser)) {
          return false;
        }
      }
      continue;
    }
    if (auto otherGeneric = dyn_cast<GenericOp>(user)) {
      for (OpOperand *input : otherGeneric.getDpsInputOperands()) {
        Value rawInput = peelStreamLayout(input->get());
        if (rawInput == sharedMemref) {
          return false;
        }
      }
      continue;
    }
  }
  return true;
}

// Get the nesting depth of an affine.for induction variable relative to the
// enclosing generic region. Returns -1 if the value is not an affine.for IV.
static int getAffineForDepth(Value val) {
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg) {
    return -1;
  }
  auto *parentOp = blockArg.getOwner()->getParentOp();
  if (!isa<affine::AffineForOp>(parentOp)) {
    return -1;
  }
  int depth = 0;
  Operation *current = parentOp;
  while (current) {
    if (isa<GenericOp>(current)) {
      return depth;
    }
    if (isa<affine::AffineForOp>(current)) {
      depth++;
    }
    current = current->getParentOp();
  }
  return -1;
}

// Structurally compare two index computation DAGs rooted at two values.
static bool areIndicesStructurallyIdentical(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef) {
    int lhsDepth = getAffineForDepth(lhs);
    int rhsDepth = getAffineForDepth(rhs);
    if (lhsDepth >= 0 && lhsDepth == rhsDepth) {
      return true;
    }
    return false;
  }

  if (lhsDef->getName() != rhsDef->getName()) {
    return false;
  }

  if (isa<CoreIndexOp, GetBlockFactorOp, IterIndexOp, BlockIndexOp>(lhsDef)) {
    if (auto lhsIdx = lhsDef->getAttrOfType<IntegerAttr>("dim")) {
      auto rhsIdx = rhsDef->getAttrOfType<IntegerAttr>("dim");
      return rhsIdx && lhsIdx.getInt() == rhsIdx.getInt();
    }
    return false;
  }

  if (lhsDef->getNumOperands() != rhsDef->getNumOperands()) {
    return false;
  }
  for (unsigned i = 0; i < lhsDef->getNumOperands(); ++i) {
    if (!areIndicesStructurallyIdentical(lhsDef->getOperand(i),
                                         rhsDef->getOperand(i))) {
      return false;
    }
  }
  return true;
}

// Check that the producer's remote_store indices match the consumer's
// remote_load indices structurally. Returns the store op if valid.
static RemoteStoreOp getMatchingStoreForShared(GenericOp producer,
                                               GenericOp consumer,
                                               Value sharedMemref) {
  SmallVector<RemoteStoreOp> producerStores;
  producer.getRegion(0).walk([&](RemoteStoreOp storeOp) {
    if (storeOp.getMemref() == sharedMemref) {
      producerStores.push_back(storeOp);
    }
  });

  if (producerStores.size() != 1) {
    return nullptr;
  }

  RemoteStoreOp storeOp = producerStores.front();

  RemoteLoadOp consumerLoad = nullptr;
  consumer.getRegion(0).walk([&](RemoteLoadOp loadOp) {
    Value rawMemref = peelStreamLayout(loadOp.getMemref());
    if (rawMemref == sharedMemref) {
      consumerLoad = loadOp;
    }
  });

  if (!consumerLoad) {
    return nullptr;
  }

  auto storeIndices = storeOp.getIndices();
  auto loadIndices = consumerLoad.getIndices();
  if (storeIndices.size() != loadIndices.size()) {
    return nullptr;
  }
  for (unsigned i = 0; i < storeIndices.size(); ++i) {
    if (!areIndicesStructurallyIdentical(storeIndices[i], loadIndices[i])) {
      return nullptr;
    }
  }

  return storeOp;
}

// Find the single outermost affine.for with d2m.blocking_loop attribute.
static affine::AffineForOp getOutermostLoop(GenericOp generic) {
  affine::AffineForOp outerLoop = nullptr;
  unsigned count = 0;
  for (Operation &op : generic.getRegion(0).front()) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
      if (forOp->hasAttr("d2m.blocking_loop")) {
        outerLoop = forOp;
        count++;
      }
    }
  }
  if (count != 1) {
    return nullptr;
  }
  return outerLoop;
}

// Count the depth of nested d2m.blocking_loop affine.for ops.
static unsigned getOuterLoopDepth(affine::AffineForOp outerLoop) {
  unsigned depth = 0;
  auto current = outerLoop;
  while (current) {
    if (current->hasAttr("d2m.blocking_loop")) {
      depth++;
    }
    affine::AffineForOp nested = nullptr;
    for (Operation &op : current.getBody()->getOperations()) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
        if (forOp->hasAttr("d2m.blocking_loop")) {
          nested = forOp;
          break;
        }
      }
    }
    current = nested;
  }
  return depth;
}

// Replace tagged constants and affine.apply ops with the original D2M ops.
static void convertFromTemporaryForm(GenericOp tempGeneric,
                                     OpBuilder &builder) {
  // Restore get_block_factor ops from tagged constants.
  SmallVector<arith::ConstantIndexOp> taggedConstants;
  tempGeneric.getRegion(0).walk([&](arith::ConstantIndexOp op) {
    if (op->hasAttr("d2m.block_factor_constant")) {
      taggedConstants.push_back(op);
    }
  });

  for (arith::ConstantIndexOp constOp : taggedConstants) {
    auto dimAttr =
        constOp->getAttrOfType<IntegerAttr>("d2m.block_factor_constant");
    int64_t dim = dimAttr.getInt();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(constOp);

    auto getBlockFactorOp =
        builder.create<GetBlockFactorOp>(constOp.getLoc(), dim);
    constOp.replaceAllUsesWith(getBlockFactorOp.getResult());
    constOp.erase();
  }

  // Restore block_index ops from tagged affine.apply identity maps.
  SmallVector<affine::AffineApplyOp> taggedApplies;
  tempGeneric.getRegion(0).walk([&](affine::AffineApplyOp op) {
    if (op->hasAttr("d2m.orig_block_index")) {
      taggedApplies.push_back(op);
    }
  });

  for (affine::AffineApplyOp applyOp : taggedApplies) {
    auto dimAttr = applyOp->getAttrOfType<IntegerAttr>("d2m.orig_block_index");
    int64_t dim = dimAttr.getInt();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(applyOp);

    auto blockIndexOp = builder.create<BlockIndexOp>(applyOp.getLoc(), dim);
    applyOp.replaceAllUsesWith(blockIndexOp.getResult());
    applyOp.erase();
  }
}

// Try to fuse a producer-consumer pair. Returns the fused generic on success.
static GenericOp tryFusePair(GenericOp producer, GenericOp consumer,
                             OpOperand *inputOperand, Value sharedMemref,
                             OpBuilder &builder) {
  // Step 1: Get outermost loops to determine superset/subset
  affine::AffineForOp producerLoop = getOutermostLoop(producer);
  affine::AffineForOp consumerLoop = getOutermostLoop(consumer);

  if (!producerLoop || !consumerLoop) {
    return nullptr;
  }

  // Step 2: Determine superset generic (one with deeper loop nesting)
  unsigned producerDepth = getOuterLoopDepth(producerLoop);
  unsigned consumerDepth = getOuterLoopDepth(consumerLoop);

  GenericOp supersetGeneric =
      (producerDepth >= consumerDepth) ? producer : consumer;
  GenericOp subsetGeneric =
      (producerDepth >= consumerDepth) ? consumer : producer;

  // Step 4: Build fused generic operands and indexing maps
  SmallVector<Value> fusedInputs;
  SmallVector<Value> fusedOutputs;
  SmallVector<AffineMap> fusedMaps;

  AffineMap supersetSharedMap =
      supersetGeneric.getIndexingMapForOperand(sharedMemref);
  AffineMap subsetOutputMap = subsetGeneric.getIndexingMap(
      subsetGeneric.getDpsInitOperand(0)->getOperandNumber());
  AffineMap subsetOutputInv = inversePermutation(subsetOutputMap);

  // Producer inputs
  for (OpOperand *pi : producer.getDpsInputOperands()) {
    fusedInputs.push_back(pi->get());
    AffineMap argMap = producer.getIndexingMap(pi->getOperandNumber());
    if (producer == subsetGeneric) {
      fusedMaps.push_back(
          argMap.compose(subsetOutputInv).compose(supersetSharedMap));
    } else {
      fusedMaps.push_back(argMap);
    }
  }

  // Shared intermediate as input
  unsigned fusedSharedInputIdx = fusedInputs.size();
  fusedInputs.push_back(sharedMemref);
  AffineMap sharedMap =
      consumer.getIndexingMap(inputOperand->getOperandNumber());
  if (consumer == subsetGeneric) {
    fusedMaps.push_back(
        sharedMap.compose(subsetOutputInv).compose(supersetSharedMap));
  } else {
    fusedMaps.push_back(sharedMap);
  }

  // Other consumer inputs
  for (OpOperand *ci : consumer.getDpsInputOperands()) {
    if (ci == inputOperand) {
      continue;
    }
    fusedInputs.push_back(ci->get());
    AffineMap argMap = consumer.getIndexingMap(ci->getOperandNumber());
    if (consumer == subsetGeneric) {
      fusedMaps.push_back(
          argMap.compose(subsetOutputInv).compose(supersetSharedMap));
    } else {
      fusedMaps.push_back(argMap);
    }
  }

  // Consumer outputs only
  for (OpOperand &co : consumer.getDpsInitsMutable()) {
    fusedOutputs.push_back(co.get());
    AffineMap argMap = consumer.getIndexingMap(co.getOperandNumber());
    if (consumer == subsetGeneric) {
      fusedMaps.push_back(
          argMap.compose(subsetOutputInv).compose(supersetSharedMap));
    } else {
      fusedMaps.push_back(argMap);
    }
  }

  // Step 5: Create fused GenericOp shell
  builder.setInsertionPoint(consumer);
  auto fusedOp = builder.create<GenericOp>(
      consumer.getLoc(), TypeRange{}, fusedInputs, fusedOutputs,
      supersetGeneric.getGrid(), supersetGeneric.getBlockFactors(),
      builder.getAffineMapArrayAttr(fusedMaps),
      supersetGeneric.getIteratorTypes(), supersetGeneric.getThreads(),
      supersetGeneric.getScratchInputsAttr(), /*regions=*/1);

  Block &fusedBlock = fusedOp.getRegion(0).emplaceBlock();

  // Step 3: Set up block arguments
  Block &prodBlock = producer.getRegion(0).front();
  Block &consBlock = consumer.getRegion(0).front();

  SmallVector<Type> fusedBlockArgTypes;
  SmallVector<Location> fusedBlockArgLocs;
  SmallVector<std::pair<Operation *, unsigned>> argSources;

  auto appendSource = [&](Operation *op, unsigned operandNumber) {
    argSources.emplace_back(op, operandNumber);
    Block *srcBlock = (op == producer.getOperation()) ? &prodBlock : &consBlock;
    BlockArgument srcArg = srcBlock->getArgument(operandNumber);
    fusedBlockArgTypes.push_back(srcArg.getType());
    fusedBlockArgLocs.push_back(srcArg.getLoc());
  };

  for (OpOperand *pi : producer.getDpsInputOperands()) {
    appendSource(producer.getOperation(), pi->getOperandNumber());
  }
  appendSource(consumer.getOperation(), inputOperand->getOperandNumber());
  for (OpOperand *ci : consumer.getDpsInputOperands()) {
    if (ci == inputOperand) {
      continue;
    }
    appendSource(consumer.getOperation(), ci->getOperandNumber());
  }
  for (OpOperand &co : consumer.getDpsInitsMutable()) {
    appendSource(consumer.getOperation(), co.getOperandNumber());
  }

  fusedBlock.addArguments(fusedBlockArgTypes, fusedBlockArgLocs);

  DenseMap<std::pair<Operation *, unsigned>, unsigned> sourceToFusedIdx;
  for (auto [idx, src] : llvm::enumerate(argSources)) {
    sourceToFusedIdx[src] = static_cast<unsigned>(idx);
  }

  IRMapping irMap;

  // Map producer block args
  for (OpOperand *pi : producer.getDpsInputOperands()) {
    unsigned fusedIdx =
        sourceToFusedIdx[{producer.getOperation(), pi->getOperandNumber()}];
    irMap.map(prodBlock.getArgument(pi->getOperandNumber()),
              fusedBlock.getArgument(fusedIdx));
  }
  unsigned prodInitIdx = producer.getDpsInitOperand(0)->getOperandNumber();
  irMap.map(prodBlock.getArgument(prodInitIdx),
            fusedBlock.getArgument(fusedSharedInputIdx));

  // Map consumer block args
  irMap.map(consBlock.getArgument(inputOperand->getOperandNumber()),
            fusedBlock.getArgument(fusedSharedInputIdx));
  for (OpOperand *ci : consumer.getDpsInputOperands()) {
    if (ci == inputOperand) {
      continue;
    }
    unsigned fusedIdx =
        sourceToFusedIdx[{consumer.getOperation(), ci->getOperandNumber()}];
    irMap.map(consBlock.getArgument(ci->getOperandNumber()),
              fusedBlock.getArgument(fusedIdx));
  }
  for (OpOperand &co : consumer.getDpsInitsMutable()) {
    unsigned fusedIdx =
        sourceToFusedIdx[{consumer.getOperation(), co.getOperandNumber()}];
    irMap.map(consBlock.getArgument(co.getOperandNumber()),
              fusedBlock.getArgument(fusedIdx));
  }

  // Step 4: Clone producer loop into fused block as sibling
  builder.setInsertionPointToEnd(&fusedBlock);
  affine::AffineForOp clonedProducerLoop =
      cast<affine::AffineForOp>(builder.clone(*producerLoop, irMap));

  // Update irMap with producer loop results
  for (auto [oldRes, newRes] : llvm::zip(producerLoop->getResults(),
                                         clonedProducerLoop->getResults())) {
    irMap.map(oldRes, newRes);
  }

  // Step 5: Clone consumer loop into fused block as sibling
  affine::AffineForOp clonedConsumerLoop =
      cast<affine::AffineForOp>(builder.clone(*consumerLoop, irMap));

  // Step 6: Convert fused generic to temporary form (replace get_block_factor
  // with constants) IMPORTANT: Create ONE constant per dimension and share it
  // across both loops to ensure both loops use the same SSA value (compatible
  // affine spaces)
  auto blockFactorsAttr = fusedOp.getBlockFactors();

  // Use distinct non-unit prime sentinel values so affine analysis sees
  // non-trivial loop trip counts regardless of the real block-factor values.
  // Each dimension gets its own unique prime so the analysis can distinguish
  // the loop bounds of different nesting levels.
  static constexpr int64_t kSentinelPrimes[] = {5,  7,  11, 13, 17,
                                                19, 23, 29, 31, 37};
  TT_assertv(blockFactorsAttr.size() <= std::size(kSentinelPrimes),
             "more block-factor dimensions ({}) than available sentinel "
             "primes ({})",
             blockFactorsAttr.size(), std::size(kSentinelPrimes));

  // Create shared constants at the start of the fused block
  builder.setInsertionPointToStart(&fusedBlock);
  SmallVector<Value> sharedConstants;
  for (size_t dim = 0; dim < blockFactorsAttr.size(); ++dim) {
    int64_t sentinel = kSentinelPrimes[dim];
    auto constOp =
        builder.create<arith::ConstantIndexOp>(fusedOp.getLoc(), sentinel);
    constOp->setAttr("d2m.block_factor_constant",
                     builder.getI64IntegerAttr(dim));
    sharedConstants.push_back(constOp.getResult());
  }

  // Replace all get_block_factor ops with the shared constants
  SmallVector<GetBlockFactorOp> blockFactorOps;
  fusedOp.getRegion(0).walk(
      [&](GetBlockFactorOp op) { blockFactorOps.push_back(op); });

  for (GetBlockFactorOp op : blockFactorOps) {
    int64_t dim = op.getDim();
    op.replaceAllUsesWith(sharedConstants[dim]);
    op.erase();
  }

  // CRITICAL: Update loop upper bounds to use shared constants
  // The cloned loops may reference get_block_factor values from original
  // contexts
  SmallVector<affine::AffineForOp> allLoops;
  fusedOp.getRegion(0).walk(
      [&](affine::AffineForOp loop) { allLoops.push_back(loop); });

  for (affine::AffineForOp loop : allLoops) {
    // Check if this is a blocking loop with symbolic upper bound
    if (!loop->hasAttr("d2m.blocking_loop")) {
      continue;
    }

    auto upperBoundMap = loop.getUpperBoundMap();
    if (upperBoundMap.getNumResults() != 1 ||
        upperBoundMap.getNumSymbols() != 1) {
      continue;
    }

    // The loop should have one upper bound operand (the symbol)
    auto ubOperands = loop.getUpperBoundOperands();
    if (ubOperands.size() != 1) {
      continue;
    }

    // Determine which dimension based on the blocking_loop attribute
    int64_t dim =
        loop->getAttrOfType<IntegerAttr>("d2m.blocking_loop").getInt();

    // Replace the upper bound operand with the shared constant
    loop->setOperand(loop.getNumControlOperands() - ubOperands.size(),
                     sharedConstants[dim]);
  }

  // Replace BlockIndexOp with affine.apply identity maps from the
  // corresponding blocking-loop induction variable.  This exposes the index
  // computation to the affine dependence analysis so that it can compute
  // correct fusion slices.
  //
  // Each BlockIndexOp is matched to its own enclosing affine.for (walking
  // upward), NOT a global dim → IV map, because producer and consumer loop
  // nests coexist in the fused block and may have different depths.

  SmallVector<BlockIndexOp> blockIndexOps;
  fusedOp.getRegion(0).walk(
      [&](BlockIndexOp op) { blockIndexOps.push_back(op); });

  AffineMap identityMap1D =
      AffineMap::get(1, 0, builder.getAffineDimExpr(0), builder.getContext());
  for (BlockIndexOp op : blockIndexOps) {
    int64_t dim = op.getDim();

    // Walk up from the BlockIndexOp to find the enclosing affine.for whose
    // d2m.blocking_loop attribute matches this dimension.
    Value iv;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto forOp = dyn_cast<affine::AffineForOp>(parent);
      if (!forOp) {
        continue;
      }
      auto loopDimAttr = forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop");
      if (loopDimAttr && loopDimAttr.getInt() == dim) {
        iv = forOp.getInductionVar();
        break;
      }
    }
    if (!iv) {
      continue;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    auto applyOp = builder.create<affine::AffineApplyOp>(
        op.getLoc(), identityMap1D, ValueRange{iv});
    applyOp->setAttr("d2m.orig_block_index", builder.getI64IntegerAttr(dim));
    op.replaceAllUsesWith(applyOp.getResult());
    op.erase();
  }

  // Step 7: Use MLIR fusion utilities
  // Determine which is source (to fuse into destination)
  bool isProducerSrc = (producerDepth >= consumerDepth);
  affine::AffineForOp srcLoop =
      isProducerSrc ? clonedProducerLoop : clonedConsumerLoop;
  affine::AffineForOp dstLoop =
      isProducerSrc ? clonedConsumerLoop : clonedProducerLoop;

  // Fuse at the full matching loop depth so all nested levels are merged.
  unsigned dstLoopDepth = std::min(producerDepth, consumerDepth);

  // Check if fusion is legal and compute slice
  affine::ComputationSliceState srcSlice;
  affine::FusionResult result = affine::canFuseLoops(
      srcLoop, dstLoop, dstLoopDepth, &srcSlice,
      affine::FusionStrategy(affine::FusionStrategy::ProducerConsumer));

  if (result.value != affine::FusionResult::Success) {
    // Fusion failed, cleanup
    fusedOp->erase();
    return nullptr;
  }

  // Perform the fusion
  affine::fuseLoops(srcLoop, dstLoop, srcSlice,
                    /*isInnermostSiblingInsertion=*/false);

  // fuseLoops copies the source body into the destination but does not erase
  // the original source loop.
  srcLoop->erase();

  // Step 8: Convert back from temporary form
  convertFromTemporaryForm(fusedOp, builder);

  // Mark as fused
  fusedOp->setAttr("d2m.affine_fused", builder.getUnitAttr());

  return fusedOp;
}

namespace {
class D2MGenericAffineLoopFusion
    : public impl::D2MGenericAffineLoopFusionBase<D2MGenericAffineLoopFusion> {
public:
  using D2MGenericAffineLoopFusionBase::D2MGenericAffineLoopFusionBase;

  void runOnOperation() final {
    bool changed = true;
    while (changed) {
      changed = false;
      // Collect all generics in the module.
      SmallVector<GenericOp> generics;
      getOperation()->walk([&](GenericOp g) { generics.push_back(g); });

      for (GenericOp consumer : generics) {
        if (consumer->hasAttr("d2m.affine_fused")) {
          continue;
        }
        if (!consumer.isUnifiedForm()) {
          continue;
        }
        if (consumer.hasSkipOpAffineLoopFusionTrait()) {
          continue;
        }
        affine::AffineForOp consumerLoop = getOutermostLoop(consumer);
        if (!consumerLoop) {
          continue;
        }

        for (OpOperand *inputOperand : consumer.getDpsInputOperands()) {
          Value rawMemref = peelStreamLayout(inputOperand->get());
          GenericOp producer = findProducerGeneric(rawMemref, consumer);
          if (!producer || producer->hasAttr("d2m.affine_fused")) {
            continue;
          }
          if (!producer.isUnifiedForm()) {
            continue;
          }
          if (producer.hasSkipOpAffineLoopFusionTrait()) {
            continue;
          }
          if (!isSoleReader(rawMemref, consumer)) {
            continue;
          }
          RemoteStoreOp storeOp =
              getMatchingStoreForShared(producer, consumer, rawMemref);
          if (!storeOp) {
            continue;
          }

          OpBuilder builder(consumer);
          GenericOp fused =
              tryFusePair(producer, consumer, inputOperand, rawMemref, builder);
          if (!fused) {
            continue;
          }

          // Erase originals.
          producer->erase();
          consumer->erase();
          changed = true;
          break; // Restart iteration — generic list is invalidated.
        }
        if (changed) {
          break; // Restart outer loop.
        }
      }
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
