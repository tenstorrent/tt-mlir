// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

#include <tuple>

#define DEBUG_TYPE "D2MGenericFusion"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICFUSION
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

// Producer's result(s) must only be consumed by the consumer op (users inside
// either op's regions are ignored, e.g. remote_load).
static bool producerResultUsedOnlyByConsumer(GenericOp producer,
                                             GenericOp consumer) {
  for (auto result : producer->getResults()) {
    for (auto *user : result.getUsers()) {
      if (producer.getOperation()->isProperAncestor(user) ||
          consumer.getOperation()->isProperAncestor(user)) {
        continue;
      }
      if (user != consumer.getOperation()) {
        return false;
      }
    }
  }
  return true;
}

// Checks shared by all fusion variants: CB/DST budget, blocking compatibility,
// and rank/permutation constraints on the indexing maps.
static bool passesSharedFusionShapeChecks(OpOperand *fusionTargetOperand,
                                          GenericOp producer,
                                          GenericOp consumer) {
  if (!producer.hasCompatibleBlocking(consumer)) {
    return false;
  }
  if (!fitsInL1PostFusion(producer, consumer)) {
    return false;
  }
  if (!fitsInDstPostFusion(producer, consumer)) {
    return false;
  }

  AffineMap consMap =
      consumer.getIndexingMap(fusionTargetOperand->getOperandNumber());
  if (consMap.getNumResults() != producer.getNumDims()) {
    return false;
  }
  // Producer result map is that of first init (assume single init/result
  // layout).
  AffineMap prodResMap = producer.getIndexingMap(
      producer.getDpsInitOperand(0)->getOperandNumber());
  return prodResMap.isPermutation();
}

static bool isElementwiseFusable(OpOperand *fusionTargetOperand,
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

  if (!producerResultUsedOnlyByConsumer(producer, consumer)) {
    return false;
  }

  if (checkConsumer && !isValidElementwiseFusionTarget(consumer)) {
    return false;
  }

  if (checkProducer && !isValidElementwiseFusionTarget(producer)) {
    return false;
  }

  return passesSharedFusionShapeChecks(fusionTargetOperand, producer, consumer);
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
                                    SmallVector<Value> &mergedAdditionalArgs,
                                    SmallVector<AffineMap> &fusedMaps,
                                    PatternRewriter &rewriter) {
  /////////////////////////////////////////////////////////////////////////////
  // Create fused op
  /////////////////////////////////////////////////////////////////////////////
  auto fusedResultTypes = TypeRange(fusedOutputs);

  auto fusedOp = rewriter.create<GenericOp>(
      consumer.getLoc(), fusedResultTypes, fusedInputs, fusedOutputs,
      mergedAdditionalArgs, consumer.getGrid(), consumer.getBlockFactors(),
      rewriter.getAffineMapArrayAttr(fusedMaps), consumer.getIteratorTypes(),
      consumer.getThreads(), consumer.getFabricConnectionConfigAttr(),
      /*regions=*/1);

  /////////////////////////////////////////////////////////////////////////////
  // Map the block arguments of the fusedOp
  /////////////////////////////////////////////////////////////////////////////
  IRMapping irMap;
  Block &fusedBlock = fusedOp.getRegion(0).emplaceBlock();
  Block &pb = producer.getRegion(0).front();
  Block &cb = consumer.getRegion(0).front();

  // For each fused operand, pick the corresponding tensor.empty type from
  // the originating op's region (producer/consumer).
  SmallVector<std::pair<Operation *, unsigned>> argSources;
  SmallVector<Type> fusedEmptyTypes;
  auto appendSource = [&](Operation *op, unsigned operandNumber) {
    argSources.emplace_back(op, operandNumber);
    Region &srcRegion = (op == producer.getOperation()) ? producer.getRegion(0)
                                                        : consumer.getRegion(0);
    Value operandAlloc = GenericOp::getOperandAlloc(srcRegion, operandNumber);
    assert(operandAlloc && "Expected alloc op for operand in region");
    fusedEmptyTypes.push_back(operandAlloc.getType());
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

  // Create tensor.empty ops in the fused block for each fused operand.
  rewriter.setInsertionPointToStart(&fusedBlock);
  SmallVector<Value> fusedTensorEmpties;
  for (Type emptyType : fusedEmptyTypes) {
    auto shapedType = mlir::cast<ShapedType>(emptyType);
    auto emptyOp = rewriter.create<mlir::tensor::EmptyOp>(
        fusedOp.getLoc(), shapedType.getShape(), shapedType.getElementType());
    fusedTensorEmpties.push_back(emptyOp.getResult());
  }

  // Map original tensor.empty values to their indices in the fused region.
  DenseMap<std::pair<Operation *, unsigned>, unsigned> sourceToFusedIdx;
  for (auto it : llvm::enumerate(argSources)) {
    sourceToFusedIdx[it.value()] = static_cast<unsigned>(it.index());
  }

  // Map consumer tensor.empty values to fused tensor.empty values.
  auto mapConsRegionEmpties = [&](GenericOp generic, Block &orig) {
    for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
      Value origEmpty = GenericOp::getOperandAlloc(*orig.getParent(), i);
      unsigned fusedIndex = sourceToFusedIdx[{generic.getOperation(), i}];
      irMap.map(origEmpty, fusedTensorEmpties[fusedIndex]);
    }
  };

  // Map all of producer's tensor.empty values to fused values.
  auto mapProdRegionEmpties = [&](GenericOp prodGeneric, Block &orig) {
    unsigned prodInitArgNum =
        prodGeneric.getDpsInitOperand(0)->getOperandNumber();
    for (unsigned i = 0; i < prodGeneric->getNumOperands(); ++i) {
      Value origEmpty = GenericOp::getOperandAlloc(*orig.getParent(), i);
      if (i != prodInitArgNum) {
        unsigned fusedIndex = sourceToFusedIdx[{prodGeneric.getOperation(), i}];
        irMap.map(origEmpty, fusedTensorEmpties[fusedIndex]);
      }
    }

    // The producer's init tensor.empty (output) is not part of the fused
    // operands. Create a fresh intermediate tensor.empty for it inside the
    // fused region rather than aliasing it to the consumer's output buffer.
    //
    // Aliasing producer-output to consumer-output is unsafe for chains of
    // 3+ ops: the consumer-output is later materialized as a single output
    // circular buffer, and the producer's pack_tile would write into that
    // CB while a downstream linalg op in the fused region reads from it.
    // The output CB's read pointer is never advanced by the compute thread
    // (only the write pointer, via cb_push_back), so on the second outer
    // loop iteration the unpack pulls stale tiles from the previous
    // iteration, corrupting the result.
    //
    // For 2-op chains this manifested as a benign aliasing because the
    // single intermediate fit in DST and was never packed to L1. With a
    // separate intermediate tensor.empty, downstream lowering is free to
    // either keep it in DST (no cost) or allocate it its own scratch CB
    // (correct).
    Value prodOutputEmpty =
        GenericOp::getOperandAlloc(*orig.getParent(), prodInitArgNum);
    auto shapedType = mlir::cast<ShapedType>(prodOutputEmpty.getType());
    auto intermediateEmpty = rewriter.create<mlir::tensor::EmptyOp>(
        fusedOp.getLoc(), shapedType.getShape(), shapedType.getElementType());
    irMap.map(prodOutputEmpty, intermediateEmpty.getResult());
  };

  mapProdRegionEmpties(producer, pb);
  mapConsRegionEmpties(consumer, cb);

  // Build the set of operand-associated tensor.empty values for producer and
  // consumer. Any tensor.empty in the original regions that is NOT in this set
  // is an intermediate tensor.empty (e.g. one we created in a previous fusion
  // for a producer's output) and must be cloned into the fused region rather
  // than skipped. If we skip an intermediate tensor.empty, downstream linalg
  // ops in the fused region will retain references to the original (about-to-
  // be-erased) value, causing the greedy rewriter's eraseOp to assert on
  // remaining uses.
  llvm::DenseSet<Value> operandAssociatedEmpties;
  auto collectOperandEmpties = [&](GenericOp generic, Block &orig) {
    for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
      if (Value origEmpty = GenericOp::getOperandAlloc(*orig.getParent(), i)) {
        operandAssociatedEmpties.insert(origEmpty);
      }
    }
  };
  collectOperandEmpties(producer, pb);
  collectOperandEmpties(consumer, cb);

  /////////////////////////////////////////////////////////////////////////////
  // Build fused region: clone producer then consumer, map fused operand to
  // producer yield value. Ensure fused block argument types match original
  // region operand argument types (shard types), not the top-level types.
  /////////////////////////////////////////////////////////////////////////////

  // Insert all cloned ops into the fused block, in order: producer then
  // consumer.
  rewriter.setInsertionPointToEnd(&fusedBlock);

  // Clone producer body (skip d2m.yield, operand-associated tensor.empty, and
  // remote_store to output). Operand-associated tensor.empty ops have already
  // been created in the fused block; intermediate tensor.empty ops (added by
  // a previous fusion iteration for a producer's output) must be cloned so
  // their replacements live inside the new fused region.
  for (Operation &op : pb) {
    if (isa<YieldOp>(op)) {
      continue;
    }
    if (auto emptyOp = dyn_cast<mlir::tensor::EmptyOp>(&op);
        emptyOp && operandAssociatedEmpties.contains(emptyOp.getResult())) {
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

  // Helper to check if a value is an operand-associated value (block arg or
  // fused tensor.empty), as opposed to an intermediate value.
  llvm::DenseSet<Value> fusedEmptySet(fusedTensorEmpties.begin(),
                                      fusedTensorEmpties.end());
  auto isOperandAssociatedValue = [&](Value v) {
    return mlir::isa<BlockArgument>(v) || fusedEmptySet.contains(v);
  };

  // First, populate reservedOutputs with any reserves from the producer.
  // This prevents duplicate reserves when the consumer also reserves the same
  // output.
  for (Operation &op : pb) {
    if (auto reserveOp = dyn_cast<d2m::ReserveOp>(&op)) {
      Value originalCB = reserveOp.getCb();
      Value mappedCB = irMap.lookupOrDefault(originalCB);
      if (isOperandAssociatedValue(mappedCB)) {
        // This producer reserve was already cloned, find the cloned result
        Value clonedResult = irMap.lookupOrDefault(reserveOp.getResult());
        if (clonedResult) {
          reservedOutputs[mappedCB] = clonedResult;
        }
      }
    }
  }
  for (Operation &op : cb.without_terminator()) {
    // Skip d2m.yield explicitly: it lacks the IsTerminator trait, so
    // Block::without_terminator() does not filter it out, and cloning would
    // accumulate stale yields across cascading fusion rounds.
    if (isa<YieldOp>(op)) {
      continue;
    }
    // Skip operand-associated tensor.empty ops - already created in the fused
    // block. Intermediate tensor.empty ops (from a prior fusion iteration)
    // are cloned normally so the new fused region owns them.
    if (auto emptyOp = dyn_cast<mlir::tensor::EmptyOp>(&op);
        emptyOp && operandAssociatedEmpties.contains(emptyOp.getResult())) {
      continue;
    }

    // Handle reserve operations specially to deduplicate reserves on the same
    // output buffer
    if (auto reserveOp = dyn_cast<d2m::ReserveOp>(&op)) {
      Value originalCB = reserveOp.getCb();
      Value mappedCB = irMap.lookupOrDefault(originalCB);
      // Check if this is a reserve on an operand-associated value.
      if (isOperandAssociatedValue(mappedCB)) {
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

      // If reserve is on an intermediate value, check if it's mapped
      Value cbOperand = irMap.lookupOrDefault(originalCB);
      if (cbOperand && !isOperandAssociatedValue(cbOperand)) {
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
      if (cbOperand && !isOperandAssociatedValue(cbOperand)) {
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

  // Tag inner linalg.generic ops with their output blocking map so that the
  // allocator's reblocking can derive the correct shape for intermediate
  // buffers that have no associated generic operand.
  for (Operation &op : fusedBlock) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(&op)) {
      AffineMap outputMap = linalgOp.getIndexingMapsArray().back();
      op.setAttr("d2m.blocking_map", AffineMapAttr::get(outputMap));
    }
  }

  return fusedOp;
}

// V1 eltwise->reduction fusion requires exactly one reduction iterator on the
// consumer so the downstream phase-1/phase-2/scratch geometry stays
// well-defined (matches the DstRegisterAnalysis shared-parallel-chunking
// fallback and the SpillAndScratch row cross-step path).
static bool hasSingleReductionDim(GenericOp gOp) {
  unsigned numReduction = 0;
  for (Attribute it : gOp.getIteratorTypes()) {
    auto itAttr = mlir::dyn_cast<ttcore::IteratorTypeAttr>(it);
    if (itAttr && itAttr.getValue() == ttcore::IteratorType::Reduction) {
      ++numReduction;
    }
  }
  return numReduction == 1;
}

static bool isValidReductionFusionConsumer(GenericOp gOp) {
  if (!gOp.isComputeOnlyForm()) {
    return false;
  }

  if (!gOp.hasPureTensorSemantics()) {
    return false;
  }

  if (!gOp.hasReduction()) {
    return false;
  }

  // V1: restrict to exactly one reduction dimension.
  if (!hasSingleReductionDim(gOp)) {
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

static bool isEltwiseReductionFusable(OpOperand *fusionTargetOperand) {
  if (!fusionTargetOperand) {
    return false;
  }

  auto producer = fusionTargetOperand->get().getDefiningOp<GenericOp>();
  auto consumer = dyn_cast<GenericOp>(fusionTargetOperand->getOwner());

  if (!producer || !consumer) {
    return false;
  }

  if (!isValidElementwiseFusionTarget(producer)) {
    return false;
  }

  if (!isValidReductionFusionConsumer(consumer)) {
    return false;
  }

  if (!producerResultUsedOnlyByConsumer(producer, consumer)) {
    return false;
  }

  return passesSharedFusionShapeChecks(fusionTargetOperand, producer, consumer);
}

// Build the fused op, redirect uses, erase the originals. Shared between the
// elementwise-only and eltwise->reduction patterns. The fused op adopts the
// consumer's iteration space; producer operands are remapped into it via
// argMap o inv(prodResMap) o consMap.
static void fuseOverOperand(OpOperand *fusedOperand, GenericOp producer,
                            GenericOp consumer, PatternRewriter &rewriter) {
  auto [fusedInputs, fusedOutputs, fusedMaps] =
      getFusedOperands(fusedOperand, producer, consumer);
  auto mergedCaptures = llvm::to_vector(llvm::concat<Value>(
      consumer.getAdditionalArgs(), producer.getAdditionalArgs()));

  GenericOp fusedOp =
      createFusedGeneric(fusedOperand, producer, consumer, fusedInputs,
                         fusedOutputs, mergedCaptures, fusedMaps, rewriter);

  for (auto [i, r] : llvm::enumerate(consumer->getResults())) {
    r.replaceAllUsesWith(fusedOp->getResult(i));
  }

  rewriter.eraseOp(consumer);
  rewriter.eraseOp(producer);
}

namespace {
struct FuseD2MEltwiseReductionOpsPattern : public OpRewritePattern<GenericOp> {
  FuseD2MEltwiseReductionOpsPattern(MLIRContext *context)
      : OpRewritePattern<GenericOp>(context) {}

  LogicalResult matchAndRewrite(GenericOp consumer,
                                PatternRewriter &rewriter) const final {
    if (!isValidReductionFusionConsumer(consumer)) {
      return failure();
    }

    assert(consumer.getNumRegions() == 1u);

    OpOperand *fusedOperand = nullptr;
    for (OpOperand *use : consumer.getDpsInputOperands()) {
      if (isEltwiseReductionFusable(use)) {
        fusedOperand = use;
        break;
      }
    }
    if (!fusedOperand) {
      return failure();
    }

    auto producer = fusedOperand->get().getDefiningOp<GenericOp>();
    assert(producer);
    fuseOverOperand(fusedOperand, producer, consumer, rewriter);
    return success();
  }
};
} // namespace

namespace {
struct FuseD2MElementwiseOpsPattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp consumer,
                                PatternRewriter &rewriter) const final {
    if (!isValidElementwiseFusionTarget(consumer)) {
      return failure();
    }

    assert(consumer.getNumRegions() == 1u);

    // Consumer was already validated above; only re-check the producer.
    OpOperand *fusedOperand = nullptr;
    for (OpOperand *use : consumer.getDpsInputOperands()) {
      if (isElementwiseFusable(use, /*checkConsumer=*/false,
                               /*checkProducer=*/true)) {
        fusedOperand = use;
        break;
      }
    }
    if (!fusedOperand) {
      return failure();
    }

    auto producer = fusedOperand->get().getDefiningOp<GenericOp>();
    assert(producer);
    fuseOverOperand(fusedOperand, producer, consumer, rewriter);
    return success();
  }
};
} // namespace

namespace {
class D2MGenericFusion
    : public tt::d2m::impl::D2MGenericFusionBase<D2MGenericFusion> {
  using D2MGenericFusionBase::D2MGenericFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseD2MElementwiseOpsPattern>(ctx);
    if (enableEltwiseReductionFusion) {
      patterns.add<FuseD2MEltwiseReductionOpsPattern>(ctx);
    }
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
