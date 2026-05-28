// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
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

#include <cassert>
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

// =============================================================================
// Tilize / untilize fusion
// =============================================================================
//
// After LowerToLayout, every `d2m.to_layout` is materialized as a
// `d2m.generic` whose body contains a single `d2m.tile_tilize_block`
// (scalar-in / tile-out) or `d2m.tile_untilize_block` (tile-in / scalar-out)
// op. When such a generic feeds an interface-tagged consumer (e.g. matmul)
// — or is fed by one — we can fold the tilize/untilize body into the
// consumer/producer so the resulting compute kernel does on-the-fly tilize
// + matmul + (optional) untilize without materializing an intermediate
// CB-to-CB format-conversion kernel.

namespace {

// === Responsibility split between this pass and `TilizedOperandInterface` ===
//
// `TilizedOperandInterface` (on compute ops like `tile_matmul`) answers a
// per-op CAPABILITY + ENVIRONMENTAL-CONSTRAINT question: "given the current
// `l1AccEnabled` setting, can I accept untilized data on operand i / emit
// untilized data on result i?". It NEVER inspects IR neighbours.
//
// The two patterns in this file own all IR-STRUCTURE concerns: finding a
// pure tilize-/untilize-shaped `d2m.generic` neighbour, enforcing single-
// use / grid-match / no-`d2m.mask` invariants, and rewriting the result.
// The pattern queries the interface only at the very end — once the
// structural shape is recognised — to get the op-specific go/no-go signal.

// Direction of the row-major↔tile conversion a single-purpose
// `d2m.generic` performs. Used to share matcher and helper code between
// the tilize-input-side and untilize-output-side fusion patterns.
enum class RMConversionDir { Tilize, Untilize };

// True iff `gOp` is a single-purpose `d2m.generic` that does only a
// row-major↔tile conversion in the requested direction:
//
//   * `Tilize`:   outer input scalar-elem, outer init tile-elem, body
//                 contains exactly one `d2m.tile_tilize_block`.
//   * `Untilize`: outer input tile-elem, outer init scalar-elem, body
//                 contains exactly one `d2m.tile_untilize_block`. We also
//                 require input/init indexing maps to be identical (no
//                 reblock); that invariant lets `fuseUntilizeFromProducer`
//                 keep the producer's init indexing map unchanged.
bool isPureRMConversionGeneric(GenericOp gOp, RMConversionDir dir) {
  if (gOp.getNumDpsInputs() != 1 || gOp.getNumDpsInits() != 1) {
    return false;
  }
  if (!gOp.hasPureTensorSemantics()) {
    return false;
  }
  auto inTy = mlir::dyn_cast<RankedTensorType>(
      gOp.getDpsInputOperand(0)->get().getType());
  auto outTy = mlir::dyn_cast<RankedTensorType>(
      gOp.getDpsInitOperand(0)->get().getType());
  if (!inTy || !outTy) {
    return false;
  }
  const bool inTiled = ttcore::isTiled(inTy);
  const bool outTiled = ttcore::isTiled(outTy);
  if (dir == RMConversionDir::Tilize) {
    if (inTiled || !outTiled) {
      return false;
    }
  } else {
    if (!inTiled || outTiled) {
      return false;
    }
    // No-reblock invariant for untilize: see fuseUntilizeFromProducer.
    unsigned inIdx = gOp.getDpsInputOperand(0)->getOperandNumber();
    unsigned outIdx = gOp.getDpsInitOperand(0)->getOperandNumber();
    if (gOp.getIndexingMap(inIdx) != gOp.getIndexingMap(outIdx)) {
      return false;
    }
  }
  if (gOp.getNumRegions() == 0 || gOp.getRegion(0).empty()) {
    return false;
  }
  // Body must contain exactly one of the requested direction's block op
  // and zero of the opposite direction's.
  unsigned want = 0;
  for (Operation &op : gOp.getRegion(0).front()) {
    if (dir == RMConversionDir::Tilize) {
      if (isa<TileTilizeBlockOp>(op)) {
        ++want;
      } else if (isa<TileUntilizeBlockOp>(op)) {
        return false;
      }
    } else {
      if (isa<TileUntilizeBlockOp>(op)) {
        ++want;
      } else if (isa<TileTilizeBlockOp>(op)) {
        return false;
      }
    }
  }
  return want == 1;
}

// Returns true if any `d2m.mask` op is reachable from `start` along forward
// def-use chains. Bounded by the visited set. Used by the fusion matchers
// to soft-bail when an unaligned workload inserted a mask op between the
// row-major converter generic and its neighbour: such workloads are
// perfectly valid IR — they just can't participate in this fusion until
// mask-aware fusion lands.
bool hasMaskOpDownstream(Value start) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  llvm::SmallVector<Value, 8> worklist;
  worklist.push_back(start);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (!visited.insert(user).second) {
        continue;
      }
      if (isa<MaskOp>(user)) {
        return true;
      }
      for (Value result : user->getResults()) {
        worklist.push_back(result);
      }
    }
  }
  return false;
}

// Shared checks for a tilize/untilize fusion candidate. Returns the
// fused operand (and producer) if the producer-consumer pair is fuseable
// over a single operand; otherwise nullptr.
struct TilizeFusionMatch {
  OpOperand *operand = nullptr;
  GenericOp producer = nullptr;
};

TilizeFusionMatch matchTilizeIntoConsumer(GenericOp consumer) {
  // Build a `outerOperand -> (tilizableOp, operand idx on tilizableOp)` map
  // by walking the consumer body ONCE. The tilizable compute op (e.g.
  // `d2m.tile_matmul`) lives inside a `linalg.generic`; each of its
  // operands is a bb-arg of that linalg, which corresponds to a
  // `d2m.remote_load(outerOperand, ...)` outside the linalg. So
  // (tilizable operand i) ↔ (linalg input k) ↔ (remote_load memref M).
  llvm::DenseMap<Value, std::pair<TilizedOperandInterface, int64_t>>
      outerOperandToTilizable;
  consumer.getRegion(0).walk([&](Operation *op) {
    auto tilizableOp = dyn_cast<TilizedOperandInterface>(op);
    if (!tilizableOp) {
      return;
    }
    auto linalgOp = op->getParentOfType<linalg::GenericOp>();
    if (!linalgOp) {
      return;
    }
    Block &linalgBlock = linalgOp.getRegion().front();
    for (auto [tilizableOperandIdx, tilizableInput] :
         llvm::enumerate(op->getOperands())) {
      auto linalgBlockArg = dyn_cast<BlockArgument>(tilizableInput);
      if (!linalgBlockArg || linalgBlockArg.getOwner() != &linalgBlock) {
        continue;
      }
      unsigned linalgInputIdx = linalgBlockArg.getArgNumber();
      if (linalgInputIdx >= linalgOp.getInputs().size()) {
        continue;
      }
      auto loadOp =
          linalgOp.getInputs()[linalgInputIdx].getDefiningOp<RemoteLoadOp>();
      if (!loadOp) {
        continue;
      }
      outerOperandToTilizable.try_emplace(
          loadOp.getMemref(), tilizableOp,
          static_cast<int64_t>(tilizableOperandIdx));
    }
  });

  // Now iterate consumer inputs and look up the tilizable op's decision
  // per input.
  for (OpOperand *use : consumer.getDpsInputOperands()) {
    auto producer = use->get().getDefiningOp<GenericOp>();
    if (!producer ||
        !isPureRMConversionGeneric(producer, RMConversionDir::Tilize)) {
      continue;
    }
    if (producer.getGrid() != consumer.getGrid()) {
      continue;
    }
    if (!producerResultUsedOnlyByConsumer(producer, consumer)) {
      continue;
    }
    auto it = outerOperandToTilizable.find(use->get());
    if (it == outerOperandToTilizable.end()) {
      continue;
    }
    auto [tilizableOp, tilizableOperandIdx] = it->second;
    if (!tilizableOp.canConsumeUntilizedOperand(tilizableOperandIdx)) {
      continue;
    }
    // Soft-bail: mask op downstream means an unaligned-shape workload that
    // mask-aware fusion would need to handle. Valid IR; just don't fuse.
    if (hasMaskOpDownstream(producer.getResult(0))) {
      continue;
    }
    return {use, producer};
  }
  return {};
}

TilizeFusionMatch matchUntilizeFromProducer(GenericOp producer,
                                            bool l1AccEnabled) {
  // `producer` is a generic whose single result is consumed by exactly one
  // untilize-shaped `d2m.generic`. The producer must contain a tilizable
  // compute op (`TilizedOperandInterface`) whose `canProduceUntilizedResult
  // (0, l1AccEnabled)` returns true.
  if (producer.getNumResults() != 1) {
    return {};
  }
  Value producerResult = producer.getResult(0);

  // The consumer of `producerResult` (outside producer's own region) must
  // be exactly one GenericOp that is a pure untilize. Uses inside the
  // consumer's own region (e.g. a `d2m.remote_load` referencing the outer
  // SSA value) are allowed — walk up to find the enclosing GenericOp.
  GenericOp untilizeConsumer;
  OpOperand *fusedOperand = nullptr;
  for (OpOperand &use : producerResult.getUses()) {
    Operation *owner = use.getOwner();
    if (producer.getOperation()->isProperAncestor(owner)) {
      continue;
    }
    Operation *ancestor = owner;
    while (ancestor && !isa<GenericOp>(ancestor)) {
      ancestor = ancestor->getParentOp();
    }
    if (!ancestor) {
      return {};
    }
    auto enclosingGeneric = cast<GenericOp>(ancestor);
    if (untilizeConsumer && untilizeConsumer != enclosingGeneric) {
      return {};
    }
    untilizeConsumer = enclosingGeneric;
    // The OpOperand for `fuseUntilizeFromProducer` must be the outer
    // operand of the consumer generic, not the body-internal use.
    if (owner == enclosingGeneric.getOperation()) {
      fusedOperand = &use;
    }
  }
  if (!untilizeConsumer || !fusedOperand) {
    return {};
  }
  if (!isPureRMConversionGeneric(untilizeConsumer, RMConversionDir::Untilize)) {
    return {};
  }
  if (producer.getGrid() != untilizeConsumer.getGrid()) {
    return {};
  }

  // Producer body must contain a tilizable op whose result feeds (via
  // remote_store) the producer's init.
  bool fuseable = false;
  producer.getRegion(0).walk([&](Operation *op) {
    auto tilizableOp = dyn_cast<TilizedOperandInterface>(op);
    if (!tilizableOp) {
      return WalkResult::advance();
    }
    if (tilizableOp.canProduceUntilizedResult(/*resultIdx=*/0, l1AccEnabled)) {
      fuseable = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!fuseable) {
    return {};
  }

  // Soft-bail: mask op downstream means an unaligned-shape workload that
  // mask-aware fusion would need to handle. Valid IR; just don't fuse.
  if (hasMaskOpDownstream(producerResult)) {
    return {};
  }
  return {fusedOperand, producer};
}

// Insert a `tensor::EmptyOp` of `dstTy` followed by either a
// `d2m.tile_tilize_block` or `d2m.tile_untilize_block` at the current
// insertion point, bridging `src` (the OPPOSITE element type) to a value
// of `dstTy`. Returns the converter op's result. Shared by both fusion
// rewriters.
template <typename ConvOp>
static Value insertRMConversion(PatternRewriter &rewriter, Location loc,
                                Value src, RankedTensorType dstTy) {
  auto scratch = rewriter.create<tensor::EmptyOp>(loc, dstTy.getShape(),
                                                  dstTy.getElementType());
  return rewriter.create<ConvOp>(loc, dstTy, src, scratch.getResult())
      .getResult();
}

// Compute the row-major equivalent of a tiled tensor type: drop the
// `!ttcore.tile<H x W, T>` element type to scalar `T` and grow the last two
// dims by the tile factor.
//
// Only valid for LOCAL SHARD types (`d2m.remote_load` / `d2m.remote_store`
// local buffers) — those have no `MetalLayoutAttr` encoding, since the
// encoding lives on the OUTER tensor only. Asserting null encoding here
// keeps callers honest: passing an outer (encoded) tensor would silently
// produce a wrong type because `MetalLayoutAttr` carries shape-dependent
// metadata (logical_shape, alignment) that doesn't survive the tile→scalar
// dimension grow without recomputation.
static RankedTensorType tiledTensorToScalar(RankedTensorType tiledTy) {
  assert(!tiledTy.getEncoding() &&
         "tiledTensorToScalar: expected unencoded local shard type");
  auto tileTy = mlir::cast<ttcore::TileType>(tiledTy.getElementType());
  ArrayRef<int64_t> tileShape = tileTy.getShape();
  SmallVector<int64_t> rmShape(tiledTy.getShape().begin(),
                               tiledTy.getShape().end());
  assert(rmShape.size() >= 2 && "tiled tensor must have rank >= 2");
  rmShape[rmShape.size() - 2] *= tileShape[0];
  rmShape[rmShape.size() - 1] *= tileShape[1];
  return RankedTensorType::get(rmShape, tileTy.getElementType());
}

// Custom helper to fold a tilize-only `producer` generic into a `consumer`
// generic over `consumerOperand`. Unlike the generic `fuseOverOperand`, this
// helper preserves the consumer's body structure (including its existing
// `d2m.block_index` calls and indexing maps) and only:
//
//   * rewires the consumer's outer operand to point at the producer's
//     scalar input;
//   * updates the consumer's indexing_map for that operand to
//     `producer_argMap ∘ inv(producer_resMap) ∘ consumer_argMap`;
//   * rewrites the consumer's `d2m.remote_load(producer_result, ...)` so it
//     reads the producer's scalar input into a scalar shard buffer; and
//   * inserts a `d2m.tile_tilize_block` immediately after the new load to
//     produce the tile shard the consumer's downstream compute expects.
//
// The producer is then erased.
static LogicalResult fuseTilizeIntoConsumer(OpOperand *consumerOperand,
                                            GenericOp producer,
                                            GenericOp consumer,
                                            PatternRewriter &rewriter) {
  unsigned consumerOperandIdx = consumerOperand->getOperandNumber();

  // Locate the consumer's remote_load that reads from the producer's
  // result. The replacement is keyed off that load.
  Value producerResult = producer.getResult(0);
  RemoteLoadOp consumerLoad;
  consumer.getRegion(0).walk([&](RemoteLoadOp op) {
    if (op.getMemref() == producerResult) {
      consumerLoad = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!consumerLoad) {
    return failure();
  }
  // V1: only handle the implicit-buffer load path.
  if (consumerLoad.getCb()) {
    return failure();
  }

  // Derive the scalar shard type from the existing tile shard buffer.
  auto tiledShardTy =
      mlir::cast<RankedTensorType>(consumerLoad.getResult().getType());
  RankedTensorType rmShardTy = tiledTensorToScalar(tiledShardTy);

  Value producerScalarInput = producer.getDpsInputOperand(0)->get();

  // 1) Create the new scalar-shard local buffer at the consumer load's
  //    insertion point. We don't rely on the old buffer's defining op
  //    being a `tensor.empty` (it could be a block-arg, cast, etc.); the
  //    old buffer becomes dead once the old load is erased and will be
  //    DCE'd by canonicalization.
  Value oldBuffer = consumerLoad.getLocalBuffer();
  rewriter.setInsertionPoint(consumerLoad);
  auto newEmpty = rewriter.create<tensor::EmptyOp>(
      consumerLoad.getLoc(), rmShardTy.getShape(), rmShardTy.getElementType());

  // 2) Recreate the remote_load with scalar types, reading from the
  //    producer's scalar input.
  SmallVector<Value> indices(consumerLoad.getIndices().begin(),
                             consumerLoad.getIndices().end());
  SmallVector<Value> mcastStartIndex(consumerLoad.getMcastStartIndex().begin(),
                                     consumerLoad.getMcastStartIndex().end());
  SmallVector<Value> mcastShape(consumerLoad.getMcastShape().begin(),
                                consumerLoad.getMcastShape().end());
  SmallVector<Value> mcastDims(consumerLoad.getMcastDims().begin(),
                               consumerLoad.getMcastDims().end());
  auto newLoadOp = rewriter.create<RemoteLoadOp>(
      consumerLoad.getLoc(),
      /*result=*/TypeRange{rmShardTy},
      /*localBuffer=*/newEmpty.getResult(),
      /*memref=*/producerScalarInput, indices,
      /*cb=*/Value{}, mcastStartIndex, mcastShape, mcastDims);
  Value newLoad = newLoadOp.getResult();

  // 3) Inject tile_tilize_block(newLoad, scratch) producing the tile-typed
  //    value the consumer's downstream linalg.generic expects.
  rewriter.setInsertionPointAfter(newLoadOp);
  Value tilized = insertRMConversion<TileTilizeBlockOp>(
      rewriter, consumerLoad.getLoc(), newLoad, tiledShardTy);

  // 4) Rewire downstream uses of the original tile-typed loaded value.
  rewriter.replaceAllUsesWith(consumerLoad.getResult(), tilized);

  // 5) Erase the old load. The old local buffer (tile-typed) becomes
  //    dead — if it's a `tensor.empty` with no other uses, erase it
  //    eagerly so the rewriter doesn't loop on stale ops; otherwise
  //    leave for canonicalization to DCE.
  rewriter.eraseOp(consumerLoad);
  if (auto oldEmpty = oldBuffer.getDefiningOp<tensor::EmptyOp>();
      oldEmpty && oldEmpty->use_empty()) {
    rewriter.eraseOp(oldEmpty);
  }

  // 6) Update the consumer's outer operand + indexing_map for that operand.
  AffineMap prodResMap = producer.getIndexingMap(
      producer.getDpsInitOperand(0)->getOperandNumber());
  AffineMap prodArgMap = producer.getIndexingMap(
      producer.getDpsInputOperand(0)->getOperandNumber());
  AffineMap consMap = consumer.getIndexingMap(consumerOperandIdx);
  // Pure tilize → prodResMap is identity-permutation (enforced by
  // `isPureRMConversionGeneric`). Future relaxation (reblocking tilize)
  // must update this composition.
  assert(prodResMap.isPermutation() &&
         "tilize fusion: producer init map must be a permutation");
  AffineMap newMap =
      prodArgMap.compose(inversePermutation(prodResMap)).compose(consMap);

  SmallVector<AffineMap> newMaps =
      llvm::to_vector(consumer.getIndexingMapsValue());
  newMaps[consumerOperandIdx] = newMap;

  rewriter.modifyOpInPlace(consumer, [&]() {
    consumer.setOperand(consumerOperandIdx, producerScalarInput);
    consumer.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
  });

  // 7) Erase the producer now that nothing references its result.
  rewriter.eraseOp(producer);

  return success();
}

// Custom helper to fold an untilize-only `untilizeConsumer` generic into a
// `producer` generic that produces tile data. Symmetric to
// `fuseTilizeIntoConsumer`:
//
//   * the untilize consumer's outer output replaces the producer's
//     tile-typed output;
//   * the producer's indexing_map for its init is left UNCHANGED. The
//     no-reblock invariant enforced by `isPureRMConversionGeneric` for the
//     `Untilize` direction (`untilize_argMap == untilize_initMap`) makes
//     the composition `prodInitMap ∘ inv(untilize_argMap) ∘ untilize_initMap`
//     collapse to `prodInitMap`. Future relaxation (reblocking untilize)
//     must drop that invariant and recompose here;
//   * the producer's body's `d2m.remote_store(producer_init, ..., %tile)`
//     is rewritten so the stored value is first piped through a
//     `d2m.tile_untilize_block` and stored as scalar shard data.
static LogicalResult fuseUntilizeFromProducer(GenericOp producer,
                                              GenericOp untilizeConsumer,
                                              PatternRewriter &rewriter) {
  assert(producer.getNumDpsInits() == 1 &&
         "untilize fusion: producer must have exactly one init");
  Value untilizeOutput = untilizeConsumer.getDpsInitOperand(0)->get();

  // Build newProducer with the consumer's scalar output as its init operand
  // and matching scalar result type. The producer's region is moved over
  // via `takeBody`, and all body surgery (insert `tile_untilize_block`,
  // rewrite `remote_store` to scalar) is done INSIDE newProducer's region
  // afterwards. This avoids leaving the old producer in a half-modified
  // state where its yield type doesn't match its result type — which makes
  // the greedy rewriter loop while trying to resolve a transiently-invalid
  // op.
  // newProducer references `untilizeOutput` (the consumer's outer init),
  // which is defined AFTER the original producer in IR order. Place
  // newProducer where untilizeConsumer was so dominance holds for the
  // rewired init operand.
  rewriter.setInsertionPoint(untilizeConsumer);
  auto newProducer = rewriter.create<GenericOp>(
      producer.getLoc(), TypeRange{untilizeOutput.getType()},
      producer.getInputs(), ValueRange{untilizeOutput},
      producer.getAdditionalArgs(), producer.getGrid(),
      producer.getBlockFactors(),
      rewriter.getAffineMapArrayAttr(producer.getIndexingMapsValue()),
      producer.getIteratorTypes(), producer.getThreads(),
      producer.getFabricConnectionConfigAttr(), /*regions=*/1);
  newProducer.getRegion(0).takeBody(producer.getRegion(0));

  // Find the producer's `remote_store` that targets the (old) producer init
  // — now living inside newProducer's region. The store's memref operand
  // still references the OLD producer's init SSA value, because that's the
  // outer value the body op was bound to. We rewrite the store to use
  // `untilizeOutput` instead and inject the `tile_untilize_block` ahead of
  // it. Then update the body yield's operand to the new store's result.
  Value oldProducerInit = producer.getDpsInitOperand(0)->get();
  RemoteStoreOp producerStore;
  newProducer.getRegion(0).walk([&](RemoteStoreOp op) {
    if (op.getMemref() == oldProducerInit) {
      producerStore = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  assert(producerStore && "producer body must have a remote_store targeting "
                          "its init operand");

  Value tileShardValue = producerStore.getLocalBuffer();
  auto tiledShardTy = mlir::cast<RankedTensorType>(tileShardValue.getType());
  RankedTensorType rmShardTy = tiledTensorToScalar(tiledShardTy);

  rewriter.setInsertionPoint(producerStore);
  Value untilized = insertRMConversion<TileUntilizeBlockOp>(
      rewriter, producerStore.getLoc(), tileShardValue, rmShardTy);

  SmallVector<Value> indices(producerStore.getIndices().begin(),
                             producerStore.getIndices().end());
  auto newStoreOp =
      rewriter.create<RemoteStoreOp>(producerStore.getLoc(),
                                     /*resultType=*/untilizeOutput.getType(),
                                     /*memref=*/untilizeOutput, indices,
                                     /*localBuffer=*/untilized);

  rewriter.replaceAllUsesWith(producerStore.getResult(),
                              newStoreOp.getResult());
  rewriter.eraseOp(producerStore);

  for (auto [i, r] : llvm::enumerate(untilizeConsumer->getResults())) {
    r.replaceAllUsesWith(newProducer.getResult(i));
  }
  rewriter.eraseOp(untilizeConsumer);
  rewriter.eraseOp(producer);
  return success();
}

// A `d2m.generic` is eligible as the CENTER of either tilize-input fusion
// (consumer side) or untilize-output fusion (producer side) when it has
// pure tensor semantics and is NOT itself a single-purpose RM↔tile
// converter — otherwise the patterns would either fire on the tilize/
// untilize generic itself or double-fire on its neighbours.
bool isFusionCenterCandidate(GenericOp gOp) {
  if (!gOp.hasPureTensorSemantics()) {
    return false;
  }
  if (isPureRMConversionGeneric(gOp, RMConversionDir::Tilize) ||
      isPureRMConversionGeneric(gOp, RMConversionDir::Untilize)) {
    return false;
  }
  return true;
}

// Single pattern handling both fusion directions on the same `d2m.generic`
// candidate: when the op is the consumer-of-a-tilize, fold the tilize in;
// when it is the producer-of-an-untilize, fold the untilize in. A given
// generic can satisfy at most one direction per invocation (the IR shape
// differs); both checks are cheap. The greedy rewriter re-fires the
// pattern after a successful fold, so the symmetric case (matmul with
// tilize input AND untilize output) lands in two iterations.
struct FuseD2MRMConversionPattern : public OpRewritePattern<GenericOp> {
  FuseD2MRMConversionPattern(MLIRContext *ctx, bool l1AccEnabled)
      : OpRewritePattern<GenericOp>(ctx), l1AccEnabled(l1AccEnabled) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (!isFusionCenterCandidate(op)) {
      return failure();
    }
    if (auto match = matchTilizeIntoConsumer(op); match.operand) {
      return fuseTilizeIntoConsumer(match.operand, match.producer, op,
                                    rewriter);
    }
    if (auto match = matchUntilizeFromProducer(op, l1AccEnabled);
        match.operand) {
      auto untilizeConsumer = cast<GenericOp>(match.operand->getOwner());
      return fuseUntilizeFromProducer(op, untilizeConsumer, rewriter);
    }
    return failure();
  }

  bool l1AccEnabled;
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
    if (enableTilizeFusion) {
      patterns.add<FuseD2MRMConversionPattern>(ctx,
                                               /*l1AccEnabled=*/enableL1Acc);
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
