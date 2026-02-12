// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

#define DEBUG_TYPE "D2MGenericAffineLoopFusion"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICAFFINELOOPFUSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Unwrap view-like ops (stream_layout/view_layout/etc.) to get the underlying
// memref (if necessary).
static Value unwrapIfViewLike(Value value) {
  while (auto viewOp = value.getDefiningOp<ViewOpInterface>()) {
    value = viewOp.getInput();
  }
  return value;
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
// GenericOps. Also check through view wrappers.
static bool isSoleReader(Value sharedMemref, GenericOp consumer) {
  for (Operation *user : sharedMemref.getUsers()) {
    if (user == consumer.getOperation()) {
      continue;
    }
    if (auto viewOp = mlir::dyn_cast<ViewOpInterface>(user)) {
      for (Operation *viewUser : viewOp.getResult().getUsers()) {
        if (viewUser == consumer.getOperation()) {
          continue;
        }
        if (mlir::isa<GenericOp>(viewUser)) {
          return false;
        }
      }
      continue;
    }
    auto otherGeneric = mlir::dyn_cast<GenericOp>(user);
    if (!otherGeneric) {
      continue;
    }
    for (OpOperand *input : otherGeneric.getDpsInputOperands()) {
      Value rawInput = unwrapIfViewLike(input->get());
      if (rawInput == sharedMemref) {
        return false;
      }
    }
  }
  return true;
}

struct FusedOperandInfo {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<AffineMap> maps;
  unsigned sharedInputIdx;
};

// Build fused GenericOp inputs/outputs/indexing maps.
static FusedOperandInfo
buildFusedOperandInfo(GenericOp producer, GenericOp consumer,
                      GenericOp supersetGeneric, GenericOp subsetGeneric,
                      OpOperand *inputOperand, Value sharedMemref) {
  FusedOperandInfo info;

  AffineMap supersetSharedMap =
      supersetGeneric.getIndexingMapForOperand(sharedMemref);
  AffineMap subsetOutputMap = subsetGeneric.getIndexingMap(
      subsetGeneric.getDpsInitOperand(0)->getOperandNumber());
  AffineMap subsetOutputInv = inversePermutation(subsetOutputMap);

  // Producer inputs.
  for (OpOperand *pi : producer.getDpsInputOperands()) {
    info.inputs.push_back(pi->get());
    AffineMap argMap = producer.getIndexingMap(pi->getOperandNumber());
    if (producer == subsetGeneric) {
      info.maps.push_back(
          argMap.compose(subsetOutputInv).compose(supersetSharedMap));
    } else {
      info.maps.push_back(argMap);
    }
  }

  // Shared intermediate as input.
  info.sharedInputIdx = info.inputs.size();
  info.inputs.push_back(sharedMemref);
  AffineMap sharedMap =
      consumer.getIndexingMap(inputOperand->getOperandNumber());
  if (consumer == subsetGeneric) {
    info.maps.push_back(
        sharedMap.compose(subsetOutputInv).compose(supersetSharedMap));
  } else {
    info.maps.push_back(sharedMap);
  }

  // Other consumer inputs.
  for (OpOperand *ci : consumer.getDpsInputOperands()) {
    if (ci == inputOperand) {
      continue;
    }
    info.inputs.push_back(ci->get());
    AffineMap argMap = consumer.getIndexingMap(ci->getOperandNumber());
    if (consumer == subsetGeneric) {
      info.maps.push_back(
          argMap.compose(subsetOutputInv).compose(supersetSharedMap));
    } else {
      info.maps.push_back(argMap);
    }
  }

  // Consumer outputs only.
  for (OpOperand &co : consumer.getDpsInitsMutable()) {
    info.outputs.push_back(co.get());
    AffineMap argMap = consumer.getIndexingMap(co.getOperandNumber());
    if (consumer == subsetGeneric) {
      info.maps.push_back(
          argMap.compose(subsetOutputInv).compose(supersetSharedMap));
    } else {
      info.maps.push_back(argMap);
    }
  }

  return info;
}

// Populate fused block arguments and seed IRMapping for producer/consumer args.
static IRMapping buildFusedBlockArgMapping(GenericOp producer,
                                           GenericOp consumer,
                                           OpOperand *inputOperand,
                                           GenericOp fusedOp,
                                           unsigned fusedSharedInputIdx,
                                           unsigned producerSharedInitIdx) {
  Block &fusedBlock = fusedOp.getRegion(0).emplaceBlock();
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

  // Map producer block args.
  for (OpOperand *pi : producer.getDpsInputOperands()) {
    unsigned fusedIdx =
        sourceToFusedIdx[{producer.getOperation(), pi->getOperandNumber()}];
    irMap.map(prodBlock.getArgument(pi->getOperandNumber()),
              fusedBlock.getArgument(fusedIdx));
  }
  irMap.map(prodBlock.getArgument(producerSharedInitIdx),
            fusedBlock.getArgument(fusedSharedInputIdx));

  // Map consumer block args.
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

  return irMap;
}

// Recursively clone loop-external, side-effect-free defs into the destination
// block so cloned loops don't retain uses into soon-to-be-erased regions.  This
// mainly avoids issues with get_block_factor() ops outside of loop being
// referenced post fusion.
static void cloneExternalDefForValue(Value value, affine::AffineForOp loopOp,
                                     OpBuilder &builder, IRMapping &irMap,
                                     DenseSet<Operation *> &visited) {
  Operation *defOp = value.getDefiningOp();
  if (irMap.contains(value) || !defOp) {
    return;
  }
  bool isLoopExternalDef =
      defOp->getBlock() == loopOp->getBlock() && defOp->isBeforeInBlock(loopOp);
  if (!isLoopExternalDef || !visited.insert(defOp).second) {
    return;
  }

  // Recursively clone operands of external defs.
  for (Value operand : defOp->getOperands()) {
    cloneExternalDefForValue(operand, loopOp, builder, irMap, visited);
  }
  if (!mlir::isMemoryEffectFree(defOp)) {
    return;
  }

  Operation *cloned = builder.clone(*defOp, irMap);
  for (auto [oldRes, newRes] :
       llvm::zip(defOp->getResults(), cloned->getResults())) {
    irMap.map(oldRes, newRes);
  }
}

// Clone values defined outside each loop (e.g. d2m.get_block_factor) into
// the fused region, so cloned loops do not keep references into operations
// that will be erased with the original producer/consumer generics.
static void cloneExternalLoopDefs(affine::AffineForOp loopOp,
                                  OpBuilder &builder, IRMapping &irMap) {
  DenseSet<Operation *> visited;
  loopOp->walk([&](affine::AffineForOp forOp) {
    for (Value operand : forOp->getOperands()) {
      cloneExternalDefForValue(operand, loopOp, builder, irMap, visited);
    }
  });
}

// Clone producer/consumer loops and run affine fusion utilities.
static bool runAffineLoopFusion(GenericOp fusedOp,
                                affine::AffineForOp producerLoop,
                                affine::AffineForOp consumerLoop,
                                unsigned producerDepth, unsigned consumerDepth,
                                OpBuilder &builder, IRMapping &irMap) {
  // Insert all newly cloned IR into the fused generic body.
  Block &fusedBlock = fusedOp.getRegion(0).front();
  builder.setInsertionPointToEnd(&fusedBlock);

  // Remap loop-external defs (e.g. block factors) so cloned loops do not
  // reference values owned by the original producer/consumer regions.
  cloneExternalLoopDefs(producerLoop, builder, irMap);
  cloneExternalLoopDefs(consumerLoop, builder, irMap);

  // Clone both loop nests into the fused generic and preserve result remaps.
  affine::AffineForOp clonedProducerLoop =
      mlir::cast<affine::AffineForOp>(builder.clone(*producerLoop, irMap));

  for (auto [oldRes, newRes] : llvm::zip(producerLoop->getResults(),
                                         clonedProducerLoop->getResults())) {
    irMap.map(oldRes, newRes);
  }

  affine::AffineForOp clonedConsumerLoop =
      mlir::cast<affine::AffineForOp>(builder.clone(*consumerLoop, irMap));

  // Treat the deeper nest as fusion source and fuse into the shallower depth.
  bool isProducerSrc = (producerDepth >= consumerDepth);
  affine::AffineForOp srcLoop =
      isProducerSrc ? clonedProducerLoop : clonedConsumerLoop;
  affine::AffineForOp dstLoop =
      isProducerSrc ? clonedConsumerLoop : clonedProducerLoop;
  unsigned dstLoopDepth = std::min(producerDepth, consumerDepth);

  affine::ComputationSliceState srcSlice;
  affine::FusionResult result = affine::canFuseLoops(
      srcLoop, dstLoop, dstLoopDepth, &srcSlice,
      affine::FusionStrategy(affine::FusionStrategy::ProducerConsumer));
  if (result.value != affine::FusionResult::Success) {
    // Bail out without mutating originals if fusion legality checks fail.
    return false;
  }

  // Fuse loops and erase original source loop nest.
  affine::fuseLoops(srcLoop, dstLoop, srcSlice,
                    /*isInnermostSiblingInsertion=*/false);
  srcLoop->erase();

  return true;
}

// Try to fuse a producer-consumer pair. Returns the fused generic on success.
static GenericOp tryFusePair(GenericOp producer, GenericOp consumer,
                             OpOperand *inputOperand, Value sharedMemref,
                             OpBuilder &builder) {
  // Get outermost loops to determine superset/subset.
  std::optional<Operation *> producerLoopOp =
      producer.getOutermostBlockingLoopOp();
  std::optional<Operation *> consumerLoopOp =
      consumer.getOutermostBlockingLoopOp();
  if (!producerLoopOp || !consumerLoopOp) {
    return nullptr;
  }
  auto producerLoop =
      mlir::dyn_cast<affine::AffineForOp>(producerLoopOp.value());
  auto consumerLoop =
      mlir::dyn_cast<affine::AffineForOp>(consumerLoopOp.value());

  if (!producerLoop || !consumerLoop) {
    return nullptr;
  }

  // Determine superset generic (one with deeper loop nesting).
  unsigned producerDepth = producer.getNumBlockFactors();
  unsigned consumerDepth = consumer.getNumBlockFactors();

  GenericOp supersetGeneric =
      (producerDepth >= consumerDepth) ? producer : consumer;
  GenericOp subsetGeneric =
      (producerDepth >= consumerDepth) ? consumer : producer;
  std::optional<unsigned> producerSharedInitIdx =
      producer.getOutputOperandIndex(sharedMemref);
  if (!producerSharedInitIdx) {
    return nullptr;
  }

  // Build fused generic op signature (inputs, outputs, indexing maps, etc).
  FusedOperandInfo fusedInfo =
      buildFusedOperandInfo(producer, consumer, supersetGeneric, subsetGeneric,
                            inputOperand, sharedMemref);

  // Create fused GenericOp shell.
  builder.setInsertionPoint(consumer);
  auto fusedOp = builder.create<GenericOp>(
      consumer.getLoc(), TypeRange{}, fusedInfo.inputs, fusedInfo.outputs,
      supersetGeneric.getGrid(), supersetGeneric.getBlockFactors(),
      builder.getAffineMapArrayAttr(fusedInfo.maps),
      supersetGeneric.getIteratorTypes(), supersetGeneric.getThreads(),
      supersetGeneric.getScratchInputsAttr(), /*regions=*/1);

  IRMapping irMap = buildFusedBlockArgMapping(producer, consumer, inputOperand,
                                              fusedOp, fusedInfo.sharedInputIdx,
                                              *producerSharedInitIdx);

  if (!runAffineLoopFusion(fusedOp, producerLoop, consumerLoop, producerDepth,
                           consumerDepth, builder, irMap)) {
    fusedOp->erase();
    return nullptr;
  }

  // Mark as fused.
  fusedOp->setAttr("d2m.affine_fused", builder.getUnitAttr());

  return fusedOp;
}

namespace {
class D2MGenericAffineLoopFusion
    : public impl::D2MGenericAffineLoopFusionBase<D2MGenericAffineLoopFusion> {
public:
  using D2MGenericAffineLoopFusionBase::D2MGenericAffineLoopFusionBase;

  void runOnOperation() final {
    if (!enable) {
      return;
    }
    auto isFusionCandidate = [](GenericOp generic) {
      return generic.isUnifiedForm() &&
             !generic.hasSkipOpAffineLoopFusionTrait();
    };

    // Iteratively fuse pairs of generics until no more can be fused.
    bool changed = true;
    while (changed) {
      changed = false;
      // Collect all generics in the module.
      SmallVector<GenericOp> generics;
      getOperation()->walk([&](GenericOp g) { generics.push_back(g); });

      for (GenericOp consumer : generics) {
        if (!isFusionCandidate(consumer) ||
            !consumer.getOutermostBlockingLoopOp().has_value()) {
          continue;
        }

        for (OpOperand *inputOperand : consumer.getDpsInputOperands()) {
          Value rawMemref = unwrapIfViewLike(inputOperand->get());
          GenericOp producer = findProducerGeneric(rawMemref, consumer);
          if (!producer || !isFusionCandidate(producer) ||
              !isSoleReader(rawMemref, consumer)) {
            continue;
          }

          OpBuilder builder(consumer);
          GenericOp fused =
              tryFusePair(producer, consumer, inputOperand, rawMemref, builder);
          if (!fused) {
            continue;
          }

          // If we made it here, we successfully fused a pair of generics. So
          // run at least one more iteration.
          changed = true;

          // Erase unfused original generics.
          producer->erase();
          consumer->erase();
          break;
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
