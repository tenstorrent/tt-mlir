// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/GenericAffineUtils.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include <optional>

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

  SmallVector<RemoteLoadOp> consumerLoads;
  consumer.getRegion(0).walk([&](RemoteLoadOp loadOp) {
    Value rawMemref = peelStreamLayout(loadOp.getMemref());
    if (rawMemref == sharedMemref) {
      consumerLoads.push_back(loadOp);
    }
  });

  if (consumerLoads.size() != 1) {
    return nullptr;
  }
  RemoteLoadOp consumerLoad = consumerLoads.front();

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

// Find the output operand number that corresponds to `target`.
static std::optional<unsigned> findOutputOperandForValue(GenericOp generic,
                                                         Value target) {
  for (OpOperand &output : generic.getOutputsMutable()) {
    if (output.get() == target) {
      return output.getOperandNumber();
    }
  }
  return std::nullopt;
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

// Clone producer/consumer loops and run affine fusion utilities.
static bool runAffineLoopFusion(GenericOp fusedOp,
                                affine::AffineForOp producerLoop,
                                affine::AffineForOp consumerLoop,
                                unsigned producerDepth, unsigned consumerDepth,
                                OpBuilder &builder, IRMapping &irMap) {
  Block &fusedBlock = fusedOp.getRegion(0).front();
  builder.setInsertionPointToEnd(&fusedBlock);
  affine::AffineForOp clonedProducerLoop =
      cast<affine::AffineForOp>(builder.clone(*producerLoop, irMap));

  for (auto [oldRes, newRes] : llvm::zip(producerLoop->getResults(),
                                         clonedProducerLoop->getResults())) {
    irMap.map(oldRes, newRes);
  }

  affine::AffineForOp clonedConsumerLoop =
      cast<affine::AffineForOp>(builder.clone(*consumerLoop, irMap));

  utils::convertToAffineCompatibilityForm(fusedOp, builder);

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
    return false;
  }

  affine::fuseLoops(srcLoop, dstLoop, srcSlice,
                    /*isInnermostSiblingInsertion=*/false);
  srcLoop->erase();

  utils::convertFromAffineCompatibilityForm(fusedOp, builder);
  return true;
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
  std::optional<unsigned> producerSharedInitIdx =
      findOutputOperandForValue(producer, sharedMemref);
  if (!producerSharedInitIdx) {
    return nullptr;
  }

  FusedOperandInfo fusedInfo =
      buildFusedOperandInfo(producer, consumer, supersetGeneric, subsetGeneric,
                            inputOperand, sharedMemref);

  // Step 5: Create fused GenericOp shell
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
        if (!isFusionCandidate(consumer) || !getOutermostLoop(consumer)) {
          continue;
        }

        for (OpOperand *inputOperand : consumer.getDpsInputOperands()) {
          Value rawMemref = peelStreamLayout(inputOperand->get());
          GenericOp producer = findProducerGeneric(rawMemref, consumer);
          if (!producer || !isFusionCandidate(producer) ||
              !isSoleReader(rawMemref, consumer) ||
              !getMatchingStoreForShared(producer, consumer, rawMemref)) {
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
