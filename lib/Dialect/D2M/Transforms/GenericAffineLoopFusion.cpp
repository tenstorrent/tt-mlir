// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

// Collect the chain of nested d2m.blocking_loop affine.for ops.
static SmallVector<affine::AffineForOp>
collectLoopChain(affine::AffineForOp outerLoop) {
  SmallVector<affine::AffineForOp> chain;
  auto cur = outerLoop;
  while (cur) {
    chain.push_back(cur);
    affine::AffineForOp nested = nullptr;
    for (Operation &op : cur.getBody()->getOperations()) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
        if (forOp->hasAttr("d2m.blocking_loop")) {
          nested = forOp;
          break;
        }
      }
    }
    cur = nested;
  }
  return chain;
}

// Try to fuse a producer-consumer pair. Returns the fused generic on success.
static GenericOp tryFusePair(GenericOp producer, GenericOp consumer,
                             OpOperand *inputOperand, Value sharedMemref,
                             OpBuilder &builder) {
  affine::AffineForOp producerLoop = getOutermostLoop(producer);
  affine::AffineForOp consumerLoop = getOutermostLoop(consumer);
  if (!producerLoop || !consumerLoop) {
    return nullptr;
  }

  unsigned producerDepth = getOuterLoopDepth(producerLoop);
  unsigned consumerDepth = getOuterLoopDepth(consumerLoop);

  GenericOp supersetGeneric =
      (producerDepth > consumerDepth) ? producer : consumer;
  GenericOp subsetGeneric =
      (producerDepth > consumerDepth) ? consumer : producer;

  // Build fused operands and indexing maps.
  // Fused inputs: producer inputs + shared intermediate + other consumer
  // inputs. Fused outputs: consumer inits only (verifier requires exactly one
  // output). The shared intermediate is kept as a fused input so that cloned
  // remote_store/remote_load ops referencing it satisfy the verifier.
  SmallVector<Value> fusedInputs;
  SmallVector<Value> fusedOutputs;
  SmallVector<AffineMap> fusedMaps;

  AffineMap supersetSharedMap =
      supersetGeneric.getIndexingMapForOperand(sharedMemref);
  AffineMap subsetOutputMap = subsetGeneric.getIndexingMap(
      subsetGeneric.getDpsInitOperand(0)->getOperandNumber());
  AffineMap subsetOutputInv = inversePermutation(subsetOutputMap);

  // Producer inputs.
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

  // Shared intermediate as fused input.
  unsigned fusedSharedInputIdx = fusedInputs.size();
  {
    fusedInputs.push_back(sharedMemref);
    AffineMap sharedMap =
        consumer.getIndexingMap(inputOperand->getOperandNumber());
    if (consumer == subsetGeneric) {
      fusedMaps.push_back(
          sharedMap.compose(subsetOutputInv).compose(supersetSharedMap));
    } else {
      fusedMaps.push_back(sharedMap);
    }
  }

  // Other consumer inputs (skip the shared intermediate).
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

  // Consumer inits only.
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

  // Create fused GenericOp using superset's attributes.
  // In memref mode, GenericOp has no results.
  builder.setInsertionPoint(consumer);
  auto fusedOp = builder.create<GenericOp>(
      consumer.getLoc(), TypeRange{}, fusedInputs, fusedOutputs,
      supersetGeneric.getGrid(), supersetGeneric.getBlockFactors(),
      builder.getAffineMapArrayAttr(fusedMaps),
      supersetGeneric.getIteratorTypes(), supersetGeneric.getThreads(),
      supersetGeneric.getScratchInputsAttr(), /*regions=*/1);

  // Set up IRMapping and block arguments.
  IRMapping irMap;
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
  // Shared intermediate — use the consumer's block arg type for this slot.
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

  // Map producer block args to fused block args.
  for (OpOperand *pi : producer.getDpsInputOperands()) {
    unsigned fusedIdx =
        sourceToFusedIdx[{producer.getOperation(), pi->getOperandNumber()}];
    irMap.map(prodBlock.getArgument(pi->getOperandNumber()),
              fusedBlock.getArgument(fusedIdx));
  }
  // Map producer's init block arg to the fused shared intermediate block arg.
  {
    unsigned prodInitIdx = producer.getDpsInitOperand(0)->getOperandNumber();
    irMap.map(prodBlock.getArgument(prodInitIdx),
              fusedBlock.getArgument(fusedSharedInputIdx));
  }

  // Map consumer block args to fused block args.
  // The shared input block arg maps to the fused shared intermediate.
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

  // Clone superset body into fused block.
  Block &subsetBlock = (subsetGeneric == producer) ? prodBlock : consBlock;
  Block &supersetBlock = (supersetGeneric == producer) ? prodBlock : consBlock;

  builder.setInsertionPointToEnd(&fusedBlock);
  for (Operation &op : supersetBlock.without_terminator()) {
    Operation *cloned = builder.clone(op, irMap);
    for (auto [oldRes, newRes] :
         llvm::zip(op.getResults(), cloned->getResults())) {
      irMap.map(oldRes, newRes);
    }
  }

  // Find the superset's outermost loop in the fused block.
  affine::AffineForOp dstLoop = nullptr;
  for (Operation &op : fusedBlock) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
      if (forOp->hasAttr("d2m.blocking_loop")) {
        dstLoop = forOp;
        break;
      }
    }
  }

  if (!dstLoop) {
    fusedOp->erase();
    return nullptr;
  }

  // Collect loop chains.
  SmallVector<affine::AffineForOp> dstLoopChain = collectLoopChain(dstLoop);

  affine::AffineForOp srcOuter = nullptr;
  for (Operation &op : subsetBlock) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
      if (forOp->hasAttr("d2m.blocking_loop")) {
        srcOuter = forOp;
        break;
      }
    }
  }
  SmallVector<affine::AffineForOp> srcLoopChain = collectLoopChain(srcOuter);

  if (srcLoopChain.empty() || dstLoopChain.size() < srcLoopChain.size()) {
    fusedOp->erase();
    return nullptr;
  }

  // Map subset induction variables to dst induction variables.
  for (unsigned d = 0; d < srcLoopChain.size(); ++d) {
    irMap.map(srcLoopChain[d].getInductionVar(),
              dstLoopChain[d].getInductionVar());
  }

  // Clone subset's non-loop ops at the top level before the dst loop.
  builder.setInsertionPoint(dstLoop);
  for (Operation &op : subsetBlock.without_terminator()) {
    if (isa<affine::AffineForOp>(&op)) {
      continue;
    }
    Operation *cloned = builder.clone(op, irMap);
    for (auto [oldRes, newRes] :
         llvm::zip(op.getResults(), cloned->getResults())) {
      irMap.map(oldRes, newRes);
    }
  }

  // Clone subset's non-loop ops at each loop level into the corresponding dst
  // loop body. For non-innermost levels, insert before the next nested loop.
  // For the innermost level, insert before the terminator.
  for (unsigned d = 0; d < srcLoopChain.size(); ++d) {
    Block *srcBody = srcLoopChain[d].getBody();
    Block *dstBody = dstLoopChain[d].getBody();
    bool isInnermost = (d == srcLoopChain.size() - 1);

    // Find insertion point: before the next nested blocking loop, or before
    // the terminator for the innermost level.
    if (isInnermost) {
      builder.setInsertionPoint(dstBody->getTerminator());
    } else {
      builder.setInsertionPoint(dstLoopChain[d + 1]);
    }

    for (Operation &op : srcBody->without_terminator()) {
      if (isa<affine::AffineForOp>(&op)) {
        continue;
      }
      Operation *cloned = builder.clone(op, irMap);
      for (auto [oldRes, newRes] :
           llvm::zip(op.getResults(), cloned->getResults())) {
        irMap.map(oldRes, newRes);
      }
    }
  }

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
        if (consumer.getNumRegions() != 1) {
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
          if (producer.getNumRegions() != 1) {
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
