// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    // Skip stream_layout ops — check their users instead.
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
    // Skip the producer itself (which writes to this memref).
    if (auto otherGeneric = dyn_cast<GenericOp>(user)) {
      // Check if this generic reads the memref (as an input, not output).
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
// Returns true if they compute the same index using the same operations.
static bool areIndicesStructurallyIdentical(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef) {
    // Handle affine.for induction variables — compare by loop nesting depth.
    int lhsDepth = getAffineForDepth(lhs);
    int rhsDepth = getAffineForDepth(rhs);
    if (lhsDepth >= 0 && lhsDepth == rhsDepth) {
      return true;
    }
    return false;
  }

  // Must be the same operation type.
  if (lhsDef->getName() != rhsDef->getName()) {
    return false;
  }

  // For index ops (core_index, get_block_factor, iter_index), compare dim attr.
  if (isa<CoreIndexOp, GetBlockFactorOp, IterIndexOp, BlockIndexOp>(lhsDef)) {
    if (auto lhsIdx = lhsDef->getAttrOfType<IntegerAttr>("dim")) {
      auto rhsIdx = rhsDef->getAttrOfType<IntegerAttr>("dim");
      return rhsIdx && lhsIdx.getInt() == rhsIdx.getInt();
    }
    return false;
  }

  // For arith ops, compare operands recursively.
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

// Check that the producer's remote_store indices for the shared operand match
// the consumer's remote_load indices structurally. Returns the store op if
// valid, nullptr otherwise.
static RemoteStoreOp getMatchingStoreForShared(GenericOp producer,
                                               GenericOp consumer,
                                               Value sharedMemref) {
  // Find all remote_store ops in producer that target the shared memref.
  SmallVector<RemoteStoreOp> producerStores;
  producer.getRegion(0).walk([&](RemoteStoreOp storeOp) {
    if (storeOp.getMemref() == sharedMemref) {
      producerStores.push_back(storeOp);
    }
  });

  // Criterion 4: at most one store to the shared memref.
  if (producerStores.size() != 1) {
    return nullptr;
  }

  RemoteStoreOp storeOp = producerStores.front();

  // Find consumer's remote_load from the shared memref (or stream_layout
  // wrapping it).
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

  // Criterion 3: structurally identical indices.
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
  // Criterion 5: exactly one outermost loop.
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
    // Find nested affine.for in body.
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

namespace {
struct FuseD2MGenericAffineLoopsPattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp consumer,
                                PatternRewriter &rewriter) const final {
    // Skip already-fused generics.
    if (consumer->hasAttr("d2m.affine_fused")) {
      return failure();
    }

    // Must have exactly one region.
    if (consumer.getNumRegions() != 1) {
      return failure();
    }

    // Criterion 5: consumer must have exactly one outermost loop.
    affine::AffineForOp consumerLoop = getOutermostLoop(consumer);
    if (!consumerLoop) {
      return failure();
    }

    // Try each consumer input operand to find a fusible producer.
    for (OpOperand *inputOperand : consumer.getDpsInputOperands()) {
      Value rawMemref = peelStreamLayout(inputOperand->get());

      // Find a producer generic that writes to this memref.
      GenericOp producer = findProducerGeneric(rawMemref, consumer);
      if (!producer) {
        continue;
      }

      // Skip already-fused producers.
      if (producer->hasAttr("d2m.affine_fused")) {
        continue;
      }

      // Producer must have exactly one region.
      if (producer.getNumRegions() != 1) {
        continue;
      }

      // Criterion 2: consumer is sole reader.
      if (!isSoleReader(rawMemref, consumer)) {
        continue;
      }

      // Criterion 3 & 4: matching indices and single store.
      RemoteStoreOp storeOp =
          getMatchingStoreForShared(producer, consumer, rawMemref);
      if (!storeOp) {
        continue;
      }

      // Criterion 5: producer must have exactly one outermost loop.
      affine::AffineForOp producerLoop = getOutermostLoop(producer);
      if (!producerLoop) {
        continue;
      }

      // Determine superset/subset: the one with more outer loops is superset.
      unsigned producerDepth = getOuterLoopDepth(producerLoop);
      unsigned consumerDepth = getOuterLoopDepth(consumerLoop);

      GenericOp supersetGeneric, subsetGeneric;

      if (producerDepth > consumerDepth) {
        supersetGeneric = producer;
        subsetGeneric = consumer;
      } else {
        // Equal or consumer has more: consumer is superset.
        supersetGeneric = consumer;
        subsetGeneric = producer;
      }

      // Build fused operands: producer inputs + consumer inputs (minus shared),
      // consumer outputs.
      SmallVector<Value> fusedInputs;
      SmallVector<Value> fusedOutputs;
      SmallVector<AffineMap> fusedMaps;

      // Compose indexing maps using the superset's iteration space.
      AffineMap supersetSharedMap =
          supersetGeneric.getIndexingMapForOperand(rawMemref);
      AffineMap subsetOutputMap = subsetGeneric.getIndexingMap(
          subsetGeneric.getDpsInitOperand(0)->getOperandNumber());
      AffineMap subsetOutputInv = inversePermutation(subsetOutputMap);

      // Producer inputs (remapped into superset iteration space).
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

      // Consumer inputs (skip the shared operand, remap if consumer is subset).
      AffineMap consumerOutputMap = consumer.getIndexingMap(
          consumer.getDpsInitOperand(0)->getOperandNumber());
      AffineMap consumerOutputInv = inversePermutation(consumerOutputMap);

      for (OpOperand *ci : consumer.getDpsInputOperands()) {
        if (ci == inputOperand) {
          continue; // Skip the shared intermediate.
        }
        fusedInputs.push_back(ci->get());
        AffineMap argMap = consumer.getIndexingMap(ci->getOperandNumber());
        if (consumer == subsetGeneric) {
          fusedMaps.push_back(
              argMap.compose(consumerOutputInv).compose(supersetSharedMap));
        } else {
          fusedMaps.push_back(argMap);
        }
      }

      // Consumer outputs.
      for (OpOperand &co : consumer.getDpsInitsMutable()) {
        fusedOutputs.push_back(co.get());
        AffineMap argMap = consumer.getIndexingMap(co.getOperandNumber());
        if (consumer == subsetGeneric) {
          fusedMaps.push_back(
              argMap.compose(consumerOutputInv).compose(supersetSharedMap));
        } else {
          fusedMaps.push_back(argMap);
        }
      }

      // Create fused GenericOp using superset's attributes.
      auto fusedOp = rewriter.create<GenericOp>(
          consumer.getLoc(), TypeRange(fusedOutputs), fusedInputs, fusedOutputs,
          supersetGeneric.getGrid(), supersetGeneric.getBlockFactors(),
          rewriter.getAffineMapArrayAttr(fusedMaps),
          supersetGeneric.getIteratorTypes(), supersetGeneric.getThreads(),
          supersetGeneric.getScratchInputsAttr(), /*regions=*/1);

      // Set up IRMapping and block arguments.
      IRMapping irMap;
      Block &fusedBlock = fusedOp.getRegion(0).emplaceBlock();
      Block &prodBlock = producer.getRegion(0).front();
      Block &consBlock = consumer.getRegion(0).front();

      // Build fused block arg types from the originating ops.
      SmallVector<Type> fusedBlockArgTypes;
      SmallVector<Location> fusedBlockArgLocs;

      // Track which source (op, operandNumber) maps to which fused arg index.
      SmallVector<std::pair<Operation *, unsigned>> argSources;
      auto appendSource = [&](Operation *op, unsigned operandNumber) {
        argSources.emplace_back(op, operandNumber);
        Block *srcBlock =
            (op == producer.getOperation()) ? &prodBlock : &consBlock;
        BlockArgument srcArg = srcBlock->getArgument(operandNumber);
        fusedBlockArgTypes.push_back(srcArg.getType());
        fusedBlockArgLocs.push_back(srcArg.getLoc());
      };

      // Producer inputs.
      for (OpOperand *pi : producer.getDpsInputOperands()) {
        appendSource(producer.getOperation(), pi->getOperandNumber());
      }

      // Consumer inputs (skip shared).
      for (OpOperand *ci : consumer.getDpsInputOperands()) {
        if (ci == inputOperand) {
          continue;
        }
        appendSource(consumer.getOperation(), ci->getOperandNumber());
      }

      // Consumer outputs.
      for (OpOperand &co : consumer.getDpsInitsMutable()) {
        appendSource(consumer.getOperation(), co.getOperandNumber());
      }

      fusedBlock.addArguments(fusedBlockArgTypes, fusedBlockArgLocs);

      // Build source-to-fused-index mapping.
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
      // Map producer's output block arg to consumer's output block arg in
      // fused.
      {
        unsigned prodInitIdx =
            producer.getDpsInitOperand(0)->getOperandNumber();
        unsigned consInitIdx =
            consumer.getDpsInitOperand(0)->getOperandNumber();
        unsigned fusedConsInitIdx =
            sourceToFusedIdx[{consumer.getOperation(), consInitIdx}];
        irMap.map(prodBlock.getArgument(prodInitIdx),
                  fusedBlock.getArgument(fusedConsInitIdx));
      }

      // Map consumer block args to fused block args.
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

      // Map the consumer's shared input block arg to the producer's output
      // block arg mapping (they share the same fused output arg).
      {
        unsigned consSharedIdx = inputOperand->getOperandNumber();
        unsigned prodInitIdx =
            producer.getDpsInitOperand(0)->getOperandNumber();
        // Map to the same fused arg as the producer's init.
        irMap.map(consBlock.getArgument(consSharedIdx),
                  irMap.lookup(prodBlock.getArgument(prodInitIdx)));
      }

      // Clone both bodies into the fused block, then manually merge the loop
      // nests. We cannot use canFuseLoops/fuseLoops because D2M remote_load/
      // remote_store implement AffineReadOpInterface/AffineWriteOpInterface
      // but reference memrefs outside the generic region, which crashes the
      // affine dependence analysis.

      // Clone superset body first (destination loop nest).
      rewriter.setInsertionPointToEnd(&fusedBlock);

      Block &subsetBlock = (subsetGeneric == producer) ? prodBlock : consBlock;
      Block &supersetBlock =
          (supersetGeneric == producer) ? prodBlock : consBlock;

      for (Operation &op : supersetBlock.without_terminator()) {
        rewriter.clone(op, irMap);
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
        rewriter.eraseOp(fusedOp);
        continue;
      }

      // Walk down the dst loop nest to find the innermost loop, and clone
      // the subset's innermost body into it. Also map subset induction vars
      // to the corresponding dst induction vars (by depth).

      // Collect the dst loop nest chain.
      SmallVector<affine::AffineForOp> dstLoopChain;
      {
        auto cur = dstLoop;
        while (cur) {
          dstLoopChain.push_back(cur);
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
      }

      // Collect subset's loop nest chain from the original region.
      SmallVector<affine::AffineForOp> srcLoopChain;
      {
        affine::AffineForOp srcOuter = nullptr;
        for (Operation &op : subsetBlock) {
          if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
            if (forOp->hasAttr("d2m.blocking_loop")) {
              srcOuter = forOp;
              break;
            }
          }
        }
        auto cur = srcOuter;
        while (cur) {
          srcLoopChain.push_back(cur);
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
      }

      if (srcLoopChain.empty() || dstLoopChain.size() < srcLoopChain.size()) {
        rewriter.eraseOp(fusedOp);
        continue;
      }

      // Map subset induction variables to dst induction variables at matching
      // depth.
      for (unsigned d = 0; d < srcLoopChain.size(); ++d) {
        irMap.map(srcLoopChain[d].getInductionVar(),
                  dstLoopChain[d].getInductionVar());
      }

      // Clone the subset's non-loop ops (everything outside the loop nest
      // but inside the region) into the fused block before the dst loop.
      rewriter.setInsertionPoint(dstLoop);
      for (Operation &op : subsetBlock.without_terminator()) {
        if (isa<affine::AffineForOp>(&op)) {
          continue; // Skip the loop nest itself.
        }
        rewriter.clone(op, irMap);
      }

      // Clone the subset innermost loop body ops into the dst innermost
      // loop body, before the yield.
      affine::AffineForOp srcInner = srcLoopChain.back();
      affine::AffineForOp dstInner = dstLoopChain[srcLoopChain.size() - 1];
      Block *dstInnerBody = dstInner.getBody();
      rewriter.setInsertionPoint(dstInnerBody->getTerminator());
      for (Operation &op : srcInner.getBody()->without_terminator()) {
        rewriter.clone(op, irMap);
      }

      // Mark as fused to prevent re-examination.
      fusedOp->setAttr("d2m.affine_fused", rewriter.getUnitAttr());

      // Erase originals. Erase producer first since it precedes consumer.
      rewriter.eraseOp(producer);
      rewriter.replaceOp(consumer, fusedOp->getResults());

      return success();
    }

    return failure();
  }
};
} // namespace

namespace {
class D2MGenericAffineLoopFusion
    : public impl::D2MGenericAffineLoopFusionBase<D2MGenericAffineLoopFusion> {
public:
  using D2MGenericAffineLoopFusionBase::D2MGenericAffineLoopFusionBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseD2MGenericAffineLoopsPattern>(&getContext());
    GreedyRewriteConfig cfg;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), cfg))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
