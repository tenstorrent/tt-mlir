// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRELEMENTWISEFUSION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttir

using namespace mlir;
using namespace mlir::tt::ttir;

namespace {

static bool hasMultiUseOperand(GenericOp gOp) {
  // TODO(mbagherbeikTT) create issue for this
  // make sure all producer inputs are also single use
  // otherwise we can end up with intermediate inputs
  // that the compiler can't handle for now
  for (OpOperand *input : gOp.getDpsInputOperands()) {
    if (llvm::range_size(input->get().getUsers()) > 1) {
      return true;
    }
  }
  return false;
}

/// Check if op should be skipped for elementwise fusion.
///
/// If op or any of the operations nested within it's regions have
/// the TTIRSkipOpEltWiseFusionTrait, the op should be skipped.
static bool shouldBeSkipped(Operation *op) {
  if(op->hasTrait<TTIRSkipOpEltWiseFusionTrait>()) {
    return true;
  }

  for (Region &region : op->getRegions()) {
    for (Operation &opInner : region.getOps()) {
      if (shouldBeSkipped(&opInner)) {
        return true;
      }
    }
  }

  return false;
}

static bool hasCompatibleBlocking(GenericOp a, GenericOp b) {
  return a.getGrid() == b.getGrid() &&
         a.getBlockFactors() == b.getBlockFactors();
}

static bool isAllParallel(ArrayAttr iteratorTypes) {
  for (Attribute it : iteratorTypes) {
    auto itAttr = cast<mlir::tt::ttcore::IteratorTypeAttr>(it);
    if (itAttr.getValue() != mlir::tt::ttcore::IteratorType::Parallel) {
      return false;
    }
  }
  return true;
}

static bool coversAllDimsAfterFusion(GenericOp producer, GenericOp consumer,
                                     OpOperand *fusedOperand) {
  // If consumer has no reductions, trivially ok as in linalg.
  SmallVector<Attribute> iters = llvm::to_vector(consumer.getIteratorTypes());
  bool hasReduction = llvm::any_of(iters, [](Attribute it) {
    auto itAttr = cast<mlir::tt::ttcore::IteratorTypeAttr>(it);
    return itAttr.getValue() == mlir::tt::ttcore::IteratorType::Reduction;
  });
  if (!hasReduction) {
    return true;
  }

  // Mark dims covered by existing (non-fused) consumer inputs.
  BitVector covered(consumer.getNumDims(), false);
  for (auto [val, mapAttr] :
       llvm::zip(consumer.getInputs(), consumer.getIndexingMaps())) {
    if (val == fusedOperand->get()) {
      continue;
    }
    AffineMap m = cast<AffineMapAttr>(mapAttr).getValue();
    for (AffineExpr e : m.getResults()) {
      if (auto d = dyn_cast<AffineDimExpr>(e)) {
        covered.set(d.getPosition());
      }
    }
  }

  // Add coverage from producer inputs remapped into consumer loops.
  AffineMap consIdx = consumer.getIndexingMap(fusedOperand->getOperandNumber());
  // Use first init operand's indexing map as result map.
  AffineMap prodResMap = producer.getIndexingMap(
      producer.getDpsInitOperand(0)->getOperandNumber());
  AffineMap inv = inversePermutation(prodResMap);
  if (!inv) {
    return false;
  }
  AffineMap consToProd = inv.compose(consIdx);
  for (OpOperand *oprd : producer.getDpsInputOperands()) {
    AffineMap argMap = producer.getIndexingMap(oprd->getOperandNumber());
    AffineMap fused = argMap.compose(consToProd);
    for (AffineExpr e : fused.getResults()) {
      if (auto d = dyn_cast<AffineDimExpr>(e)) {
        covered.set(d.getPosition());
      }
    }
  }
  return covered.all();
}

static bool isElementwiseFusable(GenericOp producer, GenericOp consumer,
                                 OpOperand *use) {  
  // operand only has 1 user
  if (llvm::range_size(use->get().getUsers()) > 1) {
    return false;
  }

  // TODO(mbagherbeikTT) create issue for this
  // make sure all producer inputs are also single use
  // otherwise we can end up with intermediate inputs
  // that the compiler can't handle for now
  if (hasMultiUseOperand(producer)) {
    return false;
  }

  // NOTE: this only works under assumption that the operand that we're
  // fusing over only has a single user!
  //
  // fused op can't have more than set amount of operands
  // due to number of CBs that can fit in L1
  //
  // TODO(mbagherbeikTT) figure out where to move this constant
  unsigned int kFusedOpOperandLimit = 32; 
  unsigned int consumerOperands = consumer->getNumOperands();
  unsigned int producerOperands = consumer->getNumOperands();

  // -2 since were removing 1 operand from both consumer and producer
  if ((consumerOperands + producerOperands - 2) > kFusedOpOperandLimit) {
    return false;
  }

  if (!hasCompatibleBlocking(producer, consumer)) {
    return false;
  }

  if (!isAllParallel(producer.getIteratorTypes())) {
    return false;
  }

  // Rank/perm checks
  AffineMap consMap = consumer.getIndexingMap(use->getOperandNumber());
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

  if (!coversAllDimsAfterFusion(producer, consumer, use)) {
    return false;
  }

  return true;
}

// Compute producer operand map in fused coordinates: argMap ∘ inv(resMap) ∘
// consMap
static AffineMap computeFusedArgMap(GenericOp producer, OpOperand *prodOpnd,
                                    AffineMap prodResMap,
                                    AffineMap consMapForFused) {
  AffineMap inv = inversePermutation(prodResMap);
  AffineMap arg = producer.getIndexingMap(prodOpnd->getOperandNumber());
  return arg.compose(inv).compose(consMapForFused);
}

// Identify which producer results must be preserved in fused op
static llvm::SmallDenseSet<int> getPreservedProducerResults(GenericOp producer,
                                                            GenericOp consumer,
                                                            OpOperand *fused) {
  llvm::SmallDenseSet<int> keep;
  // Only single output currently supported by TTIR GenericOp; be robust if
  // extended.
  for (auto it : llvm::enumerate(producer->getResults())) {
    int idx = static_cast<int>(it.index());
    // Preserve if used by something other than the consumer
    bool hasOtherUse = llvm::any_of(it.value().getUsers(), [&](Operation *u) {
      return u != consumer.getOperation();
    });
    if (hasOtherUse) {
      keep.insert(idx);
      continue;
    }
    // In doubt about dropping, conservatively keep for now.
    // TODO: Refine like linalg's invertibility concat check.
  }
  return keep;
}

struct FuseTTIRElementwiseOpsPattern : OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    if (shouldBeSkipped(consumer)) {
      return failure();
    }

    if (!consumer.isComputeOnlyForm()) {
      return failure();
    }

    // Only tensor semantics.
    if (!consumer.hasPureTensorSemantics()) {
      return failure();
    }

    if (hasMultiUseOperand(consumer)) {
      return failure();
    }


    for (OpOperand *use : consumer.getDpsInputOperands()) {
      auto producer = dyn_cast_or_null<GenericOp>(use->get().getDefiningOp());
      if (!producer) {
        continue;
      }

      if (shouldBeSkipped(producer)) {
        continue;
      }

      assert(producer.isComputeOnlyForm() && producer.hasPureTensorSemantics());
      if (!isElementwiseFusable(producer, consumer, use)) {
        continue;
      }

      // Build fused op operands, maps, results
      SmallVector<Value> fusedInputs;
      SmallVector<Value> fusedOutputs;
      SmallVector<Type> fusedResultTypes;
      SmallVector<AffineMap> fusedMaps;

      AffineMap prodResMap = producer.getIndexingMap(
          producer.getDpsInitOperand(0)->getOperandNumber());
      AffineMap consMap = consumer.getIndexingMap(use->getOperandNumber());

      // consumer inputs before fused
      auto inputs = consumer.getInputs();
      size_t fusedIdx = use->getOperandNumber();
      for (size_t i = 0; i < fusedIdx; ++i) {
        fusedInputs.push_back(inputs[i]);
        fusedMaps.push_back(consumer.getIndexingMap(i));
      }
      // producer inputs (remapped)
      for (OpOperand *pi : producer.getDpsInputOperands()) {
        fusedInputs.push_back(pi->get());
        fusedMaps.push_back(
            computeFusedArgMap(producer, pi, prodResMap, consMap));
      }
      // remaining consumer inputs after fused
      for (size_t i = fusedIdx + 1; i < inputs.size(); ++i) {
        fusedInputs.push_back(inputs[i]);
        fusedMaps.push_back(consumer.getIndexingMap(i));
      }

      // outputs: preserved producer outputs then consumer outputs
      auto keep = getPreservedProducerResults(producer, consumer, use);
      for (int i = 0, e = static_cast<int>(producer.getOutputs().size()); i < e;
           ++i) {
        if (!keep.count(i)) {
          continue;
        }
        Value outVal = producer.getOutputs()[i];
        fusedOutputs.push_back(outVal);
        // map for this output in fused coords equals producer out map remapped
        AffineMap m = computeFusedArgMap(
            producer, producer.getDpsInitOperand(i), prodResMap, consMap);
        fusedMaps.push_back(m);
        fusedResultTypes.push_back(outVal.getType());
      }
      for (OpOperand &co : consumer.getDpsInitsMutable()) {
        fusedOutputs.push_back(co.get());
        fusedMaps.push_back(consumer.getIndexingMap(co.getOperandNumber()));
        if (!isa<MemRefType>(co.get().getType())) {
          fusedResultTypes.push_back(co.get().getType());
        }
      }

      // Create fused op
      auto fused = rewriter.create<GenericOp>(
          consumer.getLoc(), fusedResultTypes, fusedInputs, fusedOutputs,
          consumer.getGrid(), consumer.getBlockFactors(),
          rewriter.getAffineMapArrayAttr(fusedMaps),
          consumer.getIteratorTypes(), consumer.getThreads(), /*regions=*/1);

      // Build fused region: clone producer then consumer, map fused operand to
      // producer yield value. Ensure fused block argument types match original
      // region operand argument types (shard types), not the top-level types.
      IRMapping map;
      Block &fusedBlock = fused.getRegion(0).emplaceBlock();
      Block &pb = producer.getRegion(0).front();
      Block &cb = consumer.getRegion(0).front();

      // For each fused operand, pick the corresponding region block-argument
      // type from the originating op (producer/consumer) to satisfy verifier.
      SmallVector<Type> fusedBlockArgTypes;
      SmallVector<Location> fusedBlockArgLocs;
      fusedBlockArgTypes.reserve(fused->getNumOperands());
      fusedBlockArgLocs.reserve(fused->getNumOperands());

      // Helper to append a source (op, operandNumber)
      SmallVector<std::pair<Operation *, unsigned>> argSources;
      auto appendSource = [&](Operation *op, unsigned operandNumber) {
        argSources.emplace_back(op, operandNumber);
        Block *srcBlock = op == producer.getOperation() ? &pb : &cb;
        BlockArgument srcArg = srcBlock->getArgument(operandNumber);
        fusedBlockArgTypes.push_back(srcArg.getType());
        fusedBlockArgLocs.push_back(srcArg.getLoc());
      };

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
      // preserved producer outputs
      for (int i = 0, e = static_cast<int>(producer.getOutputs().size()); i < e;
           ++i) {
        if (keep.count(i)) {
          appendSource(producer.getOperation(),
                       producer.getDpsInitOperand(i)->getOperandNumber());
        }
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
      auto mapRegionArgs = [&](Operation *op, Block &orig) {
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          unsigned fusedIndex = sourceToFusedIdx[{op, i}];
          map.map(orig.getArgument(i), fusedBlock.getArgument(fusedIndex));
        }
      };
      mapRegionArgs(producer.getOperation(), pb);
      mapRegionArgs(consumer.getOperation(), cb);

      // Insert all cloned ops into the fused block, in order: producer then
      // consumer.
      rewriter.setInsertionPointToEnd(&fusedBlock);

      // Clone producer body (skip explicit ttir.yield if present).
      for (Operation &op : pb) {
        if (isa<YieldOp>(op)) {
          continue;
        }
        rewriter.clone(op, map);
      }
      YieldOp prodYield = nullptr;
      for (Operation &op : pb) {
        if (auto y = dyn_cast<YieldOp>(op)) {
          prodYield = y;
        }
      }
      int prodResultNumber = cast<OpResult>(use->get()).getResultNumber();
      Value repl = map.lookupOrDefault(prodYield.getOperand(prodResultNumber));

      // Map consumer arg corresponding to fused operand number to repl.
      // If `repl` is a tensor, and consumer block arg is tensor, direct map.
      // If consumer expects a tensor but repl is produced by inner ops,
      // ensure we cloned those inner ops already (we did by cloning pb body),
      // so `repl` dominates this point.
      map.map(cb.getArgument(fusedIdx), repl);

      // Clone remaining consumer body ops (except terminator). Treat nested
      // operations opaquely; cloning preserves dominance as long as operands
      // are mapped.
      for (Operation &op : cb.without_terminator()) {
        rewriter.clone(op, map);
      }

      // Build fused yield: kept producer yields then consumer yields
      SmallVector<Value> fusedYields;
      for (auto it : llvm::enumerate(prodYield.getOperands())) {
        if (keep.count(static_cast<int>(it.index()))) {
          fusedYields.push_back(map.lookupOrDefault(it.value()));
        }
      }
      YieldOp consYield = nullptr;
      for (Operation &op : cb) {
        if (auto y = dyn_cast<YieldOp>(op)) {
          consYield = y;
        }
      }
      for (Value y : consYield.getOperands()) {
        fusedYields.push_back(map.lookupOrDefault(y));
      }
      rewriter.setInsertionPointToEnd(&fusedBlock);
      rewriter.create<YieldOp>(fused.getLoc(), fusedYields);

      // Replace uses: from producer and consumer results to fused results.
      int resIdx = 0;
      // Map preserved producer results first
      for (auto it : llvm::enumerate(producer->getResults())) {
        if (!keep.count(static_cast<int>(it.index()))) {
          continue;
        }
        it.value().replaceUsesWithIf(
            fused->getResult(resIdx++), [&](OpOperand &u) {
              return u.getOwner() != consumer.getOperation();
            });
      }
      // Consumer results
      for (auto r : consumer->getResults()) {
        r.replaceAllUsesWith(fused->getResult(resIdx++));
      }

      llvm::errs() << "Consumer to be deleted: " << consumer << "\n";
      llvm::errs() << "Producer to be deleted: " << producer << "\n\n";

      llvm::errs() << "Fused op: " << fused << "\n";

      llvm::errs() << "Consumer users = " << llvm::range_size(consumer->getUsers()) << "\n";
      llvm::errs() << "Producer users = " << llvm::range_size(producer->getUsers()) << "\n";

      rewriter.eraseOp(consumer);
      rewriter.eraseOp(producer);
      return success();
    }

    return failure();
  }
};

struct TTIRElementwiseFusion
    : public tt::ttir::impl::TTIRElementwiseFusionBase<TTIRElementwiseFusion> {
  using TTIRElementwiseFusionBase::TTIRElementwiseFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseTTIRElementwiseOpsPattern>(ctx);
    GreedyRewriteConfig cfg;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), cfg);
  }
};

} // namespace

// Factory is generated by TableGen.
