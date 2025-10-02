// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MELEMENTWISEFUSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

/// Returns true if, post-fusion, every dimension has at least one input
/// that defines it.
///
/// Ensure that the fusion does not remove size information required to
/// get the loop bounds. For non-reduction generics, this is trivially the
/// case due to the output operand. For reductions, we need to check that after
/// the fusion, each loop dimension has at least one input that defines it.
static bool coversAllDimsAfterFusion(GenericOp producer, GenericOp consumer,
                                     OpOperand *fusedOperand) {
  // TODO(mbagherbeikTT) Revisit when adressing issue
  // https://github.com/tenstorrent/tt-mlir/issues/5188 once we add reductions.
  // Currently, what's commented out is just a slightly reformatted version of
  // how the linalg fusion pass checks dimension coverage.
  //
  // already check if we're all-parallel coming into here
  assert(!consumer.hasReduction());
  return true;

  // // If consumer has no reductions, trivially ok as in linalg.
  // if (!consumer.hasReduction()) {
  //   return true;
  // }

  // // Mark dims covered by existing (non-fused) consumer inputs.
  // BitVector coveredDims(consumer.getNumDims(), false);
  // for (auto [val, mapAttr] :
  //      llvm::zip(consumer.getInputs(), consumer.getIndexingMaps())) {
  //   if (val == fusedOperand->get()) {
  //     continue;
  //   }
  //   AffineMap m = cast<AffineMapAttr>(mapAttr).getValue();
  //   for (AffineExpr e : m.getResults()) {
  //     if (auto d = dyn_cast<AffineDimExpr>(e)) {
  //       coveredDims.set(d.getPosition());
  //     }
  //   }
  // }

  // // Add coverage from producer inputs remapped into consumer loops.
  // AffineMap consIdx =
  // consumer.getIndexingMap(fusedOperand->getOperandNumber());
  // // Use first init operand's indexing map as result map.
  // AffineMap prodResMap = producer.getIndexingMap(
  //     producer.getDpsInitOperand(0)->getOperandNumber());
  // AffineMap inv = inversePermutation(prodResMap);

  // if (!inv) {
  //   return false;
  // }

  // AffineMap consToProd = inv.compose(consIdx);
  // for (OpOperand *oprd : producer.getDpsInputOperands()) {
  //   AffineMap argMap = producer.getIndexingMap(oprd->getOperandNumber());
  //   AffineMap fused = argMap.compose(consToProd);
  //   for (AffineExpr e : fused.getResults()) {
  //     if (auto d = dyn_cast<AffineDimExpr>(e)) {
  //       coveredDims.set(d.getPosition());
  //     }
  //   }
  // }
  // return coveredDims.all();
}

static int countBinaryOps(Operation *op) {
  int count = 0;

  for (Region &region : op->getRegions()) {
    for (Operation &opInner : region.getOps()) {
      if (opInner.hasTrait<D2MGenericRegionComputeOpTrait>()) {
        if (opInner.getNumOperands() > 1) {
          ++count;
        }
      }
      count += countBinaryOps(&opInner);
    }
  }
  return count;
};

static bool fitsInDstPostFusion(GenericOp producer, GenericOp consumer,
                                OpOperand *use, unsigned dstRegisterSizeTiles) {
  auto tensorType = dyn_cast<RankedTensorType>(use->get().getType());
  assert(tensorType); // use MUST be a tensorType unless something got borked

  auto shape = tensorType.getShape();
  int blockSize = shape.back();

  if (blockSize > static_cast<int>(dstRegisterSizeTiles)) {
    blockSize = dstRegisterSizeTiles;
  }

  int dstTilesRemaining = dstRegisterSizeTiles;

  // Account for number of tiles needed to store input operands after fusion
  dstTilesRemaining -= blockSize * (consumer->getNumOperands() +
                                    producer->getNumOperands() - 2 - 1);

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

static bool isElementwiseFusable(OpOperand *fusionOperand,
                                 unsigned dstRegisterSizeTiles,
                                 bool checkConsumer = true,
                                 bool checkProducer = true) {
  if (!fusionOperand) {
    return false;
  }

  auto producer = fusionOperand->get().getDefiningOp<GenericOp>();
  auto consumer = dyn_cast<GenericOp>(fusionOperand->getOwner());

  if (!producer || !consumer) {
    return false;
  }

  if (checkConsumer && !consumer.isValidElementwiseFusionTarget()) {
    return false;
  }

  if (checkProducer && !producer.isValidElementwiseFusionTarget()) {
    return false;
  }

  if (!producer.hasCompatibleBlocking(consumer)) {
    return false;
  }

  if (!fitsInDstPostFusion(producer, consumer, fusionOperand,
                           dstRegisterSizeTiles)) {
    return false;
  }

  // Rank/perm checks
  AffineMap consMap =
      consumer.getIndexingMap(fusionOperand->getOperandNumber());
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

  if (!coversAllDimsAfterFusion(producer, consumer, fusionOperand)) {
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
  // Only single output currently supported by D2M GenericOp; be robust if
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

namespace {
struct FuseD2MElementwiseOpsPattern : public OpRewritePattern<GenericOp> {
  FuseD2MElementwiseOpsPattern(MLIRContext *context,
                               unsigned dstRegisterSizeTiles)
      : OpRewritePattern<GenericOp>(context),
        dstRegisterSizeTiles(dstRegisterSizeTiles) {}

  LogicalResult matchAndRewrite(GenericOp consumer,
                                PatternRewriter &rewriter) const final {
    if (!consumer.isValidElementwiseFusionTarget()) {
      return failure();
    }

    for (OpOperand *use : consumer.getDpsInputOperands()) {
      if (!isElementwiseFusable(use, dstRegisterSizeTiles,
                                // already checked consumer outside
                                false, true)) {
        continue;
      }

      auto producer = use->get().getDefiningOp<GenericOp>();
      // we already check that producer is a valid GenericOp,
      // use assert instead of if()
      assert(producer);

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
      IRMapping irMap;
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
          irMap.map(orig.getArgument(i), fusedBlock.getArgument(fusedIndex));
        }
      };
      mapRegionArgs(producer.getOperation(), pb);
      mapRegionArgs(consumer.getOperation(), cb);

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
      int prodResultNumber = cast<OpResult>(use->get()).getResultNumber();
      Value repl =
          irMap.lookupOrDefault(prodYield.getOperand(prodResultNumber));

      // Map consumer arg corresponding to fused operand number to repl.
      // If `repl` is a tensor, and consumer block arg is tensor, direct map.
      // If consumer expects a tensor but repl is produced by inner ops,
      // ensure we cloned those inner ops already (we did by cloning pb body),
      // so `repl` dominates this point.
      irMap.map(cb.getArgument(fusedIdx), repl);

      // Clone remaining consumer body ops (except terminator). Treat nested
      // operations opaquely; cloning preserves dominance as long as operands
      // are mapped.
      for (Operation &op : cb.without_terminator()) {
        rewriter.clone(op, irMap);
      }

      // Build fused yield: kept producer yields then consumer yields
      SmallVector<Value> fusedYields;
      for (auto it : llvm::enumerate(prodYield.getOperands())) {
        if (keep.count(static_cast<int>(it.index()))) {
          fusedYields.push_back(irMap.lookupOrDefault(it.value()));
        }
      }
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

      rewriter.eraseOp(consumer);
      rewriter.eraseOp(producer);
      return success();
    }

    return failure();
  }

  unsigned dstRegisterSizeTiles;
};
} // namespace

namespace {
class D2MElementwiseFusion
    : public tt::d2m::impl::D2MElementwiseFusionBase<D2MElementwiseFusion> {
  using D2MElementwiseFusionBase::D2MElementwiseFusionBase;

  void runOnOperation() override {
    auto device = ttcore::lookupDevice(getOperation());
    assert(device && "Device not found");
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(getOperation());
    auto chipIds = device.getChipIds();
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    unsigned dstRegisterSizeTiles = chipDesc.getDstRegisterSizeTiles();
    if (maxDstRegisterSizeTiles.getValue() > 0) {
      dstRegisterSizeTiles =
          std::min(dstRegisterSizeTiles, maxDstRegisterSizeTiles.getValue());
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseD2MElementwiseOpsPattern>(ctx, dstRegisterSizeTiles);
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
