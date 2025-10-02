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

static bool isElementwiseFusable(OpOperand *fusionTargetOperand,
                                 unsigned dstRegisterSizeTiles,
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

  if (checkConsumer && !consumer.isValidElementwiseFusionTarget()) {
    return false;
  }

  if (checkProducer && !producer.isValidElementwiseFusionTarget()) {
    return false;
  }

  if (!producer.hasCompatibleBlocking(consumer)) {
    return false;
  }

  if (!fitsInDstPostFusion(producer, consumer, fusionTargetOperand,
                           dstRegisterSizeTiles)) {
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

// Compute producer operand map in fused coordinates: argMap ∘ inv(resMap) ∘
// consMap
static AffineMap computeFusedArgMap(GenericOp producer, OpOperand *prodOpnd,
                                    AffineMap prodResMap,
                                    AffineMap consMapForFused) {
  AffineMap inv = inversePermutation(prodResMap);
  AffineMap arg = producer.getIndexingMap(prodOpnd->getOperandNumber());
  return arg.compose(inv).compose(consMapForFused);
}

static void gatherAndComposeFusedOpIO(OpOperand *fusedOperand,
                                      GenericOp producer, GenericOp consumer,
                                      SmallVector<Value> &fusedInputs,
                                      SmallVector<Value> &fusedOutputs,
                                      SmallVector<Type> &fusedResultTypes,
                                      SmallVector<AffineMap> &fusedMaps) {
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
}

static void mapFusedOpIo(GenericOp fusedOp, OpOperand *fusedOperand,
                         GenericOp producer, GenericOp consumer,
                         IRMapping &irMap, Block &fusedBlock, Block &pb,
                         Block &cb) {
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
}

/// Build fused region: clone producer then consumer, map fused operand to
/// producer yield value. Ensure fused block argument types match original
/// region operand argument types (shard types), not the top-level types.
static void populateFusedOpBody(GenericOp fused, OpOperand *fusedOperand,
                                GenericOp producer, GenericOp consumer,
                                IRMapping &irMap, Block &fusedBlock, Block &pb,
                                Block &cb, PatternRewriter &rewriter) {
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
  int prodResultNumber = cast<OpResult>(fusedOperand->get()).getResultNumber();
  Value repl = irMap.lookupOrDefault(prodYield.getOperand(prodResultNumber));

  // Map consumer arg corresponding to fused operand number to repl.
  // If `repl` is a tensor, and consumer block arg is tensor, direct map.
  // If consumer expects a tensor but repl is produced by inner ops,
  // ensure we cloned those inner ops already (we did by cloning pb body),
  // so `repl` dominates this point.
  irMap.map(cb.getArgument(fusedOperand->getOperandNumber()), repl);

  // Clone remaining consumer body ops (except terminator). Treat nested
  // operations opaquely; cloning preserves dominance as long as operands
  // are mapped.
  for (Operation &op : cb.without_terminator()) {
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
  rewriter.create<YieldOp>(fused.getLoc(), fusedYields);

  // Replace uses: from producer and consumer results to fused results.
  int resIdx = 0;
  // Consumer results
  for (auto r : consumer->getResults()) {
    r.replaceAllUsesWith(fused->getResult(resIdx++));
  }

  rewriter.eraseOp(consumer);
  rewriter.eraseOp(producer);
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

      gatherAndComposeFusedOpIO(use, producer, consumer, fusedInputs,
                                fusedOutputs, fusedResultTypes, fusedMaps);

      // Create fused op
      auto fused = rewriter.create<GenericOp>(
          consumer.getLoc(), fusedResultTypes, fusedInputs, fusedOutputs,
          consumer.getGrid(), consumer.getBlockFactors(),
          rewriter.getAffineMapArrayAttr(fusedMaps),
          consumer.getIteratorTypes(), consumer.getThreads(), /*regions=*/1);

      IRMapping irMap;
      Block &fusedBlock = fused.getRegion(0).emplaceBlock();
      Block &pb = producer.getRegion(0).front();
      Block &cb = consumer.getRegion(0).front();

      mapFusedOpIo(fused, use, producer, consumer, irMap, fusedBlock, pb, cb);

      populateFusedOpBody(fused, use, producer, consumer, irMap, fusedBlock, pb,
                          cb, rewriter);

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
