// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREAD
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MSplitUnifiedThreadRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  // Check if an scf.execute_region contains any operation with
  // D2MGenericRegionDatamovementOpTrait
  static bool containsDatamovementOps(scf::ExecuteRegionOp executeRegion) {
    bool foundDatamovement = false;
    executeRegion.walk([&](Operation *op) {
      if (op->hasTrait<D2MGenericRegionDatamovementOpTrait>()) {
        foundDatamovement = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return foundDatamovement;
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Only match GenericOp with a single region (unified thread form)
    if (generic.getNumRegions() != 1) {
      return failure();
    }

    Block *originalBlock = &generic.getRegion(0).front();

    // First, find all scf.execute_region ops that contain datamovement
    // operations by walking the entire block (including nested operations)
    SmallVector<scf::ExecuteRegionOp> datamovementExecuteRegions;
    originalBlock->walk([&](scf::ExecuteRegionOp executeRegion) {
      if (containsDatamovementOps(executeRegion)) {
        datamovementExecuteRegions.push_back(executeRegion);
      }
    });

    // If no datamovement execute_region ops found, nothing to split
    if (datamovementExecuteRegions.empty()) {
      return failure();
    }

    // Create a set for quick lookup of ops to exclude from compute region
    DenseSet<Operation *> datamovementOpSet;
    for (scf::ExecuteRegionOp execRegion : datamovementExecuteRegions) {
      datamovementOpSet.insert(execRegion);
    }

    // Create new GenericOp with two regions: datamovement and compute
    SmallVector<Attribute> threads;
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));

    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(), generic.getBlockFactors(),
        generic.getIndexingMaps(), generic.getIteratorTypes(),
        rewriter.getArrayAttr(threads), /*numRegions=*/2);

    // Create datamovement region block
    Block *dmaBlock = &newGeneric.getRegion(0).emplaceBlock();
    dmaBlock->addArguments(
        originalBlock->getArgumentTypes(),
        SmallVector<mlir::Location>(originalBlock->getArgumentTypes().size(),
                                    generic.getLoc()));

    // Create compute region block
    Block *computeBlock = &newGeneric.getRegion(1).emplaceBlock();
    computeBlock->addArguments(
        originalBlock->getArgumentTypes(),
        SmallVector<mlir::Location>(originalBlock->getArgumentTypes().size(),
                                    generic.getLoc()));

    // Map original block arguments to both new blocks
    IRMapping dmaMapping, computeMapping;
    for (unsigned i = 0; i < originalBlock->getNumArguments(); ++i) {
      dmaMapping.map(originalBlock->getArgument(i), dmaBlock->getArgument(i));
      computeMapping.map(originalBlock->getArgument(i),
                         computeBlock->getArgument(i));
    }

    // Clone datamovement execute_region ops into datamovement region
    rewriter.setInsertionPointToStart(dmaBlock);
    for (scf::ExecuteRegionOp execRegion : datamovementExecuteRegions) {
      rewriter.clone(*execRegion, dmaMapping);
    }

    // Clone compute operations into compute region, excluding datamovement
    // execute_region ops. We need to manually rebuild scf.for loops without
    // the execute_region ops.
    rewriter.setInsertionPointToStart(computeBlock);
    for (Operation &op : originalBlock->getOperations()) {
      // Skip terminator - will be added at the end
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      // Skip datamovement execute_region ops - they're in datamovement region
      if (datamovementOpSet.contains(&op)) {
        continue;
      }

      // Clone the operation
      Operation *cloned = rewriter.clone(op, computeMapping);

      // Walk the cloned operation and erase any execute_region ops that
      // contain datamovement operations
      SmallVector<scf::ExecuteRegionOp> nestedToErase;
      cloned->walk([&](scf::ExecuteRegionOp nestedExecRegion) {
        if (containsDatamovementOps(nestedExecRegion)) {
          nestedToErase.push_back(nestedExecRegion);
        }
      });
      for (scf::ExecuteRegionOp execRegionToErase : nestedToErase) {
        rewriter.eraseOp(execRegionToErase);
      }
    }

    // GenericOp has NoTerminator trait, so we don't need to clone a terminator

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MSplitUnifiedThread
    : public impl::D2MSplitUnifiedThreadBase<D2MSplitUnifiedThread> {
public:
  using impl::D2MSplitUnifiedThreadBase<
      D2MSplitUnifiedThread>::D2MSplitUnifiedThreadBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MSplitUnifiedThreadRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
