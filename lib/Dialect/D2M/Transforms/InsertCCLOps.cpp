// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MINSERTCCLOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MInsertCCLOpsRewriter final : public OpRewritePattern<d2m::GenericOp> {
public:
  D2MInsertCCLOpsRewriter(MLIRContext *context)
      : OpRewritePattern<d2m::GenericOp>(context) {}

  LogicalResult matchAndRewrite(d2m::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (!checkIfCCLOpInsertionIsNeeded(op)) {
      return failure();
    }

    auto numDataMovementRegions = op.getNumOperands();
    auto numTotalRegions = op.getNumRegions() + numDataMovementRegions;
    SmallVector<Attribute> threads(
        numDataMovementRegions,
        rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
    threads.append(op.getThreads().begin(), op.getThreads().end());
    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), op.getOutputs(),
        op.getGrid(), op.getBlockFactors(), op.getIndexingMaps(),
        op.getIteratorTypes(), rewriter.getArrayAttr(threads), numTotalRegions);

    for (unsigned regionIdx = 0; regionIdx < numTotalRegions; ++regionIdx) {
      Block &block = newGeneric.getRegion(regionIdx).emplaceBlock();
      rewriter.modifyOpInPlace(newGeneric, [&] {
        block.addArguments(
            op.getRegion(0).getArgumentTypes(),
            SmallVector<mlir::Location>(
                op.getRegion(0).getArgumentTypes().size(), op.getLoc()));
      });
    }

    for (OpOperand &operand : op->getOpOperands()) {
      Block *datamovementBlock =
          &newGeneric.getRegion(operand.getOperandNumber()).front();
      rewriter.setInsertionPointToEnd(datamovementBlock);
      rewriter.create<d2m::YieldOp>(
          op.getLoc(),
          datamovementBlock->getArgument(operand.getOperandNumber()));
    }

    unsigned computeRegionIndex = numDataMovementRegions;
    auto &newRegion = newGeneric.getRegion(computeRegionIndex);
    auto &oldRegion = op.getRegion(0);
    rewriter.mergeBlocks(
        &oldRegion.front(), &newRegion.front(),
        newRegion.front().getArguments().take_front(op.getOperands().size()));
    rewriter.replaceOp(op, newGeneric);
    return success();
  }

  bool checkIfCCLOpInsertionIsNeeded(d2m::GenericOp op) const {
    // skip if it is already handled by this pass
    if (op.getThreads().size() != 1) {
      return false;
    }

    Block *computeBlock = &op.getRegions().front().front();
    for (Operation &tile_op : computeBlock->getOperations()) {
      if (mlir::isa_and_nonnull<d2m::TileAllGatherOp>(tile_op)) {
        return true;
      }
    }
    return false;
  }
};
} // namespace

namespace {
class D2MInsertCCLOps final
    : public impl::D2MInsertCCLOpsBase<D2MInsertCCLOps> {
public:
  using impl::D2MInsertCCLOpsBase<D2MInsertCCLOps>::D2MInsertCCLOpsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MInsertCCLOpsRewriter>(&getContext());
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
