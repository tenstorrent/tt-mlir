// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDMAOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MInsertDMAOpsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static bool isStream(Value operand) {
    return mlir::isa_and_nonnull<StreamLayoutOp>(operand.getDefiningOp());
  }

  static bool compatibleDeviceGrid(ttcore::DeviceAttr device,
                                   ttcore::GridAttr grid) {
    return device.getWorkerGrid().getShape().size() <= grid.getShape().size();
  }

  static BlockArgument createSemaphore(PatternRewriter &builder, Location loc,
                                       Block *block) {
    return block->addArgument(builder.getType<SemaphoreType>(), loc);
  }

  static SmallVector<ttcore::IteratorType>
  calculateMcastIterators(ttcore::GridAttr grid, ttcore::DeviceAttr device,
                          AffineMap operandIndexingMap,
                          ArrayAttr iteratorTypes) {
    assert(grid.getShape().size() == operandIndexingMap.getNumResults());
    assert(compatibleDeviceGrid(device, grid));

    bool allParallel = true;
    SmallVector<ttcore::IteratorType> mcastIterators;
    mcastIterators.reserve(grid.getShape().size());
    for (unsigned dim = 0; dim < grid.getShape().size(); dim++) {
      AffineExpr result = operandIndexingMap.getResult(dim);
      assert(mlir::isa<AffineDimExpr>(result) ||
             mlir::isa<AffineConstantExpr>(result));

      ttcore::IteratorType iteratorType;
      if (AffineConstantExpr constant =
              mlir::dyn_cast<AffineConstantExpr>(result)) {
        assert(constant.getValue() == 0);

        iteratorType = ttcore::IteratorType::Parallel;
      } else {
        auto dimId = mlir::cast<AffineDimExpr>(result);
        unsigned dimPosition = dimId.getPosition();

        iteratorType =
            mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[dimPosition])
                .getValue();
      }
      mcastIterators.push_back(iteratorType);

      // If the grid dimension is 1, we can special case it and always safely
      // fallback to mode parallel.  Reduction implies multicast, and while
      // it'll be functionally correct, a multicast with a single core to
      // itself is a redundant copy and more complicated than necessary.
      bool singleCore = grid.getShape()[dim] == 1;
      allParallel &=
          (iteratorType == ttcore::IteratorType::Parallel) || singleCore;
    }

    return allParallel ? SmallVector<ttcore::IteratorType>() : mcastIterators;
  }

  static Value createDMA(OpBuilder &builder, Location loc, Value src, Value dst,
                         std::optional<AffineMap> operandIndexingMap,
                         bool isOutput, SmallVector<Value> coreIndex = {},
                         SmallVector<Value> mcastShape = {}) {

    AffineMapAttr indexingMapAttr =
        (operandIndexingMap) ? AffineMapAttr::get(*operandIndexingMap)
                             : nullptr;
    AffineMapAttr srcIndexingMapAttr = isOutput ? nullptr : indexingMapAttr;
    AffineMapAttr dstIndexingMapAttr = isOutput ? indexingMapAttr : nullptr;

    IntegerAttr numElemsAttr = nullptr;
    // If mcast is programmed, we will implicitly set the numElems because we
    // know we already gathered the data into a contiguous buffer.
    if (!mcastShape.empty()) {
      numElemsAttr = builder.getI64IntegerAttr(
          mlir::cast<ShapedType>(dst.getType()).getNumElements());
    }

    return builder
        .create<d2m::DMAOp>(loc, src, srcIndexingMapAttr, ValueRange(), dst,
                            dstIndexingMapAttr, ValueRange(), numElemsAttr,
                            coreIndex, mcastShape)
        .getResult();
  }

  struct McastArguments {
    SmallVector<Value> senderCoreIndex;
    SmallVector<Value> mcastCoreIndex;
    SmallVector<Value> mcastShape;
    unsigned mcastVolume = 1;
    SmallVector<Value> conditions;
  };

  static McastArguments
  calculateGatherMcastArguments(PatternRewriter &rewriter, Location loc,
                                ttcore::GridAttr grid,
                                ArrayRef<ttcore::IteratorType> mcastIterators) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(1));

    McastArguments args;
    args.senderCoreIndex.reserve(grid.getShape().size());
    args.mcastCoreIndex.reserve(grid.getShape().size());
    args.mcastShape.reserve(grid.getShape().size());

    for (auto [dim, iteratorType] : llvm::enumerate(mcastIterators)) {
      Value core = rewriter.create<CoreIndexOp>(
          loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(dim));
      if (iteratorType == ttcore::IteratorType::Parallel) {
        args.senderCoreIndex.push_back(Value(core));
        args.mcastCoreIndex.push_back(Value(core));
        args.mcastShape.push_back(Value(one));
      } else {
        int64_t numDests = grid.getShape()[dim] - 1;
        Value gridDimMinusOne = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexType(), rewriter.getIndexAttr(numDests));
        assert(iteratorType == ttcore::IteratorType::Reduction);
        args.senderCoreIndex.push_back(zero);
        args.mcastCoreIndex.push_back(one);
        args.mcastShape.push_back(gridDimMinusOne);
        args.mcastVolume *= numDests;

        Value condition = rewriter.create<arith::CmpIOp>(
            loc, rewriter.getI1Type(), mlir::arith::CmpIPredicate::eq, core,
            zero);
        args.conditions.push_back(condition);
      }
    }

    return args;
  }

  // One implementation of mcast by which one core (the 0th core for the
  // respective dim) takes on the role of (the sender) gathering and sending the
  // data to all other cores (the receivers) via mcast along the same dimension.
  static void createGatherMcastDMA(
      PatternRewriter &builder, Location loc, Value src, Value dst,
      AffineMap operandIndexingMap, ttcore::GridAttr grid,
      ArrayRef<ttcore::IteratorType> mcastIterators, Block *computeBlock,
      BlockArgument receiversReadySemaphore,
      BlockArgument senderFinishedSemaphore) {
    McastArguments mcastArgs =
        calculateGatherMcastArguments(builder, loc, grid, mcastIterators);
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    Value one = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                  builder.getIndexAttr(1));
    assert(mcastArgs.mcastVolume > 0);
    Value mcastVolumeVal = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(),
        builder.getIndexAttr(mcastArgs.mcastVolume));
    assert(mcastArgs.mcastCoreIndex.size() == mcastArgs.mcastShape.size());
    assert(mcastArgs.conditions.size() == 1 &&
           "Exactly one condition supported");
    builder.create<scf::IfOp>(
        loc, mcastArgs.conditions[0],
        [&](OpBuilder &builder, Location loc) {
          bool isOutput = false;
          Value gatherMemTx =
              createDMA(builder, loc, src, dst, operandIndexingMap, isOutput);
          builder.create<d2m::DMAWaitOp>(loc, gatherMemTx);
          builder.create<d2m::SemaphoreWaitOp>(loc, receiversReadySemaphore,
                                               mcastVolumeVal, zero);
          Value mcastMemTx =
              createDMA(builder, loc, dst, dst, std::nullopt, isOutput,
                        mcastArgs.mcastCoreIndex, mcastArgs.mcastShape);
          builder.create<d2m::DMAWaitOp>(loc, mcastMemTx);
          builder.create<d2m::SemaphoreSetOp>(loc, senderFinishedSemaphore, one,
                                              mcastArgs.mcastCoreIndex,
                                              mcastArgs.mcastShape);
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<d2m::SemaphoreIncOp>(loc, receiversReadySemaphore, one,
                                              mcastArgs.senderCoreIndex);
          builder.create<d2m::SemaphoreWaitOp>(loc, senderFinishedSemaphore,
                                               one, zero);
          builder.create<scf::YieldOp>(loc);
        });
  }

  static LogicalResult buildDatamovementOps(
      PatternRewriter &builder, Location loc, Value genericOperand,
      Value blockOperand, ttcore::GridAttr grid, ttcore::DeviceAttr device,
      AffineMap operandIndexingMap, ArrayAttr iteratorTypes, bool isOutput,
      Block *computeBlock, BlockArgument receiversReadySemaphore,
      BlockArgument senderFinishedSemaphore, bool hasSemaphores) {
    // Create an execute_region to wrap all DMA-related operations for this
    // operand Save current insertion point
    OpBuilder::InsertionGuard guard(builder);
    auto executeRegionOp =
        builder.create<scf::ExecuteRegionOp>(loc, TypeRange{});
    // Prevent canonicalization from inlining the execute_region op.
    executeRegionOp->setAttr("no_inline", builder.getUnitAttr());

    // Create the execute region block
    Block *executeRegionBlock = &executeRegionOp.getRegion().emplaceBlock();

    // Set insertion point to the execute region block
    builder.setInsertionPointToStart(executeRegionBlock);

    // For stream operands, we need to insert DMA operations before wait/reserve
    if (isStream(genericOperand)) {
      // First, get the circular buffer memref
      Value cb =
          isOutput
              ? builder.create<d2m::WaitOp>(loc, blockOperand).getResult()
              : builder.create<d2m::ReserveOp>(loc, blockOperand).getResult();

      Value src = isOutput ? cb : genericOperand;
      Value dst = isOutput ? genericOperand : cb;
      SmallVector<ttcore::IteratorType> mcastIterators =
          calculateMcastIterators(grid, device, operandIndexingMap,
                                  iteratorTypes);
      bool isMcast = !mcastIterators.empty();
      if (isMcast) {
        assert(hasSemaphores && receiversReadySemaphore &&
               senderFinishedSemaphore && "multicast requires semaphores");
        // Ensure insertion point is at the end of execute region block before
        // creating multicast DMA
        builder.setInsertionPointToEnd(executeRegionBlock);
        // Create the multicast DMA operations (creates constants and scf::IfOp)
        createGatherMcastDMA(builder, loc, src, dst, operandIndexingMap, grid,
                             mcastIterators, executeRegionBlock,
                             receiversReadySemaphore, senderFinishedSemaphore);
        // After createGatherMcastDMA, the scf::IfOp should be the last
        // operation Set insertion point to end of block (after the scf::IfOp)
        builder.setInsertionPointToEnd(executeRegionBlock);
      } else {
        builder.setInsertionPointToEnd(executeRegionBlock);
        Value memTx =
            createDMA(builder, loc, src, dst, operandIndexingMap, isOutput);
        builder.create<d2m::DMAWaitOp>(loc, memTx);
      }
    } else {
      // For non-stream operands, just create wait/reserve operations
      builder.setInsertionPointToEnd(executeRegionBlock);
      if (isOutput) {
        builder.create<d2m::WaitOp>(loc, blockOperand);
      } else {
        builder.create<d2m::ReserveOp>(loc, blockOperand);
      }
    }

    // Add yield at the end of the execute region
    builder.setInsertionPointToEnd(executeRegionBlock);
    builder.create<scf::YieldOp>(loc);

    return success();
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.isComputeOnlyForm()) {
      // Already inserted, skip.
      return failure();
    }

    if (generic.getNumRegions() != 1) {
      return failure();
    }

    Block *computeBlock = &generic.getRegion(0).front();

    // Check if DMA operations or execute_regions already exist in the compute
    // region to prevent infinite loops. Check both directly in the block and
    // inside execute_regions.
    bool hasDMAOps = false;
    bool hasExecuteRegions = false;
    computeBlock->walk([&](Operation *op) {
      if (mlir::isa<d2m::DMAOp>(op)) {
        hasDMAOps = true;
        return WalkResult::interrupt();
      }
      if (mlir::isa<scf::ExecuteRegionOp>(op)) {
        hasExecuteRegions = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasDMAOps || hasExecuteRegions) {
      // DMA operations or execute_regions already inserted, skip.
      return failure();
    }
    Location loc = generic->getLoc();
    auto device = ttcore::lookupDevice(generic);

    // Check if any operand needs multicast (to determine if we need semaphores)
    bool needsSemaphores = false;
    for (OpOperand &operand : generic->getOpOperands()) {
      if (isStream(operand.get())) {
        AffineMap operandIndexingMap =
            mlir::cast<AffineMapAttr>(
                generic.getIndexingMaps()[operand.getOperandNumber()])
                .getValue();
        SmallVector<ttcore::IteratorType> mcastIterators =
            calculateMcastIterators(generic.getGrid(), device,
                                    operandIndexingMap,
                                    generic.getIteratorTypes());
        if (!mcastIterators.empty()) {
          needsSemaphores = true;
          break;
        }
      }
    }

    // Add semaphore arguments to compute block if needed
    BlockArgument receiversReadySemaphore = nullptr;
    BlockArgument senderFinishedSemaphore = nullptr;
    if (needsSemaphores) {
      rewriter.modifyOpInPlace(generic, [&] {
        receiversReadySemaphore = createSemaphore(rewriter, loc, computeBlock);
        senderFinishedSemaphore = createSemaphore(rewriter, loc, computeBlock);
      });
    }

    // Save the original insertion point and set it to the start of compute
    // block
    Operation *firstOp =
        computeBlock->empty() ? nullptr : &computeBlock->front();
    rewriter.setInsertionPointToStart(computeBlock);

    unsigned outputOperandsIndex = generic.getOutputs().getBeginOperandIndex();

    // First, insert datamovement operations for INPUT operands at the start
    for (OpOperand &operand : generic->getOpOperands()) {
      bool isOutput = operand.getOperandNumber() >= outputOperandsIndex;
      if (isOutput) {
        continue; // Skip outputs for now
      }

      AffineMap operandIndexingMap =
          mlir::cast<AffineMapAttr>(
              generic.getIndexingMaps()[operand.getOperandNumber()])
              .getValue();
      auto result = buildDatamovementOps(
          rewriter, loc, generic->getOperand(operand.getOperandNumber()),
          computeBlock->getArgument(operand.getOperandNumber()),
          generic.getGrid(), device, operandIndexingMap,
          generic.getIteratorTypes(), isOutput, computeBlock,
          receiversReadySemaphore, senderFinishedSemaphore, needsSemaphores);
      if (failed(result)) {
        return result;
      }
    }

    // Restore insertion point after the inserted input operations
    if (firstOp) {
      rewriter.setInsertionPoint(firstOp);
    } else {
      rewriter.setInsertionPointToEnd(computeBlock);
    }

    // Find the last operation in the block (before terminator if any)
    if (!computeBlock->empty()) {
      Operation *backOp = &computeBlock->back();
      // If there's a terminator, insert before it; otherwise insert at end
      if (backOp->hasTrait<mlir::OpTrait::IsTerminator>()) {
        rewriter.setInsertionPoint(backOp);
      } else {
        rewriter.setInsertionPointToEnd(computeBlock);
      }
    } else {
      rewriter.setInsertionPointToEnd(computeBlock);
    }

    // Now insert datamovement operations for OUTPUT operands at the end (after
    // compute)
    for (OpOperand &operand : generic->getOpOperands()) {
      bool isOutput = operand.getOperandNumber() >= outputOperandsIndex;
      if (!isOutput) {
        continue; // Skip inputs, already processed
      }

      AffineMap operandIndexingMap =
          mlir::cast<AffineMapAttr>(
              generic.getIndexingMaps()[operand.getOperandNumber()])
              .getValue();
      auto result = buildDatamovementOps(
          rewriter, loc, generic->getOperand(operand.getOperandNumber()),
          computeBlock->getArgument(operand.getOperandNumber()),
          generic.getGrid(), device, operandIndexingMap,
          generic.getIteratorTypes(), isOutput, computeBlock,
          receiversReadySemaphore, senderFinishedSemaphore, needsSemaphores);
      if (failed(result)) {
        return result;
      }
    }

    return success();
  }
};
} // namespace

namespace {
class D2MInsertDMAOps : public impl::D2MInsertDMAOpsBase<D2MInsertDMAOps> {
public:
  using impl::D2MInsertDMAOpsBase<D2MInsertDMAOps>::D2MInsertDMAOpsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MInsertDMAOpsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
