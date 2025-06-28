// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

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

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICGENERATEDATAMOVEMENT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericGenerateDatamovementRewriter
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static bool isStream(Value operand) {
    return mlir::isa_and_nonnull<StreamLayoutOp>(operand.getDefiningOp());
  }

  static bool compatibleDeviceGrid(ttcore::DeviceAttr device,
                                   ttcore::GridAttr grid) {
    return device.getWorkerGrid().getShape().size() == grid.getShape().size();
  }

  static BlockArgument createSemaphore(PatternRewriter &builder, Location loc,
                                       MutableArrayRef<Region> regions) {
    Block *thisBlock = builder.getBlock();
    BlockArgument semaphore = nullptr;
    for (Region &region : regions) {
      for (Block &block : region) {
        BlockArgument blockSemaphore =
            block.addArgument(builder.getType<SemaphoreType>(), loc);
        if (thisBlock == &block) {
          semaphore = blockSemaphore;
        }
      }
    }
    assert(semaphore);
    return semaphore;
  }

  static SmallVector<ttcore::IteratorType>
  calculateMcastIterators(ttcore::GridAttr grid, ttcore::DeviceAttr device,
                          AffineMap operandIndexingMap,
                          ArrayAttr iteratorTypes) {
    assert(grid.getShape().size() == 2 && "Currently only support 2D grid");
    assert(grid.getShape().size() == operandIndexingMap.getNumResults());
    assert(compatibleDeviceGrid(device, grid));

    bool allParallel = true;
    SmallVector<ttcore::IteratorType> mcastIterators;
    mcastIterators.reserve(grid.getShape().size());
    for (unsigned dim = 0; dim < grid.getShape().size(); dim++) {
      unsigned dimPosition = operandIndexingMap.getDimPosition(dim);

      ttcore::IteratorType iteratorType =
          mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[dimPosition])
              .getValue();
      mcastIterators.push_back(iteratorType);

      // If the grid dimension is 1, we can special case it and always safely
      // fallback to mode parallel.  Reduction implies multicast, and while
      // it'll be functionally correct, a multicast with a single core to itself
      // is a redundant copy and more complicated than necessary.
      bool singleCore = grid.getShape()[dim] == 1;
      allParallel &=
          (iteratorType == ttcore::IteratorType::Parallel) || singleCore;
    }

    return allParallel ? SmallVector<ttcore::IteratorType>() : mcastIterators;
  }

  static Value createDMA(OpBuilder &builder, Location loc, Value src, Value dst,
                         std::optional<AffineMap> operandIndexingMap,
                         SmallVector<Value> coreIndex = {},
                         SmallVector<Value> mcastShape = {}) {
    return builder
        .create<ttir::DMAOp>(loc, src,
                             operandIndexingMap
                                 ? AffineMapAttr::get(*operandIndexingMap)
                                 : nullptr,
                             dst, coreIndex, mcastShape)
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
  static void
  createGatherMcastDMA(PatternRewriter &builder, Location loc, Value src,
                       Value dst, AffineMap operandIndexingMap,
                       ttcore::GridAttr grid,
                       ArrayRef<ttcore::IteratorType> mcastIterators,
                       MutableArrayRef<Region> regions) {
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
    Value receiversReadySemaphore = createSemaphore(builder, loc, regions);
    Value senderFinishedSemaphore = createSemaphore(builder, loc, regions);
    assert(mcastArgs.mcastCoreIndex.size() == mcastArgs.mcastShape.size());
    assert(mcastArgs.conditions.size() == 1 &&
           "Exactly one condition supported");
    builder.create<scf::IfOp>(
        loc, mcastArgs.conditions[0],
        [&](OpBuilder &builder, Location loc) {
          Value gatherMemTx =
              createDMA(builder, loc, src, dst, operandIndexingMap);
          builder.create<ttir::DMAWaitOp>(loc, gatherMemTx);
          builder.create<ttir::SemaphoreWaitOp>(loc, receiversReadySemaphore,
                                                mcastVolumeVal, zero);
          Value mcastMemTx =
              createDMA(builder, loc, dst, dst, std::nullopt,
                        mcastArgs.mcastCoreIndex, mcastArgs.mcastShape);
          builder.create<ttir::DMAWaitOp>(loc, mcastMemTx);
          builder.create<ttir::SemaphoreSetOp>(loc, senderFinishedSemaphore,
                                               one, mcastArgs.mcastCoreIndex,
                                               mcastArgs.mcastShape);
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<ttir::SemaphoreIncOp>(loc, receiversReadySemaphore,
                                               one, mcastArgs.senderCoreIndex);
          builder.create<ttir::SemaphoreWaitOp>(loc, senderFinishedSemaphore,
                                                one, zero);
          builder.create<scf::YieldOp>(loc);
        });
  }

  static LogicalResult
  buildDatamovementBlock(PatternRewriter &builder, Location loc,
                         Value genericOperand, Value blockOperand,
                         ttcore::GridAttr grid, ttcore::DeviceAttr device,
                         AffineMap operandIndexingMap, ArrayAttr iteratorTypes,
                         bool isOutput, MutableArrayRef<Region> regions) {
    if (isOutput) {
      // Wait for compute.
      builder.create<ttir::AwaitOp>(loc, blockOperand);
    }

    if (isStream(genericOperand)) {
      assert(!isOutput && "Output streaming is not currently supported");
      Value src = isOutput ? blockOperand : genericOperand;
      Value dst = isOutput ? genericOperand : blockOperand;
      SmallVector<ttcore::IteratorType> mcastIterators =
          calculateMcastIterators(grid, device, operandIndexingMap,
                                  iteratorTypes);
      bool isMcast = !mcastIterators.empty();
      if (isMcast) {
        createGatherMcastDMA(builder, loc, src, dst, operandIndexingMap, grid,
                             mcastIterators, regions);
      } else {
        Value memTx = createDMA(builder, loc, src, dst, operandIndexingMap);
        builder.create<ttir::DMAWaitOp>(loc, memTx);
      }
    }

    if (!isOutput) {
      // Push input to compute.
      builder.create<ttir::YieldOp>(loc, blockOperand);
    }

    return success();
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.isComputeOnlyForm()) {
      // Already inserted, skip.
      return failure();
    }

    // One per operand.
    auto numDataMovementRegions = generic.getNumOperands();
    auto numTotalRegions = generic.getNumRegions() + numDataMovementRegions;
    SmallVector<Attribute> threads(
        numDataMovementRegions,
        rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
    threads.append(generic.getThreads().begin(), generic.getThreads().end());
    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(), generic.getBlockFactors(),
        generic.getIndexingMaps(), generic.getIteratorTypes(),
        rewriter.getArrayAttr(threads), numTotalRegions);

    // Preinitialize all regions so that we can modify their signatures on the
    // fly. i.e. adding semaphore arguments.
    for (unsigned regionIdx = 0; regionIdx < numTotalRegions; ++regionIdx) {
      Block &block = newGeneric.getRegion(regionIdx).emplaceBlock();
      rewriter.modifyOpInPlace(newGeneric, [&] {
        block.addArguments(generic.getRegion(0).getArgumentTypes(),
                           SmallVector<mlir::Location>(
                               generic.getRegion(0).getArgumentTypes().size(),
                               generic.getLoc()));
      });
    }

    // Insert the new data movement regions.
    unsigned outputOperandsIndex = generic.getOutputs().getBeginOperandIndex();
    unsigned outputOperandsLength = generic.getOutputs().size();
    auto device = ttcore::lookupDevice(generic);
    for (OpOperand &operand : generic->getOpOperands()) {
      Block *datamovementBlock =
          &newGeneric.getRegion(operand.getOperandNumber()).front();

      rewriter.setInsertionPointToEnd(datamovementBlock);
      bool isOutput = operand.getOperandNumber() >= outputOperandsIndex;
      AffineMap operandIndexingMap =
          mlir::cast<AffineMapAttr>(
              generic.getIndexingMaps()[operand.getOperandNumber()])
              .getValue();
      auto result = buildDatamovementBlock(
          rewriter, generic->getLoc(),
          generic->getOperand(operand.getOperandNumber()),
          datamovementBlock->getArgument(operand.getOperandNumber()),
          generic.getGrid(), device, operandIndexingMap,
          generic.getIteratorTypes(), isOutput, newGeneric.getRegions());
      if (failed(result)) {
        return result;
      }
    }

    // Copy over the original compute region.
    unsigned computeRegionIndex = numDataMovementRegions;
    auto &newRegion = newGeneric.getRegion(computeRegionIndex);
    auto &oldRegion = generic.getRegion(0);
    rewriter.mergeBlocks(&oldRegion.front(), &newRegion.front(),
                         newRegion.front().getArguments().take_front(
                             generic.getOperands().size()));

    // Await / Yield insertion to compute region.
    {
      Block *computeBlock = &newGeneric.getRegion(computeRegionIndex).front();
      rewriter.setInsertionPointToStart(computeBlock);

      // Await the inputs.
      rewriter.create<ttir::AwaitOp>(
          generic->getLoc(),
          computeBlock->getArguments().take_front(generic.getInputs().size()));

      rewriter.setInsertionPointToEnd(computeBlock);

      // Yield the outputs.
      rewriter.create<ttir::YieldOp>(
          generic->getLoc(), computeBlock->getArguments().slice(
                                 outputOperandsIndex, outputOperandsLength));
    }

    rewriter.replaceOp(generic, newGeneric);
    return success();
  }
};
} // namespace

namespace {
class TTIRGenericGenerateDatamovement
    : public impl::TTIRGenericGenerateDatamovementBase<
          TTIRGenericGenerateDatamovement> {
public:
  using impl::TTIRGenericGenerateDatamovementBase<
      TTIRGenericGenerateDatamovement>::TTIRGenericGenerateDatamovementBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericGenerateDatamovementRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
