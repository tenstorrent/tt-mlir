// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICDATAMOVEMENT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericDatamovementRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static bool isStream(Type ty) {
    return mlir::isa<StreamLayoutAttr>(mlir::cast<MemRefType>(ty).getLayout());
  }

  static bool compatibleDeviceGrid(DeviceAttr device, GridAttr grid) {
    if (grid.getShape().size() != device.getWorkerGrid().getShape().size()) {
      return false;
    }
    return true;
  }

  static BlockArgument createSemaphore(OpBuilder &builder, Location loc) {
    Block *thisBlock = builder.getBlock();
    Operation *op = thisBlock->getParentOp();
    BlockArgument semaphore = nullptr;
    for (Region &region : op->getRegions()) {
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

  static SmallVector<IteratorType>
  calculateMcastIterators(GridAttr grid, DeviceAttr device,
                          AffineMap operandIndexingMap,
                          ArrayAttr iteratorTypes) {
    assert(grid.getShape().size() == 2 && "Currently only support 2D grid");
    assert(grid.getShape().size() == operandIndexingMap.getNumResults());
    assert(compatibleDeviceGrid(device, grid));

    bool allParallel = true;
    SmallVector<IteratorType> mcastIterators;
    mcastIterators.reserve(grid.getShape().size());
    for (unsigned dim = 0; dim < grid.getShape().size(); dim++) {
      unsigned dimPosition = operandIndexingMap.getDimPosition(dim);

      IteratorType iteratorType =
          mlir::cast<IteratorTypeAttr>(iteratorTypes[dimPosition]).getValue();
      mcastIterators.push_back(iteratorType);

      // If the grid dimension is 1, we can special case it and always safely
      // fallback to mode parallel.  Reduction implies multicast, and while
      // it'll be functionally correct, a multicast with a single core to itself
      // is a redundant copy and more complicated than necessary.
      bool singleCore = grid.getShape()[dim] == 1;
      allParallel &= (iteratorType == IteratorType::Parallel) || singleCore;
    }

    return allParallel ? SmallVector<IteratorType>() : mcastIterators;
  }

  static Value createDMA(OpBuilder &builder, Location loc, Value src, Value dst,
                         std::optional<AffineMap> operandIndexingMap,
                         SmallVector<Value> coreIndex = {},
                         SmallVector<Value> mcastShape = {}) {
    return builder
        .create<ttir::DMAOp>(loc, builder.getType<MemTxType>(), src,
                             operandIndexingMap
                                 ? AffineMapAttr::get(*operandIndexingMap)
                                 : nullptr,
                             ValueRange(), dst, nullptr, ValueRange(), nullptr,
                             coreIndex, mcastShape)
        .getResult();
  }

  static std::tuple<SmallVector<Value>, SmallVector<Value>, unsigned,
                    SmallVector<Value>>
  calculateGatherMcastArguments(OpBuilder &blockBuilder, Location loc,
                                GridAttr grid,
                                ArrayRef<IteratorType> mcastIterators) {
    Value zero = blockBuilder.create<arith::ConstantOp>(
        loc, blockBuilder.getIndexType(), blockBuilder.getIndexAttr(0));
    Value one = blockBuilder.create<arith::ConstantOp>(
        loc, blockBuilder.getIndexType(), blockBuilder.getIndexAttr(1));

    SmallVector<Value> coreIndex;
    SmallVector<Value> mcastShape;
    unsigned mcastVolume = 1;
    SmallVector<Value> conditions;
    coreIndex.reserve(grid.getShape().size());
    mcastShape.reserve(grid.getShape().size());

    for (auto [dim, iteratorType] : llvm::enumerate(mcastIterators)) {
      Value gridDimMinusOne = blockBuilder.create<arith::ConstantOp>(
          loc, blockBuilder.getIndexType(),
          blockBuilder.getIndexAttr(grid.getShape()[dim] - 1));
      Value core =
          blockBuilder.create<CoreIndexOp>(loc, blockBuilder.getIndexType(),
                                           blockBuilder.getI64IntegerAttr(dim));
      if (iteratorType == IteratorType::Parallel) {
        coreIndex.push_back(Value(core));
        mcastShape.push_back(Value(one));
      } else {
        assert(iteratorType == IteratorType::Reduction);
        coreIndex.push_back(one);
        mcastShape.push_back(gridDimMinusOne);
        mcastVolume *= grid.getShape()[dim];

        Value condition = blockBuilder.create<arith::CmpIOp>(
            loc, blockBuilder.getI1Type(), mlir::arith::CmpIPredicate::eq, core,
            zero);
        conditions.push_back(condition);
      }
    }

    return std::make_tuple(coreIndex, mcastShape, mcastVolume, conditions);
  }

  // One implementation of mcast by which one core (the 0th core for the
  // respective dim) takes on the role of (the sender) gathering and sending the
  // data to all other cores (the receivers) via mcast along the same dimension.
  static void createGatherMcastDMA(OpBuilder &builder, Location loc, Value src,
                                   Value dst, AffineMap operandIndexingMap,
                                   GridAttr grid,
                                   ArrayRef<IteratorType> mcastIterators) {
    SmallVector<Value> coreIndex, mcastShape, conditions;
    unsigned mcastVolume;
    std::tie(coreIndex, mcastShape, mcastVolume, conditions) =
        calculateGatherMcastArguments(builder, loc, grid, mcastIterators);
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    Value one = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                  builder.getIndexAttr(1));
    assert(mcastVolume > 0);
    Value mcastVolumeMinusOne = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(), builder.getIndexAttr(mcastVolume - 1));
    Value receiversReadySemaphore = createSemaphore(builder, loc);
    Value senderFinishedSemaphore = createSemaphore(builder, loc);
    assert(coreIndex.size() == mcastShape.size());
    assert(conditions.size() == 1 && "Exactly one condition supported");
    builder.create<scf::IfOp>(
        loc, conditions[0],
        [&](OpBuilder &builder, Location loc) {
          Value gatherMemTx =
              createDMA(builder, loc, src, dst, operandIndexingMap);
          builder.create<ttir::DMAWaitOp>(loc, gatherMemTx);
          builder.create<ttir::SemaphoreWaitOp>(loc, receiversReadySemaphore,
                                                mcastVolumeMinusOne, zero);
          Value mcastMemTx = createDMA(builder, loc, dst, dst, std::nullopt,
                                       coreIndex, mcastShape);
          builder.create<ttir::DMAWaitOp>(loc, mcastMemTx);
          builder.create<ttir::SemaphoreSetOp>(loc, senderFinishedSemaphore,
                                               one, coreIndex, mcastShape);
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<ttir::SemaphoreIncOp>(loc, receiversReadySemaphore,
                                               one, coreIndex);
          builder.create<ttir::SemaphoreWaitOp>(loc, senderFinishedSemaphore,
                                                one, zero);
          builder.create<scf::YieldOp>(loc);
        });
  }

  static LogicalResult
  buildDatamovementBlock(OpBuilder &builder, Location loc, Value genericOperand,
                         Value blockOperand, GridAttr grid, DeviceAttr device,
                         AffineMap operandIndexingMap, ArrayAttr iteratorTypes,
                         bool isOutput) {
    if (isOutput) {
      // Wait for compute
      builder.create<ttir::AwaitOp>(loc, ValueRange(blockOperand));
    }

    if (isStream(genericOperand.getType())) {
      assert(!isOutput && "Output streaming is not currently supported");
      Value src = isOutput ? blockOperand : genericOperand;
      Value dst = isOutput ? genericOperand : blockOperand;
      SmallVector<IteratorType> mcastIterators = calculateMcastIterators(
          grid, device, operandIndexingMap, iteratorTypes);
      bool isMcast = !mcastIterators.empty();
      if (isMcast) {
        createGatherMcastDMA(builder, loc, src, dst, operandIndexingMap, grid,
                             mcastIterators);
      } else {
        Value memTx = createDMA(builder, loc, src, dst, operandIndexingMap);
        builder.create<ttir::DMAWaitOp>(loc, memTx);
      }
    }

    if (!isOutput) {
      // Push input to compute
      builder.create<ttir::YieldOp>(loc, ValueRange(blockOperand));
    }

    return success();
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getNumRegions() > 1) {
      // Already inserted, skip.
      return failure();
    }

    // One per operand
    auto numDataMovementRegions = generic.getNumOperands();
    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(), generic.getIndexingMaps(),
        generic.getIteratorTypes(),
        generic.getNumRegions() + numDataMovementRegions);

    // Insert the new data movement regions.
    auto [outputOperandsIndex, outputOperandsLength] =
        generic.getODSOperandIndexAndLength(1);
    auto device = getCurrentScopeDevice(generic);
    for (OpOperand &operand : generic->getOpOperands()) {
      Block *datamovementBlock =
          &newGeneric.getRegion(operand.getOperandNumber()).emplaceBlock();
      datamovementBlock->addArguments(
          generic.getRegion(0).getArgumentTypes(),
          SmallVector<mlir::Location>(
              generic.getRegion(0).getArgumentTypes().size(),
              generic.getLoc()));

      OpBuilder blockBuilder = OpBuilder::atBlockEnd(datamovementBlock);
      bool isOutput = operand.getOperandNumber() >= outputOperandsIndex;
      AffineMap operandIndexingMap =
          mlir::cast<AffineMapAttr>(
              generic.getIndexingMaps()[operand.getOperandNumber()])
              .getValue();
      auto result = buildDatamovementBlock(
          blockBuilder, generic->getLoc(),
          generic->getOperand(operand.getOperandNumber()),
          datamovementBlock->getArgument(operand.getOperandNumber()),
          generic.getGrid(), device, operandIndexingMap,
          generic.getIteratorTypes(), isOutput);
      if (failed(result)) {
        return result;
      }
    }

    // Copy over the original compute region.
    unsigned computeRegionIndex = numDataMovementRegions;
    auto &newRegion = newGeneric.getRegion(computeRegionIndex);
    auto &oldRegion = generic.getRegion(0);
    newRegion.takeBody(oldRegion);

    // Await / Yield insertion to compute region.
    {
      Block *computeBlock = &newGeneric.getRegion(computeRegionIndex).front();
      OpBuilder blockBuilder = OpBuilder::atBlockBegin(computeBlock);
      auto [inputOperandsIndex, inputOperandsLength] =
          generic.getODSOperandIndexAndLength(0);

      // Await the inputs
      blockBuilder.create<ttir::AwaitOp>(
          generic->getLoc(), ValueRange(computeBlock->getArguments().take_front(
                                 inputOperandsLength)));

      blockBuilder.setInsertionPointToEnd(computeBlock);

      // Yield the outputs
      blockBuilder.create<ttir::YieldOp>(
          generic->getLoc(), ValueRange(computeBlock->getArguments().take_back(
                                 outputOperandsLength)));
    }

    rewriter.replaceOp(generic, newGeneric);
    return success();
  }
};
} // namespace

namespace {
class TTIRGenericDatamovement
    : public impl::TTIRGenericDatamovementBase<TTIRGenericDatamovement> {
public:
  using impl::TTIRGenericDatamovementBase<
      TTIRGenericDatamovement>::TTIRGenericDatamovementBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericDatamovementRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
