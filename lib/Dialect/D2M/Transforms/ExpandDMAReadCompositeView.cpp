// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MEXPANDDMAREADCOMPOSITEVIEW
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

static LogicalResult expandCompositeDMAReadTiled(IRRewriter &rewriter,
                                                 DMAReadOp dmaRead,
                                                 ValueRange expandedInputs,
                                                 const int32_t concatDim) {
  assert(dmaRead.isShardLevel());
  ttcore::DeviceAttr device = ttcore::lookupDevice(dmaRead);

  int64_t gridRank = static_cast<int64_t>(dmaRead.getSrcIndices().size());
  assert(concatDim >= 0 && concatDim < gridRank);

  auto compositeType = mlir::cast<MemRefType>(dmaRead.getSrc().getType());
  TT_assert(compositeType.getRank() == 2 * gridRank);

  const int64_t concatGridDim = concatDim;
  const int64_t concatShardDim = concatDim + gridRank;

  // Number of tiles in the composite view's concat dim, potentially aligned-up.
  auto compositeDeviceShape = compositeType.getShape();
  const int64_t compositeExtent = compositeDeviceShape[concatGridDim] *
                                  compositeDeviceShape[concatShardDim];

  // Number of tiles for each & all of the composite view inputs.
  int64_t totalInputExtent = 0;
  SmallVector<int64_t> inputExtents;
  SmallVector<AffineMap> inputMemoryMaps;
  SmallVector<SmallVector<int64_t>> inputShardShapes;

  for (Value input : expandedInputs) {
    auto inputType = mlir::cast<MemRefType>(input.getType());
    TT_assert(inputType.getRank() == compositeType.getRank());
    auto inputDeviceShape = inputType.getShape();
    const int64_t inputExtent =
        inputDeviceShape[concatGridDim] * inputDeviceShape[concatShardDim];

    totalInputExtent += inputExtent;
    inputExtents.push_back(inputExtent);
    inputMemoryMaps.push_back(
        utils::getMemoryMap(device, input, /*isRemote=*/true));
    inputShardShapes.emplace_back(inputDeviceShape.drop_front(gridRank).begin(),
                                  inputDeviceShape.drop_front(gridRank).end());
  }
  TT_assert(totalInputExtent <= compositeExtent);

  Value localMemref = dmaRead.getDst();
  AffineMap localMemoryMap =
      utils::getMemoryMap(device, localMemref, /*isRemote=*/false);

  rewriter.setInsertionPoint(dmaRead);

  Location loc = dmaRead.getLoc();

  SmallVector<Value> gridIndices(dmaRead.getSrcIndices());
  SmallVector<int64_t> outputShardShape(
      mlir::cast<MemRefType>(localMemref.getType()).getShape());
  TT_assert(outputShardShape.size() == static_cast<size_t>(gridRank));

  auto emitDMAReadForInput = [&](OpBuilder &builder, Location innerLoc,
                                 ValueRange shardIters, Value globalConcatIdx,
                                 const int inputIdx,
                                 const int64_t pieceOffset) {
    // Calculate the full index (grid x shard) of the current input tile.
    SmallVector<Value> inputFullIdx(2 * gridRank);
    for (int dim = 0; dim < gridRank; dim++) {
      Value globalDimIdx = nullptr;
      if (dim == concatGridDim) {
        // Shift-back the output's tile index with this input's cumulative
        // offset so it points to the right tile of the input.
        if (pieceOffset == 0) {
          globalDimIdx = globalConcatIdx;
        } else {
          Value offsetVal = builder.create<arith::ConstantOp>(
              innerLoc, builder.getIndexType(),
              builder.getIndexAttr(pieceOffset));
          globalDimIdx = builder.create<arith::SubIOp>(
              innerLoc, globalConcatIdx, offsetVal);
        }
      } else {
        // Otherwise it's just: grid_idx * tiles_per_shard + shard_idx.
        Value outShardExtent = builder.create<arith::ConstantOp>(
            innerLoc, builder.getIndexType(),
            builder.getIndexAttr(outputShardShape[dim]));
        Value baseIdx = builder.create<arith::MulIOp>(
            innerLoc, gridIndices[dim], outShardExtent);
        globalDimIdx =
            builder.create<arith::AddIOp>(innerLoc, baseIdx, shardIters[dim]);
      }

      Value inShardExtent = builder.create<arith::ConstantOp>(
          innerLoc, builder.getIndexType(),
          builder.getIndexAttr(inputShardShapes[inputIdx][dim]));
      // grid_idx = idx // shard_size
      Value inputGridIdx =
          builder.create<arith::DivSIOp>(innerLoc, globalDimIdx, inShardExtent);
      // shard_idx = idx % shard_size
      Value inputShardIdx =
          builder.create<arith::RemSIOp>(innerLoc, globalDimIdx, inShardExtent);

      inputFullIdx[dim] = inputGridIdx;
      inputFullIdx[dim + gridRank] = inputShardIdx;
    }

    SmallVector<Value> remoteIndices = utils::applyMap(
        builder, innerLoc, inputMemoryMaps[inputIdx], inputFullIdx,
        /*isRemote=*/true);
    SmallVector<Value> localIndices = utils::applyMap(
        builder, innerLoc, localMemoryMap, shardIters, /*isRemote=*/false);

    // One tile at a time, no coalescing.
    Value dmaTx = builder.create<DMAReadOp>(
        innerLoc, expandedInputs[inputIdx], remoteIndices, localMemref,
        localIndices, builder.getI64IntegerAttr(1));
    builder.create<DMAWaitOp>(innerLoc, dmaTx);
  };

  auto [lbs, ubs, steps] =
      utils::getLoopBounds(rewriter, loc, outputShardShape);

  scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps,
      [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange shardIters) {
        // Global index along the concat dim of a given tile:
        // idx = grid_idx * tiles_per_shard + shard_idx
        Value concatShardExtent = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIndexAttr(outputShardShape[concatDim]));
        Value baseConcatIdx = loopBuilder.create<arith::MulIOp>(
            innerLoc, gridIndices[concatGridDim], concatShardExtent);
        Value globalConcatIdx = loopBuilder.create<arith::AddIOp>(
            innerLoc, baseConcatIdx, shardIters[concatDim]);

        // Recursively generate the if-else chain for all inputs.
        std::function<void(OpBuilder &, const int, const int64_t)>
            emitIfElseChain = [&](OpBuilder &builder, const int inputIdx,
                                  const int64_t startOffset) {
              // Base case: reaching the end of the chain (the last 'else').
              if (inputIdx + 1 == static_cast<int>(expandedInputs.size())) {
                emitDMAReadForInput(builder, innerLoc, shardIters,
                                    globalConcatIdx, inputIdx, startOffset);
                return;
              }

              // Recursive case:
              // - Emit DMA reads for the current input's contribution.
              // - Recurse into the 'else' branch for the next input.
              const int64_t boundary = startOffset + inputExtents[inputIdx];
              Value boundaryVal = builder.create<arith::ConstantOp>(
                  innerLoc, builder.getIndexType(),
                  builder.getIndexAttr(boundary));
              Value cond = builder.create<arith::CmpIOp>(
                  innerLoc, arith::CmpIPredicate::ult, globalConcatIdx,
                  boundaryVal);

              auto ifOp = builder.create<scf::IfOp>(innerLoc, TypeRange{}, cond,
                                                    /*hasElse=*/true);

              OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
              emitDMAReadForInput(thenBuilder, innerLoc, shardIters,
                                  globalConcatIdx, inputIdx, startOffset);

              OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
              emitIfElseChain(elseBuilder, inputIdx + 1, boundary);
            };

        // Skip out-of-bound portions if the output was aligned up.
        if (totalInputExtent < compositeExtent) {
          Value totalExtentVal = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIndexAttr(totalInputExtent));
          Value inBounds = loopBuilder.create<arith::CmpIOp>(
              innerLoc, arith::CmpIPredicate::ult, globalConcatIdx,
              totalExtentVal);
          auto guardOp = loopBuilder.create<scf::IfOp>(
              innerLoc, TypeRange{}, inBounds, /*hasElse=*/false);
          OpBuilder guardBuilder = guardOp.getThenBodyBuilder();
          emitIfElseChain(guardBuilder, /*inputIdx=*/0, /*startOffset=*/0);
        } else {
          emitIfElseChain(loopBuilder, /*inputIdx=*/0, /*startOffset=*/0);
        }
      });

  for (Operation *user : llvm::make_early_inc_range(dmaRead->getUsers())) {
    if (auto waitOp = mlir::dyn_cast<DMAWaitOp>(user)) {
      rewriter.eraseOp(waitOp);
    }
  }
  rewriter.eraseOp(dmaRead);

  return success();
}

static LogicalResult expandCompositeDMAReadRowMajor(
    IRRewriter &rewriter, DMAReadOp dmaRead, ValueRange expandedInputs,
    const int32_t concatDim, ArrayRef<int64_t> logicalSizes) {

  Value localDst = dmaRead.getDst();
  MemRefType dstType = mlir::cast<MemRefType>(localDst.getType());
  const int64_t shardRank = dstType.getRank();
  auto srcIndices = dmaRead.getSrcIndices();
  auto loc = dmaRead.getLoc();

  rewriter.setInsertionPoint(dmaRead);
  int64_t cumulativeOffset = 0;
  for (size_t i = 0; i < expandedInputs.size(); i++) {
    const int64_t pieceSize = logicalSizes[i];

    SmallVector<AffineExpr> exprs;
    for (int64_t dim = 0; dim < shardRank; dim++) {
      if (dim == concatDim) {
        exprs.push_back(rewriter.getAffineDimExpr(dim) + cumulativeOffset);
      } else {
        exprs.push_back(rewriter.getAffineDimExpr(dim));
      }
    }
    auto map = AffineMap::get(shardRank, 0, exprs, rewriter.getContext());

    auto viewType =
        MemRefType::get(dstType.getShape(), dstType.getElementType(),
                        MemRefLayoutAttrInterface(), dstType.getMemorySpace());
    auto view = rewriter.create<d2m::ViewLayoutOp>(loc, viewType, localDst, map,
                                                   /*reinterpretLayout=*/false);

    Value input = expandedInputs[i];
    auto shiftedRead =
        rewriter.create<DMAReadOp>(loc, input, srcIndices, view, ValueRange(),
                                   rewriter.getI64IntegerAttr(0));
    rewriter.create<DMAWaitOp>(loc, shiftedRead.getResult());

    cumulativeOffset += pieceSize;
  }

  auto nullTx = rewriter.create<NullTxOp>(loc, DMAType::Read);
  rewriter.replaceOp(dmaRead, nullTx.getResult());

  return success();
}

static DenseI64ArrayAttr remapScratchInputs(OpBuilder &builder,
                                            DenseI64ArrayAttr oldScratchInputs,
                                            const int64_t expandedInputIndex,
                                            const int64_t extraInputs) {
  if (!oldScratchInputs || oldScratchInputs.size() == 0) {
    return nullptr;
  }

  SmallVector<int64_t> remapped;
  remapped.reserve(oldScratchInputs.size());
  for (int64_t scratchInput : oldScratchInputs.asArrayRef()) {
    remapped.push_back(scratchInput > expandedInputIndex
                           ? scratchInput + extraInputs
                           : scratchInput);
  }
  return builder.getDenseI64ArrayAttr(remapped);
}

static LogicalResult expandCompositeViewsInGeneric(IRRewriter &rewriter,
                                                   GenericOp gOp) {
  CompositeViewOp compositeView = nullptr;
  gOp.walk([&](DMAReadOp dmaRead) {
    auto maybeCompositeView = dmaRead.getSrc().getDefiningOp<CompositeViewOp>();
    if (maybeCompositeView) {
      assert(compositeView == nullptr &&
             "Unsupported multiple composite views in one GenericOp (#7600).");
      compositeView = maybeCompositeView;
    }
  });
  assert(compositeView != nullptr);

  SmallVector<Value> compositeInputs = compositeView.getCompositeInputs();
  assert(compositeInputs.size() > 1);

  int64_t oldNumInputs = static_cast<int64_t>(gOp.getInputs().size());
  int64_t extraNumInputs = static_cast<int64_t>(compositeInputs.size() - 1);

  // Step 1: recreate the GenericOp w/ expanded list of inputs.
  SmallVector<Value> newInputs;
  newInputs.reserve(oldNumInputs + extraNumInputs);
  for (auto input : gOp.getInputs()) {
    if (input == compositeView.getResult()) {
      newInputs.append(compositeInputs.begin(), compositeInputs.end());
      continue;
    }
    newInputs.push_back(input);
  }

  int64_t compositeOperandIdx = gOp.getOperandIndex(compositeView.getResult());
  DenseI64ArrayAttr newScratchInputs =
      remapScratchInputs(rewriter, gOp.getScratchInputsAttr(),
                         compositeOperandIdx, extraNumInputs);

  rewriter.setInsertionPoint(gOp);
  // Passing empty block_factors/indexing_maps/iterator_types.
  assert(gOp.isExplicitDatamovementForm());
  auto newGOp = rewriter.create<GenericOp>(
      gOp.getLoc(), gOp.getResultTypes(), newInputs, gOp.getOutputs(),
      gOp.getAdditionalArgs(), gOp.getGrid(), rewriter.getI64ArrayAttr({}),
      rewriter.getAffineMapArrayAttr({}), rewriter.getArrayAttr({}),
      gOp.getThreads(), newScratchInputs, gOp.getFabricConnectionConfigAttr(),
      gOp.getNumRegions());

  // Step 2: clone old regions into the new GenericOp and fix up d2m.get_cb
  // operand indices that shifted due to the input list expansion.
  for (auto [oldRegion, newRegion] :
       llvm::zip(gOp.getRegions(), newGOp.getRegions())) {
    Block *oldBlock = &oldRegion.front();
    Block *newBlock = rewriter.createBlock(
        &newRegion, newRegion.end(), oldBlock->getArgumentTypes(),
        SmallVector<Location>(oldBlock->getNumArguments(), gOp.getLoc()));
    rewriter.setInsertionPointToStart(newBlock);

    IRMapping mapping;
    // Map whatever block args that still exist.
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock->getArguments(), newBlock->getArguments())) {
      mapping.map(oldArg, newArg);
    }

    for (Operation &op : oldBlock->without_terminator()) {
      rewriter.clone(op, mapping);
    }
    if (oldBlock->mightHaveTerminator()) {
      rewriter.clone(*oldBlock->getTerminator(), mapping);
    }

    // Shift the affected d2m.get_cb ops.
    for (Operation &op : newBlock->getOperations()) {
      auto getCBOp = mlir::dyn_cast<GetCBOp>(&op);
      if (!getCBOp || !getCBOp.getOperandIndexAttr()) {
        continue;
      }
      const int64_t oldIdx = getCBOp.getOperandIndexAttr().getInt();
      if (oldIdx > compositeOperandIdx) {
        const int64_t newIdx = oldIdx + extraNumInputs;
        getCBOp.setOperandIndexAttr(rewriter.getI64IntegerAttr(newIdx));
        getCBOp.setPortAttr(rewriter.getI64IntegerAttr(newIdx));
      }
    }
  }

  // Step 3: lower the composite DMA read in the new GenericOp.
  DMAReadOp clonedCompositeRead = nullptr;
  newGOp.walk([&](DMAReadOp dmaRead) {
    auto maybeCompositeView = dmaRead.getSrc().getDefiningOp<CompositeViewOp>();
    if (maybeCompositeView) {
      assert(dmaRead.getSrc() == compositeView.getResult());
      assert(clonedCompositeRead == nullptr);
      clonedCompositeRead = dmaRead;
    }
  });
  assert(clonedCompositeRead != nullptr);

  auto expandedGenericInputs =
      newGOp.getInputs().slice(compositeOperandIdx, compositeInputs.size());
  const bool isTiled = mlir::isa<ttcore::TileType>(
      mlir::cast<MemRefType>(compositeView.getResult().getType())
          .getElementType());

  LogicalResult result = success();
  if (isTiled) {
    result = expandCompositeDMAReadTiled(rewriter, clonedCompositeRead,
                                         expandedGenericInputs,
                                         compositeView.getDim());
  } else {
    result = expandCompositeDMAReadRowMajor(
        rewriter, clonedCompositeRead, expandedGenericInputs,
        compositeView.getDim(), compositeView.getLogicalSizes().value());
  }
  if (failed(result)) {
    return result;
  }

  rewriter.replaceOp(gOp, newGOp.getResults());
  return result;
}

static LogicalResult expandCompositeViews(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  SmallVector<GenericOp> genericsToExpand;

  moduleOp.walk([&](GenericOp gOp) {
    gOp.walk([&](DMAReadOp dmaRead) {
      if (dmaRead.getSrc().getDefiningOp<CompositeViewOp>()) {
        genericsToExpand.push_back(gOp);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  });

  for (GenericOp gOp : genericsToExpand) {
    if (failed(expandCompositeViewsInGeneric(rewriter, gOp))) {
      return failure();
    }
  }

  auto status = success();
  moduleOp.walk([&](CompositeViewOp compositeView) {
    if (compositeView->use_empty()) {
      compositeView.erase();
    } else {
      compositeView.emitError(
          "Composite view has remaining uses after expansion.");
      status = failure();
    }
  });
  return status;
}

namespace {
class D2MExpandDMAReadCompositeView
    : public impl::D2MExpandDMAReadCompositeViewBase<
          D2MExpandDMAReadCompositeView> {
public:
  using impl::D2MExpandDMAReadCompositeViewBase<
      D2MExpandDMAReadCompositeView>::D2MExpandDMAReadCompositeViewBase;

  void runOnOperation() final {
    if (failed(expandCompositeViews(getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
