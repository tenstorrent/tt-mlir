// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapAnalysis.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERDMATOFULLYINDEXEDFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

static std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
getLoopBounds(OpBuilder &builder, Location loc, ArrayRef<int64_t> shardShape) {
  Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                 builder.getIndexAttr(0));
  Value one = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                builder.getIndexAttr(1));
  SmallVector<Value> lbs(shardShape.size(), zero);
  SmallVector<Value> ubs(llvm::map_range(shardShape, [&](int64_t dim) {
    return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                             builder.getIndexAttr(dim));
  }));
  SmallVector<Value> step(shardShape.size(), one);
  return std::make_tuple(lbs, ubs, step);
}

static size_t getElementSizeBytes(MemRefType memref) {
  mlir::Type elementType = memref.getElementType();
  auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
  return tileType ? tileType.getSizeBytes()
                  : elementType.getIntOrFloatBitWidth() / 8;
}

static AffineMap canonicalStridedMap(MLIRContext *context,
                                     ArrayRef<int64_t> shape, Type elementType,
                                     AffineMap map) {
  assert(map.isIdentity() && "Only identity maps are supported for now.");
  auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
  int64_t elementSizeBytes = tileType ? tileType.getSizeBytes()
                                      : elementType.getIntOrFloatBitWidth() / 8;
  int64_t currentStride = elementSizeBytes;
  int64_t rank = shape.size();
  mlir::AffineExpr strideExpr = mlir::getAffineConstantExpr(0, context);
  for (int64_t i = rank - 1; i >= 0; i--) {
    mlir::AffineExpr dim = mlir::getAffineDimExpr(i, context);
    mlir::AffineExpr stride =
        mlir::getAffineConstantExpr(currentStride, context);
    strideExpr = dim * stride + strideExpr;
    currentStride *= shape[i];
  }
  return mlir::AffineMap::get(shape.size(), 0, strideExpr, context);
}

static AffineMap getMemoryMap(ttcore::DeviceAttr device, Value input,
                              bool isRemote) {
  if (isRemote) {
    // d2m::utils::getMemoryMap handles view tracing (applyViews) and
    // VGM lookup (getVirtualGridForwardMapping) internally.
    return d2m::utils::getMemoryMap(device, input, /*pageSize=*/0);
  }

  // For local memrefs (including CB values), get the underlying memref type.
  MemRefType inputType;
  if (auto cbType = mlir::dyn_cast<CBType>(input.getType())) {
    inputType = cbType.getUnderlyingAs<MemRefType>();
  } else {
    inputType = mlir::cast<MemRefType>(input.getType());
  }
  auto layoutMap = d2m::utils::resolveEffectiveAffineMap(input, inputType);
  return canonicalStridedMap(device.getContext(), inputType.getShape(),
                             inputType.getElementType(), layoutMap);
}

template <typename Builder>
static SmallVector<Value> applyMap(Builder &builder, Location loc,
                                   AffineMap map, ValueRange index,
                                   bool isRemote) {
  auto affineApply = [&](AffineMap map, ValueRange index) {
    return builder.template create<affine::AffineApplyOp>(loc, map, index);
  };

  if (isRemote) {
    assert(map.getNumResults() == 4);
    // Break the map into respective gridY, gridX, offset "single result"
    // parts. AffineApply only supports single result affine maps.
    map = map.dropResults(0); // Drop the device index.
    auto gridY = map.dropResults({1, 2});
    auto gridX = map.dropResults({0, 2});
    auto offset = map.dropResults({0, 1});
    return {affineApply(gridY, index), affineApply(gridX, index),
            affineApply(offset, index)};
  }

  assert(map.getNumResults() == 1);
  return {affineApply(map, index)};
}

// Calculates coalescing factor using analytical method with sampling fallback.
// Mirrors the logic from GenericLowerDMAs::analyzeStream.
static size_t calculateCoalescingFactorWithFallback(
    AffineMap memoryMap, ArrayRef<int64_t> gridShape,
    ArrayRef<int64_t> shardShape, size_t elemSizeBytes,
    bool debugCoalescingInference) {

  static constexpr size_t coalescingFactorSamplingFallbackThreshold = 16;

  // Compute full shape (grid + shard)
  SmallVector<int64_t> fullShape;
  fullShape.append(gridShape.begin(), gridShape.end());
  fullShape.append(shardShape.begin(), shardShape.end());

  // Try analytical method first
  size_t coalescingFactor = ttmlir::utils::computeCoalescingFactorAnalytically(
      memoryMap, fullShape, gridShape.size(), elemSizeBytes);

  // Determine if we should fallback to sampling
  size_t analyticalChunkSize = coalescingFactor * elemSizeBytes;
  bool shouldFallbackToSampling =
      analyticalChunkSize <= coalescingFactorSamplingFallbackThreshold;

  if (shouldFallbackToSampling || debugCoalescingInference) {
    if (shouldFallbackToSampling) {
      llvm::dbgs() << "Analytical coalescing factor below threshold, "
                      "falling back to sampling based coalescing factor...\n";
    } else {
      llvm::dbgs() << "--------------------------[CoalescingFactor]------------"
                      "--------------------\n";
      llvm::dbgs() << "Computing sampling based coalescing factor...\n";
    }

    size_t sampledCoalescingFactor = ttmlir::utils::calculateCoalescingFactor(
        memoryMap, fullShape, elemSizeBytes, gridShape.size());

    if (debugCoalescingInference) {
      if (coalescingFactor == sampledCoalescingFactor) {
        llvm::dbgs() << "  [✓] Analytical and sampled coalescing "
                        "factors MATCH = "
                     << coalescingFactor << "\n";
      } else if (coalescingFactor != sampledCoalescingFactor &&
                 sampledCoalescingFactor % coalescingFactor == 0) {
        llvm::dbgs() << "  [✓] Analytical coalescing factor is valid, but "
                        "smaller than the sampled coalescing factor!\n";
        llvm::dbgs() << "    analytical = " << coalescingFactor
                     << " vs sampled = " << sampledCoalescingFactor << "\n";
        llvm::dbgs() << "    Map: " << memoryMap << "\n";
        llvm::dbgs() << "    Shape: "
                     << ttmlir::utils::formatIterable(fullShape, "x") << "\n";
        llvm::dbgs() << "  Setting coalescing factor to fallback sampled "
                        "value = "
                     << sampledCoalescingFactor << "\n";
        coalescingFactor = sampledCoalescingFactor;
      }

      if (sampledCoalescingFactor % coalescingFactor != 0) {
        llvm::dbgs() << "  [ERROR] Analytical coalescing factor is not a "
                        "divisor of sampled coalescing factor! Generated DMA "
                        "indexing is likely incorrect!\n";
        llvm::dbgs() << "    Sampled coalescing factor: "
                     << sampledCoalescingFactor << "\n";
        llvm::dbgs() << "    Analytical coalescing factor: " << coalescingFactor
                     << "\n";
        llvm::dbgs() << "    Map: " << memoryMap << "\n";
        llvm::dbgs() << "    Shape: "
                     << ttmlir::utils::formatIterable(fullShape, "x") << "\n";
        llvm::dbgs() << "  Setting coalescing factor to fallback sampled "
                        "value = "
                     << sampledCoalescingFactor << "\n";
        coalescingFactor = sampledCoalescingFactor;
      }
    } else if (shouldFallbackToSampling) {
      // Not in debug mode but we need to fallback
      coalescingFactor = sampledCoalescingFactor;
    }

    llvm::dbgs() << "--------------------------------------------------------"
                    "-------------\n";
  }

  return coalescingFactor;
}

// Callback type for creating a single fully-indexed DMA op. Used by
// generateFullyIndexedDMAOps to abstract over DMAReadOp vs DMAWriteOp creation.
using CreateDMAOpFn = llvm::function_ref<Value(
    OpBuilder &builder, Location loc, SmallVector<Value> &remoteIndices,
    SmallVector<Value> &localIndices, size_t coalescingFactor)>;

// Generate fully-indexed DMA operations with proper coalescing.
// Returns the last DMA transaction value (for waiting).
// Handles both contiguous (single DMA) and strided (loop with guarded DMAs)
// cases.
static Value generateFullyIndexedDMAOps(
    OpBuilder &builder, Location loc, SmallVector<Value> gridIndices,
    ArrayRef<int64_t> shardShape, AffineMap remoteMemoryMap,
    AffineMap localMemoryMap, size_t coalescingFactor, size_t shardVolume,
    CreateDMAOpFn createDMAOp) {

  if (coalescingFactor == shardVolume) {
    // Fully contiguous: single DMA operation.
    SmallVector<Value> remoteIndices = gridIndices;
    SmallVector<Value> localIndices;

    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    for (size_t i = 0; i < shardShape.size(); ++i) {
      remoteIndices.push_back(zero);
      localIndices.push_back(zero);
    }

    remoteIndices =
        applyMap(builder, loc, remoteMemoryMap, remoteIndices, true);
    localIndices = applyMap(builder, loc, localMemoryMap, localIndices, false);

    return createDMAOp(builder, loc, remoteIndices, localIndices,
                       coalescingFactor);
  }

  // Strided/non-contiguous: generate loops with guarded DMAs.
  auto [lbs, ubs, steps] = getLoopBounds(builder, loc, shardShape);
  auto nullDmaTx = builder.create<NullTxOp>(loc);

  scf::LoopNest loopNest = scf::buildLoopNest(
      builder, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
      [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange iters,
          ValueRange args) {
        // Build full indices: grid indices + shard iteration indices.
        SmallVector<Value> remoteIndices =
            llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
        SmallVector<Value> localIndices = llvm::to_vector(iters);

        // Apply memory maps.
        remoteIndices = applyMap(loopBuilder, innerLoc, remoteMemoryMap,
                                 remoteIndices, true);
        localIndices = applyMap(loopBuilder, innerLoc, localMemoryMap,
                                localIndices, false);

        // Create guarded DMA operation based on coalescing factor.
        Value cfExpr = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIndexAttr(coalescingFactor));
        Value zero = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIntegerAttr(loopBuilder.getIndexType(), 0));

        // Construct guard function: flat_index(iters) % coalescingFactor == 0
        auto totalIterCount = zero;
        size_t currStride = 1;
        for (int i = iters.size() - 1; i >= 0; i--) {
          Value currStrideExpr = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIndexAttr(currStride));
          auto scaledCount =
              loopBuilder
                  .create<arith::MulIOp>(innerLoc, currStrideExpr, iters[i])
                  .getResult();
          totalIterCount =
              loopBuilder
                  .create<arith::AddIOp>(innerLoc, scaledCount, totalIterCount)
                  .getResult();
          currStride *= shardShape[i];
        }
        auto moduloIterCount =
            loopBuilder.create<arith::RemSIOp>(innerLoc, totalIterCount, cfExpr)
                .getResult();
        auto predicate = loopBuilder.create<arith::CmpIOp>(
            innerLoc, arith::CmpIPredicate::eq, moduloIterCount, zero);

        auto nulltx = loopBuilder.create<NullTxOp>(innerLoc);

        // Build guarded DMA.
        auto ifExpr = loopBuilder.create<scf::IfOp>(
            innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
            true /*addThenBlock*/, true /*addElseBlock*/);

        auto thenBuilder = ifExpr.getThenBodyBuilder();
        Value dmaTx = createDMAOp(thenBuilder, innerLoc, remoteIndices,
                                  localIndices, coalescingFactor);
        thenBuilder.create<scf::YieldOp>(innerLoc, dmaTx);

        auto elseBuilder = ifExpr.getElseBodyBuilder();
        elseBuilder.create<scf::YieldOp>(innerLoc, args[0]);

        return SmallVector<Value>{ifExpr.getResult(0)};
      });

  return loopNest.results.front();
}

static LogicalResult expandCompositeDMARead(IRRewriter &rewriter,
                                            DMAReadOp dmaRead,
                                            ValueRange expandedInputs,
                                            int64_t concatDim) {
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
    inputMemoryMaps.push_back(getMemoryMap(device, input, /*isRemote=*/true));
    inputShardShapes.emplace_back(inputDeviceShape.drop_front(gridRank).begin(),
                                  inputDeviceShape.drop_front(gridRank).end());
  }
  TT_assert(totalInputExtent <= compositeExtent);

  Value localMemref = dmaRead.getDst();
  AffineMap localMemoryMap =
      getMemoryMap(device, localMemref, /*isRemote=*/false);

  rewriter.setInsertionPoint(dmaRead);

  Location loc = dmaRead.getLoc();

  SmallVector<Value> gridIndices(dmaRead.getSrcIndices());
  SmallVector<int64_t> outputShardShape(
      mlir::cast<MemRefType>(localMemref.getType()).getShape());
  TT_assert(outputShardShape.size() == static_cast<size_t>(gridRank));

  auto emitDMAReadForInput = [&](OpBuilder &builder, Location innerLoc,
                                 ValueRange shardIters, Value globalConcatIdx,
                                 const int inputIdx, const int pieceOffset) {
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

    SmallVector<Value> remoteIndices =
        applyMap(builder, innerLoc, inputMemoryMaps[inputIdx], inputFullIdx,
                 /*isRemote=*/true);
    SmallVector<Value> localIndices = applyMap(
        builder, innerLoc, localMemoryMap, shardIters, /*isRemote=*/false);

    // One tile at a time, no coalescing.
    Value dmaTx = builder.create<DMAReadOp>(
        innerLoc, expandedInputs[inputIdx], remoteIndices, localMemref,
        localIndices, builder.getI64IntegerAttr(1));
    builder.create<DMAWaitOp>(innerLoc, dmaTx);
  };

  auto [lbs, ubs, steps] = getLoopBounds(rewriter, loc, outputShardShape);

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
        std::function<void(OpBuilder &, const int, const int)> emitIfElseChain =
            [&](OpBuilder &builder, const int inputIdx, const int startOffset) {
              // Base case: reaching the end of the chain (the last 'else').
              if (inputIdx + 1 == static_cast<int>(expandedInputs.size())) {
                emitDMAReadForInput(builder, innerLoc, shardIters,
                                    globalConcatIdx, inputIdx, startOffset);
                return;
              }

              // Recursive case:
              // - Emit DMA reads for the current input's contribution.
              // - Recurse into the 'else' branch for the next input.
              const int boundary = startOffset + inputExtents[inputIdx];
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
             "Unsupported multiple composite views in one GenericOp.");
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
  auto newGOp = rewriter.create<GenericOp>(
      gOp.getLoc(), gOp.getResultTypes(), newInputs, gOp.getOutputs(),
      gOp.getAdditionalArgs(), gOp.getGrid(), rewriter.getI64ArrayAttr({}),
      rewriter.getAffineMapArrayAttr({}), rewriter.getArrayAttr({}),
      gOp.getThreads(), newScratchInputs, gOp.getNumRegions());

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
  if (failed(expandCompositeDMARead(rewriter, clonedCompositeRead,
                                    expandedGenericInputs,
                                    compositeView.getDim()))) {
    return failure();
  }
  rewriter.replaceOp(gOp, newGOp.getResults());

  assert(compositeView->use_empty());
  rewriter.eraseOp(compositeView);

  return success();
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
    compositeView.emitError("Unexpanded composite view remaining.");
    status = failure();
  });
  return status;
}

namespace {
class D2MLowerDMAReadToFullyIndexed : public OpRewritePattern<DMAReadOp> {
public:
  D2MLowerDMAReadToFullyIndexed(MLIRContext *context,
                                bool debugCoalescingInference)
      : OpRewritePattern<DMAReadOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  LogicalResult matchAndRewrite(DMAReadOp op,
                                PatternRewriter &rewriter) const final {
    if (op.isFullyIndexed()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value remoteMemref = op.getSrc();
    Value localMemref = op.getDst();

    MemRefType remoteMemrefType = op.getSrcMemRefType();

    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(remoteMemref);
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          op, "remote memref must have a device layout");
    }

    ShapedType remoteShapedType =
        mlir::cast<ShapedType>(remoteMemref.getType());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    ttcore::DeviceAttr device = ttcore::lookupDevice(op);
    AffineMap remoteMemoryMap = getMemoryMap(device, remoteMemref, true);
    AffineMap localMemoryMap = getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    size_t shardVolume = ttmlir::utils::volume(shardShape);
    SmallVector<Value> gridIndices(op.getSrcIndices());

    Value newTx = generateFullyIndexedDMAOps(
        rewriter, loc, gridIndices, shardShape, remoteMemoryMap, localMemoryMap,
        coalescingFactor, shardVolume,
        [&](OpBuilder &b, Location l, SmallVector<Value> &remoteIdx,
            SmallVector<Value> &localIdx, size_t cf) {
          return b.create<DMAReadOp>(l, remoteMemref, remoteIdx, localMemref,
                                     localIdx, b.getI64IntegerAttr(cf));
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerDMAWriteToFullyIndexed : public OpRewritePattern<DMAWriteOp> {
public:
  D2MLowerDMAWriteToFullyIndexed(MLIRContext *context,
                                 bool debugCoalescingInference)
      : OpRewritePattern<DMAWriteOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  LogicalResult matchAndRewrite(DMAWriteOp op,
                                PatternRewriter &rewriter) const final {
    if (op.isFullyIndexed()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value localMemref = op.getSrc();
    Value dstMemref = op.getDst();

    ttcore::DeviceAttr device = ttcore::lookupDevice(op);

    if (op.isMcast()) {
      // Mcast write: local-to-local, compute local memory map and apply to
      // zero indices to get the fully-indexed form.
      AffineMap localMemoryMap = getMemoryMap(device, localMemref, false);

      MemRefType localType = op.getSrcMemRefType();
      ArrayRef<int64_t> shardShape = localType.getShape();
      size_t shardVolume = ttmlir::utils::volume(shardShape);

      SmallVector<Value> localIndices;
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        localIndices.push_back(zero);
      }
      localIndices =
          applyMap(rewriter, loc, localMemoryMap, localIndices, false);

      Value newTx = rewriter.create<DMAWriteOp>(
          loc, localMemref, localIndices, dstMemref, localIndices,
          op.getMcastStartIndex(), op.getMcastShape(), shardVolume);
      rewriter.replaceOp(op, newTx);
      return success();
    }

    // Non-mcast write: local src to remote dst.
    MemRefType remoteMemrefType = op.getDstMemRefType();

    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(dstMemref);
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          op, "remote memref must have a device layout");
    }

    ShapedType remoteShapedType = mlir::cast<ShapedType>(dstMemref.getType());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    AffineMap remoteMemoryMap = getMemoryMap(device, dstMemref, true);
    AffineMap localMemoryMap = getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        remoteMemoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);

    size_t shardVolume = ttmlir::utils::volume(shardShape);
    SmallVector<Value> gridIndices(op.getDstIndices());

    Value newTx = generateFullyIndexedDMAOps(
        rewriter, loc, gridIndices, shardShape, remoteMemoryMap, localMemoryMap,
        coalescingFactor, shardVolume,
        [&](OpBuilder &b, Location l, SmallVector<Value> &remoteIdx,
            SmallVector<Value> &localIdx, size_t cf) {
          return b.create<DMAWriteOp>(l, localMemref, localIdx, dstMemref,
                                      remoteIdx, cf);
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerDMAToFullyIndexedForm
    : public impl::D2MLowerDMAToFullyIndexedFormBase<
          D2MLowerDMAToFullyIndexedForm> {
public:
  using impl::D2MLowerDMAToFullyIndexedFormBase<
      D2MLowerDMAToFullyIndexedForm>::D2MLowerDMAToFullyIndexedFormBase;

  void runOnOperation() final {
    if (failed(expandCompositeViews(getOperation()))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<D2MLowerDMAReadToFullyIndexed>(&getContext(),
                                                debugCoalescingInference);
    patterns.add<D2MLowerDMAWriteToFullyIndexed>(&getContext(),
                                                 debugCoalescingInference);
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
