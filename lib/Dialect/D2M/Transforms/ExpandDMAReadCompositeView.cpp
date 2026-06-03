// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MEXPANDDMAREADCOMPOSITEVIEW
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

static LogicalResult expandCompositeDMAReadTiled(IRRewriter &rewriter,
                                                 DMAReadOp dmaRead,
                                                 ValueRange expandedInputs,
                                                 const int32_t concatDim) {
  TT_assert(dmaRead.isShardLevel());
  ttcore::DeviceAttr device = ttcore::lookupDevice(dmaRead);

  int64_t gridRank = static_cast<int64_t>(dmaRead.getSrcIndices().size());
  TT_assert((concatDim >= 0 && concatDim < gridRank));

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
    const int32_t concatDim, ArrayRef<int64_t> logicalSizes,
    const int64_t totalConcatShards,
    const DenseMap<unsigned, Value> &localFillMemrefs) {
  Value localDst = dmaRead.getDst();
  MemRefType dstType = mlir::cast<MemRefType>(localDst.getType());
  SmallVector<int64_t> outputShardShape(dstType.getShape());
  auto gridIndices = dmaRead.getSrcIndices();
  const int64_t gridRank = static_cast<int64_t>(gridIndices.size());
  const bool isWidthConcat = concatDim == gridRank - 1;

  int64_t coalescingFactor = 1;
  if (isWidthConcat) {
    // For width-concat's row-assembly, conservatively use the NoC DMA quantum.
    coalescingFactor = utils::getNocElementAlignmentL1(dmaRead, dstType);
  } else {
    // For height/outer-concat, the unit of data movement should be the
    // row-major stride of the concat dim (i.e. each step covers all trailing
    // dimensions shape[concatDim+1:]). The same factor applies to everything
    // because all shapes match except on the concatDim.
    for (int64_t d = concatDim + 1; d < gridRank; d++) {
      coalescingFactor *= outputShardShape[d];
    }
  }

  ttcore::DeviceAttr device = ttcore::lookupDevice(dmaRead);
  SmallVector<AffineMap> inputMemoryMaps;
  SmallVector<SmallVector<int64_t>> inputShardShapes;
  for (Value input : expandedInputs) {
    inputMemoryMaps.push_back(
        utils::getMemoryMap(device, input, /*isRemote=*/true));
    auto shape = mlir::cast<MemRefType>(input.getType()).getShape();
    inputShardShapes.emplace_back(shape.drop_front(gridRank).begin(),
                                  shape.drop_front(gridRank).end());
  }
  auto localMemoryMap =
      utils::getMemoryMap(device, localDst, /*isRemote=*/false);

  auto emitDMAReadForInput = [&](OpBuilder &builder, Location innerLoc,
                                 ValueRange shardIters, Value globalConcatIdx,
                                 const int inputIdx,
                                 const int64_t pieceOffset) {
    // Fill-rooted branch: source is the per-core local CB stamp (filled by
    // Stage G hook). Emit local-form DMAReadOp w/ same-core self-NoC read.
    // Loop-coalesce if pad span > fixed_shard.
    auto fillIt = localFillMemrefs.find(static_cast<unsigned>(inputIdx));
    if (fillIt != localFillMemrefs.end()) {
      Value localFill = fillIt->second;
      auto localFillType = mlir::cast<MemRefType>(localFill.getType());
      ArrayRef<int64_t> fillShardShape = localFillType.getShape();
      int64_t fillShardElems = 1;
      for (int64_t d : fillShardShape) {
        fillShardElems *= d;
      }
      // The total elements this branch needs to write (matches the read of
      // a "remote" branch at this coalescing factor).
      int64_t branchReadElems = coalescingFactor;

      // Local memory map for the fill (single L1 offset).
      AffineMap localFillMap =
          utils::getMemoryMap(device, localFill, /*isRemote=*/false);
      // Local memory map for dst (single L1 offset).
      AffineMap dstMap = localMemoryMap;

      // dst indices = shardIters mapped through dst map.
      SmallVector<Value> dstIndices = utils::applyMap(
          builder, innerLoc, dstMap, shardIters, /*isRemote=*/false);

      // Loop coalesce: read fillShardElems each time, advance dst offset.
      int64_t remaining = branchReadElems;
      int64_t dstOffsetElems = 0;
      Value dstBaseOffset = dstIndices[0];
      while (remaining > 0) {
        int64_t thisRead = std::min(remaining, fillShardElems);

        // src: zero-origin in the per-core fill (replicated content; any
        // offset yields same value).
        SmallVector<Value> zeroShardIdx(
            fillShardShape.size(),
            builder.create<arith::ConstantIndexOp>(innerLoc, 0));
        SmallVector<Value> srcIndices = utils::applyMap(
            builder, innerLoc, localFillMap, zeroShardIdx, /*isRemote=*/false);

        // dst index = base + dstOffsetElems.
        Value advancedDst = dstBaseOffset;
        if (dstOffsetElems != 0) {
          Value step =
              builder.create<arith::ConstantIndexOp>(innerLoc, dstOffsetElems);
          advancedDst =
              builder.create<arith::AddIOp>(innerLoc, dstBaseOffset, step);
        }
        SmallVector<Value> thisDstIndices = {advancedDst};

        Value dmaTx = builder.create<DMAReadOp>(
            innerLoc, localFill, srcIndices, localDst, thisDstIndices,
            builder.getI64IntegerAttr(thisRead));
        builder.create<DMAWaitOp>(innerLoc, dmaTx);

        remaining -= thisRead;
        dstOffsetElems += thisRead;
      }
      return;
    }

    // Locate the starting coordinates for the current coalescingFactor worth of
    // DMA read on the specified input.
    SmallVector<Value> inputFullIdx(2 * gridRank);
    for (int dim = 0; dim < gridRank; dim++) {
      // The global logical index of the DMA read starting point for the given
      // input at the given dim.
      Value globalDimIdx = nullptr;

      if (dim == concatDim) {
        // For the current iteration in the concat dim, shift back the global
        // logical index by the logical starting offset of the current input
        // piece, to get the index within the input.
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
        // The coordinates are identical for the non-concat dim.
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
      // Input shard index: logical starting index / shard extent.
      inputFullIdx[dim] =
          builder.create<arith::DivSIOp>(innerLoc, globalDimIdx, inShardExtent);
      // Input shard offset: logical starting index % shard extent.
      inputFullIdx[dim + gridRank] =
          builder.create<arith::RemSIOp>(innerLoc, globalDimIdx, inShardExtent);
    }

    SmallVector<Value> remoteIndices =
        utils::applyMap(builder, innerLoc, inputMemoryMaps[inputIdx],
                        inputFullIdx, /*isRemote=*/true);

    SmallVector<Value> localIndices = utils::applyMap(
        builder, innerLoc, localMemoryMap, shardIters, /*isRemote=*/false);

    Value dmaTx = builder.create<DMAReadOp>(
        innerLoc, expandedInputs[inputIdx], remoteIndices, localDst,
        localIndices, builder.getI64IntegerAttr(coalescingFactor));
    builder.create<DMAWaitOp>(innerLoc, dmaTx);
  };

  auto loc = dmaRead.getLoc();
  rewriter.setInsertionPoint(dmaRead);

  // N-D loop over all the output shard dimensions.
  SmallVector<Value> lbs, ubs, loopSteps;
  for (int dim = 0; dim < gridRank; dim++) {
    lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    ubs.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, outputShardShape[dim]));
    int64_t step;
    if (dim < concatDim) {
      // Outer dims are stepped normally.
      step = 1;
    } else if (dim == concatDim) {
      // Width-concat assembles rows, others read entire trailing strides.
      step = isWidthConcat ? coalescingFactor : 1;
    } else {
      // Inner dims only do 1 iteration, the coalescingFactor covers everything.
      step = outputShardShape[dim];
    }
    loopSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, step));
  }

  int64_t totalLogicalExtent = 0;
  for (int64_t n : logicalSizes) {
    totalLogicalExtent += n;
  }

  scf::buildLoopNest(
      rewriter, loc, lbs, ubs, loopSteps,
      [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange shardIters) {
        Value concatShardExtent = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIndexAttr(outputShardShape[concatDim]));
        Value baseConcatIdx = loopBuilder.create<arith::MulIOp>(
            innerLoc, gridIndices[concatDim], concatShardExtent);
        Value concatIter = shardIters[concatDim];

        // The expression for the current iter's logical, global position in the
        // concat dim.
        Value globalConcatIdx = loopBuilder.create<arith::AddIOp>(
            innerLoc, baseConcatIdx, concatIter);

        // Use an if-else chain to select the right input based on the logical
        // concat boundaries.
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
              const int64_t boundary = startOffset + logicalSizes[inputIdx];
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
        if (totalLogicalExtent <
            totalConcatShards * outputShardShape[concatDim]) {
          Value totalExtentVal = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIndexAttr(totalLogicalExtent));
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

// Walk the def chain through known view-style wrappers looking for a
// FillBufferOp. Returns null for any input that doesn't root in a fill_buffer
// (including chains terminating in plain memref::AllocOp, ttcore.stream_layout,
// composite_view, or any other op). Caller invokes this speculatively on every
// generic input, so unknown ops must NOT assert — they are simply not fills.
static FillBufferOp traceToFillBuffer(Value v) {
  while (v) {
    Operation *op = v.getDefiningOp();
    if (!op) {
      return nullptr;
    }
    if (auto fill = mlir::dyn_cast<FillBufferOp>(op)) {
      return fill;
    }
    if (auto view = mlir::dyn_cast<ViewLayoutOp>(op)) {
      v = view.getInput();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

// Pack `value` into a u32 stamp suitable for the experimental::fill_pad_cb
// LLK. bf16 → duplicate two bf16 halves into a u32. f32/i32 → raw 32-bit
// pattern.
static FailureOr<uint32_t> packValueForLLK(TypedAttr value, Type elementType) {
  if (auto floatTy = mlir::dyn_cast<FloatType>(elementType)) {
    auto floatAttr = mlir::dyn_cast<FloatAttr>(value);
    if (!floatAttr) {
      return failure();
    }
    APFloat ap = floatAttr.getValue();
    if (floatTy.isBF16()) {
      // bf16 occupies the upper 16 bits of a f32 bit-pattern.
      bool losesInfo = false;
      ap.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                 &losesInfo);
      uint32_t f32Bits = ap.bitcastToAPInt().getZExtValue();
      uint16_t bf16 = static_cast<uint16_t>(f32Bits >> 16);
      return (static_cast<uint32_t>(bf16) << 16) | bf16;
    }
    if (floatTy.isF32()) {
      return static_cast<uint32_t>(ap.bitcastToAPInt().getZExtValue());
    }
    return failure();
  }
  if (auto intTy = mlir::dyn_cast<IntegerType>(elementType)) {
    auto intAttr = mlir::dyn_cast<IntegerAttr>(value);
    if (!intAttr) {
      return failure();
    }
    if (intTy.getWidth() == 32) {
      return static_cast<uint32_t>(intAttr.getValue().getSExtValue());
    }
    return failure();
  }
  return failure();
}

// Emit `reserve` + `fill_pad_cb` + `push` at the top of the datamovement
// region of `genericOp` for each input traceable to a FillBufferOp.
//
// fill_buffer has been reblocked to the consumer grid by GridSelection
// (Stage C), so each core in the grid has its own L1 stamp at the same
// lock-step address. fill_pad_cb runs on each core's local CB.
//
// `localFillMemrefs` (output) maps the fill input's operand index to the
// per-core local memref returned by ReserveOp. Stage F uses these as the
// src of local-form DMAReadOps so OOB reads stay same-core (no NoC traffic
// to remote fill data).
static LogicalResult emitFillPadCBForFillBufferInputs(
    IRRewriter &rewriter, GenericOp genericOp,
    DenseMap<unsigned, Value> *localFillMemrefs = nullptr) {
  // Collect (operandIdx, FillBufferOp) for inputs that root in a fill_buffer.
  SmallVector<std::pair<unsigned, FillBufferOp>> fillInputs;
  for (auto [idx, input] : llvm::enumerate(genericOp.getInputs())) {
    if (FillBufferOp fill = traceToFillBuffer(input)) {
      fillInputs.emplace_back(static_cast<unsigned>(idx), fill);
    }
  }
  if (fillInputs.empty()) {
    return success();
  }

  Location loc = genericOp.getLoc();

  // For each region marked datamovement, prepend the reserve/fill_pad_cb/push
  // triples.
  for (auto [regionIdx, region] : llvm::enumerate(genericOp.getRegions())) {
    auto threadAttr = mlir::cast<ThreadAttr>(genericOp.getThreads()[regionIdx]);
    if (threadAttr.getThreadType() != ThreadType::Datamovement &&
        threadAttr.getThreadType() != ThreadType::Unified) {
      continue;
    }
    if (region.empty()) {
      continue;
    }
    Block &block = region.front();
    rewriter.setInsertionPointToStart(&block);

    for (auto &[operandIdx, fill] : fillInputs) {
      auto memrefType = mlir::cast<MemRefType>(fill.getResult().getType());
      Type elementType = memrefType.getElementType();
      Type llkElementType = elementType;
      if (auto tileTy = mlir::dyn_cast<ttcore::TileType>(elementType)) {
        llkElementType = tileTy.getElementType();
      }

      auto valueAttr = mlir::cast<TypedAttr>(fill.getValueAttr());
      auto packed = packValueForLLK(valueAttr, llkElementType);
      if (failed(packed)) {
        return fill.emitOpError("unsupported pad value/dtype combo for "
                                "experimental::fill_pad_cb");
      }

      // num_bytes = per-core L1 footprint of the fill. The memref shape after
      // Stage C is `grid x fixed_shard`. We only want the `fixed_shard`
      // portion since fill_pad_cb writes ONE core's CB. Use the op's
      // `fixed_shard` attr directly.
      int64_t elementCount = 1;
      for (int64_t d : fill.getFixedShard()) {
        elementCount *= d;
      }
      int64_t bytesPerElement;
      if (auto tileTy = mlir::dyn_cast<ttcore::TileType>(elementType)) {
        bytesPerElement =
            tileTy.getSizeBytes() / (tileTy.getHeight() * tileTy.getWidth());
        elementCount *= tileTy.getHeight() * tileTy.getWidth();
      } else {
        bytesPerElement =
            llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8);
      }
      int64_t numBytes = elementCount * bytesPerElement;

      // Per-core L1 memref the CB is bound to. Strip the device (shard)
      // layout: ReserveOp's result type is a plain local L1 memref of the
      // per-core shard shape — exactly what local-form DMAReadOp wants.
      MemRefType perCoreLocalType = MemRefType::get(
          memrefType.getShape().drop_front(/*gridRank=*/2),
          memrefType.getElementType(),
          /*layout=*/MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
      auto cbType = CBType::get(rewriter.getContext(), perCoreLocalType);
      Value cb = rewriter.create<GetCBOp>(
          loc, cbType, rewriter.getI64IntegerAttr(operandIdx),
          /*resolution_stage=*/nullptr);
      rewriter.create<ReserveOp>(loc, perCoreLocalType, cb).getResult();
      rewriter.create<FillPadCBOp>(
          loc, cb, rewriter.getI32IntegerAttr(static_cast<int32_t>(*packed)),
          rewriter.getI32IntegerAttr(static_cast<int32_t>(numBytes)),
          TypeAttr::get(llkElementType));
      rewriter.create<PushOp>(loc, cb);
      // Same-thread producer + consumer: must cb_wait_front before reading
      // since fill_pad_cb writes at write_ptr and pad-branch dma_reads source
      // from read_ptr. Use WaitOp's result as the local memref for downstream
      // dma_reads — its lowering emits get_read_ptr (correct post-wait).
      Value localFill =
          rewriter.create<WaitOp>(loc, perCoreLocalType, cb).getResult();
      if (localFillMemrefs) {
        (*localFillMemrefs)[operandIdx] = localFill;
      }
    }
  }
  return success();
}

static LogicalResult expandCompositeViewsInGeneric(IRRewriter &rewriter,
                                                   GenericOp gOp) {
  CompositeViewOp compositeView = nullptr;
  gOp.walk([&](DMAReadOp dmaRead) {
    auto maybeCompositeView = dmaRead.getSrc().getDefiningOp<CompositeViewOp>();
    if (maybeCompositeView) {
      TT_assertv(
          compositeView == nullptr,
          "Unsupported multiple composite views in one GenericOp (#7600).");
      compositeView = maybeCompositeView;
    }
  });
  TT_assert(compositeView != nullptr);

  SmallVector<Value> compositeInputs = compositeView.getCompositeInputs();
  TT_assert(compositeInputs.size() > 1u);

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

  rewriter.setInsertionPoint(gOp);
  // Passing empty block_factors/indexing_maps/iterator_types.
  TT_assert(gOp.isExplicitDatamovementForm());
  auto newGOp = rewriter.create<GenericOp>(
      gOp.getLoc(), gOp.getResultTypes(), newInputs, gOp.getOutputs(),
      gOp.getAdditionalArgs(), gOp.getGrid(), rewriter.getI64ArrayAttr({}),
      rewriter.getAffineMapArrayAttr({}), rewriter.getArrayAttr({}),
      gOp.getThreads(), gOp.getFabricConnectionConfigAttr(),
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
    newBlock->walk([&](GetCBOp getCBOp) {
      getCBOp.setCbOperandIdx(getCBOp.getCbOperandIdx() + extraNumInputs);
    });
  }

  // Step 3: lower the composite DMA read in the new GenericOp.
  DMAReadOp clonedCompositeRead = nullptr;
  newGOp.walk([&](DMAReadOp dmaRead) {
    auto maybeCompositeView = dmaRead.getSrc().getDefiningOp<CompositeViewOp>();
    if (maybeCompositeView) {
      TT_assert(dmaRead.getSrc() == compositeView.getResult());
      TT_assert(clonedCompositeRead == nullptr);
      clonedCompositeRead = dmaRead;
    }
  });
  TT_assert(clonedCompositeRead != nullptr);

  auto expandedGenericInputs =
      newGOp.getInputs().slice(compositeOperandIdx, compositeInputs.size());
  const bool isTiled = mlir::isa<ttcore::TileType>(
      mlir::cast<MemRefType>(compositeView.getResult().getType())
          .getElementType());

  const int32_t compositeDim = compositeView.getDim();

  // Stage G first: emit reserve+fill_pad_cb+push for every fill input at the
  // top of the DM region, capturing each fill's per-core local memref. These
  // are then used as local-form DMAReadOp sources in the per-stick expansion
  // (Stage F) below. Order matters — Stage G must precede expansion so the
  // local memrefs exist when expansion emits dma_reads.
  DenseMap<unsigned, Value> localFillMemrefsAbs;
  if (failed(emitFillPadCBForFillBufferInputs(rewriter, newGOp,
                                              &localFillMemrefsAbs))) {
    return failure();
  }
  // Translate absolute new-generic operand indices to slice-relative indices
  // (expandedInputs slice starts at compositeOperandIdx).
  DenseMap<unsigned, Value> localFillMemrefs;
  for (auto &[absIdx, val] : localFillMemrefsAbs) {
    if (static_cast<int64_t>(absIdx) >= compositeOperandIdx &&
        static_cast<int64_t>(absIdx) <
            compositeOperandIdx +
                static_cast<int64_t>(compositeInputs.size())) {
      localFillMemrefs[absIdx - static_cast<unsigned>(compositeOperandIdx)] =
          val;
    }
  }

  LogicalResult result = success();
  if (isTiled) {
    result = expandCompositeDMAReadTiled(rewriter, clonedCompositeRead,
                                         expandedGenericInputs, compositeDim);
  } else {
    // This value is only used to determine if we need the guard to skip the
    // padded portions of the output shards.
    const int64_t totalConcatShards =
        mlir::cast<MemRefType>(compositeView.getResult().getType())
            .getShape()[compositeDim];
    result = expandCompositeDMAReadRowMajor(
        rewriter, clonedCompositeRead, expandedGenericInputs, compositeDim,
        compositeView.getLogicalSizes().value(), totalConcatShards,
        localFillMemrefs);
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
