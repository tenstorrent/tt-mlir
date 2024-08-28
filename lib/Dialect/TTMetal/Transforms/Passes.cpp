// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>

#include "ttmlir/Dialect/TT/Utils/PhysicalCoreCoord.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

// This routine walks the SSA value chain to find the address of the value.
// It runs into and gets the address from one of the following:
//   - ttir::AllocOp: The address is the address of the allocation is embedded
//   in the op attributes.
//   - func::FuncOp Argument: The address is taken from ArgumentAllocationAttr.
//
// This routine is used to lookup the address of tensors for address generation
// and for CB creation.
static uint64_t lookupAddress(Value value) {
  auto blockArg = mlir::dyn_cast<BlockArgument>(value);
  if (blockArg) {
    auto funcOp = blockArg.getOwner()->getParentOp();
    if (mlir::isa<func::FuncOp>(funcOp)) {
      auto argAlloc = mlir::cast<ArgumentAllocationAttr>(
          mlir::cast<ArrayAttr>(funcOp->getDiscardableAttr(
              ArgumentAllocationAttr::name))[blockArg.getArgNumber()]);
      return argAlloc.getAddress();
    }
  }

  auto *op = value.getDefiningOp();
  if (!op) {
    return 0;
  }

  while (isa<DestinationStyleOpInterface>(op)) {
    assert(op->getResults().size() == 1);
    auto dps = cast<DestinationStyleOpInterface>(op);
    assert(dps.getNumDpsInits() == 1);
    auto *opOperand = dps.getDpsInitOperand(0);
    value = opOperand->get();
    op = value.getDefiningOp();
  }

  auto allocOp = dyn_cast<ttir::AllocOp>(op);
  if (!allocOp) {
    return 0;
  }
  return allocOp.getAddress();
}

class TTIRToTTMetalLayoutRewriter : public OpRewritePattern<ttir::ToLayoutOp> {
public:
  using OpRewritePattern<ttir::ToLayoutOp>::OpRewritePattern;

  struct NocRead {
    PhysicalCoreCoord srcCoord;
    std::int64_t srcOffset = 0;
    std::int64_t dstOffset = 0;
    std::int64_t size = 0;

    NocRead(PhysicalCoreCoord srcCoord, std::int64_t srcOffset,
            std::int64_t dstOffset, std::int64_t size)
        : srcCoord(srcCoord), srcOffset(srcOffset), dstOffset(dstOffset),
          size(size) {}

    bool isContiguous(PhysicalCoreCoord nextCoord, std::int64_t nextSrcOffset,
                      std::int64_t nextDstOffset) const {
      return (nextCoord == srcCoord) && (nextSrcOffset == srcOffset + size) &&
             (nextDstOffset == dstOffset + size);
    }
  };

  // This routine calculates the data movement for a tensor layout change by
  // tracing the walk order of the src and dst affine maps.  The sample routine
  // is just a helper function that iterates over the tensor shape and calls the
  // lambda with the current index.  It walks the shape in innermost-major
  // order. It also coalesces the noc transactions.
  //
  // The return value is a map of destination physical cores where each core has
  // an associated list of noc reads to be performed.
  llvm::MapVector<PhysicalCoreCoord, SmallVector<NocRead>>
  calculateDataMovement(ArrayRef<int64_t> tensorShape, int64_t elemSize,
                        AffineMap src, AffineMap dst) const {

    // For now it's just a simple pull model, but eventually we want to leverage
    // both NoCs and the both read and write
    llvm::MapVector<PhysicalCoreCoord, SmallVector<NocRead>> dst2srcMap;
    assert(src.getNumResults() == 4);
    assert(dst.getNumResults() == 4);

    ::ttmlir::utils::sample(tensorShape, [&dst2srcMap, src, dst, elemSize](
                                             ArrayRef<std::int64_t> index) {
      SmallVector<int64_t> srcResults = src.compose(index);
      SmallVector<int64_t> dstResults = dst.compose(index);

      assert(srcResults.size() == src.getNumResults());
      assert(dstResults.size() == dst.getNumResults());
      PhysicalCoreCoord srcCoord(srcResults);
      PhysicalCoreCoord dstCoord(dstResults);
      std::int64_t srcOffset = srcResults.back() * elemSize;
      std::int64_t dstOffset = dstResults.back() * elemSize;
      SmallVector<NocRead> &srcs = dst2srcMap[dstCoord];
      if (not srcs.empty() &&
          srcs.back().isContiguous(srcCoord, srcOffset, dstOffset)) {
        srcs.back().size += elemSize;
      } else {
        srcs.push_back(NocRead(srcCoord, srcOffset, dstOffset, elemSize));
      }
    });

    return dst2srcMap;
  }

  void buildNocAsyncRead(mlir::Location loc, std::int64_t inputBaseAddress,
                         std::int64_t outputBaseAddress,
                         std::int64_t addressAlignment, NocRead read,
                         PhysicalCoreCoordMapping const &physicalCoordMapping,
                         mlir::OpBuilder &nocBuilder) const {
    assert(read.srcOffset % addressAlignment == 0);
    assert(read.dstOffset % addressAlignment == 0);
    assert(read.size % addressAlignment == 0);
    auto [yPhys, xPhys] = physicalCoordMapping[read.srcCoord];
    auto y = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(yPhys));
    auto x = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(xPhys));
    auto srcOffset = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(inputBaseAddress + read.srcOffset));
    auto srcRemoteNocAddr =
        nocBuilder.create<ttkernel::GetNocAddrOp>(loc, x, y, srcOffset);
    auto dstLocalL1Addr = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(outputBaseAddress + read.dstOffset));
    auto size = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(read.size));
    nocBuilder.create<ttkernel::NocAsyncReadOp>(loc, srcRemoteNocAddr,
                                                dstLocalL1Addr, size);
  }

  LogicalResult relayout(ttir::ToLayoutOp op, PatternRewriter &rewriter) const {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    auto inputLayout = mlir::cast<tt::LayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::LayoutAttr>(outputTy.getEncoding());
    assert(inputLayout.getMemorySpace() == outputLayout.getMemorySpace());
    tt::DeviceAttr device = op.getDevice();
    assert(device);
    tt::SystemDescAttr systemDesc = op.getSystemDesc();
    assert(systemDesc);
    auto addressAlignment = systemDesc.getAddressAlignBytes(
        /*inputLayout.getMemorySpace() issue #407*/);
    assert(inputLayout.getPhysicalShape(inputTy.getShape()) ==
               outputLayout.getPhysicalShape(outputTy.getShape()) &&
           "Physical shapes must match for now");

    // Shape that will be traversed when calculating the data movement between
    // layouts.
    SmallVector<int64_t> inputShape =
        inputLayout.isTiled() ? inputLayout.getTiledShape(inputTy.getShape())
                              : SmallVector<int64_t>(inputTy.getShape());
    SmallVector<int64_t> outputShape =
        outputLayout.isTiled() ? outputLayout.getTiledShape(outputTy.getShape())
                               : SmallVector<int64_t>(outputTy.getShape());

    assert(inputShape == outputShape && "Shapes must be identical");

    // If tiles are reblocked, the linear maps must be identical.
    assert(!(inputLayout.isTiled() && outputLayout.isTiled()) ||
           inputLayout.getLinear() == outputLayout.getLinear());

    // When the layouts are tiled, the linear maps are the identity maps.
    // Reasoning behind this is that since it's already ensured that scalar
    // linear maps identical, tensors can be viewed simply as bags of tiles
    // omitting any affine complexity.
    AffineMap inputLinearMap = inputLayout.isTiled()
                                   ? inputLayout.getIdentityTileLinearMap()
                                   : inputLayout.getLinear();
    AffineMap outputLinearMap = outputLayout.isTiled()
                                    ? outputLayout.getIdentityTileLinearMap()
                                    : outputLayout.getLinear();

    AffineMap src =
        inputLayout.projectOnto(inputLinearMap, inputShape, device.getGrid());

    AffineMap dst = outputLayout.projectOnto(outputLinearMap, outputShape,
                                             device.getGrid());

    auto dm = calculateDataMovement(
        inputShape, inputLayout.getElementSizeBytes(), src, dst);

    auto noc0Attr =
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(ttkernel::ThreadType::Noc0);
    SmallVector<Attribute> threadTypes(dm.size(), noc0Attr);
    SmallVector<Attribute> coreRanges;
    coreRanges.reserve(dm.size());
    for (auto [dstCoord, srcs] : dm) {
      SmallVector<int64_t> offset = {dstCoord.y, dstCoord.x};
      SmallVector<int64_t> size = {1, 1};
      coreRanges.push_back(
          rewriter.getAttr<ttmetal::CoreRangeAttr>(offset, size));
    };

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), SmallVector<Type>({outputTy}),
        SmallVector<Value>({op.getInput()}),
        SmallVector<Value>({op.getOutput()}), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes), threadTypes.size());

    int i = 0;
    PhysicalCoreCoordMapping physicalCoordMapping(systemDesc.getChipDescs());
    std::int64_t inputBaseAddress = lookupAddress(op.getInput());
    std::int64_t outputBaseAddress = lookupAddress(op.getOutput());
    assert(inputBaseAddress);
    assert(outputBaseAddress);
    assert(inputBaseAddress % addressAlignment == 0);
    assert(outputBaseAddress % addressAlignment == 0);
    for (auto [dstCoord, srcs] : dm) {
      Block *nocBlock = rewriter.createBlock(&metalDispatch.getRegion(i++));
      OpBuilder nocBuilder(nocBlock, nocBlock->begin());
      for (auto s : srcs) {
        buildNocAsyncRead(op.getLoc(), inputBaseAddress, outputBaseAddress,
                          addressAlignment, s, physicalCoordMapping,
                          nocBuilder);
      }
      nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(op.getLoc());
      nocBuilder.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    }

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }

  LogicalResult reformat(ttir::ToLayoutOp op, PatternRewriter &rewriter) const {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    auto inputLayout = mlir::cast<tt::LayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::LayoutAttr>(outputTy.getEncoding());
    bool shouldTilize = not inputLayout.isTiled() && outputLayout.isTiled();
    bool shouldUntilize = inputLayout.isTiled() && not outputLayout.isTiled();
    assert(shouldTilize ^ shouldUntilize);
    assert(inputLayout.getGrid() == outputLayout.getGrid());

    auto tensixAttr = rewriter.getAttr<ttkernel::ThreadTypeAttr>(
        ttkernel::ThreadType::Tensix);
    SmallVector<Attribute> threadTypes = {tensixAttr};

    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(inputLayout.getGrid()),
    };

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), SmallVector<Type>({outputTy}),
        SmallVector<Value>({op.getInput()}),
        SmallVector<Value>({op.getOutput()}), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes), threadTypes.size());

    std::int64_t inputBaseAddress = lookupAddress(op.getInput());
    std::int64_t outputBaseAddress = lookupAddress(op.getOutput());
    Block *tensixBlock = rewriter.createBlock(&metalDispatch.getRegion(0));
    OpBuilder tensixBuilder(tensixBlock, tensixBlock->begin());
    uint64_t pageSize = inputLayout.isTiled()
                            ? inputLayout.getElementSizeBytes()
                            : outputLayout.getElementSizeBytes();
    Type inputCBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::In0, inputBaseAddress,
        mlir::cast<MemRefType>(inputLayout.getMemref()), pageSize,
        /*num_buffers*/ 1);
    Type outputCBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::Out0, outputBaseAddress,
        mlir::cast<MemRefType>(outputLayout.getMemref()), pageSize,
        /*num_buffers*/ 1);
    tensixBlock->addArgument(inputCBTy, op.getLoc());
    tensixBlock->addArgument(outputCBTy, op.getLoc());

    llvm::ArrayRef<int64_t> shardTileShape =
        (shouldTilize ? outputLayout.getMemref().getShape()
                      : inputLayout.getMemref().getShape());

    assert(shardTileShape.size() >= 2 && "Tile shape rank must be at least 2");

    // How many tiles should kernel tilize in one block.
    arith::ConstantOp numTilesPerBlock =
        tensixBuilder.create<arith::ConstantOp>(
            op.getLoc(), tensixBuilder.getI32Type(),
            tensixBuilder.getI32IntegerAttr(shardTileShape.back()));

    if (shouldTilize) {
      tensixBuilder.create<ttkernel::TilizeInitOp>(
          op.getLoc(), tensixBlock->getArgument(0), numTilesPerBlock,
          tensixBlock->getArgument(1));
    } else {
      tensixBuilder.create<ttkernel::UntilizeInitOp>(
          op.getLoc(), tensixBlock->getArgument(0),
          tensixBlock->getArgument(1));
    }

    uint64_t shardTileVolume = 1;
    for (int64_t dim : shardTileShape) {
      shardTileVolume *= dim;
    }
    const uint64_t numBlocks = shardTileVolume / shardTileShape.back();

    for (uint iblock = 0; iblock < numBlocks; ++iblock) {
      if (shouldTilize) {
        tensixBuilder.create<ttkernel::TilizeBlockOp>(
            op.getLoc(), tensixBlock->getArgument(0), numTilesPerBlock,
            tensixBlock->getArgument(1));
      } else {
        tensixBuilder.create<ttkernel::UntilizeBlockOp>(
            op.getLoc(), tensixBlock->getArgument(0), numTilesPerBlock,
            tensixBlock->getArgument(1));
      }
      tensixBuilder.create<ttkernel::CBPopFrontOp>(
          op.getLoc(), tensixBlock->getArgument(0), numTilesPerBlock);
      tensixBuilder.create<ttkernel::CBPushBackOp>(
          op.getLoc(), tensixBlock->getArgument(1), numTilesPerBlock);
    }

    tensixBuilder.create<ttkernel::ReturnOp>(op.getLoc());

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }

  LogicalResult matchAndRewrite(ttir::ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    if (not inputTy.getEncoding() || not outputTy.getEncoding()) {
      return failure();
    }
    assert(inputTy.getShape() == outputTy.getShape());
    assert(mlir::isa<tt::LayoutAttr>(inputTy.getEncoding()));
    assert(mlir::isa<tt::LayoutAttr>(outputTy.getEncoding()));
    auto inputLayout = mlir::cast<tt::LayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::LayoutAttr>(outputTy.getEncoding());

    auto [isLayoutChange, isGridChange, isFormatChange, isMemorySpaceChange,
          isMemoryLayoutChange] = op.compoundComponents();
    bool isCompound =
        (static_cast<int>(isLayoutChange) + static_cast<int>(isGridChange) +
         static_cast<int>(isFormatChange) +
         static_cast<int>(isMemorySpaceChange) +
         static_cast<int>(isMemoryLayoutChange)) > 1;
    assert(!isCompound && "Only one change is allowed");
    assert(!isMemoryLayoutChange &&
           "Tensor memory layout shouldn't change in metal backend");
    if (isMemorySpaceChange) {
      if (inputLayout.isSystemMemorySpace()) {
        assert(outputLayout.isDeviceMemorySpace());
        rewriter.replaceOpWithNewOp<ttmetal::HostWriteOp>(
            op, outputTy, op.getInput(), op.getOutput());
      } else if (outputLayout.isSystemMemorySpace()) {
        assert(inputLayout.isDeviceMemorySpace());
        rewriter.replaceOpWithNewOp<ttmetal::HostReadOp>(
            op, outputTy, op.getInput(), op.getOutput());
      } else {
        assert(false && "L1 <-> DRAM not supported yet");
      }
    } else if (isLayoutChange || isGridChange) {
      return relayout(op, rewriter);
    } else if (isFormatChange) {
      return reformat(op, rewriter);
    } else {
      assert(isMemoryLayoutChange);
      assert(false && "Changing tensor memory layout not supported yet");
    }

    return failure();
  }
};

class TTIRToTTMetalKernelRewriter : public OpRewritePattern<ttir::KernelOp> {
public:
  using OpRewritePattern<ttir::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::KernelOp op,
                                PatternRewriter &rewriter) const final {
    if (not op->use_empty()) {
      return failure();
    }
    rewriter.create<ttkernel::BuiltinOp>(op.getLoc(), op.getOpAttr(),
                                         op.getKindAttr(), op.getOperands());
    op->dropAllUses();
    rewriter.eraseOp(op);
    return success();
  }
};

class TTIRToTTMetalDispatchRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  bool hasUnloweredTTIRKernel(ttir::GenericOp op) const {
    bool exists = false;
    op->getRegion(0).walk([&exists](Operation *op) {
      if (isa<ttir::KernelOp>(op)) {
        exists = true;
      }
    });
    return exists;
  }

  ttkernel::CBPort getPort(unsigned argNumber,
                           std::int64_t numDPSInputs) const {
    std::int64_t operandInOutPartition = numDPSInputs;
    std::uint32_t portIdx = 0;
    if (argNumber < static_cast<unsigned>(operandInOutPartition)) {
      assert(argNumber < 8 && "Exceeds max 8 input ports");
      portIdx = ttmlir::utils::enum_as_int(ttkernel::CBPort::In0) + argNumber;
    } else {
      assert((argNumber - operandInOutPartition) < 8 &&
             "Exceeds max 8 output ports");
      portIdx = ttmlir::utils::enum_as_int(ttkernel::CBPort::Out0) +
                (argNumber - operandInOutPartition);
    }
    std::optional<ttkernel::CBPort> maybePort =
        ttkernel::symbolizeCBPort(portIdx);
    assert(maybePort.has_value() && "Expected legal port value");
    return maybePort.value();
  }

  // This routine evaluates the memref's affine map with it's shape to return a
  // single result affine map, e.g.:
  //  - Given: shape{2, 4} and affine_map<(d0, d1) -> (d0, d1)>
  //  - Becomes: affine_map<(d0, d1) -> (d0 * 4 + d1)
  // This is useful for evaluating iterator increment steps between each loop.
  AffineMap getAffineIterator(MemRefType memref) const {
    ArrayRef<std::int64_t> shape = memref.getShape();
    SmallVector<std::int64_t> physShape =
        memref.getLayout().getAffineMap().compose(shape);

    mlir::AffineExpr resultExpr = getAffineConstantExpr(0, memref.getContext());
    int volume = 1;
    for (int i = static_cast<int>(physShape.size()) - 1; i >= 0; i--) {
      mlir::AffineExpr dimExpr = getAffineDimExpr(i, memref.getContext());
      mlir::AffineExpr strideExpr =
          getAffineConstantExpr(volume, memref.getContext());
      resultExpr = dimExpr * strideExpr + resultExpr;
      volume *= physShape[i];
    }
    return AffineMap::get(physShape.size(), 0, resultExpr, memref.getContext());
  }

  Value i32(std::int32_t value, OpBuilder &builder) const {
    return builder
        .create<arith::ConstantOp>(builder.getUnknownLoc(),
                                   builder.getI32Type(),
                                   builder.getI32IntegerAttr(value))
        .getResult();
  }

  struct LoopNest {
    SmallVector<scf::ForOp> loops;
    SmallVector<Region *> loopRegions;
    SmallVector<unsigned> blockArgIteratorMapping;
  };

  // Creates a loop nest that walks the input/output operand tiles in the shard.
  // Converts this:
  //   %0 = arith.add(%1, %2) : tensor<2x4x!tile<f32>>, tensor<2x4x!tile<f32>>
  //          -> tensor<2x4x!tile<f32>>
  // Into this:
  //   for (%i0 = 0; %i0 < 2; %i0++)
  //     for (%i1 = 0; %i1 < 4; %i1++)
  //       %ii = %i0 * 4 + %i1
  //       %3 = ttkernel.add_tiles(%1, %2, %ii, %ii)
  LoopNest createLoopNest(ArrayRef<BlockArgument> blockArguments,
                          std::int64_t numDPSInputs, OpBuilder &builder) const {
    Value output = blockArguments[numDPSInputs];
    ttkernel::CBType outputTy = mlir::cast<ttkernel::CBType>(output.getType());
    MemRefType outputMemref = outputTy.getMemref();
    ArrayRef<std::int64_t> outputShape = outputMemref.getShape();
    AffineMap outputAffineMap = outputMemref.getLayout().getAffineMap();

    // Uniquify the iterators, i.e. operands that have identical access pattern
    // can be shared.
    llvm::MapVector<AffineMap, Value> iteratorMaps;
    auto getOrInsertIterator = [&iteratorMaps, &builder,
                                this](AffineMap affineIterator) {
      if (iteratorMaps.find(affineIterator) == iteratorMaps.end()) {
        iteratorMaps[affineIterator] = i32(0, builder);
      }
      return iteratorMaps[affineIterator];
    };

    // Map block arguments to their respective unique iterators Values
    SmallVector<Value> iterators;
    iterators.resize(blockArguments.size());
    for (BlockArgument operand : blockArguments) {
      auto cbType = mlir::cast<ttkernel::CBType>(operand.getType());
      AffineMap affineIterator = getAffineIterator(cbType.getMemref());
      assert(affineIterator.getNumDims() == outputAffineMap.getNumDims());
      iterators[operand.getArgNumber()] = getOrInsertIterator(affineIterator);
    }

    // Map block arguments to their respective unique iterator offset in the
    // map. This is needed by the caller to know how to wire the iterators into
    // the ttkernel tile operation.
    SmallVector<unsigned> blockArgIteratorMapping;
    blockArgIteratorMapping.resize(blockArguments.size());
    for (BlockArgument operand : blockArguments) {
      auto cbType = mlir::cast<ttkernel::CBType>(operand.getType());
      AffineMap affineIterator = getAffineIterator(cbType.getMemref());
      auto match = iteratorMaps.find(affineIterator);
      assert(match != iteratorMaps.end());
      blockArgIteratorMapping[operand.getArgNumber()] =
          std::distance(iteratorMaps.begin(), match);
    }

    // Convert the map data structure into a vector because it's easier to work
    // with when creating the loop nest below.
    SmallVector<Value> uniqueIterators;
    for (auto [affineMap, iterator] : iteratorMaps) {
      uniqueIterators.push_back(iterator);
    }

    // Create loop nest
    // The loop nest is created from outermost to innermost. The innermost loop
    // is special in the sense that it implements the actual iterator increment
    // and the tile operation. The outer loops are responsible for fixing up the
    // iterator offset for the current dimension if there was a stride or we're
    // accessing the tiles in non-row-major order.
    //
    // iterators are just ints that correspond to absolute offsets in the CB.
    // They walk the order defined by the affine map associated with the memref.
    LoopNest loopNest;
    loopNest.blockArgIteratorMapping = blockArgIteratorMapping;
    SmallVector<scf::ForOp> loops;
    SmallVector<Region *> loopRegions;
    SmallVector<SmallVector<Value>> iteratorsNest = {uniqueIterators};
    for (unsigned dim = 0; dim < outputAffineMap.getNumDims(); ++dim) {
      OpBuilder regionBuilder(builder);
      if (!loopNest.loopRegions.empty()) {
        regionBuilder = OpBuilder(loopNest.loopRegions.back());
      }
      // Loop variables, these are decoupled from the iterators
      Value lowerBound = i32(0, regionBuilder);
      Value upperBound = i32(outputShape[dim], regionBuilder);
      Value loopStep = i32(1, regionBuilder);
      scf::ForOp forOp = regionBuilder.create<scf::ForOp>(
          output.getLoc(), lowerBound, upperBound, loopStep,
          iteratorsNest.back());
      loopNest.loops.push_back(forOp);

      SmallVector<std::int64_t> innerIndexStep(outputAffineMap.getNumDims(), 0);
      innerIndexStep[dim] = 1;

      bool innerLoop = dim == (outputAffineMap.getNumDims() - 1);
      if (innerLoop) {
        OpBuilder innerLoopRegion(loopNest.loops.back().getRegion());
        SmallVector<Value> innerIndices;
        int i = 0;
        for (auto [affineMap, iterator] : iteratorMaps) {
          // Calculate how far a single step in the inner dim is.
          SmallVector<std::int64_t> innerOffset =
              affineMap.compose(innerIndexStep);
          assert(innerOffset.size() == 1);
          innerIndices.push_back(innerLoopRegion.create<arith::AddIOp>(
              output.getLoc(), forOp.getRegionIterArg(i),
              i32(innerOffset[0], innerLoopRegion)));
          ++i;
        }
        innerLoopRegion.create<scf::YieldOp>(output.getLoc(), innerIndices);
      }

      // Backpedal and adjust the iterator offset for the current dimension.
      if (dim > 0) {
        SmallVector<Value> outerIndices;
        SmallVector<std::int64_t> outerIndexStep(outputAffineMap.getNumDims(),
                                                 0);
        outerIndexStep[dim - 1] = 1;
        int i = 0;
        for (auto [affineMap, iterator] : iteratorMaps) {
          // Calculate how far a single step in the inner dim is.
          SmallVector<std::int64_t> innerOffset =
              affineMap.compose(innerIndexStep);
          assert(innerOffset.size() == 1);
          // Calculate how far a single step in the outer dim is.
          SmallVector<std::int64_t> outerOffset =
              affineMap.compose(outerIndexStep);
          assert(outerOffset.size() == 1);
          // Multiply by the number of steps that the inner loop took.
          // FIXME: test this for higher dims
          std::int64_t offset =
              outerOffset[0] - innerOffset[0] * outputShape[dim];
          outerIndices.push_back(regionBuilder.create<arith::AddIOp>(
              output.getLoc(), forOp.getResult(i), i32(offset, regionBuilder)));
          ++i;
        }
        regionBuilder.create<scf::YieldOp>(output.getLoc(), outerIndices);
      }

      loopNest.loopRegions.push_back(&loopNest.loops.back().getRegion());
      iteratorsNest.emplace_back(forOp.getRegionIterArgs());
    }

    return loopNest;
  }

  // Convert arith and math dialect operations into ttkernel init tile
  // operations. HLK requires the FPU to be initialized before any tile ops get
  // executed.  We separate the init tile operation from the actual tile
  // operation so that we can hoist the init tile operation outside of the loop
  // nest.
  void convertComputeInitOp(Operation &op, ArrayRef<BlockArgument> cbOperands,
                            std::int64_t numDpsInputs,
                            OpBuilder &builder) const {
    SmallVector<std::int64_t> operandIndices;
    for (OpOperand &operand : op.getOpOperands()) {
      operandIndices.push_back(operand.getOperandNumber());
    }
    if (mlir::isa<arith::AddFOp>(op)) {
      builder.create<ttkernel::BinaryOpInitCommonOp>(
          op.getLoc(), cbOperands[operandIndices[0]],
          cbOperands[operandIndices[1]], cbOperands[numDpsInputs]);
      builder.create<ttkernel::AddTilesInitOp>(op.getLoc(),
                                               cbOperands[operandIndices[0]],
                                               cbOperands[operandIndices[1]]);
    } else if (mlir::isa<arith::MulFOp>(op)) {
      builder.create<ttkernel::BinaryOpInitCommonOp>(
          op.getLoc(), cbOperands[operandIndices[0]],
          cbOperands[operandIndices[1]], cbOperands[numDpsInputs]);
      builder.create<ttkernel::MulTilesInitOp>(op.getLoc(),
                                               cbOperands[operandIndices[0]],
                                               cbOperands[operandIndices[1]]);
    } else {
      llvm_unreachable("Unhandled conversion");
    }
  }

  // Convert arith and math dialect operations into ttkernel tile operations.
  // Here iterators are the block arguments from the innermost scf.for loop.
  // The iterators are unique-ified so we need blockArgIteratorMapping to
  // recover which top level tensor operand is associated with which iterator.
  void convertComputeOp(Operation &op, ArrayRef<BlockArgument> cbOperands,
                        ArrayRef<BlockArgument> iterators,
                        SmallVector<unsigned> blockArgIteratorMapping,
                        Value dstIndex, OpBuilder &builder) const {
    SmallVector<std::int64_t> operandIndices;
    for (OpOperand &operand : op.getOpOperands()) {
      operandIndices.push_back(operand.getOperandNumber());
    }
    if (mlir::isa<arith::AddFOp>(op)) {
      builder.create<ttkernel::AddTilesOp>(
          op.getLoc(), cbOperands[operandIndices[0]],
          cbOperands[operandIndices[1]], iterators[blockArgIteratorMapping[0]],
          iterators[blockArgIteratorMapping[1]], dstIndex);
    } else if (mlir::isa<arith::MulFOp>(op)) {
      builder.create<ttkernel::MulTilesOp>(
          op.getLoc(), cbOperands[operandIndices[0]],
          cbOperands[operandIndices[1]], iterators[blockArgIteratorMapping[0]],
          iterators[blockArgIteratorMapping[1]], dstIndex);
    } else {
      llvm_unreachable("Unhandled conversion");
    }
  }

  // Convert the original block into a lowered block that contains a fully
  // expanded loop nest and inner loop that implements the underlying arith or
  // math operation as a tile operation.
  void lowerBlock(Block *origBlock, Block *computeBlock,
                  std::int64_t numDPSInputs) const {
    Block::OpListType &operations = origBlock->getOperations();
    assert(operations.size() == 2);
    Operation::user_range users = operations.front().getUsers();
    assert(users.begin() != users.end());
    assert(mlir::isa<ttir::YieldOp>(*users.begin()));
    assert(computeBlock->getNumArguments() > numDPSInputs);
    assert((computeBlock->getNumArguments() - numDPSInputs) == 1 &&
           "Expected 1 output");

    OpBuilder builder(computeBlock, computeBlock->begin());
    convertComputeInitOp(operations.front(), computeBlock->getArguments(),
                         numDPSInputs, builder);
    LoopNest loopNest =
        createLoopNest(computeBlock->getArguments(), numDPSInputs, builder);
    builder.create<ttkernel::ReturnOp>(origBlock->getTerminator()->getLoc());

    // Build the inner loop compute / unpack / pack
    {
      Value output = computeBlock->getArgument(numDPSInputs);
      Region *innerLoopRegion = loopNest.loopRegions.back();
      ArrayRef<BlockArgument> iterators =
          loopNest.loops.back().getRegionIterArgs();
      SmallVector<unsigned> blockArgIteratorMapping =
          loopNest.blockArgIteratorMapping;
      OpBuilder innerLoopBuilder(&innerLoopRegion->front(),
                                 innerLoopRegion->front().begin());
      Value dstIndex = i32(0, innerLoopBuilder);
      innerLoopBuilder.create<ttkernel::TileRegsAcquireOp>(
          computeBlock->front().getLoc());
      convertComputeOp(operations.front(), computeBlock->getArguments(),
                       iterators, blockArgIteratorMapping, dstIndex,
                       innerLoopBuilder);
      innerLoopBuilder.create<ttkernel::TileRegsCommitOp>(
          computeBlock->front().getLoc());
      innerLoopBuilder.create<ttkernel::TileRegsWaitOp>(
          computeBlock->front().getLoc());
      innerLoopBuilder.create<ttkernel::PackTileOp>(
          computeBlock->front().getLoc(), dstIndex, output,
          iterators[blockArgIteratorMapping[numDPSInputs]]);
      innerLoopBuilder.create<ttkernel::TileRegsReleaseOp>(
          computeBlock->front().getLoc());
    }
  }

  SmallVector<Type>
  getBlockArgumentTypesAsCBs(mlir::OperandRange dispatchOperands,
                             mlir::Block::BlockArgListType blockArguments,
                             std::int64_t numDPSInputs,
                             PatternRewriter &rewriter) const {
    SmallVector<Type> rewrittenBlockArgumentTypes;
    for (auto arg : blockArguments) {
      auto address = lookupAddress(dispatchOperands[arg.getArgNumber()]);
      assert(address && "Expected valid address");
      auto port = getPort(arg.getArgNumber(), numDPSInputs);
      auto tensor = mlir::cast<RankedTensorType>(arg.getType());
      auto buffer = mlir::cast<BufferAttr>(tensor.getEncoding());
      auto memref = buffer.getMemref();
      assert(buffer.getBufferAccess() == BufferAccess::Alias &&
             "Currently only alias mode is supported");
      rewrittenBlockArgumentTypes.push_back(
          rewriter.getType<ttkernel::CBType>(port, address, memref));
    }
    return rewrittenBlockArgumentTypes;
  }

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (hasUnloweredTTIRKernel(op)) {
      return failure();
    }

    SmallVector<Attribute> threadTypes = {
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(
            ttkernel::ThreadType::Tensix),
    };
    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
    };

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes), threadTypes.size());

    auto rewrittenBlockArgumentTypes = getBlockArgumentTypesAsCBs(
        op->getOperands(), op->getRegion(0).getArguments(),
        op.getNumDpsInputs(), rewriter);

    Block *tensixBlock = &metalDispatch.getRegion(0).emplaceBlock();
    for (auto ty : rewrittenBlockArgumentTypes) {
      tensixBlock->addArgument(ty, op.getLoc());
    }

    lowerBlock(&op->getRegion(0).front(), tensixBlock, op.getNumDpsInputs());

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }
};

class TTIRToTTMetalAllocRewriter : public OpRewritePattern<ttir::AllocOp> {
public:
  using OpRewritePattern<ttir::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::AllocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::AllocOp>(
        op, op.getType(), op.getAddress(), op.getSize(), op.getMemorySpace());
    return success();
  }
};

class TTIRToTTMetalDeallocRewriter : public OpRewritePattern<ttir::DeallocOp> {
public:
  using OpRewritePattern<ttir::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::DeallocOp>(op, op.getResult());
    return success();
  }
};

class ConvertTTIRToTTMetal
    : public impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal> {
public:
  using impl::ConvertTTIRToTTMetalBase<
      ConvertTTIRToTTMetal>::ConvertTTIRToTTMetalBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRToTTMetalLayoutRewriter, TTIRToTTMetalKernelRewriter,
                 TTIRToTTMetalDispatchRewriter, TTIRToTTMetalAllocRewriter,
                 TTIRToTTMetalDeallocRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }
};

void createTTIRToTTMetalBackendPipeline(OpPassManager &pm) {
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc());
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice());
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegion());
  mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
  layoutOptions.initMemorySpace = mlir::tt::MemorySpace::DeviceL1;
  layoutOptions.defaultMemorySpace = mlir::tt::MemorySpace::DeviceL1;
  pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegionOperandsToMemref());
  pm.addPass(mlir::tt::ttir::createTTIRAllocate());
  pm.addPass(createConvertTTIRToTTMetal());
}

} // namespace mlir::tt::ttmetal
