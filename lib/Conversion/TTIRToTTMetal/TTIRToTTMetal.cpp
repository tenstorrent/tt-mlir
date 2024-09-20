// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>

#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "ttmlir/Dialect/TT/Utils/PhysicalCoreCoord.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttmetal {

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
    auto *funcOp = blockArg.getOwner()->getParentOp();
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

  struct NocTx {
    enum class Type { Read, Write };

    Type type;
    PhysicalCoreCoord coreCoord;
    std::int64_t srcOffset = 0;
    std::int64_t dstOffset = 0;
    std::int64_t size = 0;

    NocTx(Type type, PhysicalCoreCoord coreCoord, std::int64_t srcOffset,
          std::int64_t dstOffset, std::int64_t size)
        : type(type), coreCoord(coreCoord), srcOffset(srcOffset),
          dstOffset(dstOffset), size(size) {}

    bool isContiguous(PhysicalCoreCoord nextCoord, std::int64_t nextSrcOffset,
                      std::int64_t nextDstOffset) const {
      return (nextCoord == coreCoord) && (nextSrcOffset == srcOffset + size) &&
             (nextDstOffset == dstOffset + size);
    }
  };

  // This routine calculates the data movement for a tensor layout change by
  // tracing the walk order of the src and dst affine maps.  The sample routine
  // is just a helper function that iterates over the tensor shape and calls the
  // lambda with the current index.  It walks the shape in innermost-major
  // order. It also coalesces the noc transactions.
  //
  // The return value is a map of physical cores where each core has
  // an associated list of noc reads/writes to be performed.
  static llvm::MapVector<PhysicalCoreCoord, mlir::SmallVector<NocTx>>
  calculateDataMovement(ArrayRef<int64_t> tensorShape, std::int64_t elemSize,
                        AffineMap src, AffineMap dst, NocTx::Type type) {
    bool read = type == NocTx::Type::Read;
    llvm::MapVector<PhysicalCoreCoord, mlir::SmallVector<NocTx>> txMap;
    assert(src.getNumResults() == MemoryMapResultIdx::NumIndices);
    assert(dst.getNumResults() == MemoryMapResultIdx::NumIndices);

    ::ttmlir::utils::sample(tensorShape, [&txMap, src, dst, elemSize, read,
                                          type](ArrayRef<std::int64_t> index) {
      SmallVector<int64_t> srcResults = src.compose(index);
      SmallVector<int64_t> dstResults = dst.compose(index);
      assert(srcResults.size() == src.getNumResults());
      assert(dstResults.size() == dst.getNumResults());
      PhysicalCoreCoord srcCoord(srcResults);
      PhysicalCoreCoord dstCoord(dstResults);
      std::int64_t srcOffset = srcResults.back() * elemSize;
      std::int64_t dstOffset = dstResults.back() * elemSize;
      SmallVector<NocTx> &txs = txMap[read ? dstCoord : srcCoord];
      if (not txs.empty() && txs.back().isContiguous(read ? srcCoord : dstCoord,
                                                     srcOffset, dstOffset)) {
        txs.back().size += elemSize;
      } else {
        txs.push_back(NocTx(type, read ? srcCoord : dstCoord, srcOffset,
                            dstOffset, elemSize));
      }
    });

    return txMap;
  }

  static void
  buildNocAsyncTx(mlir::Location loc, std::int64_t inputBaseAddress,
                  std::int64_t outputBaseAddress, std::int64_t addressAlignment,
                  NocTx nocTx,
                  PhysicalCoreCoordMapping const &physicalCoordMapping,
                  mlir::OpBuilder &nocBuilder) {
    assert(nocTx.srcOffset % addressAlignment == 0);
    assert(nocTx.dstOffset % addressAlignment == 0);
    assert(nocTx.size % addressAlignment == 0);
    auto [yPhys, xPhys] = physicalCoordMapping[nocTx.coreCoord];
    auto y = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(yPhys));
    auto x = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(xPhys));
    auto srcLocalL1Addr = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(inputBaseAddress + nocTx.srcOffset));
    auto dstLocalL1Addr = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(outputBaseAddress + nocTx.dstOffset));
    auto size = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(nocTx.size));
    if (nocTx.type == NocTx::Type::Read) {
      auto srcRemoteNocAddr =
          nocBuilder.create<ttkernel::GetNocAddrOp>(loc, x, y, srcLocalL1Addr);
      nocBuilder.create<ttkernel::NocAsyncReadOp>(loc, srcRemoteNocAddr,
                                                  dstLocalL1Addr, size);
    } else {
      auto dstRemoteNocAddr =
          nocBuilder.create<ttkernel::GetNocAddrOp>(loc, x, y, dstLocalL1Addr);
      nocBuilder.create<ttkernel::NocAsyncWriteOp>(loc, srcLocalL1Addr,
                                                   dstRemoteNocAddr, size);
    }
  }

  LogicalResult relayout(ttir::ToLayoutOp op, PatternRewriter &rewriter) const {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    auto inputLayout = mlir::cast<tt::LayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::LayoutAttr>(outputTy.getEncoding());
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

    assert(inputLayout.getMemorySpace() == MemorySpace::DeviceL1 ||
           outputLayout.getMemorySpace() == MemorySpace::DeviceL1 &&
               "DRAM <-> DRAM is not supported yet");
    NocTx::Type dataMovementType =
        outputLayout.getMemorySpace() == MemorySpace::DeviceL1
            ? NocTx::Type::Read
            : NocTx::Type::Write;

    AffineMap src = inputLayout.projectOnto(
        inputLinearMap,
        device.getMapForMemorySpace(inputLayout.getMemorySpace()), inputShape);

    AffineMap dst = outputLayout.projectOnto(
        outputLinearMap,
        device.getMapForMemorySpace(outputLayout.getMemorySpace()),
        outputShape);

    auto dm =
        calculateDataMovement(inputShape, inputLayout.getElementSizeBytes(),
                              src, dst, dataMovementType);

    auto noc0Attr =
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc0);
    SmallVector<Attribute> kernelConfigs(dm.size(), noc0Attr);
    SmallVector<Attribute> coreRanges;
    coreRanges.reserve(dm.size());
    for (auto [coreCoord, txs] : dm) {
      SmallVector<int64_t> offset = {coreCoord.y, coreCoord.x};
      SmallVector<int64_t> size = {1, 1};
      coreRanges.push_back(
          rewriter.getAttr<ttmetal::CoreRangeAttr>(offset, size));
    };
    std::int64_t inputBaseAddress = lookupAddress(op.getInput());
    std::int64_t outputBaseAddress = lookupAddress(op.getOutput());

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), SmallVector<Type>({outputTy}),
        SmallVector<Value>({op.getInput()}),
        SmallVector<Value>({op.getOutput()}), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(kernelConfigs), kernelConfigs.size());

    PhysicalCoreCoordMapping physicalCoordMapping =
        PhysicalCoreCoordMapping::getMemorySpaceMapping(
            device.getChipIds(), systemDesc.getChipDescs(),
            dataMovementType == NocTx::Type::Read
                ? inputLayout.getMemorySpace()
                : outputLayout.getMemorySpace());

    int regIdx = 0;
    for (auto [dstCoord, transactions] : dm) {
      Block *block = rewriter.createBlock(&metalDispatch.getRegion(regIdx++));
      createDataMovementThread(op->getLoc(), block, inputBaseAddress,
                               outputBaseAddress, transactions,
                               physicalCoordMapping, addressAlignment);
    }
    rewriter.replaceOp(op, metalDispatch);

    return llvm::success();
  }

  static void
  createDataMovementThread(Location loc, Block *block, int64_t inputBaseAddress,
                           int64_t outputBaseAddress,
                           ArrayRef<NocTx> transactions,
                           const PhysicalCoreCoordMapping &physicalCoordMapping,
                           std::int64_t addressAlignment) {

    assert(inputBaseAddress);
    assert(outputBaseAddress);
    assert(inputBaseAddress % addressAlignment == 0);
    assert(outputBaseAddress % addressAlignment == 0);
    OpBuilder nocBuilder(block, block->begin());
    NocTx::Type type = transactions.front().type;
    for (auto tx : transactions) {
      assert(tx.type == type);
      buildNocAsyncTx(loc, inputBaseAddress, outputBaseAddress,
                      addressAlignment, tx, physicalCoordMapping, nocBuilder);
    }
    if (type == NocTx::Type::Read) {
      nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(loc);
    } else {
      nocBuilder.create<ttkernel::NocAsyncWriteBarrierOp>(loc);
    }
    nocBuilder.create<ttkernel::ReturnOp>(loc, ValueRange());
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

    auto tensixAttr = rewriter.getAttr<ttkernel::TensixConfigAttr>(
        ttkernel::MathFidelity::HiFi4, false, false, false);
    SmallVector<Attribute> kernelConfigs = {tensixAttr};
    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(inputLayout.getGrid()),
    };

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), SmallVector<Type>({outputTy}),
        SmallVector<Value>({op.getInput()}),
        SmallVector<Value>({op.getOutput()}), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(kernelConfigs), kernelConfigs.size());

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

    auto components = op.compoundComponents();
    bool isCompound = (static_cast<int>(components.isLayoutChange) +
                       static_cast<int>(components.isGridChange ||
                                        components.isMemorySpaceChange) +
                       static_cast<int>(components.isFormatChange)) > 1;
    assert(!components.isMemoryLayoutChange &&
           "Memory layout is not used in direct to metal path");
    assert(!isCompound && "Only one change is allowed");

    assert(!isCompound && "Only one change is allowed");
    assert(!components.isMemoryLayoutChange &&
           "Tensor memory layout shouldn't change in metal backend");
    if (components.isMemorySpaceChange) {
      if (inputLayout.isSystemMemorySpace()) {
        assert(outputLayout.isDeviceMemorySpace());
        assert(false && "System memory to device memory is not supported yet");
      } else if (outputLayout.isSystemMemorySpace()) {
        assert(inputLayout.isDeviceMemorySpace());
        rewriter.replaceOpWithNewOp<ttmetal::HostReadOp>(
            op, outputTy, op.getInput(), op.getOutput());
      } else {
        return relayout(op, rewriter);
      }
    } else if (components.isLayoutChange || components.isGridChange) {
      return relayout(op, rewriter);
    } else {
      assert(components.isFormatChange);
      return reformat(op, rewriter);
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

  AffineMap getPermutedAffineMap(AffineMap map,
                                 ArrayRef<bool> reductionDims) const {
    SmallVector<uint32_t> permutation;
    for (size_t i = 0; i < reductionDims.size(); i++) {
      if (!reductionDims[i]) {
        permutation.push_back(i);
      }
    }
    if (permutation.size() == map.getNumDims()) {
      return map;
    }
    for (size_t i = 0; i < reductionDims.size(); i++) {
      if (reductionDims[i]) {
        permutation.push_back(i);
      }
    }
    return map.getPermutationMap(permutation, map.getContext());
  }

  // This routine evaluates the memref's affine map with it's shape to return a
  // single result affine map, e.g.:
  //  - Given: shape{2, 4} and affine_map<(d0, d1) -> (d0, d1)>
  //  - Becomes: affine_map<(d0, d1) -> (d0 * 4 + d1)
  // This is useful for evaluating iterator increment steps between each loop.
  AffineMap getAffineIterator(MemRefType memref,
                              ArrayRef<bool> reducedMemrefDims) const {
    ArrayRef<std::int64_t> shape = memref.getShape();
    SmallVector<int64_t> physShape(shape);
    AffineMap physShapeMap = getPermutedAffineMap(
        memref.getLayout().getAffineMap(), reducedMemrefDims);

    mlir::AffineExpr resultExpr = getAffineConstantExpr(0, memref.getContext());
    int volume = 1;
    for (int i = static_cast<int>(physShape.size()) - 1; i >= 0; i--) {
      mlir::AffineExpr dimExpr = physShapeMap.getResult(i);
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
                          ArrayRef<bool> reducedMemrefDims,
                          std::int64_t numDPSInputs, OpBuilder &builder) const {
    Value output = blockArguments[numDPSInputs];

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

    size_t operandRank = 0;
    ArrayRef<int64_t> walkingShape;

    // Map block arguments to their respective unique iterators Values
    SmallVector<Value> iterators;
    iterators.resize(blockArguments.size());
    for (BlockArgument operand : blockArguments) {
      auto cbType = mlir::cast<ttkernel::CBType>(operand.getType());
      AffineMap affineIterator =
          getAffineIterator(cbType.getMemref(), reducedMemrefDims);

      assert(operandRank == 0 || affineIterator.getNumDims() == operandRank);
      iterators[operand.getArgNumber()] = getOrInsertIterator(affineIterator);
      operandRank = affineIterator.getNumDims();

      if (walkingShape.empty()) {
        walkingShape =
            getPermutedAffineMap(cbType.getMemref().getLayout().getAffineMap(),
                                 reducedMemrefDims)
                .compose(cbType.getMemref().getShape());
      }
    }

    // Map block arguments to their respective unique iterator offset in the
    // map. This is needed by the caller to know how to wire the iterators into
    // the ttkernel tile operation.
    SmallVector<unsigned> blockArgIteratorMapping;
    blockArgIteratorMapping.resize(blockArguments.size());
    for (BlockArgument operand : blockArguments) {
      auto cbType = mlir::cast<ttkernel::CBType>(operand.getType());
      AffineMap affineIterator =
          getAffineIterator(cbType.getMemref(), reducedMemrefDims);
      auto *match = iteratorMaps.find(affineIterator);
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
    for (unsigned dim = 0; dim < operandRank; ++dim) {
      OpBuilder regionBuilder(builder);
      if (!loopNest.loopRegions.empty()) {
        regionBuilder = OpBuilder(loopNest.loopRegions.back());
      }
      // Loop variables, these are decoupled from the iterators
      Value lowerBound = i32(0, regionBuilder);
      Value upperBound = i32(walkingShape[dim], regionBuilder);
      Value loopStep = i32(1, regionBuilder);
      scf::ForOp forOp = regionBuilder.create<scf::ForOp>(
          output.getLoc(), lowerBound, upperBound, loopStep,
          iteratorsNest.back());
      loopNest.loops.push_back(forOp);

      SmallVector<std::int64_t> innerIndexStep(operandRank, 0);
      innerIndexStep[dim] = 1;
      bool innerLoop = dim == (operandRank - 1);

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
        SmallVector<std::int64_t> outerIndexStep(operandRank, 0);
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
              outerOffset[0] - innerOffset[0] * walkingShape[dim];
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

  void convertInitUnaryOp(Operation &arithOrMathOp,
                          ArrayRef<BlockArgument> cbOperands,
                          OpBuilder &builder) const {
    assert(cbOperands.size() == 2 &&
           "Expected one input and one output CB for unary op.");

    auto inCB = cbOperands[0];
    auto outCB = cbOperands[1];

    // All unary ops have common init function and specialized init function.
    builder.create<ttkernel::UnaryOpInitCommonOp>(arithOrMathOp.getLoc(), inCB,
                                                  outCB);

    if (mlir::isa<math::ExpOp>(arithOrMathOp)) {
      builder.create<ttkernel::ExpTileInitOp>(arithOrMathOp.getLoc());
    } else {
      llvm_unreachable("Unhandled unary op init conversion.");
    }
  }

  void convertInitBinaryOp(Operation &arithOrMathOp,
                           ArrayRef<BlockArgument> cbOperands,
                           OpBuilder &builder) const {
    assert(cbOperands.size() == 3 &&
           "Expected two input and one output CB for binary op.");

    auto inCB0 = cbOperands[0];
    auto inCB1 = cbOperands[1];
    auto outCB = cbOperands[2];

    // All binary ops have common init function and specialized init function.
    builder.create<ttkernel::BinaryOpInitCommonOp>(arithOrMathOp.getLoc(),
                                                   inCB0, inCB1, outCB);

    if (mlir::isa<arith::AddFOp>(arithOrMathOp)) {
      builder.create<ttkernel::AddTilesInitOp>(arithOrMathOp.getLoc(), inCB0,
                                               inCB1);
    } else if (mlir::isa<arith::MulFOp>(arithOrMathOp)) {
      builder.create<ttkernel::MulTilesInitOp>(arithOrMathOp.getLoc(), inCB0,
                                               inCB1);
    } else if (mlir::isa<arith::DivFOp>(arithOrMathOp)) {
      builder.create<ttkernel::MulTilesInitFOp>(arithOrMathOp.getLoc());
    } else {
      llvm_unreachable("Unhandled binary op init conversion.");
    }
  }

  // Convert arith and math dialect operations into ttkernel init tile
  // operations. HLK requires the FPU to be initialized before any tile ops get
  // executed.  We separate the init tile operation from the actual tile
  // operation so that we can hoist the init tile operation outside of the loop
  // nest.
  void convertComputeInitOp(Operation &arithOrMathOp,
                            ArrayRef<BlockArgument> cbOperands,
                            std::int64_t numDpsInputs,
                            ttkernel::ReduceDim reduceDim,
                            OpBuilder &builder) const {
    if (reduceDim != ttkernel::ReduceDim::None) {
      auto kernelOp = mlir::cast<ttir::KernelOp>(arithOrMathOp);
      assert(kernelOp.getOp() == "reduce");
      auto type = kernelOp.getKind() == "sum" ? ttkernel::ReduceType::ReduceSUM
                                              : ttkernel::ReduceType::ReduceMAX;
      builder.create<ttkernel::ReduceInitOp>(
          arithOrMathOp.getLoc(), cbOperands[0],
          cbOperands[0], // TODO(rpavlovic) insert proper
                         // scaling cb
          cbOperands[1],
          ttkernel::ReduceTypeAttr::get(builder.getContext(), type),
          ttkernel::ReduceDimAttr::get(builder.getContext(), reduceDim));
    } else if (numDpsInputs == 1) {
      convertInitUnaryOp(arithOrMathOp, cbOperands, builder);
    } else if (numDpsInputs == 2) {
      convertInitBinaryOp(arithOrMathOp, cbOperands, builder);
    } else {
      llvm_unreachable("Unhandled conversion for operation which is neither "
                       "unary nor binary nor reduce.");
    }
  }

  void convertComputeUnaryOp(Operation &arithOrMathOp,
                             ArrayRef<BlockArgument> cbOperands,
                             ArrayRef<BlockArgument> iterators,
                             SmallVector<unsigned> blockArgIteratorMapping,
                             OpBuilder &builder) const {
    assert(cbOperands.size() == 2 &&
           "Expected one input and one output CB for unary op.");

    auto inCBTileIndex = iterators[blockArgIteratorMapping[0]];
    auto inCB = cbOperands[0];
    auto outCBTileIndex = iterators[blockArgIteratorMapping.back()];
    auto outCB = cbOperands.back();

    auto location = arithOrMathOp.getLoc();

    // We always operate on the first and only tile in DST register.
    Value dstTileIndex = i32(0, builder);

    // MATH acquires lock on DST register.
    builder.create<ttkernel::TileRegsAcquireOp>(location);

    // For all unary ops first copy tile from input CB at inCBTileIndex to DST
    // register at dstTileIndex.
    builder.create<ttkernel::CopyTileOp>(location, inCB, inCBTileIndex,
                                         dstTileIndex);

    // Perform computation on tile in DST register on dstTileIndex (the only
    // tile in DST).
    if (mlir::isa<math::ExpOp>(arithOrMathOp)) {
      builder.create<ttkernel::ExpTileOp>(location, dstTileIndex);
    } else {
      llvm_unreachable("Unhandled unary op compute conversion.");
    }

    // MATH releases lock on DST.
    builder.create<ttkernel::TileRegsCommitOp>(location);

    // PACK acquires lock on DST register. Blocked until MATH releases it.
    builder.create<ttkernel::TileRegsWaitOp>(location);

    // Copy tile from DST at dstTileIndex to outCB at outCBTileIndex.
    // outCBTileIndex increments as loops iterate, thus placing one result tile
    // after another in outCB.
    builder.create<ttkernel::PackTileOp>(location, dstTileIndex, outCB,
                                         outCBTileIndex);

    // PACK releases lock on DST.
    builder.create<ttkernel::TileRegsReleaseOp>(location);
  }

  void convertComputeBinaryOp(Operation &arithOrMathOp,
                              ArrayRef<BlockArgument> cbOperands,
                              ArrayRef<BlockArgument> iterators,
                              SmallVector<unsigned> blockArgIteratorMapping,
                              OpBuilder &builder) const {
    assert(cbOperands.size() == 3 &&
           "Expected two input and one output CB for binary op.");

    auto inCB0TileIndex = iterators[blockArgIteratorMapping[0]];
    auto inCB0 = cbOperands[0];
    auto inCB1TileIndex = iterators[blockArgIteratorMapping[1]];
    auto inCB1 = cbOperands[1];
    auto outCB = cbOperands[2];
    auto outCBTileIndex = iterators[blockArgIteratorMapping[2]];

    auto location = arithOrMathOp.getLoc();

    // Perform computation C = A (*) B on tile A from inCB0 and tile B from
    // inCB1 and store the result C in DST register on dstTileIndex.
    if (mlir::isa<arith::AddFOp>(arithOrMathOp)) {
      Value dstIndex = i32(0, builder);
      builder.create<ttkernel::TileRegsAcquireOp>(location);
      builder.create<ttkernel::AddTilesOp>(
          location, inCB0, inCB1, inCB0TileIndex, inCB1TileIndex, dstIndex);
      builder.create<ttkernel::TileRegsCommitOp>(location);
      builder.create<ttkernel::TileRegsWaitOp>(location);
      builder.create<ttkernel::PackTileOp>(location, dstIndex, outCB,
                                           outCBTileIndex);
      builder.create<ttkernel::TileRegsReleaseOp>(location);
    } else if (mlir::isa<arith::MulFOp>(arithOrMathOp)) {
      commonComputeMulOp(arithOrMathOp, cbOperands, iterators,
                         blockArgIteratorMapping, builder);
    } else if (mlir::isa<arith::DivFOp>(arithOrMathOp)) {

      SmallVector<std::int64_t> operandIndicesRecip;
      // For DIV, input 1 is going through reciprocal.
      operandIndicesRecip.push_back(1);
      commonComputeRecipOp(arithOrMathOp, cbOperands, iterators,
                           blockArgIteratorMapping, builder,
                           operandIndicesRecip);

      Value one = i32(1, builder);
      builder.create<ttkernel::CBWaitFrontOp>(location, inCB1, one);

      builder.create<ttkernel::MulTilesInitOp>(location, inCB0, inCB1);

      commonComputeMulOp(arithOrMathOp, cbOperands, iterators,
                         blockArgIteratorMapping, builder);

      builder.create<ttkernel::CBPopFrontOp>(location, inCB1, one);
    } else {
      llvm_unreachable("Unhandled conversion for operation which is neither "
                       "unary nor binary.");
    }
  }

  void commonComputeMulOp(Operation &op, ArrayRef<BlockArgument> cbOperands,
                          ArrayRef<BlockArgument> iterators,
                          SmallVector<unsigned> blockArgIteratorMapping,
                          OpBuilder &builder) const {

    auto inCB0 = cbOperands[0];
    auto inCB1 = cbOperands[1];
    auto outCB = cbOperands[2];
    auto inCB0TileIndex = iterators[blockArgIteratorMapping[0]];
    auto inCB1TileIndex = iterators[blockArgIteratorMapping[1]];

    Value dstIndex = i32(0, builder);

    builder.create<ttkernel::TileRegsAcquireOp>(op.getLoc());
    if (mlir::isa<arith::MulFOp>(op)) {
      builder.create<ttkernel::MulTilesOp>(
          op.getLoc(), inCB0, inCB1, inCB0TileIndex, inCB1TileIndex, dstIndex);
    } else if (mlir::isa<arith::DivFOp>(op)) {
      // Source index for CB input 1 is 0(dstIndex), because of sync needed with
      // recip.
      builder.create<ttkernel::MulTilesOp>(op.getLoc(), inCB0, inCB1,
                                           inCB0TileIndex, dstIndex, dstIndex);
    } else {
      llvm_unreachable("Common compute for multiplying tiles should be called "
                       "only on MulFOp and DivFOp");
    }

    builder.create<ttkernel::TileRegsCommitOp>(op.getLoc());
    builder.create<ttkernel::TileRegsWaitOp>(op.getLoc());
    builder.create<ttkernel::PackTileOp>(op.getLoc(), dstIndex, outCB,
                                         iterators[blockArgIteratorMapping[2]]);
    builder.create<ttkernel::TileRegsReleaseOp>(op.getLoc());
  }

  void commonComputeRecipOp(Operation &op, ArrayRef<BlockArgument> cbOperands,
                            ArrayRef<BlockArgument> iterators,
                            SmallVector<unsigned> blockArgIteratorMapping,
                            OpBuilder &builder,
                            SmallVector<std::int64_t> &operandIndices) const {
    Value dstIndex = i32(0, builder);
    Value one = i32(1, builder);

    auto inputCB = cbOperands[operandIndices[0]];
    auto outputCB = inputCB;

    builder.create<ttkernel::CopyTileInitOp>(op.getLoc());
    builder.create<ttkernel::CBReserveBackOp>(op.getLoc(), inputCB, one);
    builder.create<ttkernel::TileRegsAcquireOp>(op.getLoc());
    builder.create<ttkernel::RecipTileInitOp>(op.getLoc());
    builder.create<ttkernel::CopyTileOp>(op.getLoc(), inputCB, dstIndex,
                                         dstIndex);
    builder.create<ttkernel::RecipTileOp>(op.getLoc(), dstIndex);
    builder.create<ttkernel::TileRegsCommitOp>(op.getLoc());

    builder.create<ttkernel::TileRegsWaitOp>(op.getLoc());
    builder.create<ttkernel::PackTileOp>(op.getLoc(), dstIndex, outputCB,
                                         dstIndex);
    builder.create<ttkernel::TileRegsReleaseOp>(op.getLoc());
    builder.create<ttkernel::CBPushBackOp>(op.getLoc(), outputCB, one);
  }

  // Convert arith and math dialect operations into ttkernel tile operations.
  // Here iterators are the block arguments from the innermost scf.for loop.
  // The iterators are unique-ified so we need blockArgIteratorMapping to
  // recover which top level tensor operand is associated with which iterator.
  void convertComputeOp(Block *computeBlock, Operation &arithOrMathOp,
                        LoopNest &loopNest, ArrayRef<BlockArgument> cbOperands,
                        ArrayRef<BlockArgument> iterators,
                        ttkernel::ReduceDim reduceDim,
                        SmallVector<unsigned> blockArgIteratorMapping,
                        OpBuilder &innerLoopBuilder,
                        std::int64_t numDpsInputs) const {

    if (reduceDim != ttkernel::ReduceDim::None) {

      convertReduceOp(computeBlock, arithOrMathOp, cbOperands, iterators,
                      blockArgIteratorMapping, reduceDim, loopNest);

      // auto kernelOp = mlir::cast<ttir::KernelOp>(arithOrMathOp);
      // assert(kernelOp.getOp() == "reduce");
      // auto type = kernelOp.getKind() == "sum" ?
      // ttkernel::ReduceType::ReduceSUM
      //                                         :
      //                                         ttkernel::ReduceType::ReduceMAX;
      // builder.create<ttkernel::ReduceTileOp>(
      //     arithOrMathOp.getLoc(), cbOperands[0],
      //     cbOperands[0], // TODO(rpavlovic) insert proper
      //                                    // scaling cb
      //     iterators[blockArgIteratorMapping[0]],
      //     iterators[blockArgIteratorMapping[1]], dstIndex,
      //     ttkernel::ReduceTypeAttr::get(builder.getContext(), type),
      //     ttkernel::ReduceDimAttr::get(builder.getContext(), reduceDim));
    } else if (numDpsInputs == 1) {
      convertComputeUnaryOp(arithOrMathOp, cbOperands, iterators,
                            blockArgIteratorMapping, innerLoopBuilder);
    } else if (numDpsInputs == 2) {
      convertComputeBinaryOp(arithOrMathOp, cbOperands, iterators,
                             blockArgIteratorMapping, innerLoopBuilder);
    } else {
      llvm_unreachable("Unhandled conversion for operation which is neither "
                       "unary nor binary.");
    }
  }

  void convertReduceOp(Block *computeBlock, Operation &op,
                       ArrayRef<BlockArgument> cbOperands,
                       ArrayRef<BlockArgument> iterators,
                       SmallVector<unsigned> blockArgIteratorMapping,
                       ttkernel::ReduceDim reduceDim,
                       LoopNest &loopNest) const {
    assert(reduceDim != ttkernel::ReduceDim::None);

    auto kernelOp = mlir::cast<ttir::KernelOp>(op);
    assert(kernelOp.getOp() == "reduce");
    auto type = kernelOp.getKind() == "sum" ? ttkernel::ReduceType::ReduceSUM
                                            : ttkernel::ReduceType::ReduceMAX;

    OpBuilder innerLoopBuilder(&loopNest.loopRegions.back()->front(),
                               loopNest.loopRegions.back()->front().begin());
    auto dstIndex = i32(0, innerLoopBuilder);

    innerLoopBuilder.create<ttkernel::ReduceTileOp>(
        op.getLoc(), cbOperands[0],
        cbOperands[0], // TODO(rpavlovic) insert proper
                       // scaling cb
        iterators[blockArgIteratorMapping[0]],
        iterators[blockArgIteratorMapping[0]], dstIndex,
        ttkernel::ReduceTypeAttr::get(innerLoopBuilder.getContext(), type),
        ttkernel::ReduceDimAttr::get(innerLoopBuilder.getContext(), reduceDim));

    size_t numLoopRegions = loopNest.loopRegions.size();
    size_t numReducedDims = reduceDim == ttkernel::ReduceDim::Row ||
                                    reduceDim == ttkernel::ReduceDim::Col
                                ? 1
                                : 2;

    Block *packingBlock =
        numReducedDims == numLoopRegions
            ? computeBlock
            : &loopNest.loopRegions[numLoopRegions - 1 - numReducedDims]
                   ->getBlocks()
                   .front();
    OpBuilder packingBuilder(packingBlock, packingBlock->begin());

    packingBuilder.create<ttkernel::TileRegsAcquireOp>(
        computeBlock->front().getLoc());

    Value packSingleTile = i32(0, packingBuilder);
    Value packingTileIndex =
        numReducedDims == numLoopRegions
            ? packSingleTile
            : loopNest.loops[numLoopRegions - 1 - numReducedDims]
                  .getRegionIterArgs()[blockArgIteratorMapping.back()];

    if (packingBlock->mightHaveTerminator()) {
      packingBuilder.setInsertionPoint(packingBlock->getTerminator());
    } else {
      packingBuilder.setInsertionPointToEnd(packingBlock);
    }

    packingBuilder.create<ttkernel::TileRegsCommitOp>(
        computeBlock->front().getLoc());
    packingBuilder.create<ttkernel::TileRegsWaitOp>(
        computeBlock->front().getLoc());
    packingBuilder.create<ttkernel::PackTileOp>(computeBlock->front().getLoc(),
                                                dstIndex, cbOperands.back(),
                                                packingTileIndex);
    packingBuilder.create<ttkernel::TileRegsReleaseOp>(
        computeBlock->front().getLoc());
  }

  // Builds instructions to execute before looping over tiles has started.
  void buildInitSection(Operation &arithOrMathOp, OpBuilder &builder,
                        ArrayRef<BlockArgument> cbOperands,
                        ttkernel::ReduceDim reduceDim,
                        std::int64_t numDPSInputs) const {
    convertComputeInitOp(arithOrMathOp, cbOperands, numDPSInputs, reduceDim,
                         builder);
  }

  // Builds nested loops which loop over tensor tiles after initalization is
  // done and computation to perform on each tile over which loops iterate.
  void buildLoopsAndComputation(Block *computeBlock, Operation &arithOrMathOp,
                                OpBuilder &builder,
                                ArrayRef<bool> reducedMemrefDims,
                                ttkernel::ReduceDim reduceDim,
                                ArrayRef<BlockArgument> &cbOperands,
                                std::int64_t numDPSInputs) const {
    // Create loops which iterate over tiles in tensor.
    LoopNest loopNest =
        createLoopNest(cbOperands, reducedMemrefDims, numDPSInputs, builder);
    assert(loopNest.loops.size() == 2 && "Expected only two loops!");

    // The loop nest is created from outermost to innermost. Get the inner loop
    // and place computation calls inside it.
    Region *innerLoopRegion = loopNest.loopRegions.back();
    ArrayRef<BlockArgument> iterators =
        loopNest.loops.back().getRegionIterArgs();
    SmallVector<unsigned> blockArgIteratorMapping =
        loopNest.blockArgIteratorMapping;

    OpBuilder innerLoopBuilder(&innerLoopRegion->front(),
                               innerLoopRegion->front().begin());

    // Call compute function to execute on each tile. Result will be stored in
    // DST.
    convertComputeOp(computeBlock, arithOrMathOp, loopNest, cbOperands,
                     iterators, reduceDim, blockArgIteratorMapping,
                     innerLoopBuilder, numDPSInputs);
  }

  // Builds instructions to execute after loops are finished.
  void buildEndSection(OpBuilder &dispatchBlockBuilder,
                       Block *origGenericOpBlock) const {
    // Place return op at the end of block, after loops.
    dispatchBlockBuilder.create<ttkernel::ReturnOp>(
        origGenericOpBlock->getTerminator()->getLoc());
  }

  ttkernel::ReduceDim getReduceDim(ArrayRef<bool> reducedMemrefDims) const {
    if (reducedMemrefDims.size() == 1) {
      return reducedMemrefDims[0] ? ttkernel::ReduceDim::ReduceDimSCALAR
                                  : ttkernel::ReduceDim::ReduceDimNONE;
    }
    if (reducedMemrefDims.size() == 2) {
      if (reducedMemrefDims[0] && reducedMemrefDims[1]) {
        return ttkernel::ReduceDim::ReduceDimSCALAR;
      }
      if (reducedMemrefDims[0]) {
        return ttkernel::ReduceDim::ReduceDimCOL;
      }
      if (reducedMemrefDims[1]) {
        return ttkernel::ReduceDim::ReduceDimROW;
      }
      return ttkernel::ReduceDim::ReduceDimNONE;
    }
    llvm_unreachable("Unhandled reduction dims");
  }

  // Convert the original block into a lowered block that contains a fully
  // expanded loop nest and inner loop that implements the underlying arith or
  // math operation as a tile operation.
  void lowerBlock(Block *origGenericOpBlock, Block *computeBlock,
                  ArrayAttr iteratorTypes, ArrayAttr indexingMaps,
                  std::int64_t numDPSInputs) const {
    Block::OpListType &operations = origGenericOpBlock->getOperations();
    assert(operations.size() == 2);
    Operation::user_range users = operations.front().getUsers();
    assert(users.begin() != users.end());
    assert(mlir::isa<ttir::YieldOp>(*users.begin()));
    assert(dispatchOpBlock->getNumArguments() > numDPSInputs);
    assert((dispatchOpBlock->getNumArguments() - numDPSInputs) == 1 &&
           "Expected 1 output");

    auto outputMemref = mlir::cast<ttkernel::CBType>(
                            computeBlock->getArgument(numDPSInputs).getType())
                            .getMemref()
                            .getLayout()
                            .getAffineMap();
    size_t j = iteratorTypes.size() - 1;
    uint32_t outputRank = outputMemref.getNumDims();
    SmallVector<bool> reducedMemrefDims(outputRank, false);
    uint32_t numReducedDims = 0;

    // Collect the reduction dims going from innermost to outermost.
    assert(outputRank <= iteratorTypes.size());
    for (int i = outputRank - 1; i >= 0; --i, --j) {
      auto itType = iteratorTypes[j];
      if (mlir::cast<IteratorTypeAttr>(itType).getValue() ==
          IteratorType::Reduction) {
        reducedMemrefDims[i] = true;
        ++numReducedDims;
      }
    }

    auto kernelReduceDim = getReduceDim(reducedMemrefDims);

    OpBuilder builder(computeBlock, computeBlock->begin());
    Operation &arithOrMathOp = operations.front();
    auto cbOperands = computeBlock->getArguments();

    buildInitSection(arithOrMathOp, builder, cbOperands, kernelReduceDim,
                     numDPSInputs);
    buildLoopsAndComputation(computeBlock, arithOrMathOp, builder,
                             reducedMemrefDims, kernelReduceDim, cbOperands,
                             numDPSInputs);
    buildEndSection(builder, origGenericOpBlock);
  }

  struct StreamedOperand {
    uint64_t srcAddress;
    uint64_t dstAddress;
    size_t blockArgIndex;
    llvm::MapVector<PhysicalCoreCoord,
                    SmallVector<TTIRToTTMetalLayoutRewriter::NocTx>>
        dataMovement;
    uint64_t numTiles;
    PhysicalCoreCoordMapping coordMappping;

    StreamedOperand(
        uint64_t srcAddress, uint64_t dstAddress, size_t blockArgIndex,
        llvm::MapVector<PhysicalCoreCoord,
                        SmallVector<TTIRToTTMetalLayoutRewriter::NocTx>>
            dataMovement,
        uint64_t numTiles, const PhysicalCoreCoordMapping &coordMappping)
        : srcAddress(srcAddress), dstAddress(dstAddress),
          blockArgIndex(blockArgIndex), dataMovement(dataMovement),
          numTiles(numTiles), coordMappping(coordMappping) {}
  };

  SmallVector<Type> getBlockArgumentTypesAsCBs(
      ttir::GenericOp op, mlir::Block::BlockArgListType blockArguments,
      PatternRewriter &rewriter,
      SmallVector<StreamedOperand> &streamedOperands) const {

    SmallVector<Type> rewrittenBlockArgumentTypes;
    for (auto arg : blockArguments) {
      auto port = getPort(arg.getArgNumber(), op.getInputs().size());
      auto tensor = mlir::cast<RankedTensorType>(arg.getType());
      auto buffer = mlir::cast<BufferAttr>(tensor.getEncoding());
      auto memref = buffer.getMemref();

      int32_t cbMapping = op.getOperandCbMapping()[arg.getArgNumber()];

      // Operand that is directly mapped to block argument.
      auto matchingOperand = op.getMatchingOperand(arg.getArgNumber());

      // Operand that is either directly mapped to block argument or
      // corresponding CB operand if it should be streamed.
      auto correspondingOperand =
          cbMapping == -1 ? matchingOperand : op.getCbs()[cbMapping];
      auto address = lookupAddress(correspondingOperand);
      assert(address && "Expected valid address");

      rewrittenBlockArgumentTypes.push_back(
          rewriter.getType<ttkernel::CBType>(port, address, memref));
      if (buffer.getBufferAccess() != BufferAccess::Stream) {
        continue;
      }

      assert(cbMapping != -1 && "Expected streamed operand to have CB mapping");

      auto matchingCB = op.getCbs()[cbMapping];
      uint64_t numTiles = memref.getShape()[memref.getRank() - 1] *
                          memref.getShape()[memref.getRank() - 2];
      streamedOperands.push_back(StreamedOperand(
          lookupAddress(matchingOperand), lookupAddress(matchingCB),
          arg.getArgNumber(),
          calculateDataMovement(
              mlir::cast<RankedTensorType>(matchingOperand.getType()),
              mlir::cast<RankedTensorType>(matchingCB.getType()),
              op.getDevice()),
          numTiles,
          // TODO(rpavlovic) fix the assumption that input is always in L1.
          PhysicalCoreCoordMapping::getMemorySpaceMapping(
              op.getDevice().getChipIds(), op.getSystemDesc().getChipDescs(),
              MemorySpace::DeviceL1)));
    }

    return rewrittenBlockArgumentTypes;
  }

  llvm::MapVector<PhysicalCoreCoord,
                  SmallVector<TTIRToTTMetalLayoutRewriter::NocTx>>
  calculateDataMovement(const RankedTensorType &src,
                        const RankedTensorType &dst, DeviceAttr device) const {
    auto srcLayout = mlir::cast<tt::LayoutAttr>(src.getEncoding());
    assert(srcLayout.isTiled());

    auto dstLayout = mlir::cast<tt::LayoutAttr>(dst.getEncoding());
    assert(dstLayout.isTiled());

    auto srcMap = srcLayout.getIdentityTileLinearMap();
    auto srcShape = srcLayout.getTiledShape(src.getShape());
    auto srcProjection = srcLayout.projectOnto(
        srcMap, device.getMapForMemorySpace(srcLayout.getMemorySpace()),
        srcShape);

    auto dstMap = dstLayout.getIdentityTileLinearMap();
    auto dstShape = dstLayout.getTiledShape(dst.getShape());
    auto dstProjection = dstLayout.projectOnto(
        dstMap, device.getMapForMemorySpace(dstLayout.getMemorySpace()),
        dstShape);

    auto dm = TTIRToTTMetalLayoutRewriter::calculateDataMovement(
        srcShape, srcLayout.getElementSizeBytes(), srcProjection, dstProjection,
        TTIRToTTMetalLayoutRewriter::NocTx::Type::Read);

    return dm;
  }

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Temporary fix that allows ttir::KernelOp to be lowered directly into
    // ttkernel dialect.
    // if (hasUnloweredTTIRKernel(op)) {
    //   return failure();
    // }

    auto tensixAttr = rewriter.getAttr<ttkernel::TensixConfigAttr>(
        ttkernel::MathFidelity::HiFi4, false, false, false);
    SmallVector<Attribute> kernelConfigs = {tensixAttr};
    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
    };

    SmallVector<StreamedOperand> streamedOperands;
    auto rewrittenBlockArgumentTypes = getBlockArgumentTypesAsCBs(
        op, op->getRegion(0).getArguments(), rewriter, streamedOperands);

    if (streamedOperands.size() > 1) {
      llvm::errs() << "Only 1 operand can be streamed\n";
      return failure();
    }

    if (!streamedOperands.empty()) {
      auto &dm = streamedOperands[0].dataMovement;
      for (auto [dstCoord, srcs] : dm) {
        kernelConfigs.push_back(ttkernel::NocConfigAttr::get(
            rewriter.getContext(),
            ttkernel::NocIndex::Noc0)); // TODO(rpavlovic) choose Noc0 vs Noc1
                                        // based off transaction type
        coreRanges.push_back(ttmetal::CoreRangeAttr::get(
            getContext(), {dstCoord.y, dstCoord.x}, {1, 1} /* size */));
      }
    }

    // Wire generic's operands to dispatch op's operands with respect to the CB
    // mapping.
    SmallVector<Value> inputsToDispatchOp;
    for (size_t i = 0; i < op.getInputs().size(); ++i) {
      auto operand = op.getOperandCbMapping()[i] == -1
                         ? op.getMatchingOperand(i)
                         : op.getCbs()[op.getOperandCbMapping()[i]];
      inputsToDispatchOp.push_back(operand);
    }

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op.getResults().getTypes(), inputsToDispatchOp,
        op.getOutputs(), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(kernelConfigs), kernelConfigs.size());

    Block *tensixBlock = &metalDispatch.getRegion(0).emplaceBlock();

    for (auto ty : rewrittenBlockArgumentTypes) {
      tensixBlock->addArgument(ty, op.getLoc());
    }

    // TODO(radenko) move TTIRToTTMetalLayoutRewriter::createDataMovementThreads
    // (& other data movement logic) into common place
    int dmThreadIdx = 1;

    if (!streamedOperands.empty()) {
      for (auto [dstCoord, srcs] : streamedOperands[0].dataMovement) {
        Block *block =
            rewriter.createBlock(&metalDispatch.getRegion(dmThreadIdx++));
        block->addArgument(
            rewrittenBlockArgumentTypes[streamedOperands[0].blockArgIndex],
            op.getLoc());
        TTIRToTTMetalLayoutRewriter::createDataMovementThread(
            op->getLoc(), block, streamedOperands[0].srcAddress,
            streamedOperands[0].dstAddress, srcs,
            streamedOperands[0].coordMappping,
            op.getSystemDesc().getAddressAlignBytes());
        OpBuilder b(block, block->begin());
        b.setInsertionPoint(block->getTerminator());

        auto streamedCB = block->getArgument(0);
        auto numPages = b.create<arith::ConstantOp>(
            op.getLoc(), b.getI32Type(),
            b.getI32IntegerAttr(streamedOperands[0].numTiles));
        b.create<ttkernel::CBPushBackOp>(op->getLoc(), streamedCB, numPages);
      }
    }

    lowerBlock(&op->getRegion(0).front(), tensixBlock, op.getIteratorTypes(),
               op.getIndexingMaps(), op.getInputs().size());

    if (kernelConfigs.size() > 1) {
      // There is some data movement. Let's just insert waiting command at the
      // start of compute block. We assume whole block is streamed.
      OpBuilder builder(tensixBlock, tensixBlock->begin());
      auto numPages = builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getI32Type(),
          builder.getI32IntegerAttr(streamedOperands[0].numTiles));
      auto streamedCB =
          tensixBlock->getArgument(streamedOperands[0].blockArgIndex);
      builder.create<ttkernel::CBWaitFrontOp>(op.getLoc(), streamedCB,
                                              numPages);
    }

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

class TTIRToTTMetalFillRewriter : public OpRewritePattern<ttir::FillOp> {
public:
  using OpRewritePattern<ttir::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::FillOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::HostWriteOp>(
        op, op.getResult().getType(), op.getOutput(), op.getValue());
    return success();
  }
};

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateTTIRToTTMetalPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter & /*typeConverter*/) {
  patterns.add<ttmetal::TTIRToTTMetalLayoutRewriter,
               ttmetal::TTIRToTTMetalDispatchRewriter,
               ttmetal::TTIRToTTMetalAllocRewriter,
               ttmetal::TTIRToTTMetalDeallocRewriter,
               ttmetal::TTIRToTTMetalFillRewriter>(ctx);
}

} // namespace mlir::tt
