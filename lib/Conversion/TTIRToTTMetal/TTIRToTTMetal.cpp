// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/Utils/PhysicalCoreCoord.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/Liveness.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>
#include <utility>

namespace mlir::tt::ttmetal {

// This routine walks the SSA value chain to find the address of the value.
// It runs into and gets the address from one of the following:
//   - ttir::CreateBufferOp: The address is the address of the allocation is
//   embedded in the op attributes.
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

namespace {
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
    std::int32_t numElements = 0;

    NocTx(Type type, PhysicalCoreCoord coreCoord, std::int64_t srcOffset,
          std::int64_t dstOffset, std::int64_t size, std::int32_t numElements)
        : type(type), coreCoord(coreCoord), srcOffset(srcOffset),
          dstOffset(dstOffset), size(size), numElements(numElements) {}

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
                        AffineMap src, AffineMap dst, NocTx::Type type,
                        std::int64_t dstCapacity) {
    bool read = type == NocTx::Type::Read;
    llvm::MapVector<PhysicalCoreCoord, mlir::SmallVector<NocTx>> txMap;
    assert(src.getNumResults() == MemoryMapResultIdx::NumIndices);
    assert(dst.getNumResults() == MemoryMapResultIdx::NumIndices);

    ::ttmlir::utils::sample(
        tensorShape, [&txMap, src, dst, elemSize, read, type,
                      dstCapacity](ArrayRef<std::int64_t> index) {
          SmallVector<int64_t> srcResults = src.compose(index);
          SmallVector<int64_t> dstResults = dst.compose(index);
          assert(srcResults.size() == src.getNumResults());
          assert(dstResults.size() == dst.getNumResults());
          PhysicalCoreCoord srcCoord(srcResults);
          PhysicalCoreCoord dstCoord(dstResults);
          std::int64_t srcOffset = srcResults.back() * elemSize;
          std::int64_t dstOffset = dstResults.back() * elemSize;

          SmallVector<NocTx> &txs = txMap[read ? dstCoord : srcCoord];
          if (not txs.empty() &&
              txs.back().isContiguous(read ? srcCoord : dstCoord, srcOffset,
                                      dstOffset) &&
              txs.back().size + elemSize <= dstCapacity) {
            txs.back().size += elemSize;
            txs.back().numElements++;
          } else {
            txs.push_back(NocTx(type, read ? srcCoord : dstCoord, srcOffset,
                                dstOffset, elemSize, 1));
          }
        });

    return txMap;
  }

  LogicalResult relayout(ttir::ToLayoutOp op, PatternRewriter &rewriter) const {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType(0));
    auto inputLayout = mlir::cast<tt::MetalLayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::MetalLayoutAttr>(outputTy.getEncoding());
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
        device.getMapForMemorySpace(inputLayout.getMemorySpace()));

    AffineMap dst = outputLayout.projectOnto(
        outputLinearMap,
        device.getMapForMemorySpace(outputLayout.getMemorySpace()));

    auto dm = calculateDataMovement(
        inputShape, inputLayout.getElementSizeBytes(), src, dst,
        dataMovementType, outputLayout.getMemrefSizeBytes());

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

    auto metalEnqueueProgram = rewriter.create<ttmetal::EnqueueProgramOp>(
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
      Block *block =
          rewriter.createBlock(&metalEnqueueProgram.getRegion(regIdx++));
      createDataMovementThread(op->getLoc(), block, inputBaseAddress,
                               outputBaseAddress, transactions,
                               physicalCoordMapping, addressAlignment);
      OpBuilder endBuilder = OpBuilder::atBlockEnd(block);
      endBuilder.create<ttkernel::ReturnOp>(op->getLoc());
    }
    rewriter.replaceOp(op, metalEnqueueProgram);

    return llvm::success();
  }

  static void createDataMovementThread(
      Location loc, Block *block, int64_t inputBaseAddress,
      int64_t outputBaseAddress, ArrayRef<NocTx> transactions,
      const PhysicalCoreCoordMapping &physicalCoordMapping,
      std::int64_t addressAlignment, Value *inputCB = nullptr) {

    assert(inputBaseAddress);
    assert(outputBaseAddress);
    assert(inputBaseAddress % addressAlignment == 0);
    assert(outputBaseAddress % addressAlignment == 0);
    OpBuilder nocBuilder = OpBuilder::atBlockEnd(block);
    NocTx::Type type = transactions.front().type;

    // each 'entry' is {I32:$dst, I32:$src, I32:$size, I32:<$y,$x>,
    // (I32:$numElements if have inputCB)}:
    int32_t const entrySize = 4 + (inputCB != nullptr);

    // convert 'transactions' into compile time 'NocTransactionsTableOp'
    // parameters:

    llvm::SmallVector<int32_t, 48> entries;
    for (auto const &tx : transactions) {
      // all transactions are of the same read/write type:
      assert(tx.type == type);

      assert(tx.srcOffset % addressAlignment == 0);
      assert(tx.dstOffset % addressAlignment == 0);
      assert(tx.size % addressAlignment == 0);

      entries.emplace_back(outputBaseAddress + tx.dstOffset);
      entries.emplace_back(inputBaseAddress + tx.srcOffset);
      entries.emplace_back(tx.size);

      auto const [yPhys, xPhys] = physicalCoordMapping[tx.coreCoord];
      entries.emplace_back((yPhys << 16) | xPhys); // x:lo, y:hi

      if (inputCB != nullptr) {
        entries.emplace_back(tx.numElements);
      }
    }

    mlir::IntegerType i32Type = nocBuilder.getI32Type();   // for 'scf' ops
    mlir::IndexType indexType = nocBuilder.getIndexType(); // for 'memref' ops

    auto entriesAttr = nocBuilder.getDenseI32ArrayAttr(entries);
    auto tableType = MemRefType::get(
        {static_cast<int32_t>(entries.size() / entrySize), entrySize}, i32Type,
        AffineMap::getMultiDimIdentityMap(2, nocBuilder.getContext()));
    auto tableOp = nocBuilder.create<ttkernel::NocTransactionsTableOp>(
        loc, tableType, entriesAttr);

    auto i32 = [&](int32_t value) {
      return nocBuilder.create<arith::ConstantOp>(
          loc, nocBuilder.getI32IntegerAttr(value));
    };

    auto index = [&](int64_t value) {
      return nocBuilder.create<arith::ConstantOp>(
          loc, nocBuilder.getIndexAttr(value));
    };

    auto coordWidth = i32(16);
    auto coordMask = i32(0xFFFF);

    auto loop = nocBuilder.create<scf::ForOp>(loc, i32(0),
                                              i32(transactions.size()), i32(1));
    nocBuilder.setInsertionPointToStart(loop.getBody());
    {
      // memref.load/store requires 'index'-typed indexing, but 'scf.for' can't
      // use that, so make use of arith index casting:

      auto entry = nocBuilder.create<arith::IndexCastOp>(
          loc, indexType, loop.getInductionVar());

      int32_t slot = 0;

      std::array<Value, 2> vs{entry, index(slot++)};
      auto dst = nocBuilder.create<memref::LoadOp>(loc, tableOp, vs);
      vs[1] = index(slot++);
      auto src = nocBuilder.create<memref::LoadOp>(loc, tableOp, vs);

      vs[1] = index(slot++);
      auto size = nocBuilder.create<memref::LoadOp>(loc, tableOp, vs);

      vs[1] = index(slot++);
      auto xy =
          nocBuilder.create<memref::LoadOp>(loc, tableOp, vs)->getResult(0);

      // split 'xy' into an <x,y> pair:

      auto x = nocBuilder.create<arith::AndIOp>(loc, xy, coordMask);
      auto y = nocBuilder.create<arith::ShRUIOp>(loc, xy, coordWidth);

      // emit read/write op, bracketed by CB ops if needed:

      auto rw = [&] {
        switch (type) {
        case NocTx::Type::Read: {
          auto srcRemote =
              nocBuilder.create<ttkernel::GetNocAddrXYOp>(loc, x, y, src)
                  .getResult();
          nocBuilder.create<ttkernel::NocAsyncReadOp>(loc, srcRemote, dst,
                                                      size);
        } break;
        case NocTx::Type::Write: {
          auto dstRemote =
              nocBuilder.create<ttkernel::GetNocAddrXYOp>(loc, x, y, dst)
                  .getResult();
          nocBuilder.create<ttkernel::NocAsyncWriteOp>(loc, src, dstRemote,
                                                       size);
        } break;
        }
      };

      if (!inputCB) {
        rw();
      } else {
        vs[1] = index(slot++);
        auto numElements = nocBuilder.create<memref::LoadOp>(loc, tableOp, vs);
        nocBuilder.create<ttkernel::CBReserveBackOp>(loc, *inputCB,
                                                     numElements);
        rw();
        nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(loc);
        nocBuilder.create<ttkernel::CBPushBackOp>(loc, *inputCB, numElements);
      }
    }
    nocBuilder.setInsertionPointAfter(loop);

    if (!inputCB) {
      switch (type) {
      case NocTx::Type::Read: {
        nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(loc);
      } break;
      case NocTx::Type::Write: {
        nocBuilder.create<ttkernel::NocAsyncWriteBarrierOp>(loc);
      } break;
      }
    }
  }

  LogicalResult reformat(ttir::ToLayoutOp op, PatternRewriter &rewriter) const {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType(0));
    auto inputLayout = mlir::cast<tt::MetalLayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::MetalLayoutAttr>(outputTy.getEncoding());
    bool shouldTilize = not inputLayout.isTiled() && outputLayout.isTiled();
    bool shouldUntilize = inputLayout.isTiled() && not outputLayout.isTiled();
    assert(shouldTilize ^ shouldUntilize);
    assert(inputLayout.getGrid() == outputLayout.getGrid());

    // Ensure tt-mlir/tt-metal agree on number, and set UnpackToDestMode per CB
    uint32_t chipNumCBs = op.getSystemDesc().getChipDescs()[0].getNumCBs();
    constexpr uint32_t kNumCBs = 1 + ttkernel::getMaxEnumValForCBPort();
    assert(chipNumCBs == kNumCBs && "Mismatch between tt-mlir and tt-metal "
                                    "number of CBs");

    llvm::SmallVector<ttkernel::UnpackToDestMode, kNumCBs> unpackToDestModes(
        kNumCBs, ttkernel::UnpackToDestMode::Default);

    auto tensixAttr = rewriter.getAttr<ttkernel::TensixConfigAttr>(
        ttkernel::MathFidelity::HiFi4, false, false, unpackToDestModes);
    SmallVector<Attribute> kernelConfigs = {tensixAttr};
    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(inputLayout.getGrid()),
    };

    auto metalEnqueueProgram = rewriter.create<ttmetal::EnqueueProgramOp>(
        op.getLoc(), SmallVector<Type>({outputTy}),
        SmallVector<Value>({op.getInput()}),
        SmallVector<Value>({op.getOutput()}), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(kernelConfigs), kernelConfigs.size());

    std::int64_t inputBaseAddress = lookupAddress(op.getInput());
    std::int64_t outputBaseAddress = lookupAddress(op.getOutput());
    Block *tensixBlock =
        rewriter.createBlock(&metalEnqueueProgram.getRegion(0));
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

    rewriter.replaceOp(op, metalEnqueueProgram);

    return success();
  }

  LogicalResult matchAndRewrite(ttir::ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType(0));
    if (not inputTy.getEncoding() || not outputTy.getEncoding()) {
      return failure();
    }
    assert(inputTy.getShape() == outputTy.getShape());
    assert(mlir::isa<tt::MetalLayoutAttr>(inputTy.getEncoding()));
    assert(mlir::isa<tt::MetalLayoutAttr>(outputTy.getEncoding()));
    auto inputLayout = mlir::cast<tt::MetalLayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::MetalLayoutAttr>(outputTy.getEncoding());

    auto components = op.compoundComponents();
    bool isCompound = (static_cast<int>(components.isLayoutChange) +
                       static_cast<int>(components.isGridChange ||
                                        components.isMemorySpaceChange) +
                       static_cast<int>(components.isFormatChange)) > 1;
    assert(!isCompound && "Only one change is allowed");

    assert(!isCompound && "Only one change is allowed");
    if (components.isMemorySpaceChange) {
      if (inputLayout.isSystemMemorySpace()) {
        assert(outputLayout.isDeviceMemorySpace());
        assert(false && "System memory to device memory is not supported yet");
      } else if (outputLayout.isSystemMemorySpace()) {
        assert(inputLayout.isDeviceMemorySpace());
        rewriter.replaceOpWithNewOp<ttmetal::EnqueueReadBufferOp>(
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
} // namespace

namespace {
class TTIRToTTMetalAllocRewriter : public OpRewritePattern<ttir::AllocOp> {
public:
  using OpRewritePattern<ttir::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::AllocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::CreateBufferOp>(
        op, op.getType(), op.getAddress(), op.getSize(), op.getMemorySpace());
    return success();
  }
};
} // namespace

namespace {
class TTIRToTTMetalDeallocRewriter : public OpRewritePattern<ttir::DeallocOp> {
public:
  using OpRewritePattern<ttir::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::DeallocateBufferOp>(op,
                                                             op.getResult());
    return success();
  }
};
} // namespace

namespace {
class TTIRToTTMetalFillRewriter : public OpRewritePattern<ttir::FillOp> {
public:
  using OpRewritePattern<ttir::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::FillOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueWriteBufferOp>(
        op, op.getResult().getType(), op.getOutput(), op.getValue());
    return success();
  }
};
} // namespace

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateTTIRToTTMetalPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter & /*typeConverter*/) {
  patterns.add<ttmetal::TTIRToTTMetalLayoutRewriter,
               ttmetal::TTIRToTTMetalAllocRewriter,
               ttmetal::TTIRToTTMetalDeallocRewriter,
               ttmetal::TTIRToTTMetalFillRewriter>(ctx);
}

} // namespace mlir::tt
