// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>

namespace mlir::tt::ttkernel {

static Value getCb(Value memref) {
  auto collapseOp =
      mlir::dyn_cast<memref::CollapseShapeOp>(memref.getDefiningOp());
  if (auto cb = mlir::dyn_cast<ttkernel::CBType>(
          collapseOp.getSrcMutable().get().getType())) {
    return collapseOp.getSrcMutable().get();
  }
  assert(false && "Could not match collapse op src to block argument, cannot "
                  "determine CB id. Failing.");
}

namespace {

class MemrefStoreRewriter : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, memref::StoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto index = [&](int64_t value) {
      return rewriter
          .create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(),
                                     rewriter.getIndexAttr(value))
          .getResult();
    };

    auto cb = getCb(op.getMemref());
    auto storeIdx = op.getIndices().front();
    rewriter.replaceOpWithNewOp<ttkernel::PackTileOp>(
        op, index(0), cb, storeIdx, rewriter.getBoolAttr(true));

    return success();
  };
};

} // namespace
namespace {

class TTIRComputeOpsRewriter
    : public OpTraitConversionPattern<
          mlir::tt::ttir::TTIRGenericRegionComputeOpTrait> {
public:
  using OpTraitConversionPattern<
      mlir::tt::ttir::TTIRGenericRegionComputeOpTrait>::
      OpTraitConversionPattern;

  static Value getLoadIndex(Value tile) {
    memref::LoadOp loadOp =
        mlir::dyn_cast<memref::LoadOp>(tile.getDefiningOp());
    assert(loadOp && "Expected load op, failing.");
    assert(loadOp.getIndices().size() == 1 &&
           "Expected single index in load op, failing.");
    return loadOp.getIndices().front();
  }

  static void lowerLoadToCopyTile(memref::LoadOp op, bool cbIdxAsDstIdx,
                                  PatternRewriter &rewriter) {
    auto index = [&](int64_t value) {
      return rewriter
          .create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(),
                                     rewriter.getIndexAttr(value))
          .getResult();
    };

    auto cb = getCb(op.getMemref());
    auto cbType = mlir::dyn_cast<ttkernel::CBType>(cb.getType());
    rewriter.create<ttkernel::CopyTileInitOp>(op.getLoc(), cb);
    rewriter.create<ttkernel::CopyTileOp>(
        op.getLoc(), cb, op.getIndices().front(),
        cbIdxAsDstIdx ? index(static_cast<uint32_t>(cbType.getPort()))
                      : index(0));
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i32 = [&](int32_t value) {
      return rewriter
          .create<arith::ConstantOp>(op->getLoc(), rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(value))
          .getResult();
    };

    auto index = [&](int64_t value) {
      return rewriter
          .create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexType(),
                                     rewriter.getIndexAttr(value))
          .getResult();
    };

    Operation *newOp;

    if (mlir::isa<ttir::TileMaximumOp>(op)) {
      rewriter.create<ttkernel::MaxTilesInitOp>(op->getLoc());
      newOp =
          rewriter.create<ttkernel::MaxTilesOp>(op->getLoc(), i32(0), i32(1));
    } else if (mlir::isa<ttir::TileMatmulOp>(op)) {
      auto dstIdx = index(0);
      newOp = rewriter.create<ttkernel::MatmulTilesOp>(
          op->getLoc(),
          getCb(op->getOperand(0).getDefiningOp<memref::LoadOp>().getMemref()),
          getCb(op->getOperand(1).getDefiningOp<memref::LoadOp>().getMemref()),
          getLoadIndex(op->getOperand(0)), getLoadIndex(op->getOperand(1)),
          dstIdx);
    } else {
      return failure();
    }

    rewriter.setInsertionPoint(newOp);
    if (mlir::isa<ttkernel::MatmulTilesOp>(newOp)) {
      lowerLoadToCopyTile(op->getOperand(2).getDefiningOp<memref::LoadOp>(),
                          false, rewriter);
    } else if (newOp->hasTrait<TTKernelSFPUOpTrait>()) {
      for (uint32_t i = 0; i < op->getNumOperands(); i++) {
        lowerLoadToCopyTile(op->getOperand(i).getDefiningOp<memref::LoadOp>(),
                            true, rewriter);
      }
    }

    // This is necessary to remove the invalid CollapseShapeOp that references a
    // CB once it has no more uses.
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      rewriter.eraseOp(op->getOperand(i)
                           .getDefiningOp<memref::LoadOp>()
                           .getMemref()
                           .getDefiningOp<memref::CollapseShapeOp>());
      rewriter.eraseOp(op->getOperand(i).getDefiningOp<memref::LoadOp>());
    }

    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {

template <typename T>
class TTIRAwaitYieldRewriter : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto i32 = [&](int32_t value) {
      return rewriter
          .create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(value))
          .getResult();
    };

    for (Value input : adaptor.getValues()) {
      auto cb = mlir::dyn_cast<ttkernel::CBType>(input.getType());
      assert(cb && "Expected CB input type to await/yield, failing.");
      auto memref = cb.getMemref();
      auto numPages = i32(memref.getNumElements());
      Block *block = op->getBlock();
      if (mlir::isa<ttir::AwaitOp>(op)) {
        rewriter.create<ttkernel::CBWaitFrontOp>(op.getLoc(), input, numPages);
        auto popFront = rewriter.create<ttkernel::CBPopFrontOp>(
            op.getLoc(), input, numPages);
        rewriter.moveOpBefore(popFront, block->getTerminator());
      } else if (mlir::isa<ttir::YieldOp>(op)) {
        auto reserveBack = rewriter.create<ttkernel::CBReserveBackOp>(
            op.getLoc(), input, numPages);
        rewriter.moveOpBefore(reserveBack, &block->front());
        rewriter.create<ttkernel::CBPushBackOp>(op.getLoc(), input, numPages);
        rewriter.moveOpBefore(numPages.getDefiningOp(), reserveBack);
      }
    }

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace

namespace {

class TTIRDMARewriter : public OpConversionPattern<ttir::DMAOp> {
public:
  using OpConversionPattern<ttir::DMAOp>::OpConversionPattern;

  static std::pair<Value, Value>
  getVirtualCoordsFromLogicalCoords(PatternRewriter &rewriter, Location loc,
                                    ChipDescAttr chipDesc,
                                    ValueRange dstCoreIndex) {
    auto index = [&](int64_t value) {
      return rewriter
          .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                     rewriter.getIndexAttr(value))
          .getResult();
    };

    std::pair<Value, Value> nocCoords;
    auto offset = chipDesc.getChipPhysicalCores().getWorker().front();
    nocCoords.first =
        rewriter
            .create<arith::AddIOp>(dstCoreIndex[0].getLoc(), dstCoreIndex[0],
                                   index(offset.getY()))
            .getResult();
    nocCoords.second =
        rewriter
            .create<arith::AddIOp>(dstCoreIndex[1].getLoc(), dstCoreIndex[1],
                                   index(offset.getX()))
            .getResult();
    return nocCoords;
  }

  static std::pair<Value, Value>
  getMcastEndCoords(PatternRewriter &rewriter, Location loc, Value &nocStartY,
                    Value &nocStartX, OperandRange mcastShape) {
    auto index = [&](int64_t value) {
      return rewriter
          .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                     rewriter.getIndexAttr(value))
          .getResult();
    };

    std::pair<Value, Value> nocEndCoords;
    nocEndCoords.first = rewriter.create<arith::SubIOp>(
        nocStartY.getLoc(),
        rewriter.create<arith::AddIOp>(nocStartY.getLoc(), nocStartY,
                                       mcastShape[0]),
        index(1));
    nocEndCoords.second = rewriter.create<arith::SubIOp>(
        nocStartX.getLoc(),
        rewriter.create<arith::AddIOp>(nocStartX.getLoc(), nocStartX,
                                       mcastShape[1]),
        index(1));
    return nocEndCoords;
  }

  static size_t getElementSizeBytes(MemRefType memref) {
    mlir::Type elementType = memref.getElementType();
    auto tileType = mlir::dyn_cast<TileType>(elementType);
    return tileType ? tileType.getSizeBytes()
                    : elementType.getIntOrFloatBitWidth() / 8;
  }

  static int64_t getMemrefSizeBytes(MemRefType memref) {
    return getElementSizeBytes(memref) * memref.getNumElements();
  }

  // For use on REMOTE memrefs
  static size_t getMemrefShardSizeBytes(MemRefType memref) {
    ArrayRef<int64_t> memrefShardShape =
        memref.getShape().drop_front(memref.getRank() / 2);
    return std::accumulate(memrefShardShape.begin(), memrefShardShape.end(),
                           getElementSizeBytes(memref),
                           std::multiplies<int64_t>());
  }

  // For use on REMOTE memrefs
  static size_t getMemrefShardNumElems(MemRefType memref) {
    ArrayRef<int64_t> memrefShardShape =
        memref.getShape().drop_front(memref.getRank() / 2);
    return std::accumulate(memrefShardShape.begin(), memrefShardShape.end(), 1,
                           std::multiplies<int64_t>());
  }

  static std::tuple<AffineMap, AffineMap, AffineMap>
  getIndividualResultMaps(MemRefType memref, tt::DeviceAttr device,
                          OpBuilder &builder) {
    size_t pageSize = getMemrefShardSizeBytes(memref);
    AffineMap memoryMap = device.getMemoryMap(memref, pageSize, 0)
                              .dropResult(0); // drop the device index
    assert(memoryMap.getNumResults() == 3);
    auto gridY = memoryMap.dropResults({1, 2});
    auto gridX = memoryMap.dropResults({0, 2});
    auto offset = memoryMap.dropResults({0, 1});
    return std::make_tuple(gridY, gridX, offset);
  }

  LogicalResult
  matchAndRewrite(ttir::DMAOp op, ttir::DMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto i32 = [&](int32_t value) {
      return rewriter
          .create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(value))
          .getResult();
    };

    auto isCb = [&](Value value) {
      return mlir::isa<ttkernel::CBType>(value.getType());
    };

    auto device = lookupDevice(op);
    auto chipIds = device.getChipIds();
    auto chipDescs =
        op->getParentOfType<ModuleOp>()
            ->getAttrOfType<SystemDescAttr>(tt::SystemDescAttr::name)
            .getChipDescs();
    assert((chipIds.size() == 1) && (chipDescs.size() == 1) &&
           "Chip ids and chip descs size must equal 1, failing.");
    // assert(isL1MemorySpace(mlir::cast<MemorySpaceAttr>(
    //                            op.getSrc().getType().getMemorySpace())
    //                            .getValue()) &&
    //        isL1MemorySpace(mlir::cast<MemorySpaceAttr>(
    //                            op.getDst().getType().getMemorySpace())
    //                            .getValue()) &&
    //        "Expected src and dst memory spaces to be L1, failing.");

    auto applyMap = [&](AffineMap map, ValueRange index) {
      auto apply =
          rewriter.create<affine::AffineApplyOp>(op.getLoc(), map, index);
      return apply;
    };

    if (isCb(adaptor.getSrc()) && isCb(adaptor.getDst())) {
      // local movmement, mcast

      auto srcCb = mlir::dyn_cast<ttkernel::CBType>(adaptor.getSrc().getType());

      Value srcL1Start = rewriter.create<ttkernel::GetReadPtrOp>(
          op.getLoc(), adaptor.getSrc());
      Value dstL1Start = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), adaptor.getDst());

      // AffineMap srcGridYMap, srcGridXMap, srcOffsetMap;
      // std::tie(srcGridYMap, srcGridXMap, srcOffsetMap) =
      //     getIndividualResultMaps(op.getSrcMemRefType(), device, rewriter);

      // auto srcOffset =
      //     applyMap(rewriter, op.getLoc(), srcOffsetMap, op.getSrcIndices());
      // auto srcOffsetInt = rewriter.create<arith::IndexCastOp>(
      //     op.getLoc(), rewriter.getI32Type(), srcOffset);
      // auto srcAddrInt =
      //     rewriter.create<arith::AddIOp>(op.getLoc(), srcOffsetInt,
      //     srcL1Start);

      // AffineMap dstGridYMap, dstGridXMap, dstOffsetMap;
      // std::tie(dstGridYMap, dstGridXMap, dstOffsetMap) =
      //     getIndividualResultMaps(op.getDstMemRefType(), device, rewriter);

      // auto dstOffset =
      //     applyMap(rewriter, op.getLoc(), dstOffsetMap, op.getDstIndices());
      // auto dstOffsetInt = rewriter.create<arith::IndexCastOp>(
      //     op.getLoc(), rewriter.getI32Type(), dstOffset);
      // auto dstAddrInt = rewriter.create<arith::AddIOp>(op.getLoc(),
      // dstOffsetInt, dstL1Start);

      Value transferSize = i32(getMemrefSizeBytes(srcCb.getMemref()));
      // local movement
      if (op.isMcast()) {
        // mcast
        auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
            rewriter, op.getLoc(), chipDescs.front(), op.getMcastStartIndex());
        auto [mcastEndY, mcastEndX] = getMcastEndCoords(
            rewriter, op.getLoc(), virtY, virtX, op.getMcastShape());
        auto numDestsIdx = rewriter.create<arith::MulIOp>(
            op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
        auto numDests = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getI32Type(), numDestsIdx);
        auto mcastAddr = rewriter.create<ttkernel::GetNocMulticastAddrOp>(
            op.getLoc(), virtX, virtY, mcastEndX, mcastEndY, dstL1Start,
            nullptr);
        if (adaptor.getSrc() == adaptor.getDst()) {
          // no loopback
          auto numDestsLessOne =
              rewriter.create<arith::SubIOp>(op.getLoc(), numDests, i32(1));
          rewriter.create<ttkernel::NocAsyncWriteMulticastOp>(
              op.getLoc(), srcL1Start, mcastAddr, transferSize, numDestsLessOne,
              nullptr, nullptr, nullptr);
        } else {
          // loopback
          rewriter.create<ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>(
              op.getLoc(), srcL1Start, mcastAddr, transferSize, numDests,
              nullptr, nullptr, nullptr);
        }
      } else {
        // local movement
        auto myY = rewriter.create<ttir::CoreIndexOp>(
            op.getLoc(), rewriter.getIndexType(),
            rewriter.getI64IntegerAttr(0));
        auto myX = rewriter.create<ttir::CoreIndexOp>(
            op.getLoc(), rewriter.getIndexType(),
            rewriter.getI64IntegerAttr(1));
        auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
            rewriter, op.getLoc(), chipDescs.front(), ValueRange{myY, myX});
        auto nocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
            op.getLoc(), virtX, virtY, dstL1Start);
        rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Start,
                                                   nocAddr, transferSize);
      }
    } else if (isCb(adaptor.getSrc()) && op.isDstRemote()) {
      // local to remote dram/l1
      if (!op.getOptNumElems()) {
        op.setOptNumElems(getMemrefShardNumElems(op.getDst().getType()));
      }
      auto transferSize =
          i32(op.getNumElems() * getElementSizeBytes(op.getDstMemRefType()));

      Value srcL1Start = rewriter.create<ttkernel::GetReadPtrOp>(
          op.getLoc(), adaptor.getSrc());

      // AffineMap srcGridYMap, srcGridXMap, srcOffsetMap;
      // std::tie(srcGridYMap, srcGridXMap, srcOffsetMap) =
      //     getIndividualResultMaps(op.getSrcMemRefType(), device, rewriter);

      // auto srcOffset =
      //     applyMap(rewriter, op.getLoc(), srcOffsetMap, op.getSrcIndices());
      // auto srcOffsetInt = rewriter.create<arith::IndexCastOp>(
      //     op.getLoc(), rewriter.getI32Type(), srcOffset);

      // auto srcAddrInt =
      //     rewriter.create<arith::AddIOp>(op.getLoc(), srcOffsetInt,
      //     srcL1Start);

      AffineMap dstGridYMap, dstGridXMap, dstOffsetMap;
      std::tie(dstGridYMap, dstGridXMap, dstOffsetMap) =
          getIndividualResultMaps(op.getDstMemRefType(), device, rewriter);

      auto dstGridY = applyMap(dstGridYMap, op.getDstIndices());
      auto dstGridX = applyMap(dstGridXMap, op.getDstIndices());
      auto dstOffset = applyMap(dstOffsetMap, op.getDstIndices());
      auto dstOffsetInt = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), dstOffset);

      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDescs.front(),
          ValueRange{dstGridY, dstGridX});
      auto nocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
          op.getLoc(), virtX, virtY, dstOffsetInt);
      rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Start,
                                                 nocAddr, transferSize);
    } else if (op.isSrcRemote() && isCb(adaptor.getDst())) {
      if (!op.getOptNumElems()) {
        op.setOptNumElems(getMemrefShardNumElems(op.getSrc().getType()));
      }
      Value dstL1Start = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), adaptor.getDst());

      // AffineMap dstGridYMap, dstGridXMap, dstOffsetMap;
      // std::tie(dstGridYMap, dstGridXMap, dstOffsetMap) =
      //     getIndividualResultMaps(op.getDstMemRefType(), device, rewriter);

      // auto dstOffset =
      //     applyMap(rewriter, op.getLoc(), dstOffsetMap, op.getDstIndices());
      // auto dstOffsetInt = rewriter.create<arith::IndexCastOp>(
      //     op.getLoc(), rewriter.getI32Type(), dstOffset);
      // auto dstAddrInt =
      //     rewriter.create<arith::AddIOp>(op.getLoc(), dstOffsetInt,
      //     dstL1Start);

      AffineMap srcGridYMap, srcGridXMap, srcOffsetMap;
      std::tie(srcGridYMap, srcGridXMap, srcOffsetMap) =
          getIndividualResultMaps(op.getSrcMemRefType(), device, rewriter);

      auto srcGridY = applyMap(srcGridYMap, op.getSrcIndices());
      auto srcGridX = applyMap(srcGridXMap, op.getSrcIndices());
      auto srcOffset = applyMap(srcOffsetMap, op.getSrcIndices());
      auto srcOffsetInt = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), srcOffset);
      auto size =
          i32(op.getNumElems() * getElementSizeBytes(op.getSrcMemRefType()));
      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDescs.front(),
          ValueRange{srcGridY, srcGridX});
      auto srcNocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
          op.getLoc(), virtX, virtY, srcOffsetInt);
      rewriter.create<ttkernel::NocAsyncReadOp>(op.getLoc(), srcNocAddr,
                                                dstL1Start, size);
    } else {
      assert(false && "Illegal DMA configuration");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace
namespace {

class TTIRCoreIndexRewriter : public OpConversionPattern<ttir::CoreIndexOp> {
public:
  using OpConversionPattern<ttir::CoreIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CoreIndexOp op, ttir::CoreIndexOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto index = [&](int64_t value) {
      return rewriter
          .create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(),
                                     rewriter.getIndexAttr(value))
          .getResult();
    };

    auto device = lookupDevice(op);
    auto chipIds = device.getChipIds();
    auto chipDescs =
        op->getParentOfType<ModuleOp>()
            ->getAttrOfType<SystemDescAttr>(tt::SystemDescAttr::name)
            .getChipDescs();
    assert(chipIds.size() == chipDescs.size() == 1 &&
           "Chip ids and chip descs size must equal 1, failing.");

    assert(op.getDim() == 0 ||
           op.getDim() == 1 &&
               "Expected core index dim to be in range 0-1, failing.");
    if (op.getDim()) {
      auto coreIndex = rewriter.create<ttkernel::MyXOp>(op.getLoc(), nullptr);
      auto normalizedCoreIndex =
          rewriter.create<arith::SubIOp>(op.getLoc(), coreIndex,
                                         index(chipDescs.front()
                                                   .getChipPhysicalCores()
                                                   .getWorker()
                                                   .front()
                                                   .getX()));
      rewriter.replaceOp(op, normalizedCoreIndex);
    } else {
      auto coreIndex = rewriter.create<ttkernel::MyYOp>(op.getLoc(), nullptr);
      auto normalizedCoreIndex =
          rewriter.create<arith::SubIOp>(op.getLoc(), coreIndex,
                                         index(chipDescs.front()
                                                   .getChipPhysicalCores()
                                                   .getWorker()
                                                   .front()
                                                   .getY()));
      rewriter.replaceOp(op, normalizedCoreIndex);
    }
    return success();
  }
};

} // namespace

namespace {
class TTIRDMAWaitRewriter : public OpConversionPattern<ttir::DMAWaitOp> {
public:
  using OpConversionPattern<ttir::DMAWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::DMAWaitOp op, ttir::DMAWaitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto isCb = [&](OpOperand &value) {
      return mlir::isa<ttkernel::CBType>(value.get().getType());
    };

    if (!isCb(op.getMemTx().getDefiningOp<ttir::DMAOp>().getSrcMutable())) {
      rewriter.replaceOpWithNewOp<ttkernel::NocAsyncReadBarrierOp>(op);
    } else {
      rewriter.replaceOpWithNewOp<ttkernel::NocAsyncWriteBarrierOp>(op);
    }
    return success();
  }
};
} // namespace

namespace {

class TTIRGetGlobalOperandRewriter
    : public OpConversionPattern<ttir::GetGlobalOperandOp> {
public:
  using OpConversionPattern<ttir::GetGlobalOperandOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GetGlobalOperandOp op,
                  ttir::GetGlobalOperandOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(
        op, rewriter.getI32Type(), op.getOperandIndex());
    return success();
  }
};

} // namespace

namespace {
class TTIRKernelFunctionArgsRewriter
    : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  static int64_t getSemId(func::FuncOp op, BlockArgument ttirSem,
                          PatternRewriter &rewriter) {
    auto numMemrefArgs = std::count_if(
        op.getArguments().begin(), op.getArguments().end(),
        [&](Value arg) { return mlir::isa<MemRefType>(arg.getType()); });

    return std::find(op.getArguments().begin(), op.getArguments().end(),
                     ttirSem) -
           op.getArguments().begin() - numMemrefArgs;
  }

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    if (!op->hasAttr("ttir.thread_type")) { // TODO(nsmith/jdesousa): String
                                            // constant somewhere for this?
      return failure();
    }

    // Collect the new argument types
    SmallVector<Type> newArgTypes;

    for (auto funcArg : op.getArguments()) {
      if (auto memrefType = mlir::dyn_cast<MemRefType>(funcArg.getType())) {
        auto cbType = ttkernel::CBType::get(
            rewriter.getContext(),
            ttkernel::symbolizeCBPort(funcArg.getArgNumber()).value(), 0,
            memrefType);
        newArgTypes.push_back(cbType);
      } else {
        // Keep the original type for non-memref arguments (like semaphores)
        newArgTypes.push_back(funcArg.getType());
      }
    }

    // Update the function type
    auto newFuncType =
        op.getFunctionType().clone(newArgTypes, op->getResultTypes());
    rewriter.modifyOpInPlace(op, [&]() { op.setType(newFuncType); });

    // Update the block argument types
    for (uint32_t i = 0; i < op.getNumArguments(); i++) {
      if (op.getArgument(i).getType() != newArgTypes[i]) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.getArgument(i).setType(newArgTypes[i]); });
      }
    }

    return success();
  }
};
} // namespace

} // namespace mlir::tt::ttkernel

namespace mlir::tt {

void populateTTIRToTTKernelPatterns(MLIRContext *ctx,
                                    RewritePatternSet &patterns,
                                    TypeConverter & /*typeConverter*/) {

  patterns.add<ttkernel::TTIRComputeOpsRewriter, ttkernel::MemrefStoreRewriter,
               ttkernel::TTIRAwaitYieldRewriter<ttir::AwaitOp>,
               ttkernel::TTIRAwaitYieldRewriter<ttir::YieldOp>,
               ttkernel::TTIRDMARewriter, ttkernel::TTIRDMAWaitRewriter,
               ttkernel::TTIRCoreIndexRewriter,
               ttkernel::TTIRGetGlobalOperandRewriter,
               ttkernel::TTIRKernelFunctionArgsRewriter>(ctx);
}

} // namespace mlir::tt
