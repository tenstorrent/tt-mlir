// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
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

namespace {

static Value i32(PatternRewriter &rewriter, Location loc, int32_t value) {
  return rewriter
      .create<arith::ConstantOp>(loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(value))
      .getResult();
}

static Value index(PatternRewriter &rewriter, Location loc, int64_t value) {
  return rewriter
      .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                 rewriter.getIndexAttr(value))
      .getResult();
}

} // namespace

namespace {

class MemrefStoreRewriter : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, memref::StoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto cb = adaptor.getMemref();
    auto storeIdx = op.getIndices().front();
    rewriter.replaceOpWithNewOp<ttkernel::PackTileOp>(
        op, index(rewriter, op->getLoc(), 0), cb, storeIdx,
        rewriter.getBoolAttr(true));

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

  static Value getCB(ConversionPatternRewriter &rewriter, Value cb) {
    memref::LoadOp loadOp = mlir::dyn_cast<memref::LoadOp>(cb.getDefiningOp());
    assert(loadOp && "Expected load op, failing.");
    assert(loadOp.getIndices().size() == 1 &&
           "Expected single index in load op, failing.");
    return rewriter.getRemappedValue(loadOp.getMemref());
  }

  static Value getLoadIndex(Value tile) {
    memref::LoadOp loadOp =
        mlir::dyn_cast<memref::LoadOp>(tile.getDefiningOp());
    assert(loadOp && "Expected load op, failing.");
    assert(loadOp.getIndices().size() == 1 &&
           "Expected single index in load op, failing.");
    return loadOp.getIndices().front();
  }

  static void lowerLoadToCopyTile(memref::LoadOp op, bool cbIdxAsDstIdx,
                                  ConversionPatternRewriter &rewriter) {
    auto index = [&](int64_t value) {
      return rewriter
          .create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(),
                                     rewriter.getIndexAttr(value))
          .getResult();
    };

    auto cb = rewriter.getRemappedValue(op.getMemref());
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
    Operation *newOp;

    if (mlir::isa<ttir::TileMaximumOp>(op)) {
      rewriter.create<ttkernel::MaxTilesInitOp>(op->getLoc());
      newOp = rewriter.create<ttkernel::MaxTilesOp>(
          op->getLoc(), i32(rewriter, op->getLoc(), 0),
          i32(rewriter, op->getLoc(), 1));
    } else if (mlir::isa<ttir::TileAddOp>(op)) {
      auto dstIdx = index(rewriter, op->getLoc(), 0);
      newOp = rewriter.create<ttkernel::AddTilesOp>(
          op->getLoc(), getCB(rewriter, operands[0]),
          getCB(rewriter, operands[1]), getLoadIndex(operands[0]),
          getLoadIndex(operands[1]), dstIdx);
    } else if (mlir::isa<ttir::TileMatmulOp>(op)) {
      auto dstIdx = index(rewriter, op->getLoc(), 0);
      newOp = rewriter.create<ttkernel::MatmulTilesOp>(
          op->getLoc(), getCB(rewriter, operands[0]),
          getCB(rewriter, operands[1]), getLoadIndex(operands[0]),
          getLoadIndex(operands[1]), dstIdx);
    } else if (mlir::isa<ttir::TileTilizeBlockOp>(op)) {
      assert(operands.size() == 2);
      Value src = operands[0];
      Value dst = operands[1];
      auto numTiles =
          i32(rewriter, op->getLoc(),
              mlir::cast<ttkernel::CBType>(dst.getType()).getNumTiles());
      newOp = rewriter.create<ttkernel::TilizeBlockOp>(op->getLoc(), src,
                                                       numTiles, dst);
    } else if (mlir::isa<ttir::TileUntilizeBlockOp>(op)) {
      assert(operands.size() == 2);
      Value src = operands[0];
      Value dst = operands[1];
      auto numTiles =
          i32(rewriter, op->getLoc(),
              mlir::cast<ttkernel::CBType>(src.getType()).getNumTiles());
      newOp = rewriter.create<ttkernel::UntilizeBlockOp>(op->getLoc(), src,
                                                         numTiles, dst);
    } else {
      return failure();
    }

    rewriter.setInsertionPoint(newOp);
    if (mlir::isa<ttkernel::MatmulTilesOp>(newOp)) {
      lowerLoadToCopyTile(operands[2].getDefiningOp<memref::LoadOp>(), false,
                          rewriter);
    } else if (newOp->hasTrait<TTKernelSFPUOpTrait>()) {
      for (uint32_t i = 0; i < op->getNumOperands(); i++) {
        lowerLoadToCopyTile(operands[i].getDefiningOp<memref::LoadOp>(), true,
                            rewriter);
      }
    }

    // This is necessary to remove the invalid CollapseShapeOp that references a
    // CB once it has no more uses.
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto load = operands[i].getDefiningOp<memref::LoadOp>();
      if (load) {
        rewriter.eraseOp(load);
      }
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

    for (Value input : adaptor.getValues()) {
      auto cb = mlir::dyn_cast<ttkernel::CBType>(input.getType());
      assert(cb && "Expected CB input type to await/yield, failing.");
      auto memref = cb.getMemref();
      auto numPages = i32(rewriter, op->getLoc(), memref.getNumElements());
      Block *block = op->getBlock();
      if (mlir::isa<ttir::AwaitOp>(op)) {
        rewriter.create<ttkernel::CBWaitFrontOp>(op.getLoc(), input, numPages);
        auto popFront = rewriter.create<ttkernel::CBPopFrontOp>(
            op.getLoc(), input, numPages);
        rewriter.moveOpBefore(popFront, block->getTerminator());
      } else if (mlir::isa<ttir::YieldOp>(op)) {
        auto reserveBack = rewriter.create<ttkernel::CBReserveBackOp>(
            op.getLoc(), input, numPages);
        if (mlir::isa<func::FuncOp>(block->getParentOp())) {
          rewriter.moveOpAfter(reserveBack, input.getDefiningOp());
        } else {
          rewriter.moveOpBefore(reserveBack, &block->front());
        }
        rewriter.moveOpBefore(numPages.getDefiningOp(), reserveBack);
        rewriter.create<ttkernel::CBPushBackOp>(op.getLoc(), input, numPages);
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
  TTIRDMARewriter(TypeConverter &typeConverter, MLIRContext *context,
                  ttir::AssociatedDMAWaits const *associatedDMAWaits)
      : OpConversionPattern<ttir::DMAOp>(typeConverter, context),
        associatedDMAWaits(associatedDMAWaits) {}

  static std::pair<Value, Value>
  getVirtualCoordsFromLogicalCoords(PatternRewriter &rewriter, Location loc,
                                    ChipDescAttr chipDesc,
                                    ValueRange dstCoreIndex) {
    std::pair<Value, Value> nocCoords;
    auto offset = chipDesc.getChipPhysicalCores().getWorker().front();
    nocCoords.first =
        rewriter
            .create<arith::AddIOp>(dstCoreIndex[0].getLoc(), dstCoreIndex[0],
                                   index(rewriter, loc, offset.getY()))
            .getResult();
    nocCoords.second =
        rewriter
            .create<arith::AddIOp>(dstCoreIndex[1].getLoc(), dstCoreIndex[1],
                                   index(rewriter, loc, offset.getX()))
            .getResult();
    return nocCoords;
  }

  static std::pair<Value, Value>
  getMcastEndCoords(PatternRewriter &rewriter, Location loc, Value &nocStartY,
                    Value &nocStartX, OperandRange mcastShape) {
    std::pair<Value, Value> nocEndCoords;
    nocEndCoords.first = rewriter.create<arith::SubIOp>(
        nocStartY.getLoc(),
        rewriter.create<arith::AddIOp>(nocStartY.getLoc(), nocStartY,
                                       mcastShape[0]),
        index(rewriter, loc, 1));
    nocEndCoords.second = rewriter.create<arith::SubIOp>(
        nocStartX.getLoc(),
        rewriter.create<arith::AddIOp>(nocStartX.getLoc(), nocStartX,
                                       mcastShape[1]),
        index(rewriter, loc, 1));
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
    auto device = lookupDevice(op);
    auto systemDesc = getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    auto chipDescs = systemDesc.getChipDescs();
    auto chipDescIndices = systemDesc.getChipDescIndices();
    assert(chipIds.size() == 1);
    auto chipDesc = chipDescs[chipDescIndices[chipIds[0]]];

    // TODO(jdesousa): Temporary L1 assertion until DRAM is supported
    assert(isL1MemorySpace(mlir::cast<MemorySpaceAttr>(
                               op.getSrc().getType().getMemorySpace())
                               .getValue()) &&
           isL1MemorySpace(mlir::cast<MemorySpaceAttr>(
                               op.getDst().getType().getMemorySpace())
                               .getValue()) &&
           "Expected src and dst memory spaces to be L1, failing.");

    auto applyMap = [&](AffineMap map, ValueRange index) {
      auto apply =
          rewriter.create<affine::AffineApplyOp>(op.getLoc(), map, index);
      return apply;
    };

    bool isRead = false;
    if (op.isSrcLocal() && op.isDstLocal()) {
      // local movmement, mcast

      auto srcCb = mlir::dyn_cast<ttkernel::CBType>(adaptor.getSrc().getType());

      Value srcL1Start = rewriter.create<ttkernel::GetReadPtrOp>(
          op.getLoc(), adaptor.getSrc());
      Value dstL1Start = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), adaptor.getDst());

      Value transferSize =
          i32(rewriter, op->getLoc(), getMemrefSizeBytes(srcCb.getMemref()));
      // local movement
      if (op.isMcast()) {
        // mcast
        auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
            rewriter, op.getLoc(), chipDesc, op.getMcastStartIndex());
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
          auto numDestsLessOne = rewriter.create<arith::SubIOp>(
              op.getLoc(), numDests, i32(rewriter, op->getLoc(), 1));
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
            rewriter, op.getLoc(), chipDesc, ValueRange{myY, myX});
        auto nocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
            op.getLoc(), virtX, virtY, dstL1Start);
        rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Start,
                                                   nocAddr, transferSize);
      }
    } else if (op.isSrcLocal() && op.isDstRemote()) {
      // local to remote dram/l1
      if (!op.getOptNumElems()) {
        op.setOptNumElems(getMemrefShardNumElems(op.getDst().getType()));
      }
      auto transferSize =
          i32(rewriter, op->getLoc(),
              op.getNumElems() * getElementSizeBytes(op.getDstMemRefType()));

      Value srcL1Start = rewriter.create<ttkernel::GetReadPtrOp>(
          op.getLoc(), adaptor.getSrc());

      AffineMap dstGridYMap, dstGridXMap, dstOffsetMap;
      std::tie(dstGridYMap, dstGridXMap, dstOffsetMap) =
          getIndividualResultMaps(op.getDstMemRefType(), device, rewriter);

      auto dstGridY = applyMap(dstGridYMap, op.getDstIndices());
      auto dstGridX = applyMap(dstGridXMap, op.getDstIndices());
      auto dstOffset = applyMap(dstOffsetMap, op.getDstIndices());
      auto dstOffsetInt = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), dstOffset);

      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, ValueRange{dstGridY, dstGridX});
      auto nocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
          op.getLoc(), virtX, virtY, dstOffsetInt);
      rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Start,
                                                 nocAddr, transferSize);
    } else if (op.isSrcRemote() && op.isDstLocal()) {
      if (!op.getOptNumElems()) {
        op.setOptNumElems(getMemrefShardNumElems(op.getSrc().getType()));
      }
      Value dstL1Start = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), adaptor.getDst());

      AffineMap srcGridYMap, srcGridXMap, srcOffsetMap;
      std::tie(srcGridYMap, srcGridXMap, srcOffsetMap) =
          getIndividualResultMaps(op.getSrcMemRefType(), device, rewriter);

      auto srcGridY = applyMap(srcGridYMap, op.getSrcIndices());
      auto srcGridX = applyMap(srcGridXMap, op.getSrcIndices());
      auto srcOffset = applyMap(srcOffsetMap, op.getSrcIndices());
      auto srcOffsetInt = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), srcOffset);
      auto size =
          i32(rewriter, op->getLoc(),
              op.getNumElems() * getElementSizeBytes(op.getSrcMemRefType()));
      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, ValueRange{srcGridY, srcGridX});
      auto srcNocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
          op.getLoc(), virtX, virtY, srcOffsetInt);
      rewriter.create<ttkernel::NocAsyncReadOp>(op.getLoc(), srcNocAddr,
                                                dstL1Start, size);
      isRead = true;
    } else {
      emitError(op.getLoc(), "Unsupported DMA Configuration");
      return failure();
    }

    auto dmaWaitOps = associatedDMAWaits->get(op);
    for (auto dmaWaitOp : dmaWaitOps) {
      rewriter.modifyOpInPlace(dmaWaitOp, [&]() {
        if (isRead) {
          dmaWaitOp->setDiscardableAttr("ttkernel.lowering.associated_noc_read",
                                        rewriter.getUnitAttr());
        } else {
          dmaWaitOp->setDiscardableAttr(
              "ttkernel.lowering.associated_noc_write", rewriter.getUnitAttr());
        }
      });
    }

    rewriter.replaceOpWithNewOp<ttir::NullTxOp>(op);
    return success();
  }

private:
  ttir::AssociatedDMAWaits const *associatedDMAWaits;
};
} // namespace

namespace {
class TTIRCoreIndexRewriter : public OpConversionPattern<ttir::CoreIndexOp> {
public:
  using OpConversionPattern<ttir::CoreIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CoreIndexOp op, ttir::CoreIndexOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = lookupDevice(op);
    auto systemDesc = getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    auto chipDescs = systemDesc.getChipDescs();
    auto chipDescIndices = systemDesc.getChipDescIndices();
    assert(chipIds.size() == 1);
    auto chipDesc = chipDescs[chipDescIndices[chipIds[0]]];

    assert(op.getDim() == 0 ||
           op.getDim() == 1 &&
               "Expected core index dim to be in range 0-1, failing.");
    if (op.getDim()) {
      auto coreIndex = rewriter.create<ttkernel::MyXOp>(op.getLoc(), nullptr);
      auto normalizedCoreIndex = rewriter.create<arith::SubIOp>(
          op.getLoc(), coreIndex,
          index(rewriter, op->getLoc(),
                chipDesc.getChipPhysicalCores().getWorker().front().getX()));
      rewriter.replaceOp(op, normalizedCoreIndex);
    } else {
      auto coreIndex = rewriter.create<ttkernel::MyYOp>(op.getLoc(), nullptr);
      auto normalizedCoreIndex = rewriter.create<arith::SubIOp>(
          op.getLoc(), coreIndex,
          index(rewriter, op->getLoc(),
                chipDesc.getChipPhysicalCores().getWorker().front().getY()));
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
    auto isRead =
        op->getDiscardableAttr("ttkernel.lowering.associated_noc_read");
    auto isWrite =
        op->getDiscardableAttr("ttkernel.lowering.associated_noc_write");
    assert(isRead || isWrite);

    if (isRead) {
      rewriter.create<ttkernel::NocAsyncReadBarrierOp>(op.getLoc());
    }

    if (isWrite) {
      rewriter.create<ttkernel::NocAsyncWriteBarrierOp>(op.getLoc());
    }

    rewriter.eraseOp(op);
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
        op, rewriter.getI32Type(),
        i32(rewriter, op->getLoc(), op.getOperandIndex()));
    return success();
  }
};
} // namespace

namespace {
class TTIRNullTxRewriter : public OpConversionPattern<ttir::NullTxOp> {
public:
  using OpConversionPattern<ttir::NullTxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::NullTxOp op, ttir::NullTxOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(0));
    return success();
  }
};
} // namespace

namespace {
class MemRefCollapseRewriter
    : public OpConversionPattern<memref::CollapseShapeOp> {
public:
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op,
                  memref::CollapseShapeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<CBReinterpretShapeOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getSrc());
    return success();
  }
};
} // namespace

namespace {
class TTIRKernelFunctionArgsRewriter
    : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  static void convertFunctionAttrs(Builder &builder, func::FuncOp op) {
    ttir::ThreadAttr threadAttr =
        op->getAttrOfType<ttir::ThreadAttr>(ttir::ThreadAttr::name);
    op->removeAttr(ttir::ThreadAttr::name);
    ThreadType threadType;
    switch (threadAttr.getThreadType()) {
    case ttir::ThreadType::Compute: {
      threadType = ThreadType::Tensix;
      break;
    }
    case ttir::ThreadType::Datamovement: {
      threadType = ThreadType::Noc;
      break;
    }
    }
    op->setAttr(ThreadTypeAttr::name,
                builder.getAttr<ThreadTypeAttr>(threadType));
  }

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op->hasAttr(ttir::ThreadAttr::name) ||
        (op.getFunctionType().getNumInputs() == 0)) {
      return failure();
    }

    Block *block = &op.getCallableRegion()->front();
    auto blockArgs = block->getArguments();
    assert(!blockArgs.empty());

    TypeConverter::SignatureConversion signatureConverter(op.getNumArguments());
    OpBuilder::InsertionGuard funcInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(block);
    for (auto arg : blockArgs) {
      auto cb = rewriter.create<GetCBOp>(
          op.getLoc(), getTypeConverter()->convertType(arg.getType()),
          rewriter.getI32IntegerAttr(arg.getArgNumber()));
      signatureConverter.remapInput(arg.getArgNumber(), cb);
    }

    rewriter.applySignatureConversion(block, signatureConverter,
                                      getTypeConverter());
    rewriter.modifyOpInPlace(op, [&]() {
      op.setType(rewriter.getFunctionType(TypeRange(), TypeRange()));
      convertFunctionAttrs(rewriter, op);
    });
    return success();
  }
};
} // namespace

namespace {
template <typename T>
class TTIRSemaphoreSetRewriter : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = lookupDevice(op);
    auto systemDesc = getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    auto chipDescs = systemDesc.getChipDescs();
    auto chipDescIndices = systemDesc.getChipDescIndices();
    assert(chipIds.size() == 1);
    auto chipDesc = chipDescs[chipDescIndices[chipIds[0]]];

    Value value = op.getValue();

    Value semaphoreAddr = rewriter.create<ttkernel::GetSemaphoreOp>(
        op.getLoc(), op.getSemaphore());

    if (op.getDstCoreIndex().empty()) {
      if (mlir::isa<ttir::SemaphoreIncOp>(op)) {
        emitError(op.getLoc(), "ttir.semaphore_inc to local core is illegal.");
        return failure();
      }

      // Local semaphore set
      auto semaphorePtr =
          rewriter.create<ttkernel::CastToL1PtrOp>(op.getLoc(), semaphoreAddr);

      if (mlir::isa<ttir::SemaphoreSetOp>(op)) {
        rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreSetOp>(
            op, semaphorePtr, value);
      }
    } else if (op.getMcastShape().empty()) {
      if (mlir::isa<ttir::SemaphoreSetOp>(op)) {
        emitError(op.getLoc(),
                  "ttir.semaphore_set to single remote core is illegal.");
        return failure();
      }
      auto [virtY, virtX] = TTIRDMARewriter::getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, op.getDstCoreIndex());
      auto nocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
          op.getLoc(), virtX, virtY, semaphoreAddr);
      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreIncOp>(op, nocAddr,
                                                               value);
    } else {
      if (mlir::isa<ttir::SemaphoreIncOp>(op)) {
        emitError(op.getLoc(), "ttir.semaphore_inc multicast is illegal.");
        return failure();
      }

      auto [virtY, virtX] = TTIRDMARewriter::getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, op.getDstCoreIndex());
      auto [mcastEndY, mcastEndX] = TTIRDMARewriter::getMcastEndCoords(
          rewriter, op.getLoc(), virtY, virtX, op.getMcastShape());
      Value numDestsIdx = rewriter.create<arith::MulIOp>(
          op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
      Value numDests = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), numDestsIdx);
      auto mcastAddr = rewriter.create<ttkernel::GetNocMulticastAddrOp>(
          op.getLoc(), virtX, virtY, mcastEndX, mcastEndY, semaphoreAddr,
          nullptr);

      rewriter.create<ttkernel::NocSemaphoreSetOp>(op.getLoc(), semaphoreAddr,
                                                   value);
      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreSetMulticastOp>(
          op, semaphoreAddr, mcastAddr, numDests, nullptr, nullptr);
    }

    return success();
  }
};
} // namespace

namespace {
class TTIRSemaphoreWaitRewriter
    : public OpConversionPattern<ttir::SemaphoreWaitOp> {
  using OpConversionPattern<ttir::SemaphoreWaitOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::SemaphoreWaitOp op,
                  ttir::SemaphoreWaitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value semaphoreAddr = rewriter.create<ttkernel::GetSemaphoreOp>(
        op.getLoc(), op.getSemaphore());

    rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreWaitOp>(op, semaphoreAddr,
                                                              op.getValue());
    if (op.getResetValue()) {
      rewriter.create<ttkernel::NocSemaphoreSetOp>(op.getLoc(), semaphoreAddr,
                                                   op.getResetValue());
    }

    return success();
  }
};
} // namespace

} // namespace mlir::tt::ttkernel

namespace mlir::tt {

void populateTTIRToTTKernelPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttir::AssociatedDMAWaits const &associatedDMAWaits) {
  patterns.add<ttkernel::TTIRKernelFunctionArgsRewriter,
               ttkernel::TTIRComputeOpsRewriter, ttkernel::MemrefStoreRewriter,
               ttkernel::TTIRAwaitYieldRewriter<ttir::AwaitOp>,
               ttkernel::TTIRAwaitYieldRewriter<ttir::YieldOp>,
               ttkernel::TTIRDMAWaitRewriter, ttkernel::TTIRCoreIndexRewriter,
               ttkernel::TTIRGetGlobalOperandRewriter,
               ttkernel::TTIRNullTxRewriter, ttkernel::MemRefCollapseRewriter>(
      typeConverter, ctx);

  patterns.add<ttkernel::TTIRDMARewriter>(typeConverter, ctx,
                                          &associatedDMAWaits);
}

} // namespace mlir::tt
