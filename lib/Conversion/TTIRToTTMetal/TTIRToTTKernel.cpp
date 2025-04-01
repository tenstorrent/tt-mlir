// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Dialect/TT/Utils/PhysicalCoreCoord.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>

namespace mlir::tt::ttkernel {

static Value i32(int32_t value, OpBuilder &builder) {
  return builder
      .create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(),
                                 builder.getI32IntegerAttr(value))
      .getResult();
}

static Value index(int64_t value, OpBuilder &builder) {
  return builder
      .create<arith::ConstantOp>(builder.getUnknownLoc(),
                                 builder.getIndexType(),
                                 builder.getIndexAttr(value))
      .getResult();
}

static uint32_t getCbId(Value value) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    return blockArg.getArgNumber();
  }
  memref::CollapseShapeOp collapseOp =
      mlir::dyn_cast<memref::CollapseShapeOp>(value.getDefiningOp());
  if (auto blockArg =
          mlir::dyn_cast<mlir::BlockArgument>(collapseOp.getSrc())) {
    return blockArg.getArgNumber();
  }
  assert(false && "Could not match collapse op src to block argument, cannot "
                  "determine CB id. Failing.");
}

namespace {
class TTIRGenericRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    auto coreRanges = llvm::SmallVector<Attribute>();
    coreRanges.reserve(op.getThreads().size());
    for (size_t i = 0; i < op.getThreads().size(); i++) {
      coreRanges.push_back(
          rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()));
    }
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, op->getResultTypes(), op.getInputs(), op.getOutputs(),
        op.getThreads(), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr({}));
    return success();
  };
};
} // namespace

namespace {

class MemrefStoreRewriter : public OpRewritePattern<memref::StoreOp> {
public:
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const final {
    auto cbId = index(getCbId(op.getMemref()), rewriter);
    auto storeIdx = op.getIndices().front();
    rewriter.create<ttkernel::PackTileOp>(op.getLoc(), index(0, rewriter), cbId,
                                          storeIdx, rewriter.getBoolAttr(true));

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace
namespace {

class TTIRComputeOpsRewriter
    : public OpTraitRewritePattern<
          mlir::OpTrait::tt::ttir::TTIRGenericRegionComputeOpTrait> {
public:
  using OpTraitRewritePattern<
      mlir::OpTrait::tt::ttir::TTIRGenericRegionComputeOpTrait>::
      OpTraitRewritePattern;

  static Value getLoadIndex(Value tile) {
    Operation *loadOp = tile.getDefiningOp();
    assert(mlir::isa<memref::LoadOp>(loadOp) && "Expected load op, failing.");
    return mlir::cast<memref::LoadOp>(loadOp).getIndices().front();
  }

  static bool computeOpInBlock(Block *block) {
    bool computeFound = 0;

    block->walk([&](Operation *walkOp) {
      if (walkOp->hasTrait<OpTrait::TTKernelFPUOpTrait>() ||
          walkOp->hasTrait<OpTrait::TTKernelSFPUOpTrait>()) {
        computeFound = 1;
      }
    });

    return computeFound;
  }

  static void lowerLoad(memref::LoadOp op, bool copyToDst, bool cbIndices,
                        PatternRewriter &rewriter) {
    if (copyToDst) {
      rewriter.setInsertionPoint(op);
      auto cbId = index(getCbId(op.getMemref()), rewriter);
      rewriter.create<ttkernel::CopyTileInitOp>(op.getLoc(), cbId);
      rewriter.create<ttkernel::CopyTileOp>(
          op.getLoc(), cbId, op.getIndices().front(),
          cbIndices ? cbId : i32(0, rewriter));
    }
    rewriter.eraseOp(op);
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto isComputeOpInBlock = computeOpInBlock(op->getBlock());
    Operation *newOp;

    if (mlir::isa<ttir::TileMaximumOp>(op)) {
      rewriter.create<ttkernel::MaxTilesInitOp>(op->getLoc());
      newOp = rewriter.create<ttkernel::MaxTilesOp>(
          op->getLoc(), i32(0, rewriter), i32(1, rewriter));
    } else if (mlir::isa<ttir::TileMatmulOp>(op)) {
      auto dstIdx = rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
      newOp = rewriter.create<ttkernel::MatmulTilesOp>(
          op->getLoc(),
          index(getCbId(op->getOperand(0)
                          .getDefiningOp<memref::LoadOp>()
                          .getMemref()),
              rewriter),
          index(getCbId(op->getOperand(1)
                          .getDefiningOp<memref::LoadOp>()
                          .getMemref()),
              rewriter),
          getLoadIndex(op->getOperand(0)), getLoadIndex(op->getOperand(1)),
          dstIdx);
    } else {
      return failure();
    }

    // Lower memref loads if this is the first compute op to be lowered.
    if (!isComputeOpInBlock) {
      if (mlir::isa<ttkernel::MatmulTilesOp>(newOp)) {
        lowerLoad(op->getOperand(0).getDefiningOp<memref::LoadOp>(), false,
                  false, rewriter);
        lowerLoad(op->getOperand(1).getDefiningOp<memref::LoadOp>(), false,
                  false, rewriter);
        lowerLoad(op->getOperand(2).getDefiningOp<memref::LoadOp>(), true,
                  false, rewriter);
      } else if (newOp->hasTrait<OpTrait::TTKernelSFPUOpTrait>()) {
        for (uint32_t i = 0; i < op->getNumOperands(); i++) {
          lowerLoad(op->getOperand(i).getDefiningOp<memref::LoadOp>(), true,
                    true, rewriter);
        }
      } else {
        for (uint32_t i = 0; i < op->getNumOperands(); i++) {
          lowerLoad(op->getOperand(i).getDefiningOp<memref::LoadOp>(), false,
                    false, rewriter);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {

template <typename T>
class TTIRAwaitYieldRewriter : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
    for (Value input : op.getValues()) {
      auto cbId = index(getCbId(input), rewriter);
      auto type = mlir::cast<MemRefType>(input.getType());
      auto numPages = i32(type.getNumElements(), rewriter);
      Block *block = op->getBlock();
      if (mlir::isa<ttir::AwaitOp>(op)) {
        rewriter.create<ttkernel::CBWaitFrontOp>(op.getLoc(), cbId, numPages);
        auto popFront = rewriter.create<ttkernel::CBPopFrontOp>(op.getLoc(),
                                                                cbId, numPages);
        rewriter.moveOpBefore(popFront, block->getTerminator());
      } else if (mlir::isa<ttir::YieldOp>(op)) {
        auto reserveBack = rewriter.create<ttkernel::CBReserveBackOp>(
            op.getLoc(), cbId, numPages);
        rewriter.moveOpBefore(reserveBack, &block->front());
        rewriter.create<ttkernel::CBPushBackOp>(op.getLoc(), cbId, numPages);
        rewriter.moveOpBefore(cbId.getDefiningOp(), reserveBack);
        rewriter.moveOpBefore(numPages.getDefiningOp(), reserveBack);
      }
    }

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace

namespace {

class MemrefAllocRewriter : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {

    assert(op->getAttr("address") && "No address attribute found, failing.");
    auto address = op->getAttrOfType<IntegerAttr>("address");
    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memroy space found, failing.");
    assert(mlir::isa<TileType>(op.getMemref().getType().getElementType()) &&
           "Expected memref to have tile element type, failing.");
    auto size = mlir::cast<TileType>(op.getMemref().getType().getElementType())
                    .getSizeBytes() *
                op.getMemref().getType().getNumElements();
    auto memorySpace = mlir::cast<tt::MemorySpaceAttr>(
        op.getMemref().getType().getMemorySpace());
    auto createBufferOp = rewriter.create<ttmetal::CreateBufferOp>(
        op->getLoc(), op.getMemref().getType(), address.getInt(), size,
        memorySpace.getValue());
    rewriter.replaceOp(op, createBufferOp);

    return success();
  };
};

} // namespace

namespace {

class TTIRDMARewriter : public OpRewritePattern<ttir::DMAOp> {
public:
  using OpRewritePattern<ttir::DMAOp>::OpRewritePattern;

  static std::pair<Value, Value>
  getVirtualCoordsFromLogicalCoords(PatternRewriter &rewriter,
                                    ChipDescAttr chipDesc,
                                    ValueRange dstCoreIndex) {
    std::pair<Value, Value> nocCoords;
    auto offset = chipDesc.getChipPhysicalCores().getWorker().front();
    nocCoords.first =
        rewriter
            .create<arith::AddIOp>(dstCoreIndex[0].getLoc(), dstCoreIndex[0],
                                   index(offset.getY(), rewriter))
            .getResult();
    nocCoords.second =
        rewriter
            .create<arith::AddIOp>(dstCoreIndex[1].getLoc(), dstCoreIndex[1],
                                   index(offset.getX(), rewriter))
            .getResult();
    return nocCoords;
  }

  static std::pair<Value, Value> getMcastEndCoords(PatternRewriter &rewriter,
                                                   Value &nocStartY,
                                                   Value &nocStartX,
                                                   OperandRange mcastShape) {
    std::pair<Value, Value> nocEndCoords;
    nocEndCoords.first = rewriter.create<arith::SubIOp>(
        nocStartY.getLoc(),
        rewriter.create<arith::AddIOp>(nocStartY.getLoc(), nocStartY,
                                       mcastShape[0]),
        index(1, rewriter));
    nocEndCoords.second = rewriter.create<arith::SubIOp>(
        nocStartX.getLoc(),
        rewriter.create<arith::AddIOp>(nocStartX.getLoc(), nocStartX,
                                       mcastShape[1]),
        index(1, rewriter));
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

  LogicalResult matchAndRewrite(ttir::DMAOp op,
                                PatternRewriter &rewriter) const final {
    auto device = lookupDevice(op);
    auto chipIds = device.getChipIds();
    auto chipDescs =
        op->getParentOfType<ModuleOp>()
            ->getAttrOfType<SystemDescAttr>(tt::SystemDescAttr::name)
            .getChipDescs();
    assert(chipIds.size() == chipDescs.size() == 1 &&
           "Chip ids and chip descs size must equal 1, failing.");
    assert(isL1MemorySpace(mlir::cast<MemorySpaceAttr>(
                               op.getSrc().getType().getMemorySpace())
                               .getValue()) &&
           isL1MemorySpace(mlir::cast<MemorySpaceAttr>(
                               op.getDst().getType().getMemorySpace())
                               .getValue()) &&
           "Expected src and dst memory spaces to be L1, failing.");

    if (op.isSrcLocal() && op.isDstLocal() && !op.getDstCoreIndex().size()) {
      Value srcL1Addr = rewriter.create<ttkernel::GetReadPtrOp>(
          op.getLoc(), index(getCbId(op.getSrc()), rewriter));

      Value dstL1Addr = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), index(getCbId(op.getDst()), rewriter));
      // local movement
      Value transferSize =
          i32(getMemrefSizeBytes(op.getSrcMemRefType()), rewriter);
      auto myY = rewriter.create<ttir::CoreIndexOp>(
          op.getLoc(), rewriter.getIndexType(), rewriter.getI64IntegerAttr(0));
      auto myX = rewriter.create<ttir::CoreIndexOp>(
          op.getLoc(), rewriter.getIndexType(), rewriter.getI64IntegerAttr(1));
      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, chipDescs.front(), ValueRange{myY, myX});
      auto nocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
          op.getLoc(), virtX, virtY, dstL1Addr);
      rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Addr,
                                                 nocAddr, transferSize);
    } else if (op.isSrcLocal() && op.isDstLocal()) {
      // mcast & l1 to l1 single core remote

      Value srcL1Addr = rewriter.create<ttkernel::GetReadPtrOp>(
          op.getLoc(), index(getCbId(op.getSrc()), rewriter));
      Value dstL1Addr = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), index(getCbId(op.getDst()), rewriter));
      Value transferSize =
          i32(getMemrefSizeBytes(op.getSrcMemRefType()), rewriter);
      assert(op.getDstCoreIndex().size() == 2 &&
             "Expected 2 core indices for dst core index, failing.");

      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, chipDescs.front(), op.getDstCoreIndex());

      if (op.isMcast()) {
        // mcast
        auto [mcastEndY, mcastEndX] =
            getMcastEndCoords(rewriter, virtY, virtX, op.getMcastShape());
        auto numDestsIdx = rewriter.create<arith::MulIOp>(
            op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
        auto numDests = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getI32Type(), numDestsIdx);
        auto mcastAddr = rewriter.create<ttkernel::GetNocMulticastAddrOp>(
            op.getLoc(), virtX, virtY, mcastEndX, mcastEndY, dstL1Addr,
            nullptr);
        if (op.getSrc() == op.getDst()) {
          // no loopback
          auto numDestsLessOne = rewriter.create<arith::SubIOp>(
              op.getLoc(), numDests, i32(1, rewriter));
          rewriter.create<ttkernel::NocAsyncWriteMulticastOp>(
              op.getLoc(), srcL1Addr, mcastAddr, transferSize, numDestsLessOne,
              nullptr, nullptr, nullptr);
        } else {
          // loopback
          rewriter.create<ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>(
              op.getLoc(), srcL1Addr, mcastAddr, transferSize, numDests,
              nullptr, nullptr, nullptr);
        }
      } else {
        // l1 to l1 single core "remote"
        auto nocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
            op.getLoc(), virtX, virtY, dstL1Addr);
        rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Addr,
                                                   nocAddr, transferSize);
      }
    } else if (!op.isSrcLocal() && op.isDstLocal()) {
      // read l1 from l1 or l1 from dram
      Value dstL1Addr = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), index(getCbId(op.getDst()), rewriter));

      // Fully Index the Operands
      while (op.getSrcIndices().size() !=
             static_cast<size_t>(op.getSrc().getType().getRank())) {
        op.getSrcIndicesMutable().append(index(0, rewriter));
      }
      while (op.getDstIndices().size() !=
             static_cast<size_t>(op.getDst().getType().getRank())) {
        op.getDstIndicesMutable().append(index(0, rewriter));
      }
      if (!op.getOptNumElems()) {
        op.setOptNumElems(getMemrefShardNumElems(op.getSrc().getType()));
      }

      AffineMap srcGridYMap, srcGridXMap, srcOffsetMap;
      std::tie(srcGridYMap, srcGridXMap, srcOffsetMap) =
          getIndividualResultMaps(op.getSrcMemRefType(), device, rewriter);

      auto applyMap = [](OpBuilder &builder, Location loc, AffineMap map,
                         ValueRange index) {
        auto apply = builder.create<affine::AffineApplyOp>(loc, map, index);
        return apply;
      };

      auto loc = op.getLoc();
      auto srcGridY = applyMap(rewriter, loc, srcGridYMap, op.getSrcIndices());
      auto srcGridX = applyMap(rewriter, loc, srcGridXMap, op.getSrcIndices());
      auto srcOffset =
          applyMap(rewriter, loc, srcOffsetMap, op.getSrcIndices());
      auto srcOffsetInt = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), srcOffset);
      auto size =
          i32(op.getNumElems() * getElementSizeBytes(op.getSrcMemRefType()),
              rewriter);
      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, chipDescs.front(), ValueRange{srcGridY, srcGridX});
      auto srcNocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
          loc, virtX, virtY, srcOffsetInt);
      rewriter.create<ttkernel::NocAsyncReadOp>(loc, srcNocAddr, dstL1Addr,
                                                size);
    } else if (op.isSrcLocal() && !op.isDstLocal()) {
      // write l1 to dram
      assert(false && "Unimplemented lowering l1 to dram write, failing.");
    } else {
      assert(false && "Illegal DMA op configuration.");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace
namespace {

class TTIRCoreIndexRewriter : public OpRewritePattern<ttir::CoreIndexOp> {
public:
  using OpRewritePattern<ttir::CoreIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::CoreIndexOp op,
                                PatternRewriter &rewriter) const final {
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
                                                   .getX(),
                                               rewriter));
      rewriter.replaceAllUsesWith(op.getResult(), normalizedCoreIndex);
    } else {
      auto coreIndex = rewriter.create<ttkernel::MyYOp>(op.getLoc(), nullptr);
      auto normalizedCoreIndex =
          rewriter.create<arith::SubIOp>(op.getLoc(), coreIndex,
                                         index(chipDescs.front()
                                                   .getChipPhysicalCores()
                                                   .getWorker()
                                                   .front()
                                                   .getY(),
                                               rewriter));
      rewriter.replaceAllUsesWith(op.getResult(), normalizedCoreIndex);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {
class TTIRDMAWaitRewriter : public OpRewritePattern<ttir::DMAWaitOp> {
public:
  using OpRewritePattern<ttir::DMAWaitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DMAWaitOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.getMemTx().getDefiningOp<ttir::DMAOp>().isSrcLocal()) {
      rewriter.create<ttkernel::NocAsyncReadBarrierOp>(op.getLoc());
    } else {
      rewriter.create<ttkernel::NocAsyncWriteBarrierOp>(op.getLoc());
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {

class TTIRGetGlobalOperandRewriter
    : public OpRewritePattern<ttir::GetGlobalOperandOp> {
public:
  using OpRewritePattern<ttir::GetGlobalOperandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GetGlobalOperandOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(
        op, rewriter.getI32Type(), op.getOperandIndex());
    return success();
  }
};

} // namespace

} // namespace mlir::tt::ttkernel

namespace mlir::tt {

void populateTTIRToTTKernelInnerRegionPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns,
    TypeConverter & /*typeConverter*/) {

  patterns.add<ttkernel::TTIRComputeOpsRewriter, ttkernel::MemrefStoreRewriter,
               ttkernel::TTIRAwaitYieldRewriter<ttir::AwaitOp>,
               ttkernel::TTIRAwaitYieldRewriter<ttir::YieldOp>,
               ttkernel::TTIRDMARewriter, ttkernel::TTIRDMAWaitRewriter,
               ttkernel::TTIRCoreIndexRewriter,
               ttkernel::TTIRGetGlobalOperandRewriter>(ctx);
}

void populateTTIRToTTKernelTopLevelPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter & /*typeConverter*/) {
  patterns.add<ttkernel::TTIRGenericRewriter, ttkernel::MemrefAllocRewriter>(
      ctx);
}

} // namespace mlir::tt
