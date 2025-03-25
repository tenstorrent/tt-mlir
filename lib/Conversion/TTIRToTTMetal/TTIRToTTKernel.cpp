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
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

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

static int32_t getMemrefSizeBytes(MemRefType memref) {
  if (auto elementType = mlir::dyn_cast<TileType>(memref.getElementType())) {
    return elementType.getSizeBytes() * memref.getNumElements();
  }
  return memref.getElementTypeBitWidth() / 8 * memref.getNumElements();
}

namespace {
class TTIRGenericRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    auto coreRanges = llvm::SmallVector<Attribute>();
    coreRanges.reserve(op.getKernelSymbols()->size());
    for (size_t i = 0; i < op.getKernelSymbols()->size(); i++) {
      coreRanges.push_back(
          rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()));
    }
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, op->getResultTypes(), op.getInputs(), op.getOutputs(),
        rewriter.getArrayAttr(op.getKernelSymbols()->getValue()),
        rewriter.getArrayAttr(coreRanges), rewriter.getArrayAttr({}));
    return success();
  };
};
} // namespace

namespace {

class MemrefStoreRewriter : public OpRewritePattern<memref::StoreOp> {
public:
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  static uint32_t getCbId(memref::StoreOp op) {
    memref::CollapseShapeOp collapseOp =
        llvm::cast<memref::CollapseShapeOp>(op.getMemref().getDefiningOp());
    if (auto blockArg =
            mlir::dyn_cast<mlir::BlockArgument>(collapseOp.getSrc())) {
      return blockArg.getArgNumber();
    }
    assert(false && "Could not match collapse op src to block argument, cannot "
                    "determine CB id. Failing.");
  }

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const final {
    auto cbId = i32(getCbId(op), rewriter);
    auto storeIdx = op.getIndices().front();
    rewriter.create<ttkernel::PackTileOp>(op.getLoc(), index(0, rewriter), cbId,
                                          storeIdx, rewriter.getBoolAttr(true));

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace
namespace {

class TTIRTileOpsRewriter
    : public OpTraitRewritePattern<ttir::OpTrait::TTIRGenericRegionOpTrait> {
public:
  using OpTraitRewritePattern<
      ttir::OpTrait::TTIRGenericRegionOpTrait>::OpTraitRewritePattern;

  static Value index(Value tile) {
    Operation *loadOp = tile.getDefiningOp();
    assert(mlir::isa<memref::LoadOp>(loadOp) && "Expected load op, failing.");
    return mlir::cast<memref::LoadOp>(loadOp).getIndices().front();
  }

  static uint32_t getCbId(memref::LoadOp value) {
    memref::CollapseShapeOp collapseOp =
        mlir::cast<memref::CollapseShapeOp>(value.getMemref().getDefiningOp());
    assert(
        collapseOp &&
        "Could not find assocated collapse shape op with memref load, failing");
    if (auto blockArg =
            mlir::dyn_cast<mlir::BlockArgument>(collapseOp.getSrc())) {
      return blockArg.getArgNumber();
    }
    assert(false && "Could not match collapse op src to block argument, cannot "
                    "determine CB id. Failing.");
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
    OpBuilder builder(op);
    if (copyToDst) {
      auto cbId = i32(getCbId(op), builder);
      rewriter.setInsertionPoint(op);
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
          i32(getCbId(op->getOperand(0).getDefiningOp<memref::LoadOp>()),
              rewriter),
          i32(getCbId(op->getOperand(1).getDefiningOp<memref::LoadOp>()),
              rewriter),
          index(op->getOperand(0)), index(op->getOperand(1)), dstIdx);
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

  static uint32_t getCbId(Value value) {
    memref::CollapseShapeOp collapseOp =
        llvm::cast<memref::CollapseShapeOp>(value.getDefiningOp());
    if (auto blockArg =
            mlir::dyn_cast<mlir::BlockArgument>(collapseOp.getSrc())) {
      return blockArg.getArgNumber();
    }
    assert(false && "Could not match collapse op src to block argument, cannot "
                    "determine CB id. Failing.");
  }

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
    for (Value input : op.getValues()) {
      auto cbId = i32(getCbId(input), rewriter);
      auto type = mlir::cast<MemRefType>(input.getType());
      assert(type.getShape().size() == 1 &&
             "Expected collapsed 1D memref, failing.");
      auto numPages = i32(type.getNumElements(), rewriter);
      Block *block = op->getBlock();
      if (mlir::isa<ttir::AwaitOp>(op)) {
        rewriter.create<ttkernel::CBWaitFrontOp>(op.getLoc(), cbId, numPages);
        auto popFront = rewriter.create<ttkernel::CBPopFrontOp>(op.getLoc(),
                                                                cbId, numPages);
        rewriter.moveOpBefore(popFront, block, block->end());
      } else if (mlir::isa<ttir::YieldOp>(op)) {
        auto reserveBack = rewriter.create<ttkernel::CBReserveBackOp>(
            op.getLoc(), cbId, numPages);
        rewriter.moveOpAfter(reserveBack, block, block->begin());
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

// reconfig data formats

// memref.alloc -> ttmetal.create_buffer pass

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

// core range and kernel config rewriter

// dma rewriter

namespace {

class TTIRDMARewriter : public OpRewritePattern<ttir::DMAOp> {
public:
  using OpRewritePattern<ttir::DMAOp>::OpRewritePattern;

  static std::pair<Value, Value>
  getNocCoordsFromCoreIndex(PatternRewriter &rewriter,
                            PhysicalCoreCoordMapping &mapping,
                            OperandRange dstCoreIndex) {
    std::pair<Value, Value> nocCoords;
    for (size_t i = 0; i < dstCoreIndex.size(); i++) {
      if (auto op = mlir::dyn_cast<arith::ConstantOp>(
              dstCoreIndex[i].getDefiningOp())) {
        PhysicalCoreCoord logicalCoord =
            i ? PhysicalCoreCoord(
                    0, 0, mlir::cast<IntegerAttr>(op.getValue()).getInt())
              : PhysicalCoreCoord(
                    0, mlir::cast<IntegerAttr>(op.getValue()).getInt(), 0);
        auto [virtY, virtX] = mapping[logicalCoord];
        if (i) {
          nocCoords.second = index(virtX, rewriter);
        } else {
          nocCoords.first = index(virtY, rewriter);
        }
      } else if (auto op = mlir::dyn_cast<ttir::CoreIndexOp>(
                     dstCoreIndex[i].getDefiningOp())) {
        auto nocCoord = i ? nocCoords.second : nocCoords.first;
        nocCoord = op.getResult();
      } else {
        assert(false && "Expected constant op or core index op, failing.");
      }
    }
    return nocCoords;
  }

  static std::pair<Value, Value> getMcastEndCoords(PatternRewriter &rewriter,
                                                   Value &nocStartY,
                                                   Value &nocStartX,
                                                   OperandRange mcastShape) {
    std::pair<Value, Value> nocEndCoords;
    nocEndCoords.first = rewriter.create<arith::AddIOp>(
        nocStartY.getLoc(), nocStartY, mcastShape[0]);
    nocEndCoords.second = rewriter.create<arith::AddIOp>(
        nocStartX.getLoc(), nocStartX, mcastShape[1]);
    return nocEndCoords;
  }

  static int64_t getAddressFromMemref(Value memref) {
    for (auto &use : memref.getUses()) {
      if (auto alloc = mlir::dyn_cast<memref::AllocOp>(use.getOwner())) {
        return alloc->getAttrOfType<IntegerAttr>("address").getInt();
      }
    }
    assert("Unallocated tensor found, failing.");
    return -1;
  }

  LogicalResult matchAndRewrite(ttir::DMAOp op,
                                PatternRewriter &rewriter) const final {
    auto device = op->getParentOfType<ttir::GenericOp>().getDevice();
    auto chipIds = device.getChipIds();
    auto chipDescs =
        op->getParentOfType<ttir::GenericOp>().getSystemDesc().getChipDescs();

    PhysicalCoreCoordMapping workerMapping =
        PhysicalCoreCoordMapping::getWorkerMapping(chipIds, chipDescs);
    PhysicalCoreCoordMapping dramMapping =
        PhysicalCoreCoordMapping::getDramMapping(chipIds, chipDescs);

    Value srcL1Addr = i32(getAddressFromMemref(op.getSrc()), rewriter);
    Value dstL1Addr = i32(getAddressFromMemref(op.getDst()), rewriter);
    Value transferSize =
        i32(getMemrefSizeBytes(op.getSrcMemRefType()), rewriter);

    if (op.isSrcLocal()) {
      // writes
      if (op.isDstLocal()) {
        // local write - 1
      } else {
        // remote write
        if (op.getDstCoreIndex().empty()) {
          // dram write - 2 TBD re: ADDRESSING
        } else {
          // l1 write
          assert(op.getDstCoreIndex().size() == 2 &&
                 "Expected single dst core index (2 dims) "
                 "for now, failing.");

          auto [dstNocY, dstNocX] = getNocCoordsFromCoreIndex(
              rewriter, workerMapping, op.getDstCoreIndex());

          if (op.isMcast()) {
            // mcast writes
            assert(op.getMcastShape().size() == 2 &&
                   "Expected mcast shape to have 2 dims, failing.");

            auto [mcastEndY, mcastEndX] = getMcastEndCoords(
                rewriter, dstNocY, dstNocX, op.getMcastShape());
            auto numDests = rewriter.create<arith::MulIOp>(
                op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
            auto dstNocMcastAddr =
                rewriter.create<ttkernel::GetNocMulticastAddrOp>(
                    op.getLoc(), dstNocX, dstNocY, mcastEndX, mcastEndY,
                    dstL1Addr, nullptr);
            if (op.getSrc() == op.getDst()) {
              // mcast write - 5
              auto numDestsNoLoopback = rewriter.create<arith::SubIOp>(
                  op.getLoc(), numDests, index(1, rewriter));
              rewriter.create<ttkernel::NocAsyncWriteMulticastOp>(
                  op.getLoc(), srcL1Addr, dstNocMcastAddr, transferSize,
                  numDestsNoLoopback, nullptr, nullptr, nullptr);
            } else {
              // mcast write loopback src - 6
              rewriter.create<ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>(
                  op.getLoc(), srcL1Addr, dstNocMcastAddr, transferSize,
                  numDests, nullptr, nullptr, nullptr);
            }
          } else {
            // single core write - 4
            auto dstNocAddr = rewriter.create<ttkernel::GetNocAddrXYOp>(
                op.getLoc(), dstNocX, dstNocY, dstL1Addr);
            rewriter.create<ttkernel::NocAsyncWriteOp>(
                op.getLoc(), srcL1Addr, dstNocAddr, transferSize);
          }
        }
      }
    } else {
      // dram reads - 3
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

// memref.dealloc -> ttmetal.deallocate_buffer pass?

// init cleanup pass ?

namespace {

class TTIRCoreIndexRewriter : public OpRewritePattern<ttir::CoreIndexOp> {
public:
  using OpRewritePattern<ttir::CoreIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::CoreIndexOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.getDim() == 0 ||
           op.getDim() == 1 &&
               "Expected core index dim to be in range 0-1, failing.");
    if (op.getDim()) {
      auto coreIndex = rewriter.create<ttkernel::MyYOp>(op.getLoc(), nullptr);
      rewriter.replaceOp(op, coreIndex);
    } else {
      auto coreIndex = rewriter.create<ttkernel::MyXOp>(op.getLoc(), nullptr);
      rewriter.replaceOp(op, coreIndex);
    }
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
    rewriter.replaceOpWithNewOp<ttkernel::MetalCompileTimeArgOp>(
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

  patterns.add<ttkernel::TTIRTileOpsRewriter, ttkernel::MemrefStoreRewriter,
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
