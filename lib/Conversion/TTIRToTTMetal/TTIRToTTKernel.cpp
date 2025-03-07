// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tt::ttkernel {

Value i32(int32_t value, OpBuilder &builder) {
  return builder
      .create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(),
                                 builder.getI32IntegerAttr(value))
      .getResult();
}

namespace {
class TTIRGenericRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "ttir generic rewriter\n";
    if (mlir::dyn_cast_or_null<ttkernel::CBType>(
            op->getRegion(0).getBlocks().front().getArgument(0).getType())) {
      return failure();
    }

    for (auto &region : op->getRegions()) {
      assert(region.getBlocks().size() <= 1 &&
             "Expected single block in region (temporary), failing.");
      Block &block = region.getBlocks().front();
      assert(
          block.getNumArguments() == op.getNumOperands() &&
          "Mismatch between number of operands and block arguments, failing.");
      llvm::SmallVector<ttkernel::CBType> cbTypes;
      for (uint32_t i = 0; i < block.getNumArguments(); i++) {
        auto memref = mlir::cast<MemRefType>(block.getArgument(i).getType());
        auto cbPort = ttkernel::symbolizeCBPort(i);
        assert(cbPort.has_value() && "Out of CBs, failing.");
        cbTypes.push_back(ttkernel::CBType::get(rewriter.getContext(),
                                                cbPort.value(), 0, memref));
      }
      for (uint32_t i = 0; i < block.getNumArguments(); i++) {
        for (Operation *user : block.getArgument(i).getUsers()) {
          rewriter.eraseOp(user);
        }
        rewriter.modifyOpInPlace(
            op, [&]() { block.getArgument(i).setType(cbTypes[i]); });
      }

      auto enqueueProgramOp = rewriter.create<ttmetal::EnqueueProgramOp>(
          op->getLoc(), op->getResultTypes(), op.getInputs(), op.getOutputs(),
          rewriter.getArrayAttr({}), rewriter.getArrayAttr({}),
          op->getNumRegions());
      rewriter.modifyOpInPlace(enqueueProgramOp, [&]() {
        for (uint32_t i = 0; i < op->getNumRegions(); i++) {
          auto &region = enqueueProgramOp->getRegion(i);
          region.takeBody(op->getRegion(i));
        }
      });

      rewriter.eraseOp(op);
    }

    return success();
  };
};
} // namespace

/*

========
Old Memref Load Rewriter - keeping while lowering is WIP
========

namespace {

class MemrefLoadRewriter : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  static uint32_t getCbId(memref::LoadOp op) {
    memref::CollapseShapeOp collapseOp =
        llvm::cast<memref::CollapseShapeOp>(op.getMemref().getDefiningOp());
    if (auto blockArg =
            mlir::dyn_cast<mlir::BlockArgument>(collapseOp.getSrc())) {
      return blockArg.getArgNumber();
    }
    assert(false && "Could not match collapse op src to block argument, cannot "
                    "determine CB id. Failing.");
  }

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const final {

    OpBuilder builder(op);
    // erase the load op for now? sfpu / fpu considerations
    assert(op.getIndices().size() == 1 && "Expected 1D memref load, failing.");

    int sfpuOp = -1;
    op->getBlock()->walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::TTKernelSFPUOpTrait>() && sfpuOp == -1) {
        sfpuOp = 1;
      } else if (op->hasTrait<OpTrait::TTKernelFPUOpTrait>() && sfpuOp == -1) {
        sfpuOp = 0;
      }
    });

    assert(sfpuOp != -1 && "Data Movement op only, unsupported, failing.");

    auto cbId = i32(getCbId(op), builder);

    if (sfpuOp == 1) {
      rewriter.create<ttkernel::CopyTileInitOp>(op.getLoc(), cbId);
      rewriter.create<ttkernel::CopyTileOp>(op.getLoc(), cbId,
                                            op.getIndices().front(), cbId);
    }

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace */

// memref store rewriter

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
    llvm::errs() << "memref store rewriter\n";
    OpBuilder builder(op);

    auto cbId = i32(getCbId(op), builder);
    rewriter.create<ttkernel::PackTileOp>(
        op.getLoc(), i32(0, builder), cbId,
        i32(0,
            builder)); // we should lower to a pack loop (or use
                       // matmul_pack_tile??) if input is not one tile, i.e. for
                       // matmul_block, we need to loop pack on the whole dst
                       // output. out_idx is ignore except for out of order pack

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace

// tile ops rewriter
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

  static bool isFirstComputeOp(Operation *op) {
    auto generic = op->getParentOfType<ttir::GenericOp>();
    assert(generic && "Could not find parent generic op, failing.");

    bool isFirst = 1;

    generic.walk([&](Operation *walkOp) {
      if (walkOp->hasTrait<OpTrait::TTKernelFPUOpTrait>() ||
          walkOp->hasTrait<OpTrait::TTKernelSFPUOpTrait>()) {
        isFirst = 0;
      }
    });

    return isFirst;
  }

  static void lowerLoad(memref::LoadOp op, bool copyToDst, bool cbIndices,
                        PatternRewriter &rewriter) {
    OpBuilder builder(op);

    if (copyToDst) {
      auto cbId = i32(getCbId(op), builder);
      rewriter.setInsertionPoint(op);
      rewriter.create<ttkernel::CopyTileInitOp>(op.getLoc(), cbId);
      rewriter.create<ttkernel::CopyTileOp>(op.getLoc(), cbId,
                                            op.getIndices().front(),
                                            cbIndices ? cbId : i32(0, builder));
    }
    rewriter.eraseOp(op);
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "tileops & memref load rewriter\n";
    OpBuilder builder(op);

    auto first = isFirstComputeOp(op);
    bool erase = 0;
    Operation *newOp;

    if (mlir::isa<ttir::TileMaximumOp>(op)) {
      rewriter.create<ttkernel::MaxTilesInitOp>(op->getLoc());
      newOp = rewriter.create<ttkernel::MaxTilesOp>(
          op->getLoc(), i32(0, builder), i32(1, builder));
      erase = 1;
    } else if (mlir::isa<ttir::TileMatmulOp>(op)) {
      auto dstIdx = rewriter.create<arith::ConstantOp>(
          op->getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
      newOp = rewriter.create<ttkernel::MatmulTilesOp>(
          op->getLoc(),
          i32(getCbId(op->getOperand(0).getDefiningOp<memref::LoadOp>()),
              builder),
          i32(getCbId(op->getOperand(1).getDefiningOp<memref::LoadOp>()),
              builder),
          index(op->getOperand(0)), index(op->getOperand(1)), dstIdx);
      erase = 1;
    }

    if (!erase) {
      return failure(); // new ttir trait for generic region compute vs generic region "helpers" or something
    }

    if (first) {
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

    if (erase) {
      rewriter.eraseOp(op);
    }
    return success();
  };
};
} // namespace

// ttir await/yield rewriter

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
    llvm::errs() << "ttir await/yield rewriter\n";
    OpBuilder builder(op);

    for (Value input : op.getValues()) {
      auto cbId = i32(getCbId(input), builder);
      auto type = mlir::cast<MemRefType>(input.getType());
      assert(type.getShape().size() == 1 &&
             "Expected collapsed 1D memref, failing.");
      auto numPages = i32(type.getNumElements(), builder);
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

// tile regs pass

namespace {

class TTIRTileRegsRewriter : public OpRewritePattern<ttkernel::PackTileOp> {
public:
  using OpRewritePattern<ttkernel::PackTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::PackTileOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "tile regs rewriter\n";

    if (!op->getBlock()->getOps<ttkernel::TileRegsCommitOp>().empty()) {
      return failure();
    }

    rewriter.moveOpAfter(
        rewriter.create<ttkernel::TileRegsReleaseOp>(op->getLoc()), op);
    auto regsWait = rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());
    rewriter.moveOpBefore(regsWait, op);
    rewriter.moveOpBefore(
        rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc()), regsWait);

    rewriter.moveOpAfter(
        rewriter.create<ttkernel::TileRegsAcquireOp>(op->getLoc()),
        op->getBlock(), op->getBlock()->begin());

    return success();
  };
};

} // namespace

// memref.alloc -> ttmetal.create_buffer pass

namespace {

class MemrefAllocRewriter : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "memref alloc rewriter\n";

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

// memref.dealloc -> ttmetal.deallocate_buffer pass?

// init cleanup pass ?

} // namespace mlir::tt::ttkernel

namespace mlir::tt {

void populateTTIRToTTKernelPatternsPhase1(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter & /*typeConverter*/) {

  patterns.add<ttkernel::TTIRTileOpsRewriter, ttkernel::MemrefStoreRewriter,
               ttkernel::TTIRAwaitYieldRewriter<ttir::AwaitOp>,
               ttkernel::TTIRAwaitYieldRewriter<ttir::YieldOp>,
               ttkernel::MemrefAllocRewriter>(ctx);
}

void populateTTIRToTTKernelPatternsPhase2(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter & /*typeConverter*/) {
  // patterns.add<ttkernel::MemrefLoadRewriter>(ctx);
}

void populateTTIRToTTKernelPatternsPhase3(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter & /*typeConverter*/) {
  patterns.add<ttkernel::TTIRGenericRewriter, ttkernel::TTIRTileRegsRewriter>(
      ctx);
}

} // namespace mlir::tt
