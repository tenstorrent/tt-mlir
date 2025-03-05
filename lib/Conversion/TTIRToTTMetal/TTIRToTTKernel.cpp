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
      assert(region.getBlocks().size() == 1 &&
             "Expected single block in region, failing.");
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
    }

    return success();
  };
};
} // namespace

// memref load rewriter
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

} // namespace
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
    rewriter.create<ttkernel::PackTileOp>(op.getLoc(), i32(0, builder), cbId,
                                          i32(0, builder));

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace

// tile ops rewriter
namespace {

class TTIRTileOpsRewriter
    : public OpRewritePattern<
          ttir::TileMaximumOp> { // this needs to match all generic region ops,
                                 // for now, match max specifically
public:
  using OpRewritePattern<ttir::TileMaximumOp>::OpRewritePattern;

  static Value index(Value tile) {
    Operation *loadOp = tile.getDefiningOp();
    assert(mlir::isa<memref::LoadOp>(loadOp) && "Expected load op, failing.");
    return mlir::cast<memref::LoadOp>(loadOp).getIndices().front();
  }

  LogicalResult matchAndRewrite(ttir::TileMaximumOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "tileops rewriter\n";
    OpBuilder builder(op);

    if (mlir::isa<ttir::TileMaximumOp>(op)) {
      rewriter.create<ttkernel::MaxTilesInitOp>(op->getLoc());
      rewriter.create<ttkernel::MaxTilesOp>(op->getLoc(), i32(0, builder),
                                            i32(1, builder));
    }

    rewriter.eraseOp(op);
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
      auto numPages = i32(type.getShape()[0], builder);
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

// tile regs pass

// reconfig data formats

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
    assert(mlir::isa<TileType>(op.getMemref().getType().getElementType()) && "Expected memref to have tile element type, failing.");
    auto size = mlir::cast<TileType>(op.getMemref().getType().getElementType()).getSizeBytes() * op.getMemref().getType().getNumElements();
    auto memorySpace = mlir::cast<tt::MemorySpaceAttr>(
        op.getMemref().getType().getMemorySpace());
    auto createBufferOp = rewriter.create<ttmetal::CreateBufferOp>(op->getLoc(), op.getMemref().getType(), address.getInt(), size, memorySpace.getValue());
    rewriter.replaceOp(op, createBufferOp);

    return success();
  };
};

} // namespace

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
  patterns.add<ttkernel::MemrefLoadRewriter>(ctx);
}

void populateTTIRToTTKernelPatternsPhase3(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter & /*typeConverter*/) {
  patterns.add<ttkernel::TTIRGenericRewriter, ttkernel::TTIRTileRegsRewriter>(
      ctx);
}

} // namespace mlir::tt
