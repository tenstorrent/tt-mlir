// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#include "ttmlir/Dialect/TTMetal/Passes.h.inc"

class TTIRToTTMetalLayoutRewriter : public OpRewritePattern<ttir::LayoutOp> {
public:
  using OpRewritePattern<ttir::LayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::LayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputTy = op.getInput().getType().template cast<RankedTensorType>();
    auto outputTy = op.getType().template cast<RankedTensorType>();
    if (not inputTy.getEncoding() || not outputTy.getEncoding())
      return failure();
    assert(inputTy.getEncoding().isa<tt::LayoutAttr>());
    assert(outputTy.getEncoding().isa<tt::LayoutAttr>());
    auto inputLayout = inputTy.getEncoding().template cast<tt::LayoutAttr>();
    auto outputLayout = outputTy.getEncoding().template cast<tt::LayoutAttr>();
    if (inputLayout.isSystemMemorySpace()) {
      assert(outputLayout.isDeviceMemorySpace());
      rewriter.replaceOpWithNewOp<ttmetal::HostWriteOp>(
          op, outputTy, op.getInput(), op.getOutput());
    } else if (outputLayout.isSystemMemorySpace()) {
      assert(inputLayout.isDeviceMemorySpace());
      rewriter.replaceOpWithNewOp<ttmetal::HostReadOp>(
          op, outputTy, op.getInput(), op.getOutput());
    } else {
      return failure();
    }
    return success();
  }
};

class TTIRToTTMetalKernelRewriter : public OpRewritePattern<ttir::KernelOp> {
public:
  using OpRewritePattern<ttir::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::KernelOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::KernelOp>(
        op, op.getResults().getTypes(), op.getOpAttr(), op.getKindAttr(),
        op.getOperands());
    return success();
  }
};

class TTIRToTTMetalYieldRewriter : public OpRewritePattern<ttir::YieldOp> {
public:
  using OpRewritePattern<ttir::YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::YieldOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::YieldOp>(op, op.getOperands());
    return success();
  }
};

class TTIRToTTMetalDispatchRewriter : public OpRewritePattern<ttir::DispatchOp> {
public:
  using OpRewritePattern<ttir::DispatchOp>::OpRewritePattern;

  bool hasUnloweredTTIRKernel(ttir::DispatchOp op) const {
    bool exists = false;
    op->getRegion(0).walk([&exists](Operation *op) {
      if (isa<ttir::KernelOp>(op))
        exists = true;
    });
    return exists;
  }

  uint64_t lookupAddress(Value value) const {
    auto op = value.getDefiningOp();
    if (!op)
      return 0;
    auto allocOp = dyn_cast<ttir::AllocOp>(op);
    if (!allocOp)
      return 0;
    return allocOp.getAddress();
  }

  SmallVector<Type> getBlockArgumentTypesAsCBs(
      mlir::Block::BlockArgListType blockArguments,
      SmallVector<Attribute> const &operand_cb_port_mapping,
      PatternRewriter &rewriter) const {
    SmallVector<Type> rewrittenBlockArgumentTypes;
    for (auto arg : blockArguments) {
      auto address = lookupAddress(arg);
      auto port = operand_cb_port_mapping[arg.getArgNumber()]
                      .cast<IntegerAttr>()
                      .getInt();
      auto memref = arg.getType().cast<MemRefType>();
      rewrittenBlockArgumentTypes.push_back(
          rewriter.getType<ttmetal::CBType>(address, port, memref));
    }
    return rewrittenBlockArgumentTypes;
  }

  LogicalResult matchAndRewrite(ttir::DispatchOp op,
                                PatternRewriter &rewriter) const final {
    if (hasUnloweredTTIRKernel(op))
      return failure();

    SmallVector<Attribute> threadTypes = {
        rewriter.getAttr<ttmetal::ThreadTypeAttr>(ttmetal::ThreadType::Noc0),
        rewriter.getAttr<ttmetal::ThreadTypeAttr>(ttmetal::ThreadType::Noc1),
        rewriter.getAttr<ttmetal::ThreadTypeAttr>(ttmetal::ThreadType::Tensix),
    };
    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
    };
    SmallVector<Attribute> operand_cb_port_mapping;
    for (auto &operand : op->getOpOperands()) {
      operand_cb_port_mapping.push_back(
          rewriter.getI64IntegerAttr(operand.getOperandNumber()));
    }
    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes),
        rewriter.getArrayAttr(operand_cb_port_mapping), threadTypes.size());

    auto rewrittenBlockArgumentTypes =
        getBlockArgumentTypesAsCBs(op->getRegion(0).getArguments(),
                                   operand_cb_port_mapping, rewriter);

    metalDispatch.getRegion(2).takeBody(op->getRegion(0));
    Block *tensixBlock = &metalDispatch.getRegion(2).front();
    Block *noc0Block = rewriter.createBlock(&metalDispatch.getRegion(0));
    Block *noc1Block = rewriter.createBlock(&metalDispatch.getRegion(1));

    int i = 0;
    for (auto ty : rewrittenBlockArgumentTypes) {
      noc0Block->addArgument(ty, op.getLoc());
      noc1Block->addArgument(ty, op.getLoc());
      auto arg = tensixBlock->getArgument(i++);
      arg.setType(ty);
    }

    rewriter.setInsertionPointToStart(noc0Block);
    auto push0 = rewriter.create<ttmetal::CBPushBackOp>(
        op.getLoc(), noc0Block->getArgument(0));
    push0->remove();
    noc0Block->push_back(push0);
    auto yield0 = rewriter.create<ttmetal::YieldOp>(op.getLoc(), ValueRange());
    yield0->remove();
    noc0Block->push_back(yield0);

    rewriter.setInsertionPointToStart(noc1Block);
    auto push1 = rewriter.create<ttmetal::CBPushBackOp>(
        op.getLoc(), noc1Block->getArgument(1));
    push1->remove();
    noc1Block->push_back(push1);
    auto yield1 = rewriter.create<ttmetal::YieldOp>(op.getLoc(), ValueRange());
    yield1->remove();
    noc1Block->push_back(yield1);

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
  using impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal>::ConvertTTIRToTTMetalBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRToTTMetalLayoutRewriter, TTIRToTTMetalKernelRewriter,
                 TTIRToTTMetalYieldRewriter, TTIRToTTMetalDispatchRewriter,
                 TTIRToTTMetalAllocRewriter, TTIRToTTMetalDeallocRewriter>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
  }
};

} // namespace mlir::tt::ttmetal
