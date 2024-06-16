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

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Passes.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

class TTIRToTTNNLayoutRewriter : public OpRewritePattern<ttir::LayoutOp> {
public:
  using OpRewritePattern<ttir::LayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::LayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputTy = op.getInput().getType().template cast<RankedTensorType>();
    auto outputTy = op.getType().template cast<RankedTensorType>();
    if (not inputTy.getEncoding() || not outputTy.getEncoding()) {
      return failure();
    }
    assert(inputTy.getEncoding().isa<tt::LayoutAttr>());
    assert(outputTy.getEncoding().isa<tt::LayoutAttr>());
    auto inputLayout = inputTy.getEncoding().template cast<tt::LayoutAttr>();
    auto outputLayout = outputTy.getEncoding().template cast<tt::LayoutAttr>();
    if (inputLayout.isSystemMemorySpace()) {
      assert(outputLayout.isDeviceMemorySpace());
      rewriter.replaceOpWithNewOp<ttnn::HostWriteOp>(
          op, outputTy, op.getInput(), op.getOutput());
    } else if (outputLayout.isSystemMemorySpace()) {
      assert(inputLayout.isDeviceMemorySpace());
      rewriter.replaceOpWithNewOp<ttnn::HostReadOp>(op, outputTy, op.getInput(),
                                                    op.getOutput());
    } else {
      return failure();
    }
    return success();
  }
};

class TTIRToTTNNKernelRewriter : public OpRewritePattern<ttir::KernelOp> {
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

class TTIRToTTNNReturnRewriter : public OpRewritePattern<ttir::YieldOp> {
public:
  using OpRewritePattern<ttir::YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::YieldOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttkernel::ReturnOp>(op);
    return success();
  }
};

class TTIRToTTNNDispatchRewriter : public OpRewritePattern<ttir::GenericOp> {
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

  uint64_t lookupAddress(Value value) const {
    auto *op = value.getDefiningOp();
    if (!op) {
      return 0;
    }
    auto allocOp = dyn_cast<ttir::AllocOp>(op);
    if (!allocOp) {
      return 0;
    }
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
          rewriter.getType<ttkernel::CBType>(address, port, memref));
    }
    return rewrittenBlockArgumentTypes;
  }

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (hasUnloweredTTIRKernel(op)) {
      return failure();
    }

    SmallVector<Attribute> threadTypes = {
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(ttkernel::ThreadType::Noc0),
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(ttkernel::ThreadType::Noc1),
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(
            ttkernel::ThreadType::Tensix),
    };
    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttnn::CoreRangeAttr>(op.getGrid()),
        rewriter.getAttr<ttnn::CoreRangeAttr>(op.getGrid()),
        rewriter.getAttr<ttnn::CoreRangeAttr>(op.getGrid()),
    };
    SmallVector<Attribute> operand_cb_port_mapping;
    for (auto &operand : op->getOpOperands()) {
      operand_cb_port_mapping.push_back(
          rewriter.getI64IntegerAttr(operand.getOperandNumber()));
    }
    auto metalDispatch = rewriter.create<ttnn::DispatchOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes),
        rewriter.getArrayAttr(operand_cb_port_mapping), threadTypes.size());

    auto rewrittenBlockArgumentTypes = getBlockArgumentTypesAsCBs(
        op->getRegion(0).getArguments(), operand_cb_port_mapping, rewriter);

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
    auto push0 = rewriter.create<ttkernel::CBPushBackOp>(
        op.getLoc(), noc0Block->getArgument(0));
    push0->remove();
    noc0Block->push_back(push0);
    auto return0 =
        rewriter.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    return0->remove();
    noc0Block->push_back(return0);

    rewriter.setInsertionPointToStart(noc1Block);
    auto push1 = rewriter.create<ttkernel::CBPushBackOp>(
        op.getLoc(), noc1Block->getArgument(1));
    push1->remove();
    noc1Block->push_back(push1);
    auto return1 =
        rewriter.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    return1->remove();
    noc1Block->push_back(return1);

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }
};

class TTIRToTTNNAllocRewriter : public OpRewritePattern<ttir::AllocOp> {
public:
  using OpRewritePattern<ttir::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::AllocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttnn::AllocOp>(
        op, op.getType(), op.getAddress(), op.getSize(), op.getMemorySpace());
    return success();
  }
};

class TTIRToTTNNDeallocRewriter : public OpRewritePattern<ttir::DeallocOp> {
public:
  using OpRewritePattern<ttir::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttnn::DeallocOp>(op, op.getResult());
    return success();
  }
};

class ConvertTTIRToTTNN
    : public impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNN> {
public:
  using impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNN>::ConvertTTIRToTTNNBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRToTTNNLayoutRewriter, TTIRToTTNNKernelRewriter,
                 TTIRToTTNNReturnRewriter, TTIRToTTNNDispatchRewriter,
                 TTIRToTTNNAllocRewriter, TTIRToTTNNDeallocRewriter>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
  }
};

} // namespace mlir::tt::ttnn
