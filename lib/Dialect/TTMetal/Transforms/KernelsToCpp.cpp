// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTMETALKERNELSTOEMITC
#include "ttmlir/Dialect/TTMetal/Passes.h.inc"

class TTMetalToEmitCDispatchRegionRewriter
    : public OpRewritePattern<ttmetal::DispatchOp> {
public:
  using OpRewritePattern<ttmetal::DispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttmetal::DispatchOp op,
                                PatternRewriter &rewriter) const final {
    for (auto &region : op.getRegions()) {
      auto op_begin = region.op_begin();
      if (isa<mlir::ModuleOp>(*op_begin))
        return rewriter.notifyMatchFailure(op, "Already converted");

      // Create a new func op and move the existing block into it.
      auto func = rewriter.create<func::FuncOp>(
          op.getLoc(), "kernel_main",
          rewriter.getType<FunctionType>(TypeRange(), TypeRange()));
      func.getCallableRegion()->takeBody(region);
      func->remove();

      // Rewrite the block arguments to be variables.
      rewriter.setInsertionPointToStart(&func.getCallableRegion()->front());
      auto blockArgs = func.getCallableRegion()->getArguments();
      for (auto arg : blockArgs) {
        auto cb = cast<ttkernel::CBType>(arg.getType());
        auto var = rewriter.create<emitc::VariableOp>(
            op.getLoc(), rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(cb.getPort()));
        rewriter.replaceAllUsesWith(arg, var);
      }
      func.getCallableRegion()->front().eraseArguments(0, blockArgs.size());

      // Replace the original block with a the new block containing a module op
      ttkernel::ThreadType threadType =
          op.getThreadTypes()[region.getRegionNumber()]
              .cast<ttkernel::ThreadTypeAttr>()
              .getValue();
      Block *newBlock = rewriter.createBlock(&region);
      auto module = rewriter.create<mlir::ModuleOp>(
          op.getLoc(), ttkernel::stringifyThreadType(threadType));
      module->remove();
      newBlock->push_back(module);

      // Push the function inside the module.
      Block *moduleBlock = &module.getBodyRegion().front();
      moduleBlock->push_front(func);
      rewriter.create<emitc::IncludeOp>(func.getLoc(), "ttmetal.h", "ttmetal")
          ->moveBefore(func);

      // Blocks require a terminator operation, so add an unreachable op.
      rewriter.create<ttkernel::UnreachableOp>(func.getLoc());
    }
    return success();
  }
};

class TTMetalToEmitCReturnRewriter : public OpRewritePattern<ttkernel::ReturnOp> {
public:
  using OpRewritePattern<ttkernel::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    if (not isa<func::FuncOp>(op.getOperation()->getParentOp()))
      return rewriter.notifyMatchFailure(op, "Not inside of func op");
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, ValueRange());
    return success();
  }
};

template<typename OpTy>
class TTMetalToEmitCOpaqueRewriter
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  StringRef getOpName(OpTy op) const {
    if constexpr (std::is_same_v<OpTy, ttkernel::BuiltinOp>) {
      return op.getOp();
    }
    auto name = op.getOperation()->getName().getStringRef();
    if (name.starts_with("ttmetal."))
      return name.drop_front(8);
    return name;
  }

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange(), getOpName(op), nullptr, nullptr, op->getOperands());
    return success();
  }
};

class ConvertTTMetalKernelsToEmitC
    : public impl::ConvertTTMetalKernelsToEmitCBase<
          ConvertTTMetalKernelsToEmitC> {
public:
  using impl::ConvertTTMetalKernelsToEmitCBase<
      ConvertTTMetalKernelsToEmitC>::ConvertTTMetalKernelsToEmitCBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTMetalToEmitCDispatchRegionRewriter,
                 TTMetalToEmitCOpaqueRewriter<ttkernel::BuiltinOp>,
                 TTMetalToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>,
                 TTMetalToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>,
                 TTMetalToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>,
                 TTMetalToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>,
                 TTMetalToEmitCReturnRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();

      // Dump kernels to stdout
#if 0
    ModuleOp module = getOperation();
    module->walk([&](ttmetal::DispatchOp dispatchOp) {
      for (auto &region : dispatchOp.getRegions()) {
        for (auto &op : region.getOps()) {
          if (isa<ModuleOp>(op)) {
            auto res = emitc::translateToCpp(&op, llvm::outs());
            (void)res;
          }
        }
      }
    });
#endif
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

} // namespace mlir::tt::ttmetal
