// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/ScopeExit.h"

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
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Passes.h"

namespace mlir::tt::ttnn {

class TTNNToEmitCFuncArgsRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    auto blockArgs = op.getCallableRegion()->getArguments();
    if (blockArgs.empty())
      return rewriter.notifyMatchFailure(op, "No block arguments");

    // Rewrite the block arguments to be variables.
    rewriter.setInsertionPointToStart(&op.getCallableRegion()->front());
    for (auto arg : blockArgs) {
      auto cb = cast<ttkernel::CBType>(arg.getType());
      auto var = rewriter.create<emitc::VariableOp>(
          op.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(cb.getPort()));
      arg.replaceAllUsesWith(var);
    }
    op.getCallableRegion()->front().eraseArguments(0, blockArgs.size());
    op.setType(rewriter.getType<FunctionType>(TypeRange(), TypeRange()));

    return success();
  }
};

class TTNNToEmitCReturnRewriter : public OpRewritePattern<ttkernel::ReturnOp> {
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

template <typename OpTy>
class TTNNToEmitCOpaqueRewriter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  StringRef getOpName(OpTy op) const {
    if constexpr (std::is_same_v<OpTy, ttkernel::BuiltinOp>) {
      return op.getOp();
    }
    auto name = op.getOperation()->getName().getStringRef();
    if (name.starts_with("ttnn."))
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

LogicalResult emitDispatchOpRegionAsCpp(DispatchOp origOp,
                                        unsigned regionNumber,
                                        llvm::raw_ostream &os) {
  DispatchOp op = cast<DispatchOp>(origOp->clone());
  auto cleanupDispatchClone = llvm::make_scope_exit([&op] { op->erase(); });
  Region &region = op->getRegion(regionNumber);

  OpBuilder builder(op.getOperation());

  auto threadTypeAttr =
      op.getThreadTypes()[regionNumber].cast<ttkernel::ThreadTypeAttr>();

  // Replace the original block with a the new block containing a module op
  auto module = builder.create<mlir::ModuleOp>(
      mlir::UnknownLoc::get(op.getContext()),
      ttkernel::stringifyThreadType(threadTypeAttr.getValue()));
  auto cleanupFreeModule =
      llvm::make_scope_exit([&module] { module->erase(); });
  auto &moduleBlock = module.getBodyRegion().front();
  module->setDiscardableAttr(builder.getStringAttr("ttkernel.thread_type"),
                             threadTypeAttr);
  builder.setInsertionPointToStart(&moduleBlock);

  // Create a new func op and move the existing block into it.
  auto func = builder.create<func::FuncOp>(
      module.getLoc(), "kernel_main",
      builder.getType<FunctionType>(region.getArgumentTypes(), TypeRange()));
  Block *entryBlock = func.addEntryBlock();
  Region *funcBody = entryBlock->getParent();
  IRMapping irMapper;
  funcBody->takeBody(region);

  RewritePatternSet patterns(module.getContext());
  patterns.add<TTNNToEmitCFuncArgsRewriter,
               TTNNToEmitCOpaqueRewriter<ttkernel::BuiltinOp>,
               TTNNToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>,
               TTNNToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>,
               TTNNToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>,
               TTNNToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>,
               TTNNToEmitCReturnRewriter>(module.getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(module, patternSet))) {
    return failure();
  }

  if (emitc::translateToCpp(module.getOperation(), os).failed()) {
    return failure();
  }

  return success();
}

} // namespace mlir::tt::ttnn
