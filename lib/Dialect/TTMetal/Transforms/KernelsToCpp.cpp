// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/ScopeExit.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

namespace mlir::tt::ttmetal {

class TTKernelToEmitCTypeConverter : public TypeConverter {
public:
  TTKernelToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::tt::ttkernel::NocAddrType type) -> Type {
      return Builder(ctx).getI64Type();
    });
  }
};

class TTMetalToEmitCFuncArgsRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    auto blockArgs = op.getCallableRegion()->getArguments();
    if (blockArgs.empty()) {
      return rewriter.notifyMatchFailure(op, "No block arguments");
    }

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

class TTMetalToEmitCReturnRewriter
    : public OpRewritePattern<ttkernel::ReturnOp> {
public:
  using OpRewritePattern<ttkernel::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    if (not isa<func::FuncOp>(op.getOperation()->getParentOp())) {
      return rewriter.notifyMatchFailure(op, "Not inside of func op");
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, ValueRange());
    return success();
  }
};

template <typename OpTy>
class TTMetalToEmitCOpaqueRewriter : public OpRewritePattern<OpTy> {
public:
  TTMetalToEmitCOpaqueRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                               MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx), typeConverter(&typeConverter) {}

  StringRef getOpName(OpTy op) const {
    if constexpr (std::is_same_v<OpTy, ttkernel::BuiltinOp>) {
      return op.getOp();
    }
    auto name = op.getOperation()->getName().getStringRef();
    if (name.starts_with("ttkernel.")) {
      return name.drop_front(9);
    }
    return name;
  }

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    SmallVector<Type, 4> resultTypes;
    for (auto type : op->getResultTypes()) {
      resultTypes.push_back(typeConverter->convertType(type));
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultTypes, getOpName(op), nullptr, nullptr, op->getOperands());
    return success();
  }

  TTKernelToEmitCTypeConverter *typeConverter;
};

LogicalResult emitDispatchOpRegionAsCpp(DispatchOp origOp,
                                        unsigned regionNumber,
                                        llvm::raw_ostream &os) {
  DispatchOp op = cast<DispatchOp>(origOp->clone());
  auto cleanupDispatchClone = llvm::make_scope_exit([&op] { op->erase(); });
  Region &region = op->getRegion(regionNumber);

  OpBuilder builder(op.getOperation());

  auto threadTypeAttr =
      mlir::cast<ttkernel::ThreadTypeAttr>(op.getThreadTypes()[regionNumber]);

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

  builder.create<emitc::IncludeOp>(module.getLoc(), "cstdint",
                                   /*isStandard=*/true);
  if (threadTypeAttr.getValue() == ttkernel::ThreadType::Noc0 ||
      threadTypeAttr.getValue() == ttkernel::ThreadType::Noc1) {
    builder.create<emitc::IncludeOp>(module.getLoc(), "dataflow_api.h",
                                     /*isStandard=*/false);
  }
  if (threadTypeAttr.getValue() == ttkernel::ThreadType::Tensix) {
    builder.create<emitc::IncludeOp>(module.getLoc(),
                                     "compute_kernel_api/common.h",
                                     /*isStandard=*/false);
    builder.create<emitc::IncludeOp>(module.getLoc(),
                                     "compute_kernel_api/tilize.h",
                                     /*isStandard=*/false);
    builder.create<emitc::IncludeOp>(module.getLoc(),
                                     "compute_kernel_api/untilize.h",
                                     /*isStandard=*/false);
  }

  if (threadTypeAttr.getValue() == ttkernel::ThreadType::Tensix) {
    builder.create<emitc::VerbatimOp>(module.getLoc(), "namespace NAMESPACE {");
  }

  // Create a new func op and move the existing block into it.
  auto func = builder.create<func::FuncOp>(
      module.getLoc(), "kernel_main",
      builder.getType<FunctionType>(region.getArgumentTypes(), TypeRange()));
  Block *entryBlock = func.addEntryBlock();
  Region *funcBody = entryBlock->getParent();
  IRMapping irMapper;
  funcBody->takeBody(region);

  if (threadTypeAttr.getValue() == ttkernel::ThreadType::Tensix) {
    builder.create<emitc::VerbatimOp>(module.getLoc(),
                                      "void MAIN { kernel_main(); }");
    builder.create<emitc::VerbatimOp>(module.getLoc(), "}");
  }

  // Apply arith to emitc conversion first
  {
    ConversionTarget target(*module.getContext());
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    RewritePatternSet arithPatterns(module.getContext());
    TypeConverter arithTypeConverter;
    arithTypeConverter.addConversion([](Type type) { return type; });
    populateArithToEmitCPatterns(arithTypeConverter, arithPatterns);
    if (failed(
            applyPartialConversion(module, target, std::move(arithPatterns)))) {
      return failure();
    }
  }

  TTKernelToEmitCTypeConverter typeConverter(module.getContext());
  RewritePatternSet patterns(module.getContext());

  patterns.add<TTMetalToEmitCFuncArgsRewriter, TTMetalToEmitCReturnRewriter>(
      module.getContext());
  patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::BuiltinOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeInitOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeInitOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeBlockOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeBlockOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::GetNocAddrOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadBarrierOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteOp>,
               TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteBarrierOp>>(
      typeConverter, module.getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(module, patternSet))) {
    return failure();
  }

  if (emitc::translateToCpp(module.getOperation(), os).failed()) {
    return failure();
  }

  return success();
}

} // namespace mlir::tt::ttmetal
