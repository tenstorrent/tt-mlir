// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/Cpp/CppEmitter.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
using namespace tt;

namespace mlir::tt::ttkernel {

#define GEN_PASS_DEF_CONVERTTTKERNELTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttkernel

emitc::OpaqueAttr convertCBPort(Builder &builder, ttkernel::CBPort port) {
  switch (port) {
  case ttkernel::CBPort::In0:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in0");
  case ttkernel::CBPort::In1:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in1");
  case ttkernel::CBPort::In2:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in2");
  case ttkernel::CBPort::In3:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in3");
  case ttkernel::CBPort::In4:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in4");
  case ttkernel::CBPort::In5:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in5");
  case ttkernel::CBPort::In6:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in6");
  case ttkernel::CBPort::In7:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_in7");
  case ttkernel::CBPort::DataFlow0:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow0");
  case ttkernel::CBPort::DataFlow1:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow1");
  case ttkernel::CBPort::DataFlow2:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow2");
  case ttkernel::CBPort::DataFlow3:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow3");
  case ttkernel::CBPort::DataFlow4:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow4");
  case ttkernel::CBPort::DataFlow5:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow5");
  case ttkernel::CBPort::DataFlow6:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow6");
  case ttkernel::CBPort::DataFlow7:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::dataflow7");
  case ttkernel::CBPort::Out0:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out0");
  case ttkernel::CBPort::Out1:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out1");
  case ttkernel::CBPort::Out2:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out2");
  case ttkernel::CBPort::Out3:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out3");
  case ttkernel::CBPort::Out4:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out4");
  case ttkernel::CBPort::Out5:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out5");
  case ttkernel::CBPort::Out6:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out6");
  case ttkernel::CBPort::Out7:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_out7");
  case ttkernel::CBPort::Intermed0:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed0");
  case ttkernel::CBPort::Intermed1:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed1");
  case ttkernel::CBPort::Intermed2:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed2");
  case ttkernel::CBPort::Intermed3:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed3");
  case ttkernel::CBPort::Intermed4:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed4");
  case ttkernel::CBPort::Intermed5:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed5");
  case ttkernel::CBPort::Intermed6:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed6");
  case ttkernel::CBPort::Intermed7:
    return builder.getType<emitc::OpaqueAttr>("::tt::CB::c_intermed7");
  }
  llvm_unreachable("Unknown CBPort");
  return nullptr;
}

class TTKernelToEmitCTypeConverter : public TypeConverter {
public:
  TTKernelToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::tt::ttkernel::NocAddrType type) -> Type {
      return Builder(ctx).getI64Type();
    });
    addConversion([ctx](mlir::tt::ttkernel::CBType type) -> Type {
      return Builder(ctx).getType<emitc::OpaqueType>("::tt::CB");
    });
    // addSourceMaterialization([](OpBuilder &builder, Type spTp,
    //                             ValueRange inputs,
    //                             Location loc) -> std::optional<Value> {
    //   return builder
    //       .create<UnrealizedConversionCastOp>(loc, TypeRange(spTp), inputs)
    //       .getResult(0);
    // });
    // addTargetMaterialization([&](OpBuilder &builder, Type resultType,
    //                              ValueRange inputs,
    //                              Location loc) -> std::optional<Value> {
    //   if (inputs.size() != 1) {
    //     return std::nullopt;
    //   }
    //   return builder.create<UnrealizedConversionCastOp>(loc, resultType,
    //   inputs)
    //       .getResult(0);
    // });
    addTargetMaterialization([&](mlir::OpBuilder &builder,
                                 mlir::Type resultType, mlir::ValueRange inputs,
                                 mlir::Location loc) { return inputs[0]; });
  }
};

class TTMetalToEmitCFuncArgsRewriter
    : public OpConversionPattern<func::FuncOp> {
public:
  TTMetalToEmitCFuncArgsRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                                 MLIRContext *ctx)
      : OpConversionPattern<func::FuncOp>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::outs() << "Rewriting func op\n";

    auto blockArgs = op.getCallableRegion()->getArguments();
    if (blockArgs.empty()) {
      return rewriter.notifyMatchFailure(op, "No block arguments");
    }
    rewriter.startOpModification(op);
    // // Rewrite the block arguments to be variables.
    rewriter.setInsertionPointToStart(&op.getCallableRegion()->front());
    for (auto arg : blockArgs) {
      llvm::outs() << "blockArgs: ";
      arg.print(llvm::outs());
      llvm::outs() << "\n";
      auto cb = cast<ttkernel::CBType>(arg.getType());
      auto cbType = getTypeConverter()->convertType(cb);
      auto var = rewriter.create<emitc::VariableOp>(
          op.getLoc(), cbType, convertCBPort(rewriter, cb.getPort()));
      arg.replaceAllUsesWith(var);
    }
    op.getCallableRegion()->front().eraseArguments(0, blockArgs.size());
    op.setType(rewriter.getType<FunctionType>(TypeRange(), TypeRange()));
    rewriter.finalizeOpModification(op);

    return success();
  }

  // TTKernelToEmitCTypeConverter *typeConverter;
};

class TTMetalToEmitCReturnRewriter
    : public OpConversionPattern<ttkernel::ReturnOp> {
public:
  TTMetalToEmitCReturnRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                               MLIRContext *ctx)
      : OpConversionPattern<ttkernel::ReturnOp>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(ttkernel::ReturnOp op, ttkernel::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa<func::FuncOp>(op.getOperation()->getParentOp())) {
      return rewriter.notifyMatchFailure(op, "Not inside of func op");
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, ValueRange());
    return success();
  }
};

template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class TTMetalToEmitCOpaqueRewriter : public OpConversionPattern<SourceOp> {
public:
  TTMetalToEmitCOpaqueRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                               MLIRContext *ctx)
      : OpConversionPattern<SourceOp>(typeConverter, ctx) {}

  StringRef getOpName(SourceOp op) const {
    if constexpr (std::is_same_v<SourceOp, ttkernel::BuiltinOp>) {
      return op.getOp();
    }
    auto name = op.getOperation()->getName().getStringRef();
    if (name.starts_with("ttkernel.")) {
      return name.drop_front(9);
    }
    return name;
  }

  LogicalResult
  matchAndRewrite(SourceOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 4> resultTypes;
    // llvm::outs() << "numOperands: " << adaptor.getOperands().size() << "\n";
    for (Type type : op->getResultTypes()) {
      Type ct = this->getTypeConverter()->convertType(type);
      if (!ct) {
        return rewriter.notifyMatchFailure(op, "Failed to convert type ");
      }
      // resultTypes.push_back(this->getTypeConverter()->convertType(type));

      resultTypes.push_back(ct);
    }
    // llvm::outs() << "Rewriting opaque op: " << getOpName(op) << "\n";
    // llvm::outs() << "resultTypes: " << resultTypes.size() << "\n";
    // for (Type type : resultTypes) {
    //   type.print(llvm::outs());
    // }
    // llvm::outs() << "\n";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultTypes, getOpName(op), nullptr, nullptr,
        adaptor.getOperands());
    return success();
  }
};

// rename to converTTDispatchOpToEmitC
struct ConvertTTKernelToEmitCPass
    : public ttkernel::impl::ConvertTTKernelToEmitCBase<
          ConvertTTKernelToEmitCPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    // mlir::ConversionTarget target(getContext());

    // target.addLegalDialect<emitc::EmitCDialect>();

    moduleOp.walk([&](ttmetal::DispatchOp dispatchOp) {
      llvm::outs() << "Found dispatch op\n";
      dispatchOp->print(llvm::outs());

      // auto cleanupDispatchClone =
      //     llvm::make_scope_exit([&dispatchOp] { dispatchOp->erase(); });

      OpBuilder builder(dispatchOp.getOperation());

      int regionNumber = 0;
      for (auto &region : dispatchOp->getRegions()) {
        auto threadTypeAttr = mlir::cast<ttkernel::ThreadTypeAttr>(
            dispatchOp.getThreadTypes()[regionNumber++]);
        llvm::outs() << "region number: " << regionNumber
                     << " ---- starting conversion --------\n\n";

        // builder.setInsertionPoint(&region.getBlocks().front(),
        // region.getBlocks().front().begin()); issue is every next region is
        // inserted at the end of the previous we must separate them in
        // different functions

        // Replace the original block with a the new block containing a module
        // op
        auto module = builder.create<mlir::ModuleOp>(
            mlir::UnknownLoc::get(dispatchOp.getContext()),
            ttkernel::stringifyThreadType(threadTypeAttr.getValue()));
        // // cleanup causes hang in compile time
        // auto cleanupFreeModule =
        //     llvm::make_scope_exit([&module] { module->erase(); });
        auto &moduleBlock = module.getBodyRegion().front();

        module->setDiscardableAttr(
            builder.getStringAttr("ttkernel.thread_type"), threadTypeAttr);
        // start inserting in the module itself, at the end we will move the
        // insertion point after this module
        builder.setInsertionPointToStart(&moduleBlock);

        assert(module);

        // builder.setInsertionPoint(region.takeBody(Region ))

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
          builder.create<emitc::IncludeOp>(
              module.getLoc(), "compute_kernel_api/eltwise_binary.h",
              /*isStandard=*/false);
        }

        if (threadTypeAttr.getValue() == ttkernel::ThreadType::Tensix) {
          builder.create<emitc::VerbatimOp>(module.getLoc(),
                                            "namespace NAMESPACE {");
        }

        // Create a new func op and move the existing block into it.
        auto func = builder.create<func::FuncOp>(
            module.getLoc(), "kernel_main",
            builder.getType<FunctionType>(region.getArgumentTypes(),
                                          TypeRange()));

        llvm::outs() << "created funcOp ";
        func.print(llvm::outs());
        llvm::outs() << "\n";

        for (auto arg : func.getArguments()) {
          arg.print(llvm::outs());
          llvm::outs() << "\n";
        }

        Block *entryBlock = func.addEntryBlock();
        Region *funcBody = entryBlock->getParent();
        IRMapping irMapper;

        funcBody->takeBody(region);
        llvm::outs() << "funcBody->dump() before conversion\n";
        funcBody->front().print(llvm::outs());
        llvm::outs() << "\n";

        if (threadTypeAttr.getValue() == ttkernel::ThreadType::Tensix) {
          builder.create<emitc::VerbatimOp>(module.getLoc(),
                                            "void MAIN { kernel_main(); }");
          builder.create<emitc::VerbatimOp>(module.getLoc(), "}");
        }

        // Apply arith to emitc conversion first
        {
          llvm::outs() << "Applying arith to emitc conversion\n";
          ConversionTarget target(*module.getContext());
          target.addLegalDialect<emitc::EmitCDialect>();
          target.addIllegalDialect<arith::ArithDialect>();
          RewritePatternSet arithPatterns(module.getContext());
          TypeConverter arithTypeConverter;
          arithTypeConverter.addConversion([](Type type) { return type; });
          populateArithToEmitCPatterns(arithTypeConverter, arithPatterns);
          if (failed(applyPartialConversion(module, target,
                                            std::move(arithPatterns)))) {
            signalPassFailure();
            return;
          }
        }

        // Apply scf to emitc conversion next
        {
          llvm::outs() << "Applying scf to emitc conversion\n";
          ConversionTarget target(*module.getContext());
          target.addLegalDialect<emitc::EmitCDialect>();
          target.addIllegalDialect<scf::SCFDialect>();
          RewritePatternSet scfPatterns(module.getContext());
          populateSCFToEmitCConversionPatterns(scfPatterns);
          if (failed(applyPartialConversion(module, target,
                                            std::move(scfPatterns)))) {
            signalPassFailure();
            return;
          }
        }

        {
          llvm::outs() << "Applying ttmetal to emitc conversion\n";
          TTKernelToEmitCTypeConverter typeConverter(module.getContext());
          RewritePatternSet patterns(module.getContext());
          ConversionTarget target(*module.getContext());
          target.addLegalDialect<emitc::EmitCDialect>();
          target.addLegalDialect<ttmetal::TTMetalDialect>();
          target.addLegalOp<mlir::ModuleOp>();
          // target.addLegalOp<func::FuncOp>();
          // target.addIllegalOp<func::FuncOp>();
          target.addDynamicallyLegalOp<func::FuncOp>(
              [&](func::FuncOp op) -> bool {
                // Converting func op (kernel main) will result it having 0
                // arguments. At that point it becomes legal.
                return op.getNumArguments() == 0;
              });
          target.addLegalOp<func::ReturnOp>();
          // target.addLegalOp<mlir::UnrealizedConversionCastOp>();
          target.addIllegalDialect<ttkernel::TTKernelDialect>();

          patterns.add<TTMetalToEmitCFuncArgsRewriter>(typeConverter,
                                                       module.getContext());
          patterns.add<TTMetalToEmitCReturnRewriter>(typeConverter,
                                                     module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::BuiltinOp>>(
              typeConverter, module.getContext());
          patterns
              .add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsAcquireOp>>(
                  typeConverter, module.getContext());
          patterns
              .add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsCommitOp>>(
                  typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsWaitOp>>(
              typeConverter, module.getContext());
          patterns
              .add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsReleaseOp>>(
                  typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::PackTileOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeInitOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeInitOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeBlockOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeBlockOp>>(
              typeConverter, module.getContext());
          patterns.add<
              TTMetalToEmitCOpaqueRewriter<ttkernel::BinaryOpInitCommonOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::AddTilesInitOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::MulTilesInitOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::AddTilesOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::MulTilesOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::GetNocAddrOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadOp>>(
              typeConverter, module.getContext());
          patterns.add<
              TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadBarrierOp>>(
              typeConverter, module.getContext());
          patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteOp>>(
              typeConverter, module.getContext());
          patterns.add<
              TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteBarrierOp>>(
              typeConverter, module.getContext());
          if (failed(
                  applyFullConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
            return;
          }
        }

        builder.setInsertionPointAfter(module);
      }
    });
  }
};

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTKernelToEmitCPass() {
  return std::make_unique<ConvertTTKernelToEmitCPass>();
}

LogicalResult emitDispatchOpRegionsAsCpp(ttmetal::DispatchOp dispatchOp,
                                         std::vector<std::string> &cppStrings) {
  auto pm = PassManager::on<ttmetal::DispatchOp>(dispatchOp.getContext());
  pm.addPass(createConvertTTKernelToEmitCPass());

  // maybe we should wrap region in a module op before running the pass
  if (pm.run(dispatchOp).failed()) {
    return failure();
  }

  for (auto &reg : dispatchOp->getRegions()) {
    auto &block = reg.getBlocks().front();
    auto modOp = cast<ModuleOp>(block.getOperations().front());
    assert(modOp && "expected module op");

    std::string cppString;
    llvm::raw_string_ostream cppStream(cppString);
    if (emitc::translateToCpp(modOp.getOperation(), cppStream).failed()) {
      return failure();
    }

    cppStrings.push_back(cppString);
  }

  return success();
}

} // namespace mlir::tt
