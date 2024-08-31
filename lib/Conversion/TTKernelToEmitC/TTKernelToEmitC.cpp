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
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
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
      llvm::errs() << "Not inside of func op "
                   << op.getOperation()->getParentOp()->getName() << "\n";
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

  /*
    this conversion should work solely on tt kernel ops
    they shouuld be wrapped inside module op
    for testing we can create suitable mlir with those ops
    for production we will carve out each dispatch op's region in a new module
    op and run conversion
  */

  void runOnOperation() override {
    auto moduleOp = getOperation();

    OpBuilder builder(moduleOp);

    // module->setDiscardableAttr(builder.getStringAttr("ttkernel.thread_type"),
    //                            ttkernel::ThreadTypeAttr::get(moduleOp->getContext(),
    //                            threadType));

    // assert(module);

    Block *firstBlock = moduleOp.getBody();

    llvm::SmallVector<Operation *> fBodyOps;

    IRMapping irMapper;
    for (Operation &op : firstBlock->getOperations()) {
      fBodyOps.push_back(builder.clone(op, irMapper));
    }

    for (auto &[k, v] : irMapper.getValueMap()) {
      llvm::outs() << "key: ";
      k.print(llvm::outs());
      llvm::outs() << " value: ";
      v.print(llvm::outs());
      llvm::outs() << "\n";
    }

    for (auto fbodyOp : fBodyOps) {
      fbodyOp->print(llvm::outs());
      llvm::outs() << "\n";
      for (auto operand : fbodyOp->getOperands()) {
        operand.print(llvm::outs());
        llvm::outs() << "\n";
        if (irMapper.contains(operand)) {
          llvm::outs() << "found in irMapper\n";
          irMapper.lookup(operand).print(llvm::outs());
          llvm::outs() << "\n";
        }
      }
    }

    for (auto &op : firstBlock->getOperations()) {
      if (irMapper.contains(&op)) {
        llvm::outs() << "found in irMapper\n";
        irMapper.lookup(&op)->print(llvm::outs());
        llvm::outs() << "\n";

        op.replaceAllUsesWith(irMapper.lookup(&op));
      }
    }

    builder.createBlock(&moduleOp.getBodyRegion());

    // builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    builder.create<emitc::IncludeOp>(moduleOp.getLoc(), "cstdint",
                                     /*isStandard=*/true);
    if (threadType.getValue() == ttkernel::ThreadType::Noc0 ||
        threadType.getValue() == ttkernel::ThreadType::Noc1) {
      builder.create<emitc::IncludeOp>(moduleOp.getLoc(), "dataflow_api.h",
                                       /*isStandard=*/false);
    }
    if (threadType.getValue() == ttkernel::ThreadType::Tensix) {
      builder.create<emitc::IncludeOp>(moduleOp.getLoc(),
                                       "compute_kernel_api/common.h",
                                       /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(moduleOp.getLoc(),
                                       "compute_kernel_api/tilize.h",
                                       /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(moduleOp.getLoc(),
                                       "compute_kernel_api/untilize.h",
                                       /*isStandard=*/false);
      builder.create<emitc::IncludeOp>(moduleOp.getLoc(),
                                       "compute_kernel_api/eltwise_binary.h",
                                       /*isStandard=*/false);
    }

    if (threadType.getValue() == ttkernel::ThreadType::Tensix) {
      builder.create<emitc::VerbatimOp>(moduleOp.getLoc(),
                                        "namespace NAMESPACE {");
    }

    auto &region = moduleOp->getRegions().front();

    // moduleOp.emitRemark("region size: " +
    // std::to_string(moduleOp->getNumRegions())); llvm::errs() << "region size:
    // " << moduleOp->getNumRegions() << "\n";

    // Create a new func op and move the existing block into it.
    // builder.createBlock(&region);

    auto func = builder.create<func::FuncOp>(
        moduleOp.getLoc(), "kernel_main",
        builder.getType<FunctionType>(region.getArgumentTypes(), TypeRange()));

    Block *entryBlock = func.addEntryBlock();
    // Region *funcBody = entryBlock->getParent();
    // IRMapping irMapper;

    for (Operation *fop : fBodyOps) {
      entryBlock->push_back(fop);
    }

    // func->emitRemark("threadType: " +
    // stringifyThreadType(threadType.getValue()));

    // funcBody->takeBody(region);

    // region.cloneInto(funcBody, irMapper);
    // std::string f;
    // llvm::raw_string_ostream ff(f);
    // func->print(ff);
    // func.emitRemark("my body : \n" + ff.str());

    // func->emitRemark("threadType: " +
    // stringifyThreadType(threadType.getValue()));

    OpBuilder funcBuilder = OpBuilder::atBlockEnd(entryBlock);
    funcBuilder.setInsertionPointAfter(func);
    if (threadType.getValue() == ttkernel::ThreadType::Tensix) {
      funcBuilder.create<emitc::VerbatimOp>(moduleOp.getLoc(),
                                            "void MAIN { kernel_main(); }");
      funcBuilder.create<emitc::VerbatimOp>(moduleOp.getLoc(), "}");
    }

    firstBlock->erase();

    // Apply arith to emitc conversion first
    {
      llvm::errs() << "Applying arith to emitc conversion\n";
      ConversionTarget target(*moduleOp.getContext());
      target.addLegalDialect<emitc::EmitCDialect>();
      target.addIllegalDialect<arith::ArithDialect>();
      RewritePatternSet arithPatterns(moduleOp.getContext());
      TypeConverter arithTypeConverter;
      arithTypeConverter.addConversion([](Type type) { return type; });
      populateArithToEmitCPatterns(arithTypeConverter, arithPatterns);
      if (failed(
              applyPartialConversion(func, target, std::move(arithPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Apply scf to emitc conversion next
    {
      llvm::errs() << "Applying scf to emitc conversion\n";
      ConversionTarget target(*moduleOp.getContext());
      target.addLegalDialect<emitc::EmitCDialect>();
      target.addIllegalDialect<scf::SCFDialect>();
      RewritePatternSet scfPatterns(moduleOp.getContext());
      populateSCFToEmitCConversionPatterns(scfPatterns);
      if (failed(
              applyPartialConversion(func, target, std::move(scfPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    {
      llvm::outs() << "Applying ttmetal to emitc conversion\n";
      TTKernelToEmitCTypeConverter typeConverter(moduleOp.getContext());
      RewritePatternSet patterns(moduleOp.getContext());
      ConversionTarget target(*moduleOp.getContext());
      target.addLegalDialect<emitc::EmitCDialect>();
      target.addLegalDialect<ttmetal::TTMetalDialect>();
      target.addLegalOp<mlir::ModuleOp>();
      // target.addLegalOp<func::FuncOp>();
      // target.addIllegalOp<func::FuncOp>();
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) -> bool {
        // Converting func op (kernel main) will result it having 0
        // arguments. At that point it becomes legal.
        return op.getNumArguments() == 0;
      });
      target.addLegalOp<func::ReturnOp>();
      // target.addLegalOp<mlir::UnrealizedConversionCastOp>();
      target.addIllegalDialect<ttkernel::TTKernelDialect>();

      patterns.add<TTMetalToEmitCFuncArgsRewriter>(typeConverter,
                                                   moduleOp.getContext());
      patterns.add<TTMetalToEmitCReturnRewriter>(typeConverter,
                                                 moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::BuiltinOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsAcquireOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsCommitOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsWaitOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsReleaseOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::PackTileOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeInitOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeInitOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeBlockOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeBlockOp>>(
          typeConverter, moduleOp.getContext());
      patterns
          .add<TTMetalToEmitCOpaqueRewriter<ttkernel::BinaryOpInitCommonOp>>(
              typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::AddTilesInitOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::MulTilesInitOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::AddTilesOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::MulTilesOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::GetNocAddrOp>>(
          typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadOp>>(
          typeConverter, moduleOp.getContext());
      patterns
          .add<TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadBarrierOp>>(
              typeConverter, moduleOp.getContext());
      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteOp>>(
          typeConverter, moduleOp.getContext());
      patterns
          .add<TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteBarrierOp>>(
              typeConverter, moduleOp.getContext());
      if (failed(applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // builder.setInsertionPointAfter(module);

    // newDispatchOp->moveBefore(dispatchOp);
    // dispatchOp->replaceAllUsesWith(newDispatchOp);
    // dispatchOp->erase();

    // for (Operation &op : block->getOperations()) {
    //   op.erase();
    // }
    // firstBlock->erase();
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
