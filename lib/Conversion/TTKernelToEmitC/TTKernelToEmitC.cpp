// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

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
      auto cbOpaqueType = Builder(ctx).getType<emitc::OpaqueType>("::tt::CB");
      return emitc::LValueType::get(cbOpaqueType);
    });
    addConversion([ctx](mlir::tt::ttkernel::L1AddrType type) -> Type {
      return Builder(ctx).getI32Type();
    });
    addConversion(
        [ctx](mlir::tt::ttkernel::L1AddrPtrType type) -> emitc::PointerType {
          return emitc::PointerType::get(
              emitc::OpaqueType::get(ctx, "volatile tt_l1_ptr uint32_t"));
        });
  }
};

class TTKernelStoreToL1OpToEmitCOpRewriter
    : public OpConversionPattern<ttkernel::StoreToL1Op> {

public:
  TTKernelStoreToL1OpToEmitCOpRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ttkernel::StoreToL1Op>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(ttkernel::StoreToL1Op op,
                  ttkernel::StoreToL1Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto subscriptOp = rewriter.create<emitc::SubscriptOp>(
        op->getLoc(),
        emitc::LValueType::get(
            op.getContext(),
            mlir::cast<emitc::PointerType>(adaptor.getL1Ptr().getType())
                .getPointee()),
        adaptor.getL1Ptr(), adaptor.getOffset());

    // Cast rhs to volatile tt_l1_ptr uint32_t to match the pointed type.
    // This is because assignment requires the types to match. This compiles
    // in metal, but it looks ugly.
    auto casted = rewriter.create<emitc::CastOp>(
        op->getLoc(),
        emitc::OpaqueType::get(op.getContext(), "volatile tt_l1_ptr uint32_t"),
        adaptor.getValue());
    rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscriptOp, casted);
    return success();
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
    auto blockArgs = op.getCallableRegion()->getArguments();
    if (blockArgs.empty()) {
      return rewriter.notifyMatchFailure(op, "No block arguments");
    }
    rewriter.startOpModification(op);
    rewriter.setInsertionPointToStart(&op.getCallableRegion()->front());
    for (auto arg : blockArgs) {
      // Skip initialization if the argument is not a CBType (SemaphoreType)
      if (!mlir::isa<ttkernel::CBType>(arg.getType())) {
        continue;
      }
      auto cb = cast<ttkernel::CBType>(arg.getType());
      // Get opaque type i.e emitc::LValueType<emitc::OpaqueType>
      auto cbType = getTypeConverter()->convertType(cb);
      // Create a variable of type emitc::LValueType<emitc::OpaqueType>
      auto lValueVar = rewriter.create<emitc::VariableOp>(
          op.getLoc(), cbType, convertCBPort(rewriter, cb.getPort()));
      // Get the emitc::OpaqueType from the emitc::LValueType<emitc::OpaqueType>
      auto opaqueType = cast<emitc::LValueType>(cbType).getValueType();
      // Load the value from the lvalue variable this
      // will allow use to use the value
      auto var =
          rewriter.create<emitc::LoadOp>(op.getLoc(), opaqueType, lValueVar);
      arg.replaceAllUsesWith(var);
    }
    op.getCallableRegion()->front().eraseArguments(0, blockArgs.size());
    op.setType(rewriter.getType<FunctionType>(TypeRange(), TypeRange()));
    rewriter.finalizeOpModification(op);

    return success();
  }
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
                               MLIRContext *ctx, std::string opName = "")
      : OpConversionPattern<SourceOp>(typeConverter, ctx), opName(opName) {}

  StringRef getOpName(SourceOp op) const {
    if constexpr (std::is_same_v<SourceOp, ttkernel::BuiltinOp>) {
      return op.getOp();
    }
    auto name =
        opName.empty() ? op.getOperation()->getName().getStringRef() : opName;
    if (name.starts_with("ttkernel.")) {
      return name.drop_front(9);
    }
    return name;
  }

  template <typename ReduceKindOp>
  std::pair<StringRef, StringRef> getReduceTypeAndDim(ReduceKindOp op) const {
    StringRef reduceType =
        op.getReduceTypeAttr().getValue() == ttkernel::ReduceType::Max
            ? "PoolType::MAX"
            : "PoolType::SUM";
    StringRef reduceDim =
        op.getReduceDimAttr().getValue() == ttkernel::ReduceDim::Col
            ? "ReduceDim::REDUCE_COL"
        : op.getReduceDimAttr().getValue() == ttkernel::ReduceDim::Row
            ? "ReduceDim::REDUCE_ROW"
            : "ReduceDim::REDUCE_SCALAR";
    return {reduceType, reduceDim};
  }

  ArrayAttr getTemplateArgs(SourceOp op) const {
    if constexpr (std::is_same_v<SourceOp, ttkernel::ReduceInitOp> ||
                  std::is_same_v<SourceOp, ttkernel::ReduceTileOp>) {
      SmallVector<Attribute, 4> template_args;
      StringRef reduceType, reduceDim;
      if (mlir::isa<ttkernel::ReduceInitOp>(op)) {
        auto reduceInitOp = mlir::cast<ttkernel::ReduceInitOp>(op);
        template_args.push_back(emitc::OpaqueAttr::get(
            op.getContext(), "true")); // "at_start" template argument
        std::tie(reduceType, reduceDim) =
            getReduceTypeAndDim<ttkernel::ReduceInitOp>(reduceInitOp);
      } else {
        auto reduceOp = mlir::cast<ttkernel::ReduceTileOp>(op);
        std::tie(reduceType, reduceDim) =
            getReduceTypeAndDim<ttkernel::ReduceTileOp>(reduceOp);
      }
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), reduceType));
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), reduceDim));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::GetArgValOp>) {
      SmallVector<Attribute, 1> template_args;

      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), "uint32_t"));
      return ArrayAttr::get(op.getContext(), template_args);
    }
    return ArrayAttr();
  }

  LogicalResult
  matchAndRewrite(SourceOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 4> resultTypes;
    for (Type type : op->getResultTypes()) {
      Type ct = this->getTypeConverter()->convertType(type);
      if (!ct) {
        return rewriter.notifyMatchFailure(op, "Failed to convert type ");
      }
      resultTypes.push_back(ct);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultTypes, getOpName(op), nullptr, getTemplateArgs(op),
        adaptor.getOperands());

    return success();
  }

private:
  std::string opName;
};

template <typename Op, typename Adaptor = typename Op::Adaptor>
class TTKernelMacroOpToEmitCOpRewriter : public OpConversionPattern<Op> {
public:
  TTKernelMacroOpToEmitCOpRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                                   MLIRContext *ctx)
      : OpConversionPattern<Op>(typeConverter, ctx) {}

  std::string getMacroName(Op op) const {
    auto name = op.getOperation()->getName().getStringRef();
    name = name.drop_front(9);
    return name.upper();
  }

  LogicalResult
  matchAndRewrite(Op op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        op, op->getResultTypes(),
        emitc::OpaqueAttr::get(op->getContext(), getMacroName(op)));

    return success();
  }
};

class ConvertTTKernelToEmitCPass
    : public ttkernel::impl::ConvertTTKernelToEmitCBase<
          ConvertTTKernelToEmitCPass> {
public:
  using ConvertTTKernelToEmitCBase<
      ConvertTTKernelToEmitCPass>::ConvertTTKernelToEmitCBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Apply arith to emitc conversion first
    {
      ConversionTarget target(*funcOp.getContext());
      target.addLegalDialect<emitc::EmitCDialect>();
      target.addIllegalDialect<arith::ArithDialect>();
      RewritePatternSet arithPatterns(funcOp.getContext());
      TypeConverter arithTypeConverter;
      arithTypeConverter.addConversion([](Type type) { return type; });
      populateArithToEmitCPatterns(arithTypeConverter, arithPatterns);
      if (failed(applyPartialConversion(funcOp, target,
                                        std::move(arithPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Apply scf to emitc conversion next
    {
      ConversionTarget target(*funcOp.getContext());
      target.addLegalDialect<emitc::EmitCDialect>();
      target.addIllegalDialect<scf::SCFDialect>();
      RewritePatternSet scfPatterns(funcOp.getContext());
      populateSCFToEmitCConversionPatterns(scfPatterns);
      if (failed(
              applyPartialConversion(funcOp, target, std::move(scfPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    {
      TTKernelToEmitCTypeConverter typeConverter(funcOp.getContext());
      RewritePatternSet patterns(funcOp.getContext());
      ConversionTarget target(*funcOp.getContext());
      target.addLegalDialect<emitc::EmitCDialect>();
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) -> bool {
        // Converting func op (kernel main) will result it having 0
        // arguments. At that point it becomes legal.
        return op.getNumArguments() == 0;
      });
      target.addLegalOp<func::ReturnOp>();
      target.addIllegalDialect<ttkernel::TTKernelDialect>();

      patterns.add<
          TTMetalToEmitCFuncArgsRewriter, TTMetalToEmitCReturnRewriter,
          TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosBaseOp>,
          TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosSizeOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::BuiltinOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::GetArgValOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::CastToL1PtrOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::GetSemaphoreOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocSemaphoreWaitMinOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocSemaphoreIncOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocSemaphoreWaitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetMulticastOp>,
          TTMetalToEmitCOpaqueRewriter<
              ttkernel::NocSemaphoreSetMulticastLoopbackOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::CopyTileInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::RecipTileInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::RecipTileOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsAcquireOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsCommitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsWaitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::TileRegsReleaseOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::PackTileOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::TilizeBlockOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::UntilizeBlockOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::BinaryOpInitCommonOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::AddTilesInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::MulTilesInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::MulTilesInitFOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::MaxTilesInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::AddTilesOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::MulTilesOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::MaxTilesOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::ReduceInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::ReduceTileOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::GetNocAddrOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadOp>,
          TTMetalToEmitCOpaqueRewriter<
              ttkernel::NocAsyncReadOnePacketSetStateOp>,
          TTMetalToEmitCOpaqueRewriter<
              ttkernel::NocAsyncReadOnePacketWithStateOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncReadBarrierOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteBarrierOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::GetNocMulticastAddrOp>,
          TTMetalToEmitCOpaqueRewriter<
              ttkernel::NocAsyncWriteMulticastOnePacketOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteMulticastOp>,
          TTMetalToEmitCOpaqueRewriter<
              ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::UnaryOpInitCommonOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::CopyTileOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::ExpTileInitOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::ExpTileOp>,
          TTMetalToEmitCOpaqueRewriter<ttkernel::GetWritePtrOp>>(
          typeConverter, funcOp.getContext());

      patterns.add<TTMetalToEmitCOpaqueRewriter<ttkernel::GetNocAddrXYOp>>(
          typeConverter, funcOp.getContext(), "get_noc_addr");

      patterns.add<TTKernelStoreToL1OpToEmitCOpRewriter>(typeConverter,
                                                         funcOp.getContext());

      if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

namespace mlir::tt {

std::unique_ptr<::mlir::Pass> createConvertTTKernelToEmitC() {
  return std::make_unique<ConvertTTKernelToEmitCPass>();
}

// Class used to add includes and other boilerplate code to the generated
// kernel.
class ThreadConfigHelper {
public:
  ThreadConfigHelper(OpBuilder *builder, Location loc,
                     ttkernel::ThreadType threadType)
      : builder(builder), loc(loc), threadType(threadType) {
    builder->create<emitc::IncludeOp>(loc, "cstdint",
                                      /*isStandard=*/true);
    if (threadType == ttkernel::ThreadType::Noc) {

      builder->create<emitc::IncludeOp>(loc, "dataflow_api.h",
                                        /*isStandard=*/false);
    }
    if (threadType == ttkernel::ThreadType::Tensix) {
      builder->create<emitc::IncludeOp>(loc, "llk_defs.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/common.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/tilize.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/untilize.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc,
                                        "compute_kernel_api/eltwise_binary.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api.h", // max ops
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc,
                                        "compute_kernel_api/tile_move_copy.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/eltwise_unary.h",
          /*isStandard=*/false);
      // TODO (kmitrovic) exp.h is an ExpOp-specific include. Every op has one,
      // should be handled in general, not like this.
      // Issue: https://github.com/tenstorrent/tt-mlir/issues/772
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/exp.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/sfpu_split_includes.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/recip.h",
          /*isStandard=*/false);
      // Must define macros REDUCE_OP and REDUCE_DIM before including reduce.h
      // because they are default template parameters values in reduce api.
      builder->create<emitc::VerbatimOp>(loc,
                                         "#define REDUCE_OP PoolType::SUM");
      builder->create<emitc::VerbatimOp>(
          loc, "#define REDUCE_DIM ReduceDim::REDUCE_COL");
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/reduce.h",
                                        /*isStandard=*/false);
      builder->create<emitc::VerbatimOp>(loc, "namespace NAMESPACE {");
    }
  }

  ~ThreadConfigHelper() {
    if (threadType == ttkernel::ThreadType::Tensix) {
      builder->create<emitc::VerbatimOp>(loc, "void MAIN { kernel_main(); }");
      builder->create<emitc::VerbatimOp>(loc,
                                         "}"); // close namespace NAMESPACE
    }
  }

private:
  OpBuilder *builder;
  Location loc;
  ttkernel::ThreadType threadType;
};

LogicalResult
convertTTKernelRegionToEmitC(OpBuilder &builder, Region *region,
                             const ttkernel::ThreadType &threadType) {
  ThreadConfigHelper threadConfigHelper(&builder, region->getLoc(), threadType);

  auto funcOp = builder.create<func::FuncOp>(
      region->getLoc(), "kernel_main",
      builder.getType<FunctionType>(region->getArgumentTypes(), TypeRange()));

  IRMapping irMapper;
  region->cloneInto(&funcOp.getBody(), irMapper);

  auto pm = PassManager::on<func::FuncOp>(region->getContext());
  pm.addPass(createConvertTTKernelToEmitC());

  if (pm.run(funcOp).failed()) {
    return failure();
  }

  return success();
}

LogicalResult emitOpRegionAsCpp(Region *region, std::string &regionCpp,
                                const ttkernel::ThreadType &threadType) {

  llvm::raw_string_ostream os(regionCpp);
  return emitOpRegionAsCpp(region, os, threadType);
}

LogicalResult emitOpRegionAsCpp(Region *region, llvm::raw_ostream &os,
                                const ttkernel::ThreadType &threadType) {

  // We must load the EmitC dialect before we can emit any EmitC code. This
  // dialect won't be loaded by MLIR until pass manager starts a pass that
  // depends on it. Because we want to emit EmitC code before that, we need to
  // load it here.
  region->getContext()->getOrLoadDialect<emitc::EmitCDialect>();

  OpBuilder builder(region->getContext());
  // We will wrap everything in a module op so that we can run the
  // translation.
  auto moduleWrapper =
      builder.create<mlir::ModuleOp>(region->getLoc(), "module_wrapper");
  builder.setInsertionPointToStart(moduleWrapper.getBody());

  if (convertTTKernelRegionToEmitC(builder, region, threadType).failed()) {
    return failure();
  }

  if (emitc::translateToCpp(moduleWrapper, os).failed()) {
    return failure();
  }

  return success();
}

LogicalResult
emitEnqueueProgramOpRegionsAsCpp(ttmetal::EnqueueProgramOp enqueueProgramOp,
                                 llvm::SmallVector<std::string> &cppStrings) {
  assert(cppStrings.size() == enqueueProgramOp.getNumRegions() &&
         "cppStrings size must match number of regions");

  for (auto &reg : enqueueProgramOp->getRegions()) {
    auto kernelConfig = mlir::cast<ttkernel::KernelConfigInterface>(
        enqueueProgramOp.getKernelConfigs()[reg.getRegionNumber()]);
    if (emitOpRegionAsCpp(&reg, cppStrings[reg.getRegionNumber()],
                          kernelConfig.getThreadType())
            .failed()) {
      return llvm::failure();
    }
  }

  return success();
}

LogicalResult emitKernelAsCpp(mlir::ModuleOp op, llvm::raw_ostream &os,
                              const ttkernel::ThreadType &threadType) {
  llvm::SmallVector<func::FuncOp, 1> ops;
  op->walk([&](func::FuncOp entry) { ops.push_back(entry); });

  for (const auto &op : ops) {
    for (auto &reg : op->getRegions()) {
      if (emitOpRegionAsCpp(&reg, os, threadType).failed()) {
        return llvm::failure();
      }
    }
  }
  return llvm::success();
}

} // namespace mlir::tt
