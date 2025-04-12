// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>
#include <unordered_map>

using namespace mlir;
using namespace tt;

namespace mlir::tt::ttkernel {

#define GEN_PASS_DEF_CONVERTTTKERNELTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttkernel

// ............................................................................

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

// A no-op type converter:
// (note that the trivial T->T conversion is necessary)
namespace {
class NullTypeConverter : public TypeConverter {
public:
  NullTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](mlir::tt::ttir::MemTxType type) {
      return IntegerType::get(type.getContext(), 32,
          IntegerType::SignednessSemantics::Unsigned);
    });
  }
};
} // namespace

// Type converter used for TTKernel/TTMetal conversions:
namespace {
class TTKernelToEmitCTypeConverter : public NullTypeConverter {
public:
  TTKernelToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::tt::ttkernel::NocAddrType type) -> Type {
      return Builder(ctx).getI64Type();
    });
    addConversion([ctx](mlir::tt::ttkernel::CBType type) -> Type {
      return Builder(ctx).getType<emitc::OpaqueType>("::tt::CB");
    });
    addConversion([ctx](mlir::tt::ttkernel::L1AddrType type) -> Type {
      return Builder(ctx).getI32Type();
    });
    addConversion(
        [ctx](mlir::tt::ttkernel::L1AddrPtrType type) -> emitc::PointerType {
          return emitc::PointerType::get(
              emitc::OpaqueType::get(ctx, "volatile tt_l1_ptr uint32_t"));
        });
    addConversion([ctx](mlir::tt::ttkernel::DataFormatType type) -> Type {
      return emitc::OpaqueType::get(ctx, "DataFormat");
    });
    addConversion(
        [ctx](mlir::tt::ttkernel::InterleavedAddrGenFastType type) -> Type {
          // There is never a case in metal kernel code where template is false.
          return emitc::OpaqueType::get(ctx, "InterleavedAddrGenFast<true>");
        });
    addConversion([ctx](mlir::tt::ttir::MemTxType type) -> Type {
      return emitc::OpaqueType::get(ctx, "int");
    });
  }
};
} // namespace

namespace {
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
} // namespace

namespace {
class TTMetalToEmitCFuncArgsRewriter
    : public OpConversionPattern<func::FuncOp> {
public:
  TTMetalToEmitCFuncArgsRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                                 MLIRContext *ctx)
      : OpConversionPattern<func::FuncOp>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Block *block = &op.getCallableRegion()->front();
    auto blockArgs = block->getArguments();
    if (blockArgs.empty()) {
      return rewriter.notifyMatchFailure(op, "No block arguments");
    }

    TypeConverter::SignatureConversion signatureConverter(op.getNumArguments());
    OpBuilder::InsertionGuard funcInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(block);
    for (auto arg : blockArgs) {
      auto cb = cast<ttkernel::CBType>(arg.getType());
      auto cbType = getTypeConverter()->convertType(cb);
      auto cbPort = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), cbType, convertCBPort(rewriter, cb.getPort()));
      signatureConverter.remapInput(arg.getArgNumber(), cbPort.getResult());
    }

    rewriter.applySignatureConversion(block, signatureConverter,
                                      getTypeConverter());
    rewriter.modifyOpInPlace(op, [&]() {
      op.setType(rewriter.getFunctionType(TypeRange(), TypeRange()));
    });

    return success();
  }
};
} // namespace

namespace {
class TTKernelToEmitCGetCBOpRewriter
    : public OpConversionPattern<ttkernel::GetCBOp> {
public:
  using OpConversionPattern<ttkernel::GetCBOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::GetCBOp op, ttkernel::GetCBOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        op, this->getTypeConverter()->convertType(op.getCb().getType()),
        convertCBPort(rewriter, *ttkernel::symbolizeCBPort(op.getCbIndex())));
    return success();
  }
};
} // namespace

namespace {
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class TTKernelToEmitCOpaqueRewriter : public OpConversionPattern<SourceOp> {
public:
  TTKernelToEmitCOpaqueRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                                MLIRContext *ctx, std::string opName = "")
      : OpConversionPattern<SourceOp>(typeConverter, ctx), opName(opName) {}

  StringRef getOpName(SourceOp op) const {
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
    } else if constexpr (std::is_same_v<SourceOp,
                                        ttkernel::GetNocAddrFromBankIDOp>) {
      SmallVector<Attribute, 1> template_args;

      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), "true")); // default to DRAM
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
} // namespace

namespace {
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
        op, this->getTypeConverter()->convertType(op->getResultTypes()[0]),
        emitc::OpaqueAttr::get(op->getContext(), getMacroName(op)));
    return success();
  }
};
} // namespace

namespace {
template <typename Op, typename Adaptor = typename Op::Adaptor>
class TTKernelLiteralRewriter : public OpConversionPattern<Op> {
public:
  TTKernelLiteralRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                          MLIRContext *ctx, std::string literal)
      : OpConversionPattern<Op>(typeConverter, ctx), literal(literal) {}

  LogicalResult
  matchAndRewrite(Op op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::LiteralOp>(
        op, this->getTypeConverter()->convertType(op->getResultTypes()[0]),
        literal);
    return success();
  }

private:
  std::string literal;
};
} // namespace

namespace {
template <typename Op, typename Adaptor = typename Op::Adaptor>
class TTKernelToEmitCPassthroughRewriter : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace {
class TTKernelGetInterleavedAddrGenFastOpRewriter
    : public OpConversionPattern<ttkernel::GetInterleavedAddrGenFastOp> {
  using Op = ttkernel::GetInterleavedAddrGenFastOp;

public:
  TTKernelGetInterleavedAddrGenFastOpRewriter(
      const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(Op op, ttkernel::GetInterleavedAddrGenFastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getResult().getUses().empty()) {
      rewriter.eraseOp(op);
    } else {
      mlir::Type opaqueStructType =
          this->getTypeConverter()->convertType(op->getResultTypes()[0]);

      mlir::Type lvalueType = emitc::LValueType::get(opaqueStructType);

      // Declare the struct variable and then assign to its members
      auto varOp = rewriter.create<emitc::VariableOp>(
          op->getLoc(), lvalueType,
          emitc::OpaqueAttr::get(op.getContext(), ""));

      // Create an lvalue for all struct field accesses
      auto lvalueBankBaseAddr = rewriter.create<emitc::MemberOp>(
          op->getLoc(),
          emitc::LValueType::get(adaptor.getBankBaseAddress().getType()),
          "bank_base_address", varOp);
      auto lvaluePageSize = rewriter.create<emitc::MemberOp>(
          op->getLoc(), emitc::LValueType::get(adaptor.getPageSize().getType()),
          "page_size", varOp);
      auto lvalueDataFormat = rewriter.create<emitc::MemberOp>(
          op->getLoc(),
          emitc::LValueType::get(adaptor.getDataFormat().getType()),
          "data_format", varOp);

      // Assign corresponding values to the struct members
      rewriter.create<emitc::AssignOp>(op->getLoc(), lvalueBankBaseAddr,
                                       adaptor.getBankBaseAddress());
      rewriter.create<emitc::AssignOp>(op->getLoc(), lvaluePageSize,
                                       adaptor.getPageSize());
      rewriter.create<emitc::AssignOp>(op->getLoc(), lvalueDataFormat,
                                       adaptor.getDataFormat());

      // Load the value from the lvalue variable
      auto loadOp =
          rewriter.create<emitc::LoadOp>(op->getLoc(), opaqueStructType, varOp);

      // Replace the original operation with the loaded value so it can be used.
      rewriter.replaceOp(op, loadOp.getResult());
    }
    return success();
  }
};
} // namespace

namespace {
class ConvertTTKernelToEmitCPass
    : public ttkernel::impl::ConvertTTKernelToEmitCBase<
          ConvertTTKernelToEmitCPass> {
public:
  using ConvertTTKernelToEmitCBase<
      ConvertTTKernelToEmitCPass>::ConvertTTKernelToEmitCBase;

  void runOnOperation() final {
    auto wrapper = getOperation();

    wrapper.walk([&, this](func::FuncOp funcOp) { visit(funcOp); });
  }

  void visit(func::FuncOp funcOp) {
    if (!funcOp->hasAttr(ttkernel::ThreadTypeAttr::name)) {
      return;
    }

    ConversionTarget target(*funcOp.getContext());
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();
    target.addIllegalDialect<ttkernel::TTKernelDialect>();

    TTKernelToEmitCTypeConverter typeConverter(funcOp.getContext());
    RewritePatternSet patterns(funcOp.getContext());

    populateArithToEmitCPatterns(typeConverter, patterns);
    populateSCFToEmitCConversionPatterns(patterns, typeConverter);
    populateMemRefToEmitCTypeConversion(typeConverter);
    populateMemRefToEmitCConversionPatterns(patterns, typeConverter);

    // typeConverter.addConversion([](mlir::IndexType type) -> Type {
    //   return IntegerType::get(type.getContext(), 32,
    //                           IntegerType::SignednessSemantics::Unsigned);
    // });

    patterns.add<
        TTKernelToEmitCGetCBOpRewriter,
        TTKernelToEmitCPassthroughRewriter<ttkernel::CBReinterpretShapeOp>,
        TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosBaseOp>,
        TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosSizeOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetArgValOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CastToL1PtrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetSemaphoreOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreWaitMinOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreIncOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreWaitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetMulticastOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocSemaphoreSetMulticastLoopbackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsAcquireOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsCommitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsWaitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsReleaseOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PackTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TilizeInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UntilizeInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TilizeBlockOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UntilizeBlockOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryOpInitCommonOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulTilesInitFOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MaxTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MaxTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocAddrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadTileOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketSetStateOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketWithStateOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadBarrierOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteBarrierOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocMulticastAddrOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteMulticastOnePacketOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteMulticastOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryOpInitCommonOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetWritePtrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetReadPtrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetTileSizeOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetCompileArgValOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocAddrFromBankIDOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetDataFormatOp>>(
        typeConverter, funcOp.getContext());

    patterns.add<TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocAddrXYOp>>(
        typeConverter, funcOp.getContext(), "get_noc_addr");

    patterns.add<TTKernelLiteralRewriter<ttkernel::MyXOp>>(
        typeConverter, funcOp.getContext(), "NOC_X(my_x[noc_index])");
    patterns.add<TTKernelLiteralRewriter<ttkernel::MyYOp>>(
        typeConverter, funcOp.getContext(), "NOC_Y(my_y[noc_index])");

    patterns.add<TTKernelStoreToL1OpToEmitCOpRewriter>(typeConverter,
                                                       funcOp.getContext());

    patterns.add<TTKernelGetInterleavedAddrGenFastOpRewriter>(
        typeConverter, funcOp.getContext());

    if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<::mlir::Pass> createConvertTTKernelToEmitC() {
  return std::make_unique<ConvertTTKernelToEmitCPass>();
}

} // namespace mlir::tt
