// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

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

emitc::OpaqueAttr
datatypeToDataformatEnumValue(Builder &builder,
                              ::mlir::tt::ttcore::DataType dtype) {
  std::string expression =
      "static_cast<std::underlying_type_t<DataFormat>>(DataFormat::";
  switch (dtype) {
  case ::mlir::tt::ttcore::DataType::Float32:
    expression += "Float32";
    break;
  case ::mlir::tt::ttcore::DataType::Float16:
    expression += "Float16";
    break;
  case ::mlir::tt::ttcore::DataType::BFloat16:
    expression += "Float16_b";
    break;
  case ::mlir::tt::ttcore::DataType::BFP_Float8:
    expression += "Bfp8";
    break;
  case ::mlir::tt::ttcore::DataType::BFP_BFloat8:
    expression += "Bfp8_b";
    break;
  case ::mlir::tt::ttcore::DataType::BFP_Float4:
    expression += "Bfp4";
    break;
  case ::mlir::tt::ttcore::DataType::BFP_BFloat4:
    expression += "Bfp4_b";
    break;
  case ::mlir::tt::ttcore::DataType::BFP_Float2:
    expression += "Bfp2";
    break;
  case ::mlir::tt::ttcore::DataType::BFP_BFloat2:
    expression += "Bfp2_b";
    break;
  case ::mlir::tt::ttcore::DataType::UInt32:
    expression += "UInt32";
    break;
  case ::mlir::tt::ttcore::DataType::UInt16:
    expression += "UInt16";
    break;
  case ::mlir::tt::ttcore::DataType::UInt8:
    expression += "UInt8";
    break;
  case ::mlir::tt::ttcore::DataType::Int32:
    expression += "Int32";
    break;
  }
  expression += ")";
  return builder.getType<emitc::OpaqueAttr>(expression.c_str());
}

// Type converter used for TTKernel/TTMetal conversions:
namespace {
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
    addConversion([ctx](mlir::tt::ttkernel::SemaphoreType type) -> Type {
      // Convert semaphore to an address type. (i32)
      return Builder(ctx).getI32Type();
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
    addConversion(
        [ctx](mlir::tt::ttkernel::TensorAccessorArgsType type) -> Type {
          return emitc::OpaqueType::get(ctx, "TensorAccessorArgs");
        });
    addConversion([ctx](mlir::tt::ttkernel::TensorAccessorType type) -> Type {
      return emitc::OpaqueType::get(ctx, "TensorAccessor");
    });
    addConversion(
        [ctx](mlir::tt::ttkernel::TensorAccessorPageMappingType type) -> Type {
          return emitc::OpaqueType::get(ctx, "PageMapping");
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

  std::pair<StringRef, StringRef>
  reduceTypeAndDimToString(ttkernel::ReduceTypeAttr reduceTypeAttr,
                           ttkernel::ReduceDimAttr reduceDimAttr) const {
    StringRef reduceType =
        reduceTypeAttr.getValue() == ttkernel::ReduceType::Max
            ? "PoolType::MAX"
            : "PoolType::SUM";
    StringRef reduceDim = reduceDimAttr.getValue() == ttkernel::ReduceDim::Col
                              ? "ReduceDim::REDUCE_COL"
                          : reduceDimAttr.getValue() == ttkernel::ReduceDim::Row
                              ? "ReduceDim::REDUCE_ROW"
                              : "ReduceDim::REDUCE_SCALAR";
    return {reduceType, reduceDim};
  }

  StringRef getBroadcastType(ttkernel::BcastType bcastType) const {
    switch (bcastType) {
    case ttkernel::BcastType::Row:
      return "BroadcastType::ROW";
    case ttkernel::BcastType::Col:
      return "BroadcastType::COL";
    case ttkernel::BcastType::Scalar:
      return "BroadcastType::SCALAR";
    default:
      return "BroadcastType::NONE";
    }
  }

  ArrayAttr getTemplateArgs(Builder &builder, SourceOp op) const {
    if constexpr (std::is_same_v<SourceOp, ttkernel::ReduceInitOp> ||
                  std::is_same_v<SourceOp, ttkernel::ReduceTileOp>) {
      SmallVector<Attribute, 3> template_args;
      StringRef reduceType, reduceDim;
      std::tie(reduceType, reduceDim) = reduceTypeAndDimToString(
          op.getReduceTypeAttr(), op.getReduceDimAttr());
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), reduceType));
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), reduceDim));
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), op.getFullFp32() ? "true" : "false"));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::UnaryBcastInitOp> ||
                         std::is_same_v<SourceOp, ttkernel::UnaryBcastTileOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), getBroadcastType(op.getBcastType())));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::GetArgValOp> ||
                         std::is_same_v<SourceOp,
                                        ttkernel::GetCommonArgValOp>) {
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
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::PackTileOp>) {
      SmallVector<Attribute, 1> template_args;

      auto packTileOp = mlir::cast<ttkernel::PackTileOp>(op);

      template_args.push_back(packTileOp.getOutOfOrderAttr());
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::TypecastTileOp>) {
      SmallVector<Attribute, 2> template_args;
      template_args.push_back(
          datatypeToDataformatEnumValue(builder, op.getInDtype()));
      template_args.push_back(
          datatypeToDataformatEnumValue(builder, op.getOutDtype()));
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
        op, resultTypes, getOpName(op), nullptr, getTemplateArgs(rewriter, op),
        adaptor.getOperands());

    return success();
  }

private:
  std::string opName;
};
} // namespace

namespace {
class TTKernelToEmitCGetCompileArgValRewriter
    : public OpConversionPattern<ttkernel::GetCompileArgValOp> {
public:
  using OpConversionPattern<ttkernel::GetCompileArgValOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::GetCompileArgValOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::LiteralOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        (Twine("get_compile_time_arg_val(") + Twine(op.getArgIndex()) + ")")
            .str());
    return success();
  }
};
} // namespace

namespace {
class TTKernelToEmitCDPrintRewriter
    : public OpConversionPattern<ttkernel::DPrintOp> {
public:
  using OpConversionPattern<ttkernel::DPrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::DPrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto stringlit = [&](StringRef str) {
      return rewriter
          .create<emitc::LiteralOp>(
              op.getLoc(), rewriter.getType<emitc::OpaqueType>("const char[]"),
              (Twine("\"") + str + "\"").str())
          .getResult();
    };

    auto operandsIter = adaptor.getOperands().begin();
    auto operandsEnd = adaptor.getOperands().end();
    StringRef rest, fmt = op.getFmt();
    SmallVector<Value> vargs;
    do {
      std::tie(fmt, rest) = fmt.split("{}");
      if (!fmt.empty()) {
        vargs.push_back(stringlit(fmt));
      }
      if (operandsIter != operandsEnd) {
        vargs.push_back(*operandsIter++);
      }
      fmt = rest;
    } while (!fmt.empty());

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange(), "ttmlir::dprint", nullptr, nullptr, vargs);
    return success();
  }
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
class TTKernelConstantRewriter : public OpConversionPattern<Op> {
public:
  TTKernelConstantRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                           MLIRContext *ctx, std::string opaque)
      : OpConversionPattern<Op>(typeConverter, ctx), opaque(opaque) {}

  LogicalResult
  matchAndRewrite(Op op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        op, this->getTypeConverter()->convertType(op->getResultTypes()[0]),
        rewriter.getAttr<emitc::OpaqueAttr>(opaque));
    return success();
  }

private:
  std::string opaque;
};
} // namespace

namespace {
class TTKernelInvokeSFPIOpRewriter
    : public OpConversionPattern<ttkernel::InvokeSFPIOp> {
public:
  using OpConversionPattern<ttkernel::InvokeSFPIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::InvokeSFPIOp op,
                  ttkernel::InvokeSFPIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    assert(op.getRegion().hasOneBlock());
    rewriter.create<emitc::VerbatimOp>(op->getLoc(),
                                       "experimental::invoke_sfpi([=]() {");
    auto endScope = rewriter.create<emitc::VerbatimOp>(op->getLoc(), "});");
    rewriter.inlineBlockBefore(&op.getRegion().front(), endScope);
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::string opaque;
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
class TTKernelTensorAccessorArgsOpRewriter
    : public OpConversionPattern<ttkernel::TensorAccessorArgsOp> {
  using Op = ttkernel::TensorAccessorArgsOp;

public:
  TTKernelTensorAccessorArgsOpRewriter(const TypeConverter &typeConverter,
                                       MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(Op op, ttkernel::TensorAccessorArgsOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getResult().getUses().empty()) {
      rewriter.eraseOp(op);
    } else {
      auto name = op.getOperation()->getName().getStringRef().drop_front(9);

      // cta and crta are both passed through the template instead of operands
      ValueRange operands;
      SmallVector<Attribute, 2> template_args;
      auto cta_base = op.getCtaBase();
      auto crta_base = op.getCrtaBase();
      auto cta_base_attr = cta_base.getDefiningOp<arith::ConstantOp>();
      auto crta_base_attr = crta_base.getDefiningOp<arith::ConstantOp>();
      if (!cta_base_attr || !crta_base_attr) {
        llvm_unreachable(
            "MakeTensorAccessorArgsOp should have constant operands");
      }
      template_args.push_back(cta_base_attr.getValue());
      template_args.push_back(crta_base_attr.getValue());

      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          op, this->getTypeConverter()->convertType(op->getResultTypes()[0]),
          name, nullptr, ArrayAttr::get(op.getContext(), template_args),
          operands);
    }

    return success();
  }
};
} // namespace

namespace {
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class TTKernelTensorAccessorOpsRewriter : public OpConversionPattern<SourceOp> {
public:
  TTKernelTensorAccessorOpsRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                                    MLIRContext *ctx)
      : OpConversionPattern<SourceOp>(typeConverter, ctx) {}
  LogicalResult
  matchAndRewrite(SourceOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Drop "ttkernel.tensor_accessor_" prefix
    auto name = op.getOperation()->getName().getStringRef().drop_front(25);

    auto operands = adaptor.getOperands();
    if (operands.empty()) {
      return rewriter.notifyMatchFailure(
          op, "Expected TensorAccessor as first operand");
    }

    SmallVector<Type, 2> resultTypes;
    for (Type resultType : op->getResultTypes()) {
      Type convertedType = this->getTypeConverter()->convertType(resultType);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(op, "Failed to convert result type");
      }
      resultTypes.push_back(convertedType);
    }

    // Calling class/struct member function is difficult to do in EmitC..
    // Create a unique variable name based on SSA number.
    std::string ssaName;
    llvm::raw_string_ostream os(ssaName);
    mlir::OpPrintingFlags flags;
    op->getResult(0).printAsOperand(os, flags);
    os.flush();
    std::string varName = "temp_" + ssaName.substr(1);

    // Call the member function using verbatim with placeholders {} for args.
    std::string callStr = "uint32_t " + varName + " = {}." + name.str() + "(";
    for (size_t i = 0; i < operands.size() - 1; i++) {
      if (i > 0) {
        callStr += ", ";
      }
      callStr += "{}";
    }
    callStr += ");";

    rewriter.create<emitc::VerbatimOp>(
        op->getLoc(), rewriter.getStringAttr(callStr), operands);

    // create a literal referencing the temp variable to be used later.
    auto literalOp =
        rewriter.create<emitc::LiteralOp>(op->getLoc(), resultTypes, varName);

    rewriter.replaceOp(op, literalOp.getResult());

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
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (failed(visit(funcOp))) {
        signalPassFailure();
      }
    });
  }

  static LogicalResult visit(func::FuncOp funcOp) {
    if (!funcOp->hasAttr(ttkernel::ThreadTypeAttr::name)) {
      return success();
    }

    ConversionTarget target(*funcOp.getContext());
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<ttkernel::TTKernelDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) -> bool {
      // Converting func op (kernel main) will result it having 0
      // arguments. At that point it becomes legal.
      return op.getNumArguments() == 0;
    });

    TTKernelToEmitCTypeConverter typeConverter(funcOp.getContext());
    RewritePatternSet patterns(funcOp.getContext());

    populateArithToEmitCPatterns(typeConverter, patterns);
    populateSCFToEmitCConversionPatterns(patterns, typeConverter);
    populateMemRefToEmitCTypeConversion(typeConverter);
    populateMemRefToEmitCConversionPatterns(patterns, typeConverter);

    patterns.add<
        TTKernelToEmitCGetCompileArgValRewriter, TTKernelToEmitCDPrintRewriter,
        TTKernelToEmitCPassthroughRewriter<ttkernel::CBReinterpretShapeOp>,
        TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosBaseOp>,
        TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosSizeOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetArgValOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetCommonArgValOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CastToL1PtrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetSemaphoreOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreWaitMinOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreIncOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreWaitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetMulticastOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocSemaphoreSetMulticastLoopbackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsAcquireOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsCommitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsWaitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsReleaseOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBPushBackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBPopFrontOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBReserveBackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CBWaitFrontOp>,

        // Compute kernel hardware startup
        TTKernelToEmitCOpaqueRewriter<ttkernel::ComputeKernelHWStartupOp>,

        // Tilize & untilize
        TTKernelToEmitCOpaqueRewriter<ttkernel::TilizeInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TilizeUninitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UntilizeInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UntilizeUninitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TilizeBlockOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalTilizeBlockOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UntilizeBlockOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalUntilizeBlockOp>,

        // Datamovement
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PackTileOp>,

        // FPU Ops
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryOpInitCommonOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryOpInitCommonOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MatmulInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MatmulInitShortOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MatmulTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MatmulBlockInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MatmulBlockInitShortOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalMatmulBlockOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubTilesOp>,

        // Transpose Ops
        TTKernelToEmitCOpaqueRewriter<ttkernel::TransposeInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TransposeTileOp>,

        // SFPU Ops
        TTKernelToEmitCOpaqueRewriter<ttkernel::InitSFPUOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AbsTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AbsTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AbsTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CeilTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CeilTileF32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyDestValuesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyDestValuesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CosTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CosTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::DivBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::DivBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FloorTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FloorTileF32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FillTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FillTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogicalNotUnaryTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogicalNotUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogicalNotUnaryTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::EqzTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::EqzTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::EqzTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NezTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NezTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NezTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GtzTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GtzTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GtzTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GezTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GezTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GezTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LtzTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LtzTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LtzTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LezTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LezTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LezTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MaxTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MaxTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NegativeTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NegativeTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RoundingTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RsqrtTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RsqrtTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SqrtTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SqrtTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SigmoidTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SigmoidTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SinTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SinTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TypecastTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TypecastTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryBcastInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryBcastTileOp>,

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
            ttkernel::ExperimentalGetNocMulticastAddrOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteMulticastOnePacketOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteMulticastOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetWritePtrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetReadPtrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetTileSizeOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocAddrFromBankIDOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetDataFormatOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TensorAccessorOp>>(
        typeConverter, funcOp.getContext());

    patterns.add<TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocAddrOp>>(
        typeConverter, funcOp.getContext(), "get_noc_addr");

    patterns.add<TTKernelInvokeSFPIOpRewriter>(typeConverter,
                                               funcOp.getContext());

    patterns.add<TTKernelConstantRewriter<ttkernel::MyXOp>>(
        typeConverter, funcOp.getContext(), "my_x[noc_index]");
    patterns.add<TTKernelConstantRewriter<ttkernel::MyYOp>>(
        typeConverter, funcOp.getContext(), "my_y[noc_index]");

    patterns.add<TTKernelStoreToL1OpToEmitCOpRewriter>(typeConverter,
                                                       funcOp.getContext());

    patterns.add<TTKernelGetInterleavedAddrGenFastOpRewriter>(
        typeConverter, funcOp.getContext());

    patterns.add<TTKernelTensorAccessorArgsOpRewriter>(typeConverter,
                                                       funcOp.getContext());

    patterns.add<
        TTKernelTensorAccessorOpsRewriter<ttkernel::TensorAccessorGetNocAddrOp>,
        TTKernelTensorAccessorOpsRewriter<
            ttkernel::TensorAccessorGetShardNocAddrOp>,
        TTKernelTensorAccessorOpsRewriter<
            ttkernel::TensorAccessorGetBankAndOffsetOp>,
        TTKernelTensorAccessorOpsRewriter<
            ttkernel::TensorAccessorIsLocalBankOp>,
        TTKernelTensorAccessorOpsRewriter<
            ttkernel::TensorAccessorIsLocalAddrOp>,
        TTKernelTensorAccessorOpsRewriter<
            ttkernel::TensorAccessorIsLocalPageOp>,
        TTKernelTensorAccessorOpsRewriter<
            ttkernel::TensorAccessorIsLocalShardOp>>(typeConverter,
                                                     funcOp.getContext());

    return applyFullConversion(funcOp, target, std::move(patterns));
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<::mlir::Pass> createConvertTTKernelToEmitC() {
  return std::make_unique<ConvertTTKernelToEmitCPass>();
}

} // namespace mlir::tt
