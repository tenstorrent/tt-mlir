// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

#include "ttmlir/Asserts.h"
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
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;
using namespace tt;

namespace mlir::tt::ttkernel {

#define GEN_PASS_DEF_CONVERTTTKERNELTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttkernel

// ............................................................................

static std::string datatypeToDataformatStr(ttcore::DataType dtype) {
  std::string expression = "DataFormat::";
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
  case ::mlir::tt::ttcore::DataType::Bool:
    llvm_unreachable("Bool DataType is not supported in TTKernel DataFormat");
    break;
  }
  return expression;
}

static emitc::OpaqueAttr
datatypeToDataformatEnumNameOpaqueAttr(Builder &builder,
                                       ttcore::DataType dtype) {
  std::string expression = datatypeToDataformatStr(dtype);
  return builder.getType<emitc::OpaqueAttr>(expression.c_str());
}

static emitc::OpaqueAttr floatTypeToDataformatOpaqueAttr(Builder &builder,
                                                         Type type) {
  if (type.isF32()) {
    return builder.getType<emitc::OpaqueAttr>("DataFormat::Float32");
  }
  if (type.isBF16()) {
    return builder.getType<emitc::OpaqueAttr>("DataFormat::Float16_b");
  }
  llvm_unreachable("Unsupported float type for DataFormat conversion");
}

static emitc::OpaqueAttr
datatypeToDataformatEnumValueOpaqueAttr(Builder &builder,
                                        ttcore::DataType dtype) {
  std::string expression = "static_cast<std::underlying_type_t<DataFormat>>(";
  expression += datatypeToDataformatStr(dtype);
  expression += ")";
  return builder.getType<emitc::OpaqueAttr>(expression.c_str());
}

// Type converter used for TTKernel/TTMetal conversions:
namespace {
class TTKernelToEmitCTypeConverter : public TypeConverter {
public:
  TTKernelToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [ctx](BFloat16Type type) -> Type { return Float32Type::get(ctx); });
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
    addConversion(
        [ctx](mlir::tt::ttkernel::FabricConnectionManagerType type) -> Type {
          return emitc::OpaqueType::get(
              ctx, "experimental::FabricConnectionManager");
        });
  }
};
} // namespace

namespace {
class ArithConstantBF16ToF32Rewriter
    : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto floatAttr = dyn_cast<FloatAttr>(op.getValue());
    if (!floatAttr || !floatAttr.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a bf16 float constant");
    }
    double val = floatAttr.getValueAsDouble();
    auto f32Attr = rewriter.getF32FloatAttr(static_cast<float>(val));
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, f32Attr);
    return success();
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
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::TypecastTileOp> ||
                         std::is_same_v<SourceOp,
                                        ttkernel::TypecastTileInitOp>) {
      SmallVector<Attribute, 2> template_args;
      template_args.push_back(
          datatypeToDataformatEnumValueOpaqueAttr(builder, op.getInDtype()));
      template_args.push_back(
          datatypeToDataformatEnumValueOpaqueAttr(builder, op.getOutDtype()));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp,
                                        ttkernel::BinaryDestReuseTilesInitOp> ||
                         std::is_same_v<SourceOp,
                                        ttkernel::BinaryDestReuseTilesOp>) {
      SmallVector<Attribute, 2> template_args;
      StringRef eltwiseType;
      switch (op.getEltwiseBinaryType()) {
      case ttkernel::EltwiseBinaryType::Add:
        eltwiseType = "ELWADD";
        break;
      case ttkernel::EltwiseBinaryType::Sub:
        eltwiseType = "ELWSUB";
        break;
      case ttkernel::EltwiseBinaryType::Mul:
        eltwiseType = "ELWMUL";
        break;
      }
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), eltwiseType));
      StringRef reuseType =
          op.getReuseType() == ttkernel::BinaryDestReuseType::DestToSrcA
              ? "EltwiseBinaryReuseDestType::DEST_TO_SRCA"
              : "EltwiseBinaryReuseDestType::DEST_TO_SRCB";
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), reuseType));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::WhereTileOp> ||
                         std::is_same_v<SourceOp,
                                        ttkernel::BitwiseAndBinaryTilesOp> ||
                         std::is_same_v<SourceOp,
                                        ttkernel::BitwiseOrBinaryTilesOp> ||
                         std::is_same_v<SourceOp,
                                        ttkernel::BitwiseXorBinaryTilesOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(
          datatypeToDataformatEnumNameOpaqueAttr(builder, op.getDtype()));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (
        std::is_same_v<SourceOp, ttkernel::ExperimentalWriteRowMaskTileOp> ||
        std::is_same_v<SourceOp, ttkernel::ExperimentalWriteColMaskTileOp>) {
      auto cbType = mlir::cast<ttkernel::CBType>(op.getCb().getType());
      auto tileType = mlir::cast<ttcore::TileType>(cbType.getElementType());
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(datatypeToDataformatEnumNameOpaqueAttr(
          builder, tileType.getDataType()));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp,
                                        ttkernel::ExperimentalTileFillOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(
          floatTypeToDataformatOpaqueAttr(builder, op.getValue().getType()));
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
    StringRef fmt = op.getFmt();

    auto stringlit = [&](StringRef str) {
      return rewriter
          .create<emitc::LiteralOp>(
              op.getLoc(), rewriter.getType<emitc::OpaqueType>("const char[]"),
              (Twine("\"") + str + "\"").str())
          .getResult();
    };

    auto operandsIter = adaptor.getOperands().begin();
    auto operandsEnd = adaptor.getOperands().end();
    StringRef rest;
    SmallVector<Value> vargs;
    do {
      std::tie(fmt, rest) = fmt.split("{}");
      if (!fmt.empty()) {
        vargs.push_back(stringlit(fmt));
      }
      if (operandsIter != operandsEnd) {
        if (mlir::isa<ttkernel::CBType>(
                op.getOperands()[operandsIter.getIndex()].getType()) &&
            op->getParentOfType<func::FuncOp>()
                    ->getAttrOfType<ttkernel::ThreadTypeAttr>(
                        ttkernel::ThreadTypeAttr::name)
                    .getValue() == ttkernel::ThreadType::Compute) {
          auto cbPrinter =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      op.getLoc(),
                      rewriter.getType<emitc::OpaqueType>("ttmlir::CBPrinter"),
                      "ttmlir::CBPrinter", nullptr, nullptr,
                      ValueRange{*operandsIter++})
                  .getResult(0);
          vargs.push_back(cbPrinter);
        } else {
          vargs.push_back(*operandsIter++);
        }
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
      return success();
    }

    // Generate unique variable name from SSA number (pattern from
    // TTKernelClassMethodRewriter).
    std::string ssaName;
    llvm::raw_string_ostream os(ssaName);
    mlir::OpPrintingFlags flags;
    op->getResult(0).printAsOperand(os, flags);
    os.flush();
    std::string varName = "tensor_accessor_args_" + ssaName.substr(1);

    // Build CTA/CRTA expression with priority: expr attr > chaining > literal.
    auto buildArgExpr = [&](StringAttr exprAttr, Value baseValue,
                            StringRef chainMethodName) -> std::string {
      if (exprAttr) {
        // Explicit constexpr string expression (overrides chaining).
        return exprAttr.getValue().str();
      }
      if (op.getPrevArgs()) {
        // Chaining from previous accessor.
        auto prevLiteral =
            adaptor.getPrevArgs().getDefiningOp<emitc::LiteralOp>();
        TT_assertv(prevLiteral,
                   "prev_args should be emitc.literal after conversion.");
        return prevLiteral.getValue().str() + "." + chainMethodName.str() +
               "()";
      }
      // Literal integer constant (verifier ensures this is a constant).
      auto baseAttr = baseValue.getDefiningOp<arith::ConstantOp>();
      TT_assertv(baseAttr, "base should be constant.");
      return std::to_string(cast<IntegerAttr>(baseAttr.getValue()).getInt());
    };

    std::string ctaArg = buildArgExpr(op.getCtaExprAttr(), op.getCtaBase(),
                                      "next_compile_time_args_offset");
    std::string crtaArg = buildArgExpr(op.getCrtaExprAttr(), op.getCrtaBase(),
                                       "next_common_runtime_args_offset");

    // Emit: auto tensor_accessor_args_N = TensorAccessorArgs<ctaArg,
    // crtaArg>();
    std::string code = "auto " + varName + " = TensorAccessorArgs<" + ctaArg +
                       ", " + crtaArg + ">();";
    rewriter.create<emitc::VerbatimOp>(op.getLoc(), code);

    // Create literal to reference the variable (pattern from
    // TTKernelClassMethodRewriter).
    auto resultType =
        this->getTypeConverter()->convertType(op->getResultTypes()[0]);
    auto literalOp =
        rewriter.create<emitc::LiteralOp>(op.getLoc(), resultType, varName);

    rewriter.replaceOp(op, literalOp.getResult());
    return success();
  }
};
} // namespace

namespace {
class TTKernelCreateFabricConnectionManagerOpRewriter
    : public OpConversionPattern<ttkernel::CreateFabricConnectionManagerOp> {
  using Op = ttkernel::CreateFabricConnectionManagerOp;

public:
  TTKernelCreateFabricConnectionManagerOpRewriter(
      const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(Op op,
                  ttkernel::CreateFabricConnectionManagerOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getResult().getUses().empty()) {
      rewriter.eraseOp(op);
    } else {
      mlir::Type opaqueStructType =
          this->getTypeConverter()->convertType(op->getResultTypes()[0]);

      mlir::Type lvalueType = emitc::LValueType::get(opaqueStructType);

      // Declare the struct variable
      auto varOp = rewriter.create<emitc::VariableOp>(
          op->getLoc(), lvalueType,
          emitc::OpaqueAttr::get(op.getContext(), ""));

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
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class TTKernelClassMethodRewriter : public OpConversionPattern<SourceOp> {
public:
  TTKernelClassMethodRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                              MLIRContext *ctx)
      : OpConversionPattern<SourceOp>(typeConverter, ctx) {}

  static std::string typeAsString(Type ty) {
    if (auto i = mlir::dyn_cast<IntegerType>(ty)) {
      if (i.getWidth() == 1) {
        return "bool";
      }

      if (i.getWidth() == 32) {
        return "uint32_t";
      }

      if (i.getWidth() == 64) {
        return "uint64_t";
      }

      llvm_unreachable(
          "unsupported integer type in TTKernelClassMethodRewriter");
    }

    if (auto opaque = mlir::dyn_cast<emitc::OpaqueType>(ty)) {
      return opaque.getValue().str();
    }

    llvm_unreachable("unsupported emitc type in TTKernelClassMethodRewriter");
  }

  LogicalResult
  matchAndRewrite(SourceOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Drop "ttkernel.class_name." prefix
    auto [className, methodName] =
        op.getOperation()->getName().getStringRef().rsplit('.');
    if (methodName.empty()) {
      return failure();
    }

    auto operands = adaptor.getOperands();
    if (operands.empty()) {
      return rewriter.notifyMatchFailure(
          op, "Expected class self as first operand");
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
    TT_assert(resultTypes.size() == 1u);
    std::string callStr = typeAsString(resultTypes[0]) + " " + varName +
                          " = {}." + methodName.str() + "(";
    for (size_t i = 1; i < operands.size(); i++) {
      if (i > 1) {
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
// Arith FloorDivSIOp doesn't have an emitc lowering, probably because of the
// spec which says:
//   Signed integer division. Rounds towards negative infinity, i.e. 5 / -2 = -3
//
// However we know our index type will map to size_t which is unsigned, making a
// negative denominator impossible, so as long as we assert that this floordiv
// is working on values of `index` type it's safe to map this op to regular
// divi.
class ArithFloorDivRewriter : public OpConversionPattern<arith::FloorDivSIOp> {
public:
  using OpConversionPattern<arith::FloorDivSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::FloorDivSIOp op, arith::FloorDivSIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!mlir::isa<IndexType>(op.getResult().getType())) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, op.getResult().getType(),
                                                op.getOperands());

    return success();
  }
};

// Convert arith.bitcast to a call to float_to_bits helper.
// This is needed for scalar tile ops that pass float values as integer params.
// The helper function is defined in TTKernelToCpp.cpp during code generation.
class ArithBitcastRewriter : public OpConversionPattern<arith::BitcastOp> {
public:
  using OpConversionPattern<arith::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::BitcastOp op, arith::BitcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }

    // Call the float_to_bits helper which uses memcpy to bitcast float to int.
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultType, "float_to_bits",
        /*args=*/nullptr,
        /*templateArgs=*/nullptr, adaptor.getOperands());

    return success();
  }
};

// Rewriter for scalar unary tile ops (add_unary_tile, mul_unary_tile, etc).
// These ops take a tile index and a scalar parameter. The custom GCC may not
// see the data dependency between the scalar value and the SFPU intrinsic,
// potentially optimizing away the scalar computation.
//
// We bounce the scalar through a volatile variable to prevent this:
//   volatile int32_t __scalar = param;
//   mul_unary_tile(idx, __scalar);
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class TTKernelScalarUnaryTileOpRewriter : public OpConversionPattern<SourceOp> {
public:
  TTKernelScalarUnaryTileOpRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                                    MLIRContext *ctx)
      : OpConversionPattern<SourceOp>(typeConverter, ctx) {}

  StringRef getOpName(SourceOp op) const {
    auto name = op.getOperation()->getName().getStringRef();
    if (name.starts_with("ttkernel.")) {
      return name.drop_front(9);
    }
    return name;
  }

  LogicalResult
  matchAndRewrite(SourceOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    // Expect (dst_index, scalar_param).
    if (operands.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Expected exactly 2 operands for scalar unary tile op");
    }

    Value dstIndex = operands[0];
    Value scalarParam = operands[1];

    // Use verbatim to emit the volatile bounce directly.
    // This works around EmitC's strict type checking, and avoid sfpi-gcc bug.
    //
    // Emits: { volatile int32_t __s = <scalar>; <op>(<idx>, __s); }
    // Note that apparently "{{" produces "{" but "}" is not escaped in EmitC.
    std::string code =
        "{{ volatile int32_t __s = {}; " + getOpName(op).str() + "({}, __s); }";
    rewriter.create<emitc::VerbatimOp>(op->getLoc(),
                                       rewriter.getStringAttr(code),
                                       ValueRange{scalarParam, dstIndex});
    rewriter.eraseOp(op);

    return success();
  }
};

// PackReconfigL1AccOp must be wrapped in the PACK((...)) macro to ensure it
// only executes on the TRISC_PACK thread.
class TTKernelToEmitCPackReconfigL1AccToEmitCRewriter
    : public OpConversionPattern<ttkernel::PackReconfigL1AccOp> {
public:
  using OpConversionPattern<ttkernel::PackReconfigL1AccOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::PackReconfigL1AccOp op,
                  ttkernel::PackReconfigL1AccOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.create<emitc::VerbatimOp>(
        op->getLoc(),
        rewriter.getStringAttr("PACK((llk_pack_reconfig_l1_acc({})));"),
        ValueRange{adaptor.getL1AccEn()});
    rewriter.eraseOp(op);
    return success();
  }
};

// Arith MaxUIOp doesn't have an emitc lowering. We can lower it to a call to
// std::max.
class ArithMaxUIRewriter : public OpConversionPattern<arith::MaxUIOp> {
public:
  using OpConversionPattern<arith::MaxUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MaxUIOp op, arith::MaxUIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultType, "std::max<size_t>", adaptor.getOperands());

    return success();
  }
};

// Arith MinUIOp doesn't have an emitc lowering. We can lower it to a call to
// std::min.
class ArithMinUIRewriter : public OpConversionPattern<arith::MinUIOp> {
public:
  using OpConversionPattern<arith::MinUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MinUIOp op, arith::MinUIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }

    // Explicit type template needed for some edge cases where emitc might lower
    // an int literal into the call with a size_t arg, creating sfpi compiler
    // error.
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultType, "std::min<size_t>", adaptor.getOperands());

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

    patterns.add<ArithConstantBF16ToF32Rewriter>(typeConverter,
                                                 funcOp.getContext(),
                                                 /*benefit=*/2);
    populateArithToEmitCPatterns(typeConverter, patterns);
    populateSCFToEmitCConversionPatterns(patterns, typeConverter);
    populateMemRefToEmitCTypeConversion(typeConverter);
    populateMemRefToEmitCConversionPatterns(patterns, typeConverter);

    patterns.add<
        TTKernelToEmitCGetCompileArgValRewriter, TTKernelToEmitCDPrintRewriter,
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
        TTKernelToEmitCPackReconfigL1AccToEmitCRewriter,

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
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryDestReuseTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryDestReuseTilesOp>,

        // Transpose Ops
        TTKernelToEmitCOpaqueRewriter<ttkernel::TransposeInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TransposeTileOp>,

        // SFPU Ops
        TTKernelToEmitCOpaqueRewriter<ttkernel::InitSFPUOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AbsTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AbsTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AbsTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryBitwiseTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinopWithScalarTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BitwiseAndBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BitwiseNotTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BitwiseNotTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BitwiseOrBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BitwiseXorBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SignTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SignTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CeilTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyDestValuesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyDestValuesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CosTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CosTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddBinaryTilesOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::AddUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::DivBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::DivBinaryTilesOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::DivUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfcTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfcTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FloorTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FillTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FillTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GeluTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GeluTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::HardsigmoidTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::HardsigmoidTileOp>,
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
        TTKernelScalarUnaryTileOpRewriter<ttkernel::MulUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubBinaryTilesOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::SubUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMaxTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMaxTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMinTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMinTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NegativeTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NegativeTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowerTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceUninitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReluTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReluTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReluTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RoundingTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RsqrtTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RsqrtTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SqrtTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SqrtTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SigmoidTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SigmoidTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SiluTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SiluTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SinTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SinTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanhTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanhTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TypecastTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TypecastTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalTileFillOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalWriteRowMaskTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalWriteColMaskTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalFillArangeTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryBcastInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryBcastTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::WhereTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::WhereTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ClampScalarTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ClampScalarTileOp>,

        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocAddrOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadTileOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketSetStateOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketWithStateOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketWithStateWithTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadSetTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadBarrierOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadBarrierWithTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteSetTridOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteOnePacketWithTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteBarrierOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteBarrierWithTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ResetNocTridBarrierCounterOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocMulticastAddrOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::ExperimentalGetNocMulticastAddrOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteMulticastOnePacketOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteMulticastOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ConvertLogicalXToTranslatedOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ConvertLogicalYToTranslatedOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetMyDeviceIdOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FabricWriteOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FabricMulticastWriteOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FabricSemIncOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::CreateFabricConnectionManagerOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SetupFabricConnectionsOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CloseFabricConnectionsOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetLogicalMeshPositionOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::GetDeviceIdFromLogicalMeshPositionOp>,
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
    patterns.add<TTKernelConstantRewriter<ttkernel::MyLogicalXOp>>(
        typeConverter, funcOp.getContext(), "get_absolute_logical_x()");
    patterns.add<TTKernelConstantRewriter<ttkernel::MyLogicalYOp>>(
        typeConverter, funcOp.getContext(), "get_absolute_logical_y()");

    patterns.add<TTKernelStoreToL1OpToEmitCOpRewriter>(typeConverter,
                                                       funcOp.getContext());

    patterns.add<TTKernelGetInterleavedAddrGenFastOpRewriter>(
        typeConverter, funcOp.getContext());

    patterns.add<TTKernelTensorAccessorArgsOpRewriter>(typeConverter,
                                                       funcOp.getContext());

    patterns.add<TTKernelCreateFabricConnectionManagerOpRewriter>(
        typeConverter, funcOp.getContext());

    patterns.add<
        TTKernelClassMethodRewriter<ttkernel::TensorAccessorGetNocAddrOp>,
        TTKernelClassMethodRewriter<ttkernel::TensorAccessorGetShardNocAddrOp>,
        TTKernelClassMethodRewriter<ttkernel::TensorAccessorGetBankAndOffsetOp>,
        TTKernelClassMethodRewriter<ttkernel::TensorAccessorIsLocalBankOp>,
        TTKernelClassMethodRewriter<ttkernel::TensorAccessorIsLocalAddrOp>,
        TTKernelClassMethodRewriter<ttkernel::TensorAccessorIsLocalPageOp>,
        TTKernelClassMethodRewriter<ttkernel::TensorAccessorIsLocalShardOp>,
        TTKernelClassMethodRewriter<
            ttkernel::InterleavedAddrGenFastGetNocAddrOp>>(typeConverter,
                                                           funcOp.getContext());

    patterns.add<ArithFloorDivRewriter, ArithBitcastRewriter,
                 ArithMaxUIRewriter, ArithMinUIRewriter>(typeConverter,
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
