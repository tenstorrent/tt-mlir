// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
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

static std::string getTTKernelCalleeName(llvm::StringRef opName) {
  if (opName.starts_with("ttkernel.")) {
    opName = opName.drop_front(9);
  }
  if (opName.starts_with("experimental.")) {
    return ("experimental::" + opName.drop_front(13)).str();
  }
  return opName.str();
}

static emitc::OpaqueAttr
datatypeToDataformatEnumNameOpaqueAttr(Builder &builder,
                                       ttcore::DataType dtype) {
  std::string expression = datatypeToDataformatStr(dtype);
  return builder.getType<emitc::OpaqueAttr>(expression.c_str());
}

static emitc::OpaqueAttr
datatypeToDataformatEnumValueOpaqueAttr(Builder &builder,
                                        ttcore::DataType dtype) {
  std::string expression = "static_cast<std::underlying_type_t<DataFormat>>(";
  expression += datatypeToDataformatStr(dtype);
  expression += ")";
  return builder.getType<emitc::OpaqueAttr>(expression.c_str());
}

static std::string getCBName(Value cb) {
  std::string prefix = "";
  IntegerAttr cbIdxAttr = nullptr;

  if (auto *defOp = cb.getDefiningOp()) {
    if (auto attr =
            defOp->getAttrOfType<IntegerAttr>("ttkernel.cb_ctarg_idx")) {
      prefix = "cb_ctarg_";
      cbIdxAttr = attr;
    } else if (auto attr =
                   defOp->getAttrOfType<IntegerAttr>("ttkernel.cb_arg_idx")) {
      prefix = "cb_arg_";
      cbIdxAttr = attr;
    }
  }

  TT_assertv(cbIdxAttr, "CB value must have a stable lowering name");
  return prefix + std::to_string(cbIdxAttr.getInt());
}

// Assign deterministic names to non-compile-time CB arguments before dialect
// conversion. Runtime and common runtime arguments use their own common index
// space, derived from the function-local source-order to be stable &
// non-conflicting.
static void assignRuntimeCBArgIndices(func::FuncOp funcOp) {
  int32_t nextCBArgIndex = 0;
  funcOp.walk([&](Operation *op) {
    if (!isa<ttkernel::GetArgValOp, ttkernel::GetCommonArgValOp>(op)) {
      return;
    }

    if (!isa<ttkernel::CBType>(op->getResult(0).getType())) {
      return;
    }

    op->setAttr("ttkernel.cb_arg_idx",
                IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                 nextCBArgIndex++));
  });
}

static bool isCBDeclaration(emitc::VerbatimOp verbatim,
                            llvm::StringRef cbName) {
  std::string prefix = "CircularBuffer " + cbName.str() + "(";
  return verbatim.getValue().starts_with(prefix);
}

static void setInsertionPointAfterDefOrBlockStart(Value value,
                                                  OpBuilder &builder) {
  if (Operation *defOp = value.getDefiningOp()) {
    builder.setInsertionPointAfter(defOp);
    return;
  }

  builder.setInsertionPointToStart(value.getParentBlock());
}

// Check whether a matching verbatim op exists & dominates `useOp`, to avoid
// duplicating declarations. To handle declarations that live in outer enclosing
// blocks, walk upward through the lexical parent chain and scan ops that appear
// before the current use at each nesting level. Since the IR is all-SCF at this
// stage, this strategy is equivalent but faster than full-funcOp level
// dominance analysis.
template <typename Predicate>
static bool hasDominatingVerbatim(Operation *useOp, Predicate predicate) {
  Operation *currentOp = useOp;
  Block *currentBlock = currentOp->getBlock();

  while (currentBlock) {
    for (Operation &op : *currentBlock) {
      if (&op == currentOp) {
        break;
      }
      if (auto verbatimOp = mlir::dyn_cast<emitc::VerbatimOp>(op);
          verbatimOp && predicate(verbatimOp)) {
        return true;
      }
    }

    if (Operation *parentOp = currentBlock->getParentOp()) {
      currentOp = parentOp;
      currentBlock = parentOp->getBlock();
    } else {
      break;
    }
  }

  return false;
}

// Lazily emit a CB declaration only when a CB method is actually invoked
// (compute API only uses CB IDs).
static std::string ensureCBDeclaration(Value cb, Operation *useOp,
                                       ConversionPatternRewriter &rewriter) {
  std::string cbName = getCBName(cb);
  if (hasDominatingVerbatim(useOp, [&](emitc::VerbatimOp verbatimOp) {
        return isCBDeclaration(verbatimOp, cbName);
      })) {
    return cbName;
  }

  // Place the declaration right after the value's definition so it dominates
  // every use site, including those in nested regions (e.g. loop bodies).
  OpBuilder::InsertionGuard guard(rewriter);
  setInsertionPointAfterDefOrBlockStart(cb, rewriter);

  std::string cbDecl = "CircularBuffer " + cbName + "({});";
  rewriter.create<emitc::VerbatimOp>(useOp->getLoc(), cbDecl, ValueRange{cb});
  return cbName;
}

static StringRef getL1PtrOpaqueTypeName(unsigned elementWidth) {
  switch (elementWidth) {
  case 8:
    return "tt_l1_ptr uint8_t";
  case 16:
    return "tt_l1_ptr uint16_t";
  case 32:
    return "tt_l1_ptr uint32_t";
  default:
    llvm_unreachable("unsupported L1AddrPtr element width");
  }
}

static bool hasDominatingVerbatimWithPrefix(Operation *useOp,
                                            llvm::StringRef prefix) {
  return hasDominatingVerbatim(useOp, [&](emitc::VerbatimOp verbatimOp) {
    return verbatimOp.getValue().starts_with(prefix);
  });
}

static FailureOr<int64_t> extractNocIndex(Attribute value) {
  auto nocIdxAttr = mlir::dyn_cast_if_present<IntegerAttr>(value);
  if (!nocIdxAttr) {
    return failure();
  }

  const int64_t nocIdx = nocIdxAttr.getInt();
  if (nocIdx != 0 && nocIdx != 1) {
    return failure();
  }

  return nocIdx;
}

static FailureOr<int64_t> getStaticNocIndex(Operation *useOp,
                                            Value nocId = {}) {
  if (nocId) {
    if (auto constantOp = nocId.getDefiningOp<arith::ConstantOp>()) {
      return extractNocIndex(constantOp.getValue());
    }
    if (auto constantOp = nocId.getDefiningOp<emitc::ConstantOp>()) {
      return extractNocIndex(constantOp.getValue());
    }
    return failure();
  }

  auto funcOp = useOp->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return failure();
  }

  StringRef kernelSymbol = funcOp.getSymName();
  FailureOr<int64_t> nocIdx = failure();
  for (Operation *scope = funcOp->getParentOp(); scope && failed(nocIdx);
       scope = scope->getParentOp()) {
    scope->walk([&](ttmetal::EnqueueProgramOp enqueueProgram) {
      if (succeeded(nocIdx)) {
        return;
      }

      for (Attribute kernelConfig : enqueueProgram.getKernelConfigs()) {
        auto nocConfig = mlir::dyn_cast<ttmetal::NocConfigAttr>(kernelConfig);
        if (!nocConfig ||
            nocConfig.getKernelSymbol().getRootReference() != kernelSymbol) {
          continue;
        }

        switch (nocConfig.getNocIndex()) {
        case ttcore::NocIndex::Noc0:
          nocIdx = 0;
          return;
        case ttcore::NocIndex::Noc1:
          nocIdx = 1;
          return;
        }
      }
    });
  }

  return nocIdx;
}

static void setInsertionPointToFunctionStart(Operation *useOp,
                                             OpBuilder &builder) {
  if (auto funcOp = useOp->getParentOfType<func::FuncOp>()) {
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    return;
  }

  builder.setInsertionPointToStart(useOp->getBlock());
}

static std::string ensureFunctionScopedDeclaration(
    Operation *useOp, ConversionPatternRewriter &rewriter,
    llvm::StringRef declaration, llvm::StringRef name,
    llvm::StringRef duplicateCheckPrefix = {}) {
  llvm::StringRef prefix =
      duplicateCheckPrefix.empty() ? declaration : duplicateCheckPrefix;
  if (hasDominatingVerbatimWithPrefix(useOp, prefix)) {
    return name.str();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  setInsertionPointToFunctionStart(useOp, rewriter);
  rewriter.create<emitc::VerbatimOp>(useOp->getLoc(), declaration);
  return name.str();
}

static std::string getResultVariableName(Value result, llvm::StringRef prefix) {
  std::string ssaName;
  llvm::raw_string_ostream os(ssaName);
  mlir::OpPrintingFlags flags;
  result.printAsOperand(os, flags);
  os.flush();
  return (prefix + ssaName.substr(1)).str();
}

// Resolves the kernel `Noc` C++ object to use for a NoC op and ensures it is
// declared at the top of the enclosing kernel function.
//
// When the NoC index is statically known, emit an explicitly-indexed object
// `Noc nocN(N);`, so users can use both the `noc0` and `noc1` in the same
// kernel at their own risk.
//
// For a non-constant `nocId` (determined at runtime), splice into an inline
// temporary `Noc({})`.
//
// When the optional `noc` op operand is absent and there is no way to
// statically resolve the NoC index, fall back to the NoC the kernel ultimately
// launches on (its `noc_index` global variable) with the explicit `Noc
// noc(noc_index);` declaration.
static std::string ensureNocDeclaration(Operation *useOp,
                                        ConversionPatternRewriter &rewriter,
                                        SmallVectorImpl<Value> &operands,
                                        Value nocId = {}) {
  FailureOr<int64_t> nocIdx = getStaticNocIndex(useOp, nocId);
  if (succeeded(nocIdx)) {
    std::string nocName = "noc" + std::to_string(*nocIdx);
    std::string declaration =
        "Noc " + nocName + "(" + std::to_string(*nocIdx) + ");";
    return ensureFunctionScopedDeclaration(useOp, rewriter, declaration,
                                           nocName);
  }

  if (nocId) {
    // Explicit but non-constant nocId: splice the runtime value inline.
    operands.push_back(nocId);
    return "Noc({})";
  }

  // Unresolvable and no per-op override: construct from the `noc_index` kernel
  // global variable, defined by the Metalium DM core config and the
  // `-DNOC_INDEX` SFPI cmdline flag.
  return ensureFunctionScopedDeclaration(useOp, rewriter, "Noc noc(noc_index);",
                                         "noc",
                                         /*duplicateCheckPrefix=*/"Noc noc(");
}

static std::string
ensureEndpointDeclaration(Operation *useOp, ConversionPatternRewriter &rewriter,
                          llvm::StringRef type, llvm::StringRef name) {
  std::string decl = (type + " " + name + ";").str();
  return ensureFunctionScopedDeclaration(useOp, rewriter, decl, name);
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
    addConversion([ctx](mlir::tt::ttkernel::LocalSemaphoreType type) -> Type {
      // Convert semaphore to an address type. (i32)
      return Builder(ctx).getI32Type();
    });
    addConversion([ctx](mlir::tt::ttkernel::L1AddrType type) -> Type {
      return Builder(ctx).getI32Type();
    });
    addConversion(
        [ctx](mlir::tt::ttkernel::L1AddrPtrType type) -> emitc::PointerType {
          return emitc::PointerType::get(emitc::OpaqueType::get(
              ctx, getL1PtrOpaqueTypeName(type.getElementWidth())));
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
    addConversion(
        [ctx](IndexType type) -> Type { return emitc::SizeTType::get(ctx); });
    addConversion([ctx](Float16Type type) -> Type {
      return IntegerType::get(ctx, 16, IntegerType::Unsigned);
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
class TTKernelCastToL1PtrOpToEmitCOpRewriter
    : public OpConversionPattern<ttkernel::CastToL1PtrOp> {

public:
  TTKernelCastToL1PtrOpToEmitCOpRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ttkernel::CastToL1PtrOp>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(ttkernel::CastToL1PtrOp op,
                  ttkernel::CastToL1PtrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ptrType = mlir::cast<ttkernel::L1AddrPtrType>(op.getL1Ptr().getType());
    std::string castName =
        "reinterpret_cast<" +
        getL1PtrOpaqueTypeName(ptrType.getElementWidth()).str() + "*>";

    Type resultType = getTypeConverter()->convertType(op.getL1Ptr().getType());
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultType, castName, nullptr, ArrayAttr(), adaptor.getOperands());
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

    auto pointeeType =
        mlir::cast<emitc::PointerType>(adaptor.getL1Ptr().getType())
            .getPointee();
    auto casted = rewriter.create<emitc::CastOp>(op->getLoc(), pointeeType,
                                                 adaptor.getValue());
    rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscriptOp, casted);
    return success();
  }
};
} // namespace

namespace {
class TTKernelLoadFromL1OpToEmitCOpRewriter
    : public OpConversionPattern<ttkernel::LoadFromL1Op> {

public:
  TTKernelLoadFromL1OpToEmitCOpRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ttkernel::LoadFromL1Op>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(ttkernel::LoadFromL1Op op,
                  ttkernel::LoadFromL1Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto pointerType =
        mlir::cast<emitc::PointerType>(adaptor.getL1Ptr().getType());
    auto subscriptOp = rewriter.create<emitc::SubscriptOp>(
        op->getLoc(),
        emitc::LValueType::get(op.getContext(), pointerType.getPointee()),
        adaptor.getL1Ptr(), adaptor.getOffset());

    auto loaded = rewriter.create<emitc::LoadOp>(
        op->getLoc(), pointerType.getPointee(), subscriptOp);
    rewriter.replaceOpWithNewOp<emitc::CastOp>(
        op, getTypeConverter()->convertType(op.getValue().getType()), loaded);
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

  std::string getOpName(SourceOp op) const {
    auto name =
        opName.empty() ? op.getOperation()->getName().getStringRef() : opName;
    return getTTKernelCalleeName(name);
  }

  StringRef getReduceType(ttkernel::ReduceType reduceType) const {
    switch (reduceType) {
    case ttkernel::ReduceType::Max:
      return "PoolType::MAX";
    case ttkernel::ReduceType::Avg:
      return "PoolType::AVG";
    case ttkernel::ReduceType::Sum:
      return "PoolType::SUM";
    }
  }

  StringRef getReduceDim(ttkernel::ReduceDim reduceDim) const {
    switch (reduceDim) {
    case ttkernel::ReduceDim::Col:
      return "ReduceDim::REDUCE_COL";
    case ttkernel::ReduceDim::Row:
      return "ReduceDim::REDUCE_ROW";
    case ttkernel::ReduceDim::Scalar:
      return "ReduceDim::REDUCE_SCALAR";
    }
  }

  std::pair<StringRef, StringRef>
  reduceTypeAndDimToString(ttkernel::ReduceTypeAttr reduceTypeAttr,
                           ttkernel::ReduceDimAttr reduceDimAttr) const {
    StringRef reduceType = getReduceType(reduceTypeAttr.getValue());
    StringRef reduceDim = getReduceDim(reduceDimAttr.getValue());
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

  StringRef getInputClamping(ttkernel::InputClamping inputClamping) const {
    switch (inputClamping) {
    case ttkernel::InputClamping::None:
      return "InputClamping::None";
    case ttkernel::InputClamping::ClampToNegative:
      return "InputClamping::ClampToNegative";
    }
  }

  static bool hasNonDefaultExpTileScale(IntegerAttr scaleAttr) {
    constexpr uint32_t defaultScale = 0x3F80u; // 1.0 encoded for exp_tile().
    return scaleAttr &&
           static_cast<uint32_t>(scaleAttr.getInt()) != defaultScale;
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
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::ReduceUninitOp>) {
      // The default `reduce_uninit()` signature already resolves to <false>,
      // so emit a template arg only when full_fp32 is set.
      if (!op.getFullFp32()) {
        return ArrayAttr();
      }
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(op.getContext(), "true"));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp,
                                        ttkernel::NocSemaphoreIncOp> ||
                         std::is_same_v<SourceOp,
                                        ttkernel::NocSemaphoreIncMulticastOp>) {
      // The metal C signature defaults `posted` to false, so emit a template
      // arg only when the producer explicitly opts into posted semantics.
      auto posted = op.getPosted();
      if (!posted || !*posted) {
        return ArrayAttr();
      }
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(op.getContext(), "true"));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp,
                                        ttkernel::NocInlineDwWriteOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), "InlineWriteDst::L1"));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::SFPUReduceInitOp>) {
      // sfpu_reduce_init<PoolType, DataFormat>()
      SmallVector<Attribute, 2> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), getReduceType(op.getReduceType())));
      template_args.push_back(
          datatypeToDataformatEnumNameOpaqueAttr(builder, op.getDataFormat()));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::SFPUReduceTileOp>) {
      // sfpu_reduce<PoolType, DataFormat, ReduceDim>(dst_index)
      SmallVector<Attribute, 3> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), getReduceType(op.getReduceType())));
      template_args.push_back(
          datatypeToDataformatEnumNameOpaqueAttr(builder, op.getDataFormat()));
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), getReduceDim(op.getReduceDim())));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::CopyDestValuesOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(
          datatypeToDataformatEnumNameOpaqueAttr(builder, op.getDataFormat()));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::UnaryBcastInitOp> ||
                         std::is_same_v<SourceOp, ttkernel::UnaryBcastTileOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), getBroadcastType(op.getBcastType())));
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
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::AddIntTileOp> ||
                         std::is_same_v<SourceOp, ttkernel::SubIntTileOp> ||
                         std::is_same_v<SourceOp, ttkernel::MulIntTileInitOp> ||
                         std::is_same_v<SourceOp, ttkernel::MulIntTileOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(
          datatypeToDataformatEnumNameOpaqueAttr(builder, op.getDtype()));
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
        eltwiseType = "EltwiseBinaryType::ELWADD";
        break;
      case ttkernel::EltwiseBinaryType::Sub:
        eltwiseType = "EltwiseBinaryType::ELWSUB";
        break;
      case ttkernel::EltwiseBinaryType::Mul:
        eltwiseType = "EltwiseBinaryType::ELWMUL";
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
    } else if constexpr (
        std::is_same_v<SourceOp, ttkernel::WhereTileOp> ||
        std::is_same_v<SourceOp, ttkernel::BitwiseAndBinaryTilesOp> ||
        std::is_same_v<SourceOp, ttkernel::BitwiseOrBinaryTilesOp> ||
        std::is_same_v<SourceOp, ttkernel::BitwiseXorBinaryTilesOp> ||
        std::is_same_v<SourceOp, ttkernel::BinaryLeftShiftTileOp> ||
        std::is_same_v<SourceOp, ttkernel::BinaryRightShiftTileOp> ||
        std::is_same_v<SourceOp, ttkernel::BinaryLogicalRightShiftTileOp> ||
        std::is_same_v<SourceOp, ttkernel::LogicalNotTileOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(
          datatypeToDataformatEnumNameOpaqueAttr(builder, op.getDtype()));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (
        std::is_same_v<SourceOp, ttkernel::ExperimentalWriteRowMaskTileOp> ||
        std::is_same_v<SourceOp, ttkernel::ExperimentalWriteColMaskTileOp> ||
        std::is_same_v<SourceOp, ttkernel::ExperimentalFillArangeTileOp>) {
      auto cbType = mlir::cast<ttkernel::CBType>(op.getCb().getType());
      auto tileType = mlir::cast<ttcore::TileType>(cbType.getElementType());
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(datatypeToDataformatEnumNameOpaqueAttr(
          builder, tileType.getDataType()));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp,
                                        ttkernel::PackUntilizeInitOp> ||
                         std::is_same_v<
                             SourceOp,
                             ttkernel::ExperimentalPackUntilizeBlockOp>) {
      SmallVector<Attribute, 2> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), std::to_string(op.getColsPerDstPass())));
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(), std::to_string(op.getTotalColTiles())));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::FillTileIntOp>) {
      SmallVector<Attribute, 1> template_args;
      template_args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), "DataFormat::Int32"));
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::ExpTileInitOp>) {
      // exp_tile_init<bool approx, uint32_t scale, InputClamping
      // input_clamping>() Emit template args only up to the last explicitly-set
      // parameter, filling in metal defaults for any preceding unset parameter.
      // When nothing is set the op lowers to a bare `exp_tile_init()`.
      auto approxAttr = op.getApproxAttr();
      auto scaleAttr = op.getScaleAttr();
      auto clampAttr = op.getInputClampingAttr();
      int lastSet = -1;
      if (approxAttr) {
        lastSet = 0;
      }
      if (scaleAttr) {
        lastSet = 1;
      }
      if (clampAttr) {
        lastSet = 2;
      }
      if (lastSet < 0) {
        return ArrayAttr();
      }
      SmallVector<Attribute, 3> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(),
          (approxAttr && approxAttr.getValue()) ? "true" : "false"));
      if (lastSet >= 1) {
        uint32_t scale =
            scaleAttr ? static_cast<uint32_t>(scaleAttr.getInt()) : 0x3F800000u;
        template_args.push_back(
            emitc::OpaqueAttr::get(op.getContext(), std::to_string(scale)));
      }
      if (lastSet >= 2) {
        ttkernel::InputClamping inputClamping =
            clampAttr ? clampAttr.getValue()
                      : ttkernel::InputClamping::ClampToNegative;
        template_args.push_back(emitc::OpaqueAttr::get(
            op.getContext(), getInputClamping(inputClamping)));
      }
      return ArrayAttr::get(op.getContext(), template_args);
    } else if constexpr (std::is_same_v<SourceOp, ttkernel::ExpTileOp>) {
      // exp_tile<bool approx, bool scale_en, InputClamping input_clamping,
      //          int iterations>(idst, vector_mode, scale)
      // Emit template args only up to the last explicitly-set parameter,
      // filling in metal defaults for any preceding unset parameter. When
      // nothing is set the op lowers to a bare `exp_tile(idst)`. The runtime
      // `scale` argument is handled separately by getCallArgs().
      auto approxAttr = op.getApproxAttr();
      auto scaleAttr = op.getScaleAttr();
      bool scaleEn = hasNonDefaultExpTileScale(scaleAttr);
      auto clampAttr = op.getInputClampingAttr();
      auto iterationsAttr = op.getIterationsAttr();
      int lastSet = -1;
      if (approxAttr) {
        lastSet = 0;
      }
      if (scaleEn) {
        lastSet = 1;
      }
      if (clampAttr) {
        lastSet = 2;
      }
      if (iterationsAttr) {
        lastSet = 3;
      }
      if (lastSet < 0) {
        return ArrayAttr();
      }
      SmallVector<Attribute, 4> template_args;
      template_args.push_back(emitc::OpaqueAttr::get(
          op.getContext(),
          (approxAttr && approxAttr.getValue()) ? "true" : "false"));
      if (lastSet >= 1) {
        template_args.push_back(emitc::OpaqueAttr::get(
            op.getContext(), scaleEn ? "true" : "false"));
      }
      if (lastSet >= 2) {
        ttkernel::InputClamping inputClamping =
            clampAttr ? clampAttr.getValue()
                      : ttkernel::InputClamping::ClampToNegative;
        template_args.push_back(emitc::OpaqueAttr::get(
            op.getContext(), getInputClamping(inputClamping)));
      }
      if (lastSet >= 3) {
        int64_t iterations = iterationsAttr ? iterationsAttr.getInt() : 8;
        template_args.push_back(emitc::OpaqueAttr::get(
            op.getContext(), std::to_string(iterations)));
      }
      return ArrayAttr::get(op.getContext(), template_args);
    }
    return ArrayAttr();
  }

  // Build the positional call `args` interleaving. Returns a null ArrayAttr to
  // pass all operands in their natural order (the common case). Op-specific
  // overrides can insert literal runtime arguments alongside operand
  // references (operand indices are encoded as IndexType IntegerAttrs).
  ArrayAttr getCallArgs(Builder &builder, SourceOp op) const {
    if constexpr (std::is_same_v<SourceOp, ttkernel::ExpTileOp>) {
      // exp_tile(idst, vector_mode, scale): only materialize the runtime
      // vector_mode/scale arguments when an explicit scale is provided.
      // Otherwise fall back to the bare exp_tile(idst) call.
      auto scaleAttr = op.getScaleAttr();
      if (!hasNonDefaultExpTileScale(scaleAttr)) {
        return ArrayAttr();
      }
      SmallVector<Attribute, 3> args;
      args.push_back(builder.getIndexAttr(0)); // idst (operand 0)
      args.push_back(
          emitc::OpaqueAttr::get(op.getContext(), "(int)VectorMode::RC"));
      args.push_back(emitc::OpaqueAttr::get(
          op.getContext(),
          std::to_string(static_cast<uint32_t>(scaleAttr.getInt()))));
      return ArrayAttr::get(op.getContext(), args);
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
        op, resultTypes, getOpName(op), getCallArgs(rewriter, op),
        getTemplateArgs(rewriter, op), adaptor.getOperands());

    return success();
  }

private:
  std::string opName;
};
} // namespace

namespace {
class TTKernelBitcastOpRewriter
    : public OpConversionPattern<ttkernel::BitcastOp> {
public:
  using OpConversionPattern<ttkernel::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = op.getResult().getType();
    Value input = adaptor.getInput();

    // Integer and index types: static_cast handles reinterpretation correctly.
    // IndexType lowers to emitc::SizeTType (size_t) via the type converter.
    if (mlir::isa<IntegerType, IndexType>(resultType)) {
      Type convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported integer/index result type");
      }
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, convertedType, input);
      return success();
    }

    // Float types: bit-cast via __builtin_memcpy (well-defined, avoids UB).
    // Derive a unique variable name from the SSA result number.
    std::string ssaName;
    {
      llvm::raw_string_ostream os(ssaName);
      mlir::OpPrintingFlags flags;
      op.getResult().printAsOperand(os, flags);
      os.flush();
    }
    std::string varName = "_rc" + ssaName.substr(1);

    // Emits: uint32_t <var>_src = <srcInit>; float <var>;
    //        __builtin_memcpy(&<var>, &<var>_src, sizeof(<var>));
    // then replaces op with a LiteralOp referencing <var>.
    auto emitMemcpyBitcast = [&](std::string srcInit) -> LogicalResult {
      std::string code = "uint32_t " + varName + "_src = " + srcInit +
                         "; float " + varName + "; __builtin_memcpy(&" +
                         varName + ", &" + varName + "_src, sizeof(" + varName +
                         "));";
      rewriter.create<emitc::VerbatimOp>(
          op.getLoc(), rewriter.getStringAttr(code), ValueRange{input});
      rewriter.replaceOp(
          op, rewriter
                  .create<emitc::LiteralOp>(
                      op.getLoc(), Float32Type::get(op.getContext()), varName)
                  .getResult());
      return success();
    };

    if (mlir::isa<Float32Type>(resultType)) {
      // uint32 -> float: store input in an addressable temporary, then memcpy.
      // "{{" escapes to "{" in EmitC verbatim; "}" needs no escaping.
      return emitMemcpyBitcast("{}");
    }

    if (mlir::isa<BFloat16Type>(resultType)) {
      // BFloat16 arg is packed in the lower 16 bits of the uint32.
      // BFloat16 == upper 16 bits of float32, so shift left by 16 to get the
      // correct float32 bit pattern. The type converter maps bf16 -> float.
      return emitMemcpyBitcast("static_cast<uint32_t>({}) << 16");
    }

    if (mlir::isa<Float16Type>(resultType)) {
      // Float16 arg is packed in the lower 16 bits of the uint32.
      // Truncate to uint16_t to preserve the bit pattern.
      // Type converter maps Float16Type -> ui16 consistently.
      Type convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(op, "unsupported f16 result type");
      }
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, convertedType, input);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported float type");
  }
};
} // namespace

namespace {
template <typename SourceOp>
class TTKernelToEmitCArgValRewriter : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  std::string getOpName(SourceOp op) const {
    auto name = op.getOperation()->getName().getStringRef();
    return getTTKernelCalleeName(name);
  }

  ArrayAttr getTemplateArgs(Builder &builder, SourceOp op) const {
    if constexpr (std::is_same_v<SourceOp, ttkernel::GetArgValOp> ||
                  std::is_same_v<SourceOp, ttkernel::GetCommonArgValOp>) {
      SmallVector<Attribute, 1> templateArgs;
      templateArgs.push_back(
          emitc::OpaqueAttr::get(builder.getContext(), "uint32_t"));
      return ArrayAttr::get(op.getContext(), templateArgs);
    }
    return ArrayAttr();
  }

  Value createRawValue(SourceOp op, typename SourceOp::Adaptor adaptor,
                       Type resultType,
                       ConversionPatternRewriter &rewriter) const {
    if constexpr (std::is_same_v<SourceOp, ttkernel::GetCompileArgValOp>) {
      auto literal = rewriter.create<emitc::LiteralOp>(
          op.getLoc(), resultType,
          (Twine("get_compile_time_arg_val(") + Twine(op.getArgIndex()) + ")")
              .str());
      if (mlir::isa<ttkernel::CBType>(op.getResult().getType())) {
        literal->setAttr("ttkernel.cb_ctarg_idx",
                         rewriter.getI32IntegerAttr(op.getArgIndex()));
      }
      return literal.getResult();
    }

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), resultType, getOpName(op), nullptr,
        getTemplateArgs(rewriter, op), adaptor.getOperands());

    // Relay the preassigned CB runtime arg index.
    if (mlir::isa<ttkernel::CBType>(op.getResult().getType())) {
      auto idxAttr =
          op->template getAttrOfType<IntegerAttr>("ttkernel.cb_arg_idx");
      assert(idxAttr && "CB runtime arg must have a stable lowering name");
      call->setAttr("ttkernel.cb_arg_idx", idxAttr);
    }

    return call.getResult(0);
  }

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }

    Value rawValue = createRawValue(op, adaptor, resultType, rewriter);
    rewriter.replaceOp(op, rawValue);
    return success();
  }
};
} // namespace

namespace {
template <typename SourceOp>
class TTKernelToEmitCCBVoidMethodRewriter
    : public OpConversionPattern<SourceOp> {
public:
  TTKernelToEmitCCBVoidMethodRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx,
      std::string methodName)
      : OpConversionPattern<SourceOp>(typeConverter, ctx),
        methodName(std::move(methodName)) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    TT_assert(operands.size() == 2u);

    std::string callStr =
        ensureCBDeclaration(operands.front(), op.getOperation(), rewriter) +
        "." + methodName + "({});";

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), callStr,
                                       operands.drop_front());
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::string methodName;
};
} // namespace

namespace {
template <typename SourceOp>
class TTKernelToEmitCCBResultMethodRewriter
    : public OpConversionPattern<SourceOp> {
public:
  TTKernelToEmitCCBResultMethodRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx,
      std::string methodName)
      : OpConversionPattern<SourceOp>(typeConverter, ctx),
        methodName(std::move(methodName)) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }

    std::string cbName =
        ensureCBDeclaration(adaptor.getCb(), op.getOperation(), rewriter);
    rewriter.replaceOpWithNewOp<emitc::LiteralOp>(
        op, resultType, cbName + "." + methodName + "()");
    return success();
  }

private:
  std::string methodName;
};
} // namespace

namespace {
class TTKernelToEmitCGetNocAddrRewriter
    : public OpConversionPattern<ttkernel::GetNocAddrOp> {
public:
  using OpConversionPattern<ttkernel::GetNocAddrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::GetNocAddrOp op,
                  ttkernel::GetNocAddrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    TT_assert(resultType);

    SmallVector<Value, 1> nocOperands;
    std::string nocName = ensureNocDeclaration(op.getOperation(), rewriter,
                                               nocOperands, adaptor.getNoc());
    std::string endpoint = ensureEndpointDeclaration(
        op.getOperation(), rewriter, "UnicastEndpoint", "unicast_ep");
    SmallVector<Value, 4> operands = {adaptor.getX(), adaptor.getY(),
                                      adaptor.getL1Address()};
    operands.append(nocOperands);

    std::string varName = getResultVariableName(op->getResult(0), "noc_addr_");
    std::string callStr =
        "uint64_t " + varName + " = " + endpoint +
        ".get_noc_unicast_addr(static_cast<uint32_t>({}), "
        "static_cast<uint32_t>({}), static_cast<uint32_t>({}), " +
        nocName + ".get_noc_id());";

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), callStr, operands);
    rewriter.replaceOp(
        op, rewriter.create<emitc::LiteralOp>(op.getLoc(), resultType, varName)
                .getResult());
    return success();
  }
};

class TTKernelToEmitCNocAtomicBarrierRewriter
    : public OpConversionPattern<ttkernel::NocAsyncAtomicBarrierOp> {
public:
  using OpConversionPattern<
      ttkernel::NocAsyncAtomicBarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::NocAsyncAtomicBarrierOp op,
                  ttkernel::NocAsyncAtomicBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 1> operands;
    std::string nocName = ensureNocDeclaration(op.getOperation(), rewriter,
                                               operands, adaptor.getNoc());
    std::string callStr = nocName + ".async_atomic_barrier();";

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), callStr, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename SourceOp>
class TTKernelToEmitCNocFullBarrierRewriter
    : public OpConversionPattern<SourceOp> {
public:
  TTKernelToEmitCNocFullBarrierRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx,
      std::string methodName)
      : OpConversionPattern<SourceOp>(typeConverter, ctx),
        methodName(std::move(methodName)) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 1> operands;
    std::string nocName = ensureNocDeclaration(op.getOperation(), rewriter,
                                               operands, adaptor.getNoc());
    std::string callStr = nocName + "." + methodName + "();";

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), callStr, operands);
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::string methodName;
};

template <typename SourceOp>
class TTKernelToEmitCNocTridBarrierRewriter
    : public OpConversionPattern<SourceOp> {
public:
  TTKernelToEmitCNocTridBarrierRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx,
      std::string methodName)
      : OpConversionPattern<SourceOp>(typeConverter, ctx),
        methodName(std::move(methodName)) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 2> operands;
    std::string nocName = ensureNocDeclaration(op.getOperation(), rewriter,
                                               operands, adaptor.getNoc());
    operands.push_back(adaptor.getTrid());
    std::string callStr = nocName + "." + methodName +
                          "<NocOptions::TXN_ID>(NocOptVals{{.trid = {}}});";

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), callStr, operands);
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::string methodName;
};

template <typename SourceOp>
class TTKernelToEmitCNocAsyncTransferRewriter
    : public OpConversionPattern<SourceOp> {
  static constexpr bool isRead =
      std::is_same_v<SourceOp, ttkernel::NocAsyncReadOp>;

public:
  TTKernelToEmitCNocAsyncTransferRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<SourceOp>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value localL1Addr = nullptr;
    ValueRange coreXY;
    ValueRange bankId;
    Value remoteAddr = nullptr;
    if constexpr (isRead) {
      localL1Addr = adaptor.getDstLocalL1Addr();
      coreXY = adaptor.getSrcCoreXY();
      bankId = adaptor.getSrcBankId();
      remoteAddr = adaptor.getSrcAddress();
    } else {
      localL1Addr = adaptor.getSrcLocalL1Addr();
      coreXY = adaptor.getDstCoreXY();
      bankId = adaptor.getDstBankId();
      remoteAddr = adaptor.getDstAddress();
    }

    SmallVector<Value, 1> nocOperands;
    std::string nocName = ensureNocDeclaration(op.getOperation(), rewriter,
                                               nocOperands, adaptor.getNoc());
    // A dynamic (explicit & non-constant) NoC object is spliced into callStr
    // and would break operand ordering for async transfers.
    if (!nocOperands.empty()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic NoC ID is not supported for async read/write");
    }
    SmallVector<Value, 5> operands{localL1Addr, adaptor.getSize()};
    std::string callStr;

    if (!coreXY.empty()) {
      TT_assert(coreXY.size() == 2u);
      std::string endpoint = ensureEndpointDeclaration(
          op.getOperation(), rewriter, "UnicastEndpoint", "unicast_ep");
      operands.append(coreXY.begin(), coreXY.end());
      operands.push_back(remoteAddr);
      if constexpr (isRead) {
        callStr = nocName + ".async_read(" + endpoint +
                  ", CoreLocalMem<uint32_t>({}), {}, "
                  "{{.noc_x = {}, .noc_y = {}, "
                  ".addr = static_cast<uint32_t>({})}, {{});";
      } else {
        callStr = nocName +
                  ".async_write("
                  "CoreLocalMem<uint32_t>({}), " +
                  endpoint +
                  ", {}, {{} , {{.noc_x = {}, .noc_y = {}, "
                  ".addr = static_cast<uint32_t>({})});";
      }
    } else {
      TT_assert(bankId.size() == 1u);
      std::string endpoint = ensureEndpointDeclaration(
          op.getOperation(), rewriter, "AllocatorBank<AllocatorBankType::DRAM>",
          "dram_ep");
      operands.push_back(bankId.front());
      operands.push_back(remoteAddr);
      if constexpr (isRead) {
        callStr = nocName + ".async_read(" + endpoint +
                  ", CoreLocalMem<uint32_t>({}), {}, "
                  "{{.bank_id = static_cast<uint32_t>({}), "
                  ".addr = static_cast<uint32_t>({})}, {{});";
      } else {
        callStr = nocName +
                  ".async_write("
                  "CoreLocalMem<uint32_t>({}), " +
                  endpoint +
                  ", {}, {{} , {{.bank_id = static_cast<uint32_t>({}), "
                  ".addr = static_cast<uint32_t>({})});";
      }
    }

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), callStr, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename SourceOp>
class TTKernelToEmitCNocAsyncWriteMulticastRewriter
    : public OpConversionPattern<SourceOp> {
public:
  TTKernelToEmitCNocAsyncWriteMulticastRewriter(
      TTKernelToEmitCTypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<SourceOp>(typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 9> operands;
    std::string nocName = ensureNocDeclaration(op.getOperation(), rewriter,
                                               operands, adaptor.getNoc());
    std::string endpoint = ensureEndpointDeclaration(
        op.getOperation(), rewriter, "MulticastEndpoint", "mcast_ep");

    // EXCLUDE_SRC maps to default NocOptions (no MCAST_INCL_SRC flag), so we
    // omit the template argument entirely. INCLUDE_SRC maps to
    // NocOptions::MCAST_INCL_SRC.
    std::string templateArg =
        std::is_same_v<SourceOp, ttkernel::NocAsyncWriteMulticastOp>
            ? ""
            : "<NocOptions::MCAST_INCL_SRC>";
    bool linked = op.getLinked().value_or(false);

    operands.append({adaptor.getSrcLocalL1Addr(), adaptor.getSize(),
                     adaptor.getNumDests(), adaptor.getDstNocXStart(),
                     adaptor.getDstNocYStart(), adaptor.getDstNocXEnd(),
                     adaptor.getDstNocYEnd(), adaptor.getDstLocalL1Addr()});
    std::string dstArgs =
        "noc_traits_t<MulticastEndpoint>::"
        "dst_args_mcast_type{{.noc_x_start = {}, .noc_y_start = {}, "
        ".noc_x_end = {}, .noc_y_end = {}, "
        ".addr = static_cast<uint32_t>({})}";

    std::string callStr = nocName + ".async_write_multicast" + templateArg +
                          "(CoreLocalMem<uint32_t>({}), " + endpoint +
                          ", {}, {}, {{} , " + dstArgs + ", " +
                          (linked ? "true" : "false") + ");";

    rewriter.create<emitc::VerbatimOp>(op.getLoc(), callStr, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {
class TTKernelToEmitCGetMyLogicalMeshPositionOpRewriter
    : public OpConversionPattern<ttkernel::GetMyLogicalMeshPositionOp> {
public:
  using OpConversionPattern<
      ttkernel::GetMyLogicalMeshPositionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::GetMyLogicalMeshPositionOp op,
                  ttkernel::GetMyLogicalMeshPositionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value> operands;
    operands.push_back(adaptor.getFcm());
    operands.push_back(rewriter
                           .create<emitc::LiteralOp>(
                               op.getLoc(),
                               rewriter.getType<emitc::OpaqueType>("uint64_t"),
                               std::to_string(op.getDim()))
                           .getResult());

    std::string opName =
        getTTKernelCalleeName(op.getOperation()->getName().getStringRef());
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()), opName,
        nullptr, nullptr, operands);
    return success();
  }
};
} // namespace

namespace {
class TTKernelToEmitCGetDeviceIdFromLogicalMeshPositionOpRewriter
    : public OpConversionPattern<
          ttkernel::GetDeviceIdFromLogicalMeshPositionOp> {
public:
  using OpConversionPattern<
      ttkernel::GetDeviceIdFromLogicalMeshPositionOp>::OpConversionPattern;

  // Creates a named value for an opaque initializer list.
  Value
  callOpaqueInitializerList(ConversionPatternRewriter &rewriter,
                            ttkernel::GetDeviceIdFromLogicalMeshPositionOp op,
                            emitc::OpaqueType type, std::string callee,
                            SmallVector<Value> initializerList) const {
    std::string varName =
        getResultVariableName(op.getResult(), "logical_mesh_position_");
    std::string initStr = callee + " " + varName + " = " + callee;
    initStr += "{{";
    for (size_t i = 0; i < initializerList.size(); ++i) {
      if (i > 0) {
        initStr += ", ";
      }
      initStr += "{}";
    }
    initStr += "};";
    rewriter.create<emitc::VerbatimOp>(op.getLoc(), initStr, initializerList);

    return rewriter.create<emitc::LiteralOp>(op.getLoc(), type, varName)
        .getResult();
  }

  LogicalResult
  matchAndRewrite(ttkernel::GetDeviceIdFromLogicalMeshPositionOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // Call std::array constructor to create an array out of the indices
    auto arrTypeStr = "std::array<uint32_t, " +
                      std::to_string(adaptor.getPositionIndices().size()) + ">";
    auto arrType = emitc::OpaqueType::get(op.getContext(), arrTypeStr);
    Value meshPositionArray = callOpaqueInitializerList(
        rewriter, op, arrType, arrTypeStr, adaptor.getPositionIndices());

    // Call get_device_id_from_logical_mesh_position
    std::string opName =
        getTTKernelCalleeName(op.getOperation()->getName().getStringRef());
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()), opName,
        nullptr, nullptr, ValueRange{adaptor.getFcm(), meshPositionArray});
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
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    mlir::Type opaqueStructType =
        this->getTypeConverter()->convertType(op->getResultTypes()[0]);
    if (!opaqueStructType) {
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }

    std::string varName =
        getResultVariableName(op.getResult(), "interleaved_addr_gen_");
    rewriter.create<emitc::VerbatimOp>(
        op.getLoc(), "InterleavedAddrGenFast<true> " + varName + ";");
    rewriter.create<emitc::VerbatimOp>(
        op.getLoc(), varName + ".bank_base_address = {};",
        ValueRange{adaptor.getBankBaseAddress()});
    rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                       varName + ".page_size = {};",
                                       ValueRange{adaptor.getPageSize()});
    rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                       varName + ".data_format = {};",
                                       ValueRange{adaptor.getDataFormat()});

    rewriter.replaceOp(op, rewriter
                               .create<emitc::LiteralOp>(
                                   op.getLoc(), opaqueStructType, varName)
                               .getResult());
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
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    std::string varName =
        getResultVariableName(op->getResult(0), "tensor_accessor_args_");

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
      return success();
    }

    mlir::Type opaqueStructType =
        this->getTypeConverter()->convertType(op->getResultTypes()[0]);
    auto opaqueType =
        mlir::dyn_cast_if_present<emitc::OpaqueType>(opaqueStructType);
    if (!opaqueType) {
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }

    std::string varName =
        getResultVariableName(op.getResult(), "fabric_connection_manager_");
    rewriter.create<emitc::VerbatimOp>(
        op.getLoc(), (opaqueType.getValue() + " " + varName + ";").str());
    rewriter.replaceOp(op, rewriter
                               .create<emitc::LiteralOp>(
                                   op.getLoc(), opaqueStructType, varName)
                               .getResult());
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

    // Calling class/struct member function is difficult to do in EmitC.
    std::string varName = getResultVariableName(op->getResult(0), "temp_");

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

// Boolean arith.andi / arith.ori represent logical operations. Lowering them to
// bitwise EmitC ops emits C++ expressions like `(a != b) | (c != d)`, which GCC
// warns about as an ambiguous mix of comparisons and bitwise operators.
template <typename SourceOp, typename EmitCOp>
class ArithBoolBinaryRewriter : public OpConversionPattern<SourceOp> {
public:
  ArithBoolBinaryRewriter(const TypeConverter &typeConverter,
                          MLIRContext *context,
                          PatternBenefit benefit = PatternBenefit(2))
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto resultType = dyn_cast<IntegerType>(op.getResult().getType());
    if (!resultType || resultType.getWidth() != 1) {
      return failure();
    }

    Type convertedType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType) {
      return failure();
    }

    ValueRange operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<EmitCOp>(op, convertedType, operands[0],
                                         operands[1]);
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
//
// The TTKernelToCpp pass then checks the VerbatimOps for inserting kernel API
// headers, on top of the CallOpaqueOps.
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class TTKernelScalarUnaryTileOpRewriter : public OpConversionPattern<SourceOp> {
public:
  TTKernelScalarUnaryTileOpRewriter(TTKernelToEmitCTypeConverter &typeConverter,
                                    MLIRContext *ctx)
      : OpConversionPattern<SourceOp>(typeConverter, ctx) {}

  std::string getOpName(SourceOp op) const {
    auto name = op.getOperation()->getName().getStringRef();
    return getTTKernelCalleeName(name);
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
        "{{ volatile int32_t __s = {}; " + getOpName(op) + "({}, __s); }";
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

class PackReconfigDataFormatOpConversion
    : public OpConversionPattern<ttkernel::PackReconfigDataFormatOp> {
public:
  using OpConversionPattern<
      ttkernel::PackReconfigDataFormatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttkernel::PackReconfigDataFormatOp op,
                  ttkernel::PackReconfigDataFormatOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.create<emitc::CallOpaqueOp>(op->getLoc(), TypeRange{},
                                         "pack_reconfig_data_format",
                                         ValueRange{adaptor.getOutCb()});
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

// Arith MaxSIOp / MinSIOp don't have an emitc lowering. Lower to
// std::max<int32_t> / std::min<int32_t>, mirroring the unsigned versions.
class ArithMaxSIRewriter : public OpConversionPattern<arith::MaxSIOp> {
public:
  using OpConversionPattern<arith::MaxSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MaxSIOp op, arith::MaxSIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultType, "std::max<int32_t>", adaptor.getOperands());
    return success();
  }
};

class ArithMinSIRewriter : public OpConversionPattern<arith::MinSIOp> {
public:
  using OpConversionPattern<arith::MinSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MinSIOp op, arith::MinSIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultType, "std::min<int32_t>", adaptor.getOperands());
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

    assignRuntimeCBArgIndices(funcOp);

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
        TTKernelBitcastOpRewriter,
        TTKernelToEmitCArgValRewriter<ttkernel::GetCompileArgValOp>,
        TTKernelToEmitCArgValRewriter<ttkernel::GetArgValOp>,
        TTKernelToEmitCArgValRewriter<ttkernel::GetCommonArgValOp>,
        TTKernelToEmitCDPrintRewriter,
        TTKernelToEmitCGetDeviceIdFromLogicalMeshPositionOpRewriter,
        TTKernelToEmitCGetMyLogicalMeshPositionOpRewriter,
        TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosBaseOp>,
        TTKernelMacroOpToEmitCOpRewriter<ttkernel::MemZerosSizeOp>,
        TTKernelCastToL1PtrOpToEmitCOpRewriter,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetSemaphoreOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SemaphoreWaitMinOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreIncOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SemaphoreWaitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreSetMulticastOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocSemaphoreSetMulticastLoopbackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocSemaphoreIncMulticastOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnpackStallOnPackOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsAcquireOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsCommitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsWaitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TileRegsReleaseOp>,

        // Numeric
        TTKernelToEmitCOpaqueRewriter<ttkernel::Bfloat16GreaterOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Float32GreaterOp>,

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
        TTKernelToEmitCOpaqueRewriter<ttkernel::PackUntilizeInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PackUntilizeUninitOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::ExperimentalPackUntilizeBlockOp>,

        // Datamovement
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CopyBlockMatmulPartialsOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PackTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PackTileBlockOp>,
        TTKernelToEmitCPackReconfigL1AccToEmitCRewriter,
        PackReconfigDataFormatOpConversion,

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
        TTKernelToEmitCOpaqueRewriter<ttkernel::MatmulBlockOp>,
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
        TTKernelToEmitCOpaqueRewriter<ttkernel::AcosTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AcosTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AsinTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AsinTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AtanTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AtanTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Atan2BinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Atan2BinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryBitwiseTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryLeftShiftTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryLogicalRightShiftTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryRightShiftTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryShiftTileInitOp>,
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
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddIntTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::AddIntTileOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::AddUnaryTileOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::AddUnaryTileInt32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::DivBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::DivBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::EqBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::EqBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NeBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NeBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GtBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GtBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LtBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LtBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GeBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GeBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LeBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LeBinaryTilesOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::DivUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfcTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ErfcTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExpTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Exp2TileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Exp2TileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Expm1TileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Expm1TileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FloorTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FillTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FillTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FillTileIntOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FracTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GeluTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GeluTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::HardsigmoidTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::HardsigmoidTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Log1pTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::Log1pTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogicalNotTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::LogicalNotTileOp>,
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
        TTKernelToEmitCOpaqueRewriter<ttkernel::LezTileI32Op>>(
        typeConverter, funcOp.getContext());

    // The remaining patterns are added in a separate `patterns.add<>` call to
    // avoid exceeding clang's default fold-expression nesting depth (256
    // template args) in `RewritePatternSet::add`.
    patterns.add<
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulIntTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::MulIntTileOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::MulUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubIntTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SubIntTileOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::SubUnaryTileOp>,
        TTKernelScalarUnaryTileOpRewriter<ttkernel::SubUnaryTileInt32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMaxTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMaxTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMaxInt32TileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMaxInt32TileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMinTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMinTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMinInt32TileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::BinaryMinInt32TileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NegativeTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NegativeTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NegativeTileInt32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowBinaryTilesInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowBinaryTilesOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowerTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::PowUnaryTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RecipTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReduceUninitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SFPUReduceInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SFPUReduceTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReluTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReluTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ReluTileI32Op>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RoundingTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SeluTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SeluTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RandTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RandTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RsqrtTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::RsqrtTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SqrtTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SqrtTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SignbitTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SignbitTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SigmoidTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SigmoidTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SiluTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SiluTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SinTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SinTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SoftsignTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SoftsignTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SquareTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SquareTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanhTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TanhTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TruncTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TypecastTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TypecastTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalWriteRowMaskTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalWriteColMaskTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ExperimentalFillArangeTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryBcastInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::UnaryBcastTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::WhereTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::WhereTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ClampScalarTileInitOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ClampScalarTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ClampScalarTileInt32Op>,

        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadTileOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketSetStateOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketWithStateOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncReadOnePacketWithStateWithTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncReadSetTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteTileOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::NocAsyncWriteSetTridOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteOnePacketWithTridOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ResetNocTridBarrierCounterOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocMulticastAddrOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::NocAsyncWriteMulticastOnePacketOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ConvertLogicalXToTranslatedOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::ConvertLogicalYToTranslatedOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetMyDeviceIdOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FabricWriteOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FabricMulticastWriteOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FabricSemIncOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::FabricMulticastSemIncOp>,
        TTKernelToEmitCOpaqueRewriter<
            ttkernel::CreateFabricConnectionManagerOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::SetupFabricConnectionsOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::CloseFabricConnectionsOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetTileSizeOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetNocAddrFromBankIDOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::GetDataFormatOp>,
        TTKernelToEmitCOpaqueRewriter<ttkernel::TensorAccessorOp>>(
        typeConverter, funcOp.getContext());

    patterns.add<TTKernelToEmitCCBVoidMethodRewriter<ttkernel::CBPushBackOp>>(
        typeConverter, funcOp.getContext(), "push_back");
    patterns.add<TTKernelToEmitCCBVoidMethodRewriter<ttkernel::CBPopFrontOp>>(
        typeConverter, funcOp.getContext(), "pop_front");
    patterns
        .add<TTKernelToEmitCCBVoidMethodRewriter<ttkernel::CBReserveBackOp>>(
            typeConverter, funcOp.getContext(), "reserve_back");
    patterns.add<TTKernelToEmitCCBVoidMethodRewriter<ttkernel::CBWaitFrontOp>>(
        typeConverter, funcOp.getContext(), "wait_front");
    patterns
        .add<TTKernelToEmitCCBResultMethodRewriter<ttkernel::GetWritePtrOp>>(
            typeConverter, funcOp.getContext(), "get_write_ptr");
    patterns.add<TTKernelToEmitCCBResultMethodRewriter<ttkernel::GetReadPtrOp>>(
        typeConverter, funcOp.getContext(), "get_read_ptr");

    patterns.add<TTKernelToEmitCOpaqueRewriter<ttkernel::RemoteSramWriteU32Op>>(
        typeConverter, funcOp.getContext(), "noc_semaphore_set_remote");
    patterns.add<TTKernelToEmitCOpaqueRewriter<ttkernel::NocInlineDwWriteOp>>(
        typeConverter, funcOp.getContext());

    patterns
        .add<TTKernelToEmitCGetNocAddrRewriter,
             TTKernelToEmitCNocAtomicBarrierRewriter,
             TTKernelToEmitCNocAsyncTransferRewriter<ttkernel::NocAsyncReadOp>,
             TTKernelToEmitCNocAsyncTransferRewriter<ttkernel::NocAsyncWriteOp>,
             TTKernelToEmitCNocAsyncWriteMulticastRewriter<
                 ttkernel::NocAsyncWriteMulticastOp>,
             TTKernelToEmitCNocAsyncWriteMulticastRewriter<
                 ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>>(
            typeConverter, funcOp.getContext());
    patterns.add<
        TTKernelToEmitCNocFullBarrierRewriter<ttkernel::NocAsyncReadBarrierOp>>(
        typeConverter, funcOp.getContext(), "async_read_barrier");
    patterns.add<TTKernelToEmitCNocFullBarrierRewriter<
        ttkernel::NocAsyncWriteBarrierOp>>(typeConverter, funcOp.getContext(),
                                           "async_write_barrier");
    patterns.add<TTKernelToEmitCNocTridBarrierRewriter<
        ttkernel::NocAsyncReadBarrierWithTridOp>>(
        typeConverter, funcOp.getContext(), "async_read_barrier");
    patterns.add<TTKernelToEmitCNocTridBarrierRewriter<
        ttkernel::NocAsyncWriteBarrierWithTridOp>>(
        typeConverter, funcOp.getContext(), "async_write_barrier");

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
    patterns.add<TTKernelLoadFromL1OpToEmitCOpRewriter>(typeConverter,
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

    patterns
        .add<ArithFloorDivRewriter, ArithBitcastRewriter, ArithMaxUIRewriter,
             ArithMinUIRewriter, ArithMaxSIRewriter, ArithMinSIRewriter,
             ArithBoolBinaryRewriter<arith::AndIOp, emitc::LogicalAndOp>,
             ArithBoolBinaryRewriter<arith::OrIOp, emitc::LogicalOrOp>>(
            typeConverter, funcOp.getContext());

    return applyFullConversion(funcOp, target, std::move(patterns));
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<::mlir::Pass> createConvertTTKernelToEmitC() {
  return std::make_unique<ConvertTTKernelToEmitCPass>();
}

} // namespace mlir::tt
