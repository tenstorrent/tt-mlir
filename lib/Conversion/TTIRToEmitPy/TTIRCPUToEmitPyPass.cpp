// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ConvertTTIRCPUToEmitPy pass
// ===========================
//
// Lowers TTIR ops inside CPU-hoisted functions to EmitPy CallOpaqueOp calls
// targeting `ttir_cpu.<op>` — pure-torch implementations that run on the host.
//
// The generated code operates at two type levels:
//
//   Function boundary (signature, args, returns): ttnn.Tensor
//     Callers live in the TTNN world and pass/receive ttnn tensors.
//
//   Function body (internal ops): torch.Tensor
//     The ttir_cpu.* implementations are pure torch, so ops use torch tensors.
//
// The bridge is explicit:
//   - FuncOpBoundaryPattern  rewrites signatures to ttnn.Tensor and inserts
//     ttnn.to_torch at function entry.
//   - ReturnOpBoundaryPattern inserts ttnn.from_torch before each return.
//   - All other patterns use the type converter (MLIR tensor -> torch.Tensor)
//     to produce internal CallOpaqueOp results.
//

#include "ttmlir/Conversion/TTIRToEmitPy/TTIRToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRCPUTOEMITPY
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

// ============================================================================
// Type converter
// ============================================================================

// Converts MLIR tensor types to emitpy::OpaqueType("torch.Tensor").
// This is the type used by internal ops (CallOpaqueOp results) — it reflects
// the fact that ttir_cpu.* functions operate on torch tensors at runtime.
//
// Function boundaries use ttnn.Tensor instead; that conversion is handled
// separately by FuncOpBoundaryPattern / ReturnOpBoundaryPattern, which
// bypass the type converter.
class EmitPyTypeConverter : public TypeConverter {
public:
  EmitPyTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::TensorType type) -> emitpy::OpaqueType {
      return emitpy::OpaqueType::get(ctx, "torch.Tensor");
    });
  }
};

// ============================================================================
// Helpers
// ============================================================================

// Derives the Python callee name from a TTIR op: ttir.add -> ttir_cpu.add.
static std::string getCallee(Operation *op) {
  auto name = op->getName().getStringRef();
  assert(name.starts_with("ttir.") &&
         "getCallee expects a TTIR op; got unexpected dialect");
  return ("ttir_cpu." + name.drop_front(5)).str();
}

static std::string formatI32List(ArrayRef<int32_t> arr) {
  std::string s = "[";
  llvm::raw_string_ostream os(s);
  llvm::interleaveComma(arr, os);
  os << "]";
  return s;
}

static std::string formatI64List(ArrayRef<int64_t> arr) {
  std::string s = "[";
  llvm::raw_string_ostream os(s);
  llvm::interleaveComma(arr, os);
  os << "]";
  return s;
}

static std::pair<int32_t, int32_t> getI32Pair(Attribute attr) {
  if (auto dense = dyn_cast<DenseI32ArrayAttr>(attr)) {
    return {dense.asArrayRef()[0], dense.asArrayRef()[1]};
  }
  int32_t val = cast<IntegerAttr>(attr).getInt();
  return {val, val};
}

static SmallVector<int32_t, 4> getI32Quad(Attribute attr) {
  if (auto dense = dyn_cast<DenseI32ArrayAttr>(attr)) {
    auto a = dense.asArrayRef();
    if (a.size() == 4) {
      return {a[0], a[1], a[2], a[3]};
    }
    assert(a.size() == 2 && "expected 2 or 4 element array");
    return {a[0], a[1], a[0], a[1]};
  }
  int32_t val = cast<IntegerAttr>(attr).getInt();
  return {val, val, val, val};
}

static std::string formatScalarAttr(Attribute attr) {
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    return std::to_string(fAttr.getValueAsDouble());
  }
  return std::to_string(cast<IntegerAttr>(attr).getInt());
}

static SmallVector<int32_t> arrayAttrToI32Vec(ArrayAttr arr) {
  SmallVector<int32_t> v;
  for (auto a : arr) {
    v.push_back(cast<IntegerAttr>(a).getInt());
  }
  return v;
}

static std::string elemTypeToTorchDtypeStr(Type elemType) {
  if (elemType.isF32()) {
    return "torch.float32";
  }
  if (elemType.isBF16()) {
    return "torch.bfloat16";
  }
  if (elemType.isF16()) {
    return "torch.float16";
  }
  if (elemType.isInteger(32)) {
    return "torch.int32";
  }
  if (elemType.isInteger(64)) {
    return "torch.int64";
  }
  if (elemType.isInteger(16)) {
    return "torch.int16";
  }
  if (elemType.isInteger(8)) {
    return "torch.int8";
  }
  if (elemType.isInteger(1)) {
    return "torch.bool";
  }
  return "None";
}

// ============================================================================
// EmitPyCallBuilder
// ============================================================================

// Helper for building an emitpy::CallOpaqueOp from a TTIR op. Each TTIR
// conversion pattern constructs one of these, adds positional operands
// (SSA values), literal arguments (inline Python expressions), and keyword
// arguments, then calls replaceOp() to emit the final CallOpaqueOp.
class EmitPyCallBuilder {
public:
  EmitPyCallBuilder(Operation *srcOp, const TypeConverter *typeConverter,
                    StringRef callee)
      : srcOp(srcOp), ctx(srcOp->getContext()), typeConverter(typeConverter),
        callee(callee.str()) {}

  void addOperand(Value val) {
    unsigned idx = operands.size();
    operands.push_back(val);
    args.push_back(IntegerAttr::get(IndexType::get(ctx), idx));
    kwargNames.push_back(StringAttr::get(ctx, ""));
  }

  void addLiteral(StringRef value) {
    args.push_back(emitpy::OpaqueAttr::get(ctx, value));
    kwargNames.push_back(StringAttr::get(ctx, ""));
  }

  void addKwarg(StringRef name, StringRef value) {
    args.push_back(emitpy::OpaqueAttr::get(ctx, value));
    kwargNames.push_back(StringAttr::get(ctx, name));
  }

  void addKwarg(StringRef name, Value val) {
    unsigned idx = operands.size();
    operands.push_back(val);
    args.push_back(IntegerAttr::get(IndexType::get(ctx), idx));
    kwargNames.push_back(StringAttr::get(ctx, name));
  }

  void replaceOp(ConversionPatternRewriter &rewriter) {
    SmallVector<Type> resultTypes;
    for (Type t : srcOp->getResultTypes()) {
      resultTypes.push_back(typeConverter->convertType(t));
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        srcOp, resultTypes, callee, operands, ArrayAttr::get(ctx, args),
        ArrayAttr::get(ctx, kwargNames));
  }

private:
  Operation *srcOp;
  MLIRContext *ctx;
  const TypeConverter *typeConverter;
  std::string callee;
  SmallVector<Value> operands;
  SmallVector<Attribute> args;
  SmallVector<Attribute> kwargNames;
};

// ============================================================================
// Function boundary patterns
// ============================================================================

// Rewrites the function signature to ttnn.Tensor and inserts ttnn.to_torch
// for each argument so internal ops receive torch tensors.
class FuncOpBoundaryPattern : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto ttnnType = emitpy::OpaqueType::get(ctx, "ttnn.Tensor");
    auto torchType = emitpy::OpaqueType::get(ctx, "torch.Tensor");

    // Build the new signature: all tensor args/results become ttnn.Tensor.
    auto oldType = op.getFunctionType();
    SmallVector<Type> newArgTypes(oldType.getNumInputs(), ttnnType);
    SmallVector<Type> newResultTypes(oldType.getNumResults(), ttnnType);
    auto newType = FunctionType::get(ctx, newArgTypes, newResultTypes);

    // Create a new function with the ttnn.Tensor signature and move the body.
    auto newOp =
        rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), newType);
    newOp->setAttrs(op->getAttrs());
    newOp.setFunctionType(newType);

    // Inline the old body into the new function, converting block arg types.
    Block &oldEntry = op.getBody().front();
    Block &newEntry = newOp.getBody().emplaceBlock();

    // Add ttnn.Tensor block args and insert to_torch for each.
    rewriter.setInsertionPointToStart(&newEntry);
    SmallVector<Value> torchArgs;
    for (unsigned i = 0; i < oldEntry.getNumArguments(); ++i) {
      auto newArg = newEntry.addArgument(ttnnType, op.getLoc());
      auto toTorch = rewriter.create<emitpy::CallOpaqueOp>(
          op.getLoc(), torchType, "ttnn.to_torch", ValueRange{newArg}, nullptr,
          nullptr);
      torchArgs.push_back(toTorch.getResult(0));
    }

    // Merge the old block, replacing old block args with torch values.
    rewriter.mergeBlocks(&oldEntry, &newEntry, torchArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

// Inserts ttnn.from_torch before each return, converting torch.Tensor
// results back to ttnn.Tensor for the caller.
class ReturnOpBoundaryPattern : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ttnnType =
        emitpy::OpaqueType::get(rewriter.getContext(), "ttnn.Tensor");

    SmallVector<Value> newOperands;
    for (Value operand : adaptor.getOperands()) {
      auto fromTorch = rewriter.create<emitpy::CallOpaqueOp>(
          op.getLoc(), ttnnType, "ttnn.from_torch", ValueRange{operand},
          nullptr, nullptr);
      newOperands.push_back(fromTorch.getResult(0));
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, newOperands);
    return success();
  }
};

// ============================================================================
// Conversion patterns
// ============================================================================

// Generic pattern for elementwise unary ops (single input, no extra attrs).
template <typename TTIROp>
class TTIRUnaryToEmitPy : public OpConversionPattern<TTIROp> {
public:
  using OpConversionPattern<TTIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROp op, typename TTIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, this->getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.replaceOp(rewriter);
    return success();
  }
};

// Generic pattern for elementwise binary ops (lhs + rhs, no extra attrs).
template <typename TTIROp>
class TTIRBinaryToEmitPy : public OpConversionPattern<TTIROp> {
public:
  using OpConversionPattern<TTIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROp op, typename TTIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, this->getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getLhs());
    b.addOperand(adaptor.getRhs());
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Where ---
class TTIRWhereToEmitPy : public OpConversionPattern<ttir::WhereOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getFirst());
    b.addOperand(adaptor.getSecond());
    b.addOperand(adaptor.getThird());
    b.replaceOp(rewriter);
    return success();
  }
};

// Generic pattern for reduction ops (input + dim + keepdim).
template <typename TTIROp>
class TTIRReductionToEmitPy : public OpConversionPattern<TTIROp> {
public:
  using OpConversionPattern<TTIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROp op, typename TTIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, this->getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());

    auto dimArg = adaptor.getDimArg();
    if (dimArg) {
      b.addKwarg("dim", formatI32List(arrayAttrToI32Vec(*dimArg)));
    } else {
      b.addKwarg("dim", "None");
    }
    b.addKwarg("keepdim", adaptor.getKeepDim() ? "True" : "False");
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Softmax ---
class TTIRSoftmaxToEmitPy : public OpConversionPattern<ttir::SoftmaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addKwarg("dim", std::to_string(adaptor.getDimension()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Reshape ---
class TTIRReshapeToEmitPy : public OpConversionPattern<ttir::ReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(formatI32List(arrayAttrToI32Vec(adaptor.getShape())));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Permute ---
class TTIRPermuteToEmitPy : public OpConversionPattern<ttir::PermuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(formatI64List(adaptor.getPermutation()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Concat ---
class TTIRConcatToEmitPy : public OpConversionPattern<ttir::ConcatOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto listOp = rewriter.create<emitpy::CallOpaqueOp>(
        op.getLoc(),
        emitpy::OpaqueType::get(rewriter.getContext(), "[torch.Tensor]"),
        "util_create_list", adaptor.getInputs(), nullptr, nullptr);

    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(listOp.getResult(0));
    b.addKwarg("dim", std::to_string(adaptor.getDim()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Matmul ---
class TTIRMatmulToEmitPy : public OpConversionPattern<ttir::MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getA());
    b.addOperand(adaptor.getB());
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Linear ---
class TTIRLinearToEmitPy : public OpConversionPattern<ttir::LinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getA());
    b.addOperand(adaptor.getB());
    if (adaptor.getBias()) {
      b.addOperand(adaptor.getBias());
    } else {
      b.addKwarg("bias", "None");
    }
    b.addKwarg("transpose_a", adaptor.getTransposeA() ? "True" : "False");
    b.addKwarg("transpose_b", adaptor.getTransposeB() ? "True" : "False");
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Embedding ---
class TTIREmbeddingToEmitPy : public OpConversionPattern<ttir::EmbeddingOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addOperand(adaptor.getWeight());
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Squeeze ---
class TTIRSqueezeToEmitPy : public OpConversionPattern<ttir::SqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(std::to_string(adaptor.getDim()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Unsqueeze ---
class TTIRUnsqueezeToEmitPy : public OpConversionPattern<ttir::UnsqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(std::to_string(adaptor.getDim()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Transpose ---
class TTIRTransposeToEmitPy : public OpConversionPattern<ttir::TransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(std::to_string(adaptor.getDim0()));
    b.addLiteral(std::to_string(adaptor.getDim1()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Repeat ---
class TTIRRepeatToEmitPy : public OpConversionPattern<ttir::RepeatOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(formatI64List(adaptor.getRepeatDimensions()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Pad ---
class TTIRPadToEmitPy : public OpConversionPattern<ttir::PadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(formatI32List(adaptor.getPadding()));
    b.addLiteral(std::to_string(adaptor.getValue().convertToFloat()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- SliceStatic ---
class TTIRSliceStaticToEmitPy
    : public OpConversionPattern<ttir::SliceStaticOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceStaticOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(formatI32List(arrayAttrToI32Vec(adaptor.getBegins())));
    b.addLiteral(formatI32List(arrayAttrToI32Vec(adaptor.getEnds())));
    b.addLiteral(formatI32List(arrayAttrToI32Vec(adaptor.getStep())));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Broadcast ---
class TTIRBroadcastToEmitPy : public OpConversionPattern<ttir::BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    b.addLiteral(formatI64List(resultType.getShape()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- NamedFull (zeros, ones) ---
template <typename TTIROp>
class TTIRNamedFullToEmitPy : public OpConversionPattern<TTIROp> {
public:
  using OpConversionPattern<TTIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROp op, typename TTIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, this->getTypeConverter(), getCallee(op));
    b.addKwarg("shape", formatI32List(adaptor.getShape()));
    auto elemType =
        cast<RankedTensorType>(op.getResult().getType()).getElementType();
    b.addKwarg("dtype", elemTypeToTorchDtypeStr(elemType));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Full ---
class TTIRFullToEmitPy : public OpConversionPattern<ttir::FullOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addKwarg("shape", formatI32List(adaptor.getShape()));
    b.addKwarg("fill_value", formatScalarAttr(adaptor.getFillValue()));
    auto elemType =
        cast<RankedTensorType>(op.getResult().getType()).getElementType();
    b.addKwarg("dtype", elemTypeToTorchDtypeStr(elemType));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Empty ---
class TTIREmptyToEmitPy : public OpConversionPattern<ttir::EmptyOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addKwarg("shape", formatI64List(resultType.getShape()));
    b.addKwarg("dtype", elemTypeToTorchDtypeStr(resultType.getElementType()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Arange ---
class TTIRArangeToEmitPy : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addLiteral(std::to_string(adaptor.getStart()));
    b.addLiteral(std::to_string(adaptor.getEnd()));
    b.addLiteral(std::to_string(adaptor.getStep()));
    b.addKwarg("arange_dimension",
               std::to_string(adaptor.getArangeDimension()));
    b.addKwarg("shape", formatI64List(resultType.getShape()));
    b.addKwarg("dtype", elemTypeToTorchDtypeStr(resultType.getElementType()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- GlobalAvgPool2d ---
class TTIRGlobalAvgPool2dToEmitPy
    : public OpConversionPattern<ttir::GlobalAvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GlobalAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.replaceOp(rewriter);
    return success();
  }
};

// --- LeakyRelu ---
class TTIRLeakyReluToEmitPy : public OpConversionPattern<ttir::LeakyReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LeakyReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addKwarg("negative_slope",
               std::to_string(adaptor.getParameter().convertToFloat()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Constant ---
class TTIRConstantToEmitPy : public OpConversionPattern<ttir::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.getValue();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto elemType = resultType.getElementType();

    auto denseAttr = dyn_cast<DenseElementsAttr>(value);
    if (!denseAttr) {
      return rewriter.notifyMatchFailure(
          op, "only DenseElementsAttr constants are supported");
    }

    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addKwarg("shape", formatI64List(resultType.getShape()));
    b.addKwarg("dtype", elemTypeToTorchDtypeStr(elemType));

    if (denseAttr.isSplat()) {
      if (isa<FloatType>(elemType)) {
        b.addKwarg("fill_value",
                   std::to_string(
                       denseAttr.getSplatValue<APFloat>().convertToDouble()));
      } else {
        b.addKwarg(
            "fill_value",
            std::to_string(denseAttr.getSplatValue<APInt>().getSExtValue()));
      }
    } else {
      // Non-splat: serialize as a flat Python list and let the runtime reshape.
      std::string data = "[";
      llvm::raw_string_ostream os(data);
      bool first = true;
      if (isa<FloatType>(elemType)) {
        for (APFloat v : denseAttr.getValues<APFloat>()) {
          if (!first) {
            os << ", ";
          }
          os << v.convertToDouble();
          first = false;
        }
      } else {
        for (APInt v : denseAttr.getValues<APInt>()) {
          if (!first) {
            os << ", ";
          }
          os << v.getSExtValue();
          first = false;
        }
      }
      os << "]";
      b.addKwarg("data", data);
    }

    b.replaceOp(rewriter);
    return success();
  }
};

// --- CumSum ---
class TTIRCumSumToEmitPy : public OpConversionPattern<ttir::CumSumOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CumSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addKwarg("dim", std::to_string(adaptor.getDim()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- ConcatenateHeads ---
class TTIRConcatenateHeadsToEmitPy
    : public OpConversionPattern<ttir::ConcatenateHeadsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatenateHeadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.replaceOp(rewriter);
    return success();
  }
};

// --- ClampScalar ---
class TTIRClampScalarToEmitPy
    : public OpConversionPattern<ttir::ClampScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ClampScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addLiteral(formatScalarAttr(adaptor.getMin()));
    b.addLiteral(formatScalarAttr(adaptor.getMax()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- ClampTensor ---
class TTIRClampTensorToEmitPy
    : public OpConversionPattern<ttir::ClampTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ClampTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addOperand(adaptor.getMin());
    b.addOperand(adaptor.getMax());
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Typecast ---
class TTIRTypecastToEmitPy : public OpConversionPattern<ttir::TypecastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TypecastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());

    auto elemType =
        cast<RankedTensorType>(op.getResult().getType()).getElementType();
    b.addKwarg("dtype", elemTypeToTorchDtypeStr(elemType));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- LayerNorm ---
class TTIRLayerNormToEmitPy : public OpConversionPattern<ttir::LayerNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    if (adaptor.getWeight()) {
      b.addKwarg("weight", adaptor.getWeight());
    } else {
      b.addKwarg("weight", "None");
    }
    if (adaptor.getBias()) {
      b.addKwarg("bias", adaptor.getBias());
    } else {
      b.addKwarg("bias", "None");
    }
    b.addKwarg("epsilon",
               std::to_string(adaptor.getEpsilon().convertToFloat()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- DotGeneral ---
class TTIRDotGeneralToEmitPy : public OpConversionPattern<ttir::DotGeneralOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getLhs());
    b.addOperand(adaptor.getRhs());
    b.addKwarg("batch_dims_lhs", formatI64List(adaptor.getBatchDimsLhs()));
    b.addKwarg("contract_dims_lhs",
               formatI64List(adaptor.getContractDimsLhs()));
    b.addKwarg("batch_dims_rhs", formatI64List(adaptor.getBatchDimsRhs()));
    b.addKwarg("contract_dims_rhs",
               formatI64List(adaptor.getContractDimsRhs()));
    b.replaceOp(rewriter);
    return success();
  }
};

// --- SplitQueryKeyValueAndSplitHeads ---
class TTIRSplitQKVToEmitPy
    : public OpConversionPattern<ttir::SplitQueryKeyValueAndSplitHeadsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SplitQueryKeyValueAndSplitHeadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInputTensor());
    if (adaptor.getKvInputTensor()) {
      b.addOperand(adaptor.getKvInputTensor());
    } else {
      b.addKwarg("kv_input_tensor", "None");
    }
    b.addKwarg("num_heads", std::to_string(adaptor.getNumHeads()));
    auto numKvHeads = adaptor.getNumKvHeads();
    if (numKvHeads) {
      b.addKwarg("num_kv_heads", std::to_string(*numKvHeads));
    } else {
      b.addKwarg("num_kv_heads", "None");
    }
    b.addKwarg("transpose_key", adaptor.getTransposeKey() ? "True" : "False");
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Pooling helper ---
template <typename PoolOp>
static void addPoolSpatialDims(PoolOp op, typename PoolOp::Adaptor adaptor,
                               EmitPyCallBuilder &b) {
  int64_t batchSize, inputH, inputW, channels;
  auto flatInfo = adaptor.getFlattenedCompatInfo();
  auto inputType = cast<RankedTensorType>(op.getInput().getType());

  if (flatInfo) {
    batchSize = flatInfo.getBatchSize();
    inputH = flatInfo.getInputHeight();
    inputW = flatInfo.getInputWidth();
    channels = inputType.getDimSize(3);
  } else {
    auto shape = inputType.getShape();
    batchSize = shape[0];
    inputH = shape[1];
    inputW = shape[2];
    channels = shape[3];
  }
  b.addLiteral(std::to_string(batchSize));
  b.addLiteral(std::to_string(inputH));
  b.addLiteral(std::to_string(inputW));
  b.addLiteral(std::to_string(channels));
}

// --- MaxPool2d ---
class TTIRMaxPool2dToEmitPy : public OpConversionPattern<ttir::MaxPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [kH, kW] = getI32Pair(adaptor.getKernel());
    auto [sH, sW] = getI32Pair(adaptor.getStride());
    auto [dH, dW] = getI32Pair(adaptor.getDilation());
    auto padding = getI32Quad(adaptor.getPadding());

    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    addPoolSpatialDims(op, adaptor, b);
    b.addLiteral(formatI32List({kH, kW}));
    b.addLiteral(formatI32List({sH, sW}));
    b.addLiteral(formatI32List(padding));
    b.addLiteral(formatI32List({dH, dW}));
    b.addLiteral(adaptor.getCeilMode() ? "True" : "False");
    b.replaceOp(rewriter);
    return success();
  }
};

// --- MaxPool2dWithIndices ---
class TTIRMaxPool2dWithIndicesToEmitPy
    : public OpConversionPattern<ttir::MaxPool2dWithIndicesOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxPool2dWithIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [kH, kW] = getI32Pair(adaptor.getKernel());
    auto [sH, sW] = getI32Pair(adaptor.getStride());
    auto [dH, dW] = getI32Pair(adaptor.getDilation());
    auto padding = getI32Quad(adaptor.getPadding());

    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    addPoolSpatialDims(op, adaptor, b);
    b.addLiteral(formatI32List({kH, kW}));
    b.addLiteral(formatI32List({sH, sW}));
    b.addLiteral(formatI32List(padding));
    b.addLiteral(formatI32List({dH, dW}));
    b.addLiteral(adaptor.getCeilMode() ? "True" : "False");
    b.replaceOp(rewriter);
    return success();
  }
};

// --- AvgPool2d ---
class TTIRAvgPool2dToEmitPy : public OpConversionPattern<ttir::AvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [kH, kW] = getI32Pair(adaptor.getKernel());
    auto [sH, sW] = getI32Pair(adaptor.getStride());
    auto padding = getI32Quad(adaptor.getPadding());

    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    addPoolSpatialDims(op, adaptor, b);
    b.addLiteral(formatI32List({kH, kW}));
    b.addLiteral(formatI32List({sH, sW}));
    b.addLiteral(formatI32List(padding));
    b.addKwarg("ceil_mode", adaptor.getCeilMode() ? "True" : "False");
    b.addKwarg("count_include_pad",
               adaptor.getCountIncludePad() ? "True" : "False");
    b.replaceOp(rewriter);
    return success();
  }
};

// --- Conv2d ---
class TTIRConv2dToEmitPy : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [sH, sW] = getI32Pair(adaptor.getStride());
    auto [dH, dW] = getI32Pair(adaptor.getDilation());
    auto padding = getI32Quad(adaptor.getPadding());

    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addOperand(adaptor.getWeight());
    if (adaptor.getBias()) {
      b.addOperand(adaptor.getBias());
    } else {
      b.addKwarg("bias", "None");
    }
    b.addKwarg("stride", formatI32List({sH, sW}));
    b.addKwarg("padding", formatI32List(padding));
    b.addKwarg("dilation", formatI32List({dH, dW}));
    b.addKwarg("groups", std::to_string(adaptor.getGroups()));
    b.addKwarg("batch_dim", std::to_string(adaptor.getBatchDim()));
    b.addKwarg("height_dim", std::to_string(adaptor.getHeightDim()));
    b.addKwarg("width_dim", std::to_string(adaptor.getWidthDim()));
    b.addKwarg("channel_dim", std::to_string(adaptor.getChannelDim()));
    b.replaceOp(rewriter);
    return success();
  }
};

// ============================================================================
// Pass definition
// ============================================================================

struct ConvertTTIRCPUToEmitPyPass
    : public ::mlir::tt::ttir::impl::ConvertTTIRCPUToEmitPyBase<
          ConvertTTIRCPUToEmitPyPass> {

  using ::mlir::tt::ttir::impl::ConvertTTIRCPUToEmitPyBase<
      ConvertTTIRCPUToEmitPyPass>::ConvertTTIRCPUToEmitPyBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalDialect<emitpy::EmitPyDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();
    target.addDynamicallyLegalOp<ModuleOp>(
        [](ModuleOp op) { return op->getAttrs().empty(); });

    if (module.getBodyRegion().empty()) {
      signalPassFailure();
      return;
    }

    EmitPyTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    // Function boundary: signature becomes ttnn.Tensor, with to_torch /
    // from_torch bridging to the torch.Tensor world used by internal ops.
    patterns.add<FuncOpBoundaryPattern, ReturnOpBoundaryPattern>(typeConverter,
                                                                 &getContext());
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      return llvm::all_of(op.getArgumentTypes(),
                          llvm::IsaPred<emitpy::OpaqueType>) &&
             llvm::all_of(op.getResultTypes(),
                          llvm::IsaPred<emitpy::OpaqueType>);
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
      return llvm::all_of(op.getOperandTypes(),
                          llvm::IsaPred<emitpy::OpaqueType>);
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    // Elementwise unary.
    patterns.add<
        TTIRUnaryToEmitPy<ttir::AbsOp>, TTIRUnaryToEmitPy<ttir::AcosOp>,
        TTIRUnaryToEmitPy<ttir::AsinOp>, TTIRUnaryToEmitPy<ttir::AtanOp>,
        TTIRUnaryToEmitPy<ttir::BitwiseNotOp>, TTIRUnaryToEmitPy<ttir::CbrtOp>,
        TTIRUnaryToEmitPy<ttir::CeilOp>, TTIRUnaryToEmitPy<ttir::CosOp>,
        TTIRUnaryToEmitPy<ttir::ExpOp>, TTIRUnaryToEmitPy<ttir::Expm1Op>,
        TTIRUnaryToEmitPy<ttir::ErfOp>, TTIRUnaryToEmitPy<ttir::ErfcOp>,
        TTIRUnaryToEmitPy<ttir::FloorOp>, TTIRUnaryToEmitPy<ttir::GeluOp>,
        TTIRUnaryToEmitPy<ttir::HardsigmoidOp>,
        TTIRUnaryToEmitPy<ttir::IsFiniteOp>, TTIRUnaryToEmitPy<ttir::LogOp>,
        TTIRUnaryToEmitPy<ttir::Log1pOp>, TTIRUnaryToEmitPy<ttir::LogicalNotOp>,
        TTIRUnaryToEmitPy<ttir::MishOp>, TTIRUnaryToEmitPy<ttir::NegOp>,
        TTIRUnaryToEmitPy<ttir::ReciprocalOp>, TTIRUnaryToEmitPy<ttir::ReluOp>,
        TTIRUnaryToEmitPy<ttir::Relu6Op>, TTIRUnaryToEmitPy<ttir::RsqrtOp>,
        TTIRUnaryToEmitPy<ttir::SigmoidOp>, TTIRUnaryToEmitPy<ttir::SignOp>,
        TTIRUnaryToEmitPy<ttir::SiluOp>, TTIRUnaryToEmitPy<ttir::SinOp>,
        TTIRUnaryToEmitPy<ttir::SqrtOp>, TTIRUnaryToEmitPy<ttir::TanOp>,
        TTIRUnaryToEmitPy<ttir::TanhOp>>(typeConverter, &getContext());

    // Elementwise unary with parameters.
    patterns.add<TTIRLeakyReluToEmitPy>(typeConverter, &getContext());

    // Elementwise binary.
    patterns.add<
        TTIRBinaryToEmitPy<ttir::AddOp>, TTIRBinaryToEmitPy<ttir::SubtractOp>,
        TTIRBinaryToEmitPy<ttir::MultiplyOp>, TTIRBinaryToEmitPy<ttir::DivOp>,
        TTIRBinaryToEmitPy<ttir::EqualOp>, TTIRBinaryToEmitPy<ttir::NotEqualOp>,
        TTIRBinaryToEmitPy<ttir::GreaterThanOp>,
        TTIRBinaryToEmitPy<ttir::GreaterEqualOp>,
        TTIRBinaryToEmitPy<ttir::LessThanOp>,
        TTIRBinaryToEmitPy<ttir::LessEqualOp>,
        TTIRBinaryToEmitPy<ttir::LogicalAndOp>,
        TTIRBinaryToEmitPy<ttir::LogicalOrOp>,
        TTIRBinaryToEmitPy<ttir::LogicalXorOp>,
        TTIRBinaryToEmitPy<ttir::MaximumOp>,
        TTIRBinaryToEmitPy<ttir::MinimumOp>, TTIRBinaryToEmitPy<ttir::Atan2Op>,
        TTIRBinaryToEmitPy<ttir::RemainderOp>, TTIRBinaryToEmitPy<ttir::PowOp>,
        TTIRBinaryToEmitPy<ttir::BitwiseAndOp>,
        TTIRBinaryToEmitPy<ttir::BitwiseOrOp>,
        TTIRBinaryToEmitPy<ttir::BitwiseXorOp>,
        TTIRBinaryToEmitPy<ttir::LogicalLeftShiftOp>,
        TTIRBinaryToEmitPy<ttir::LogicalRightShiftOp>,
        TTIRBinaryToEmitPy<ttir::GeluBackwardOp>>(typeConverter, &getContext());

    // Ternary / reductions / named ops.
    patterns.add<TTIRWhereToEmitPy>(typeConverter, &getContext());

    patterns.add<
        TTIRReductionToEmitPy<ttir::SumOp>, TTIRReductionToEmitPy<ttir::MeanOp>,
        TTIRReductionToEmitPy<ttir::MaxOp>, TTIRReductionToEmitPy<ttir::MinOp>,
        TTIRReductionToEmitPy<ttir::ProdOp>,
        TTIRReductionToEmitPy<ttir::ArgMaxOp>,
        TTIRReductionToEmitPy<ttir::ReduceOrOp>>(typeConverter, &getContext());

    // Creation ops.
    patterns.add<TTIRNamedFullToEmitPy<ttir::ZerosOp>,
                 TTIRNamedFullToEmitPy<ttir::OnesOp>, TTIRFullToEmitPy,
                 TTIREmptyToEmitPy, TTIRArangeToEmitPy, TTIRConstantToEmitPy>(
        typeConverter, &getContext());

    // Named ops.
    patterns.add<
        TTIRSoftmaxToEmitPy, TTIRReshapeToEmitPy, TTIRPermuteToEmitPy,
        TTIRConcatToEmitPy, TTIRMatmulToEmitPy, TTIRLinearToEmitPy,
        TTIREmbeddingToEmitPy, TTIRSqueezeToEmitPy, TTIRUnsqueezeToEmitPy,
        TTIRTransposeToEmitPy, TTIRRepeatToEmitPy, TTIRPadToEmitPy,
        TTIRSliceStaticToEmitPy, TTIRBroadcastToEmitPy,
        TTIRGlobalAvgPool2dToEmitPy, TTIRClampScalarToEmitPy,
        TTIRClampTensorToEmitPy, TTIRTypecastToEmitPy, TTIRLayerNormToEmitPy,
        TTIRDotGeneralToEmitPy, TTIRSplitQKVToEmitPy, TTIRMaxPool2dToEmitPy,
        TTIRMaxPool2dWithIndicesToEmitPy, TTIRAvgPool2dToEmitPy,
        TTIRConv2dToEmitPy, TTIRCumSumToEmitPy, TTIRConcatenateHeadsToEmitPy>(
        typeConverter, &getContext());

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRCPUToEmitPyPass() {
  return std::make_unique<ConvertTTIRCPUToEmitPyPass>();
}

} // namespace mlir::tt
