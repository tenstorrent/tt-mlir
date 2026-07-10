// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ConvertTTIRCPUToEmitPy pass
// ===========================
//
// Lowers each CPU-hoisted function to a ttnn-typed wrapper around a nested
// pure-torch body, so the body can run shard-by-shard for multi-chip meshes:
//
//   def <name>(ttnn.Tensor...):                  # wrapper, keeps the symbol
//     def <name>_impl(torch.Tensor...): ...      # nested ttir_cpu.* body
//     return utils.execute_cpu_hoisted_function([...], <name>_impl)
//
// In two steps: dialect conversion lowers the function in place to torch.Tensor
// (stock signature/return conversion plus per-op ttir -> ttir_cpu patterns),
// then wrapAndNest builds the ttnn wrapper and nests the body as an
// emitpy.nested_func.
//

#include "ttmlir/Conversion/TTIRToEmitPy/TTIRToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/FunctionTypes.h"

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

// Converts MLIR tensor types to emitpy::OpaqueType("torch.Tensor"). The whole
// CPU-hoisted function is lowered to this torch.Tensor world (signature,
// internal ttir_cpu ops, and return); the ttnn.Tensor boundary is introduced
// later by wrapAndNest, which wraps the torch body in a ttnn-typed function.
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

static std::string formatAPFloat(llvm::APFloat value) {
  // APFloat::toString prints non-finite values as "Inf"/"-Inf"/"NaN", which
  // are not valid Python literals. Emit float('...') for those cases.
  if (value.isInfinity()) {
    return value.isNegative() ? "float('-inf')" : "float('inf')";
  }
  if (value.isNaN()) {
    return "float('nan')";
  }
  // Default precision (0) prints enough decimal digits that parsing the
  // string back yields the exact same float bits.
  llvm::SmallString<32> buf;
  value.toString(buf);
  return std::string(buf);
}

static std::string formatScalarAttr(Attribute attr) {
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    return formatAPFloat(fAttr.getValue());
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
    b.addLiteral(formatAPFloat(adaptor.getValue()));
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
    b.addKwarg("negative_slope", formatAPFloat(adaptor.getParameter()));
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
                   formatAPFloat(denseAttr.getSplatValue<APFloat>()));
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
          os << formatAPFloat(v);
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
    b.addKwarg("epsilon", formatAPFloat(adaptor.getEpsilon()));
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

// Wraps each in-place-lowered (now torch-typed) CPU-hoisted function in a
// ttnn-typed wrapper that defers to utils.execute_cpu_hoisted_function, with
// the torch body emitted as a nested `<name>_impl` def. Given
//
//   func @cpu_hoisted_X(torch...) -> torch... { ...ttir_cpu... }   //
//   forward_cpu
//
// it produces
//
//   def cpu_hoisted_X(a, b):                      # ttnn wrapper
//     def cpu_hoisted_X_impl(a, b): ...ttir_cpu...
//     return utils.execute_cpu_hoisted_function([a, b], cpu_hoisted_X_impl)
//
// Keeping the body nested means the impl is not a module-level symbol (no
// symbol-DCE concern, nothing for module linking to relocate separately).
static void wrapAndNest(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  auto ttnnType = emitpy::OpaqueType::get(ctx, "ttnn.Tensor");

  SmallVector<func::FuncOp> impls;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (ttmlir::utils::isForwardCPUFunc(func)) {
      impls.push_back(func);
    }
  }

  for (func::FuncOp impl : impls) {
    auto origName = impl.getSymName().str();
    auto implName = origName + "_impl";
    unsigned numInputs = impl.getNumArguments();
    unsigned numResults = impl.getFunctionType().getNumResults();

    OpBuilder builder(impl);

    // The body lives in a nested function named <name>_impl; the wrapper keeps
    // the original symbol and a ttnn.Tensor signature so callers are unchanged.
    impl.setSymName(implName);
    auto wrapType =
        FunctionType::get(ctx, SmallVector<Type>(numInputs, ttnnType),
                          SmallVector<Type>(numResults, ttnnType));
    auto wrapOp =
        builder.create<func::FuncOp>(impl.getLoc(), origName, wrapType);
    wrapOp->setAttrs(impl->getAttrs());
    wrapOp.setSymName(origName);
    wrapOp.setFunctionType(wrapType);

    Block &wrapEntry = wrapOp.getBody().emplaceBlock();
    SmallVector<Value> wrapArgs;
    for (unsigned i = 0; i < numInputs; ++i) {
      wrapArgs.push_back(wrapEntry.addArgument(ttnnType, impl.getLoc()));
    }
    builder.setInsertionPointToStart(&wrapEntry);

    // Nested torch body: move the impl's body in and swap its terminator for an
    // emitpy.nested_func_return.
    auto funcOp = builder.create<emitpy::NestedFuncOp>(impl.getLoc(), implName);
    funcOp.getBody().takeBody(impl.getBody());
    auto returnOp =
        cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
    {
      OpBuilder retBuilder(returnOp);
      retBuilder.create<emitpy::NestedFuncReturnOp>(returnOp.getLoc(),
                                                    returnOp.getOperands());
      returnOp.erase();
    }

    // result = utils.execute_cpu_hoisted_function([arg0, ...], <name>_impl)
    builder.setInsertionPointToEnd(&wrapEntry);
    auto listOp = builder.create<emitpy::CallOpaqueOp>(
        impl.getLoc(), emitpy::OpaqueType::get(ctx, "[ttnn.Tensor]"),
        "util_create_list", wrapArgs, nullptr, nullptr);
    SmallVector<Value> callOperands{listOp.getResult(0)};
    SmallVector<Attribute> callArgs{IntegerAttr::get(IndexType::get(ctx), 0),
                                    emitpy::OpaqueAttr::get(ctx, implName)};
    SmallVector<Attribute> callKwargs{StringAttr::get(ctx, ""),
                                      StringAttr::get(ctx, "")};
    auto callOp = builder.create<emitpy::CallOpaqueOp>(
        impl.getLoc(), SmallVector<Type>(numResults, ttnnType),
        "utils.execute_cpu_hoisted_function", callOperands,
        ArrayAttr::get(ctx, callArgs), ArrayAttr::get(ctx, callKwargs));
    builder.create<func::ReturnOp>(impl.getLoc(), callOp.getResults());

    impl.erase();
  }
}

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

    // Lower the CPU-hoisted function in place to the torch.Tensor world used by
    // the internal ttir_cpu ops, using stock signature/return type conversion.
    // The ttnn-typed wrapper and the nested impl are built afterwards by
    // wrapAndNest.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
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
        TTIRUnaryToEmitPy<ttir::AsinOp>, TTIRUnaryToEmitPy<ttir::AsinhOp>,
        TTIRUnaryToEmitPy<ttir::AtanOp>, TTIRUnaryToEmitPy<ttir::BitwiseNotOp>,
        TTIRUnaryToEmitPy<ttir::CbrtOp>, TTIRUnaryToEmitPy<ttir::CeilOp>,
        TTIRUnaryToEmitPy<ttir::CosOp>, TTIRUnaryToEmitPy<ttir::ExpOp>,
        TTIRUnaryToEmitPy<ttir::Expm1Op>, TTIRUnaryToEmitPy<ttir::ErfOp>,
        TTIRUnaryToEmitPy<ttir::ErfcOp>, TTIRUnaryToEmitPy<ttir::FloorOp>,
        TTIRUnaryToEmitPy<ttir::GeluOp>, TTIRUnaryToEmitPy<ttir::HardsigmoidOp>,
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

    wrapAndNest(getOperation());
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRCPUToEmitPyPass() {
  return std::make_unique<ConvertTTIRCPUToEmitPyPass>();
}

} // namespace mlir::tt
