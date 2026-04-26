// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToSMT/TTIRToSMT.h"

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

#define GEN_PASS_DEF_CONVERTTTIRTOSMT
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

static void addTTIRToSMTTypeConversions(TypeConverter &converter) {
  // tensor<1xiN>            -> smt::BitVectorType(N)
  // tensor<KxiN> (rank 1)   -> smt::ArrayType(bv<ceil_log2(K)>, bv<N>)
  // tensor<AxBx...xiN>      -> flatten to tensor<(A*B*...)xiN> -> array
  converter.addConversion([](RankedTensorType t) -> std::optional<Type> {
    auto intTy = dyn_cast<IntegerType>(t.getElementType());
    if (!intTy)
      return std::nullopt;
    unsigned width = intTy.getWidth();
    int64_t numElems = t.getNumElements();
    if (numElems == 1)
      return smt::BitVectorType::get(t.getContext(), width);
    // Multi-element tensor (any rank) -> SMT array indexed by total size.
    unsigned idxBits = llvm::Log2_64_Ceil(numElems);
    if (idxBits == 0)
      idxBits = 1;
    auto idxTy = smt::BitVectorType::get(t.getContext(), idxBits);
    auto elemTy = smt::BitVectorType::get(t.getContext(), width);
    return smt::ArrayType::get(t.getContext(), idxTy, elemTy);
  });

  // Keep SMT types as-is (for mixed modules)
  converter.addConversion([](smt::BitVectorType t) { return t; });
  converter.addConversion([](smt::BoolType t) { return t; });
  converter.addConversion([](smt::ArrayType t) { return t; });

  // Bool <-> bv<1> materializations
  converter.addTargetMaterialization(
      [](OpBuilder &builder, smt::BitVectorType type, ValueRange inputs,
         Location loc) -> Value {
        if (type.getWidth() != 1 || inputs.size() != 1)
          return {};
        if (isa<smt::BoolType>(inputs[0].getType())) {
          auto one = smt::BVConstantOp::create(builder, loc, 1, 1);
          auto zero = smt::BVConstantOp::create(builder, loc, 0, 1);
          return smt::IteOp::create(builder, loc, inputs[0], one, zero);
        }
        return {};
      });

  converter.addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange inputs,
         Location loc) -> Value {
        if (!isa<smt::BoolType>(type) || inputs.size() != 1)
          return {};
        if (auto bvTy = dyn_cast<smt::BitVectorType>(inputs[0].getType())) {
          if (bvTy.getWidth() == 1) {
            auto zero = smt::BVConstantOp::create(builder, loc, 0, 1);
            return smt::DistinctOp::create(builder, loc, inputs[0], zero);
          }
        }
        return {};
      });
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Direct 1:1 mapping for binary ops: ttir.op(lhs, rhs) -> smt.bv.op(lhs, rhs)
template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = this->getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();
    rewriter.replaceOpWithNewOp<TargetOp>(op, resultTy, adaptor.getLhs(),
                                          adaptor.getRhs());
    return success();
  }
};

/// Unary op: ttir.op(input) -> smt.bv.op(input)
template <typename SourceOp, typename TargetOp>
struct UnaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = this->getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();
    rewriter.replaceOpWithNewOp<TargetOp>(op, resultTy, adaptor.getInput());
    return success();
  }
};

/// ttir.subtract -> smt.bv.neg + smt.bv.add (two's complement subtraction)
struct SubtractOpConversion
    : OpConversionPattern<ttir::SubtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SubtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();
    auto neg =
        smt::BVNegOp::create(rewriter, op.getLoc(), resultTy, adaptor.getRhs());
    rewriter.replaceOpWithNewOp<smt::BVAddOp>(op, resultTy, adaptor.getLhs(),
                                              neg);
    return success();
  }
};

/// ttir.constant -> smt.bv.constant (scalar) or array build (multi-element).
/// For splat arrays we use smt.array.broadcast; otherwise a chain of
/// smt.array.store starting from an array_broadcast(0).
struct ConstantOpConversion
    : OpConversionPattern<ttir::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr)
      return failure();

    Location loc = op.getLoc();

    // Scalar (single-element tensor) case.
    if (auto bvTy = dyn_cast<smt::BitVectorType>(resultTy)) {
      APInt val = denseAttr.isSplat() ? denseAttr.getSplatValue<APInt>()
                                      : *denseAttr.value_begin<APInt>();
      if (val.getBitWidth() != bvTy.getWidth())
        val = val.zextOrTrunc(bvTy.getWidth());
      rewriter.replaceOpWithNewOp<smt::BVConstantOp>(op, val);
      return success();
    }

    // Multi-element array case.
    auto arrTy = dyn_cast<smt::ArrayType>(resultTy);
    if (!arrTy)
      return failure();
    auto elemBvTy = cast<smt::BitVectorType>(arrTy.getRangeType());
    auto idxBvTy = cast<smt::BitVectorType>(arrTy.getDomainType());
    unsigned elemWidth = elemBvTy.getWidth();

    if (denseAttr.isSplat()) {
      APInt val = denseAttr.getSplatValue<APInt>();
      if (val.getBitWidth() != elemWidth)
        val = val.zextOrTrunc(elemWidth);
      auto elemConst = smt::BVConstantOp::create(rewriter, loc, val);
      rewriter.replaceOpWithNewOp<smt::ArrayBroadcastOp>(op, resultTy,
                                                         elemConst);
      return success();
    }

    // Non-splat: build with broadcast(0) base + N stores.
    auto zero = smt::BVConstantOp::create(rewriter, loc, 0, elemWidth);
    Value arr = smt::ArrayBroadcastOp::create(rewriter, loc, resultTy, zero);
    auto values = denseAttr.getValues<APInt>();
    unsigned i = 0;
    for (APInt val : values) {
      if (val.getBitWidth() != elemWidth)
        val = val.zextOrTrunc(elemWidth);
      auto idx = smt::BVConstantOp::create(rewriter, loc, i, idxBvTy.getWidth());
      auto elemConst = smt::BVConstantOp::create(rewriter, loc, val);
      arr = smt::ArrayStoreOp::create(rewriter, loc, arr, idx, elemConst);
      ++i;
    }
    rewriter.replaceOp(op, arr);
    return success();
  }
};

/// ttir.full -> smt.bv.constant (scalar) or smt.array.broadcast (multi-elem).
/// fill_value is an attribute (I32 or F32), not a dense tensor.
struct FullOpConversion : OpConversionPattern<ttir::FullOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    auto intAttr = dyn_cast<IntegerAttr>(op.getFillValue());
    if (!intAttr)
      return failure(); // skip f32 fills

    APInt val = intAttr.getValue();

    if (auto bvTy = dyn_cast<smt::BitVectorType>(resultTy)) {
      if (val.getBitWidth() != bvTy.getWidth())
        val = val.zextOrTrunc(bvTy.getWidth());
      rewriter.replaceOpWithNewOp<smt::BVConstantOp>(op, val);
      return success();
    }

    auto arrTy = dyn_cast<smt::ArrayType>(resultTy);
    if (!arrTy)
      return failure();
    auto elemBvTy = cast<smt::BitVectorType>(arrTy.getRangeType());
    if (val.getBitWidth() != elemBvTy.getWidth())
      val = val.zextOrTrunc(elemBvTy.getWidth());
    auto elemConst = smt::BVConstantOp::create(rewriter, op.getLoc(), val);
    rewriter.replaceOpWithNewOp<smt::ArrayBroadcastOp>(op, resultTy, elemConst);
    return success();
  }
};

/// Materialize an SMT bool as a bitvector of `width` bits: 0 for false,
/// 1 for true. Used by comparison op lowerings.
static Value boolToBitVector(OpBuilder &b, Location loc, Value boolVal,
                             unsigned width) {
  auto bvTy = smt::BitVectorType::get(b.getContext(), width);
  auto one = smt::BVConstantOp::create(b, loc, 1, width);
  auto zero = smt::BVConstantOp::create(b, loc, 0, width);
  return smt::IteOp::create(b, loc, bvTy, boolVal, one, zero);
}

/// ttir.eq -> smt.eq (bool) -> bitvector of result width.
struct EqOpConversion : OpConversionPattern<ttir::EqualOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast_or_null<smt::BitVectorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultTy)
      return failure();
    auto boolResult = smt::EqOp::create(rewriter, op.getLoc(),
                                        adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(
        op, boolToBitVector(rewriter, op.getLoc(), boolResult,
                            resultTy.getWidth()));
    return success();
  }
};

/// ttir.ne -> smt.distinct (bool) -> bitvector of result width.
struct NeOpConversion : OpConversionPattern<ttir::NotEqualOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::NotEqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast_or_null<smt::BitVectorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultTy)
      return failure();
    auto boolResult = smt::DistinctOp::create(
        rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(
        op, boolToBitVector(rewriter, op.getLoc(), boolResult,
                            resultTy.getWidth()));
    return success();
  }
};

/// Helper to determine the SMT BV comparison predicate from a TTIR comparison
/// op's operand types.
static smt::BVCmpPredicate
getTTIRCmpPredicate(Type operandType, bool isLess, bool isStrict) {
  auto tensorTy = dyn_cast<RankedTensorType>(operandType);
  bool isUnsigned = false;
  if (tensorTy) {
    if (auto intTy = dyn_cast<IntegerType>(tensorTy.getElementType()))
      isUnsigned = intTy.isUnsigned();
  }

  if (isLess && isStrict)
    return isUnsigned ? smt::BVCmpPredicate::ult : smt::BVCmpPredicate::slt;
  if (isLess && !isStrict)
    return isUnsigned ? smt::BVCmpPredicate::ule : smt::BVCmpPredicate::sle;
  if (!isLess && isStrict)
    return isUnsigned ? smt::BVCmpPredicate::ugt : smt::BVCmpPredicate::sgt;
  return isUnsigned ? smt::BVCmpPredicate::uge : smt::BVCmpPredicate::sge;
}

/// Generic comparison op conversion: ttir.lt/le/gt/ge -> smt.bv.cmp.
/// Result is a bitvector of the converted result-type width (typically i1
/// or, after width promotion, i8).
template <typename SourceOp, bool IsLess, bool IsStrict>
struct CmpOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast_or_null<smt::BitVectorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultTy)
      return failure();
    auto pred = getTTIRCmpPredicate(op.getLhs().getType(), IsLess, IsStrict);
    auto boolResult = smt::BVCmpOp::create(rewriter, op.getLoc(), pred,
                                           adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(
        op, boolToBitVector(rewriter, op.getLoc(), boolResult,
                            resultTy.getWidth()));
    return success();
  }
};

/// ttir.where(cond, true_val, false_val) -> smt.ite(cond_bool, true, false)
struct WhereOpConversion : OpConversionPattern<ttir::WhereOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // first=condition, second=true_val, third=false_val
    // The condition is a bitvector (was tensor<1xiN>). We need to convert to
    // bool: nonzero -> true.
    Value cond = adaptor.getFirst();
    auto condBvTy = dyn_cast<smt::BitVectorType>(cond.getType());
    if (!condBvTy)
      return failure();

    // Compare condition != 0
    auto zero = smt::BVConstantOp::create(rewriter, op.getLoc(), 0,
                                          condBvTy.getWidth());
    auto condBool =
        smt::DistinctOp::create(rewriter, op.getLoc(), cond, zero);

    rewriter.replaceOpWithNewOp<smt::IteOp>(
        op, resultTy, condBool, adaptor.getSecond(), adaptor.getThird());
    return success();
  }
};

/// ttir.typecast(input) -> smt.bv.extract or zero-extend (concat with zeros)
struct TypecastOpConversion
    : OpConversionPattern<ttir::TypecastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TypecastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    auto srcBvTy = dyn_cast<smt::BitVectorType>(adaptor.getInput().getType());
    auto dstBvTy = dyn_cast<smt::BitVectorType>(resultTy);
    if (!srcBvTy || !dstBvTy)
      return failure();

    unsigned srcWidth = srcBvTy.getWidth();
    unsigned dstWidth = dstBvTy.getWidth();

    if (srcWidth == dstWidth) {
      // Same width — identity
      rewriter.replaceOp(op, adaptor.getInput());
    } else if (srcWidth > dstWidth) {
      // Truncate: extract low bits
      rewriter.replaceOpWithNewOp<smt::ExtractOp>(op, dstBvTy, 0,
                                                   adaptor.getInput());
    } else {
      // Zero-extend: concat zeros on the high side
      unsigned extBits = dstWidth - srcWidth;
      auto zeros = smt::BVConstantOp::create(rewriter, op.getLoc(), 0, extBits);
      // smt.bv.concat: lhs is high bits, rhs is low bits
      rewriter.replaceOpWithNewOp<smt::ConcatOp>(op, dstBvTy, zeros,
                                                  adaptor.getInput());
    }
    return success();
  }
};

/// ttir.reshape -> identity for single-element tensors
struct ReshapeOpConversion
    : OpConversionPattern<ttir::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // For single-element tensors, reshape is a no-op in SMT
    if (adaptor.getInput().getType() == resultTy) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }
    return failure();
  }
};

/// ttir.concat(a, b, ...) -> smt.bv.concat chain
struct ConcatOpConversion : OpConversionPattern<ttir::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    auto inputs = adaptor.getInputs();
    if (inputs.empty())
      return failure();

    // Chain binary concat: a ++ b ++ c ++ ...
    Value result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
      result = smt::ConcatOp::create(rewriter, op.getLoc(), result, inputs[i]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// ttir.slice_static -> smt.array.select (single element) or array build
/// (multi-element). Only handles 1-D slices with stride 1.
struct SliceStaticOpConversion
    : OpConversionPattern<ttir::SliceStaticOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceStaticOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    ArrayAttr beginsAttr = op.getBeginsAttr();
    ArrayAttr endsAttr = op.getEndsAttr();
    ArrayAttr stepAttr = op.getStepAttr();
    if (beginsAttr.size() != 1 || endsAttr.size() != 1 || stepAttr.size() != 1)
      return rewriter.notifyMatchFailure(op, "only 1-D slices supported");

    int64_t begin = cast<IntegerAttr>(beginsAttr[0]).getInt();
    int64_t end = cast<IntegerAttr>(endsAttr[0]).getInt();
    int64_t step = cast<IntegerAttr>(stepAttr[0]).getInt();
    if (step != 1)
      return rewriter.notifyMatchFailure(op, "only stride-1 slices supported");
    int64_t numElems = end - begin;
    if (numElems <= 0)
      return failure();

    Value inArr = adaptor.getInput();
    Location loc = op.getLoc();

    // Single-element slice: smt.array.select at begin index.
    if (auto bvTy = dyn_cast<smt::BitVectorType>(resultTy)) {
      auto inArrTy = dyn_cast<smt::ArrayType>(inArr.getType());
      if (!inArrTy)
        return failure();
      unsigned idxBits =
          cast<smt::BitVectorType>(inArrTy.getDomainType()).getWidth();
      auto idx = smt::BVConstantOp::create(rewriter, loc, begin, idxBits);
      rewriter.replaceOpWithNewOp<smt::ArraySelectOp>(op, bvTy, inArr, idx);
      return success();
    }

    // Multi-element slice: build a new array by stores.
    auto outArrTy = dyn_cast<smt::ArrayType>(resultTy);
    if (!outArrTy)
      return failure();
    auto inArrTy = dyn_cast<smt::ArrayType>(inArr.getType());
    if (!inArrTy)
      return failure();
    unsigned inIdxBits =
        cast<smt::BitVectorType>(inArrTy.getDomainType()).getWidth();
    unsigned outIdxBits =
        cast<smt::BitVectorType>(outArrTy.getDomainType()).getWidth();
    auto elemBvTy = cast<smt::BitVectorType>(outArrTy.getRangeType());
    auto zeroElem = smt::BVConstantOp::create(rewriter, loc, 0,
                                              elemBvTy.getWidth());
    Value outArr =
        smt::ArrayBroadcastOp::create(rewriter, loc, outArrTy, zeroElem);
    for (int64_t i = 0; i < numElems; ++i) {
      auto inIdx = smt::BVConstantOp::create(rewriter, loc, begin + i, inIdxBits);
      auto outIdx = smt::BVConstantOp::create(rewriter, loc, i, outIdxBits);
      auto elem = smt::ArraySelectOp::create(rewriter, loc, elemBvTy, inArr,
                                             inIdx);
      outArr =
          smt::ArrayStoreOp::create(rewriter, loc, outArr, outIdx, elem);
    }
    rewriter.replaceOp(op, outArr);
    return success();
  }
};

/// ttir.empty -> removed (dead code in SMT context)
struct EmptyOpConversion : OpConversionPattern<ttir::EmptyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // EmptyOp has no SMT equivalent — if it has no users, erase it.
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

void mlir::tt::populateTTIRToSMTTypeConverter(TypeConverter &converter) {
  addTTIRToSMTTypeConversions(converter);
}

void mlir::tt::populateTTIRToSMTConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // clang-format off
  patterns.add<
    // Constants
    ConstantOpConversion,
    FullOpConversion,
    EmptyOpConversion,
    // Bitwise
    BinaryOpConversion<ttir::BitwiseAndOp, smt::BVAndOp>,
    BinaryOpConversion<ttir::BitwiseOrOp, smt::BVOrOp>,
    BinaryOpConversion<ttir::BitwiseXorOp, smt::BVXOrOp>,
    UnaryOpConversion<ttir::BitwiseNotOp, smt::BVNotOp>,
    // Logical (same as bitwise for bv<1>)
    UnaryOpConversion<ttir::LogicalNotOp, smt::BVNotOp>,
    BinaryOpConversion<ttir::LogicalAndOp, smt::BVAndOp>,
    BinaryOpConversion<ttir::LogicalOrOp, smt::BVOrOp>,
    // Arithmetic
    BinaryOpConversion<ttir::AddOp, smt::BVAddOp>,
    SubtractOpConversion,
    BinaryOpConversion<ttir::MultiplyOp, smt::BVMulOp>,
    // Shifts
    BinaryOpConversion<ttir::LogicalLeftShiftOp, smt::BVShlOp>,
    BinaryOpConversion<ttir::LogicalRightShiftOp, smt::BVLShrOp>,
    // Comparisons
    EqOpConversion,
    NeOpConversion,
    CmpOpConversion<ttir::LessThanOp, /*IsLess=*/true, /*IsStrict=*/true>,
    CmpOpConversion<ttir::LessEqualOp, /*IsLess=*/true, /*IsStrict=*/false>,
    CmpOpConversion<ttir::GreaterThanOp, /*IsLess=*/false, /*IsStrict=*/true>,
    CmpOpConversion<ttir::GreaterEqualOp, /*IsLess=*/false, /*IsStrict=*/false>,
    // Conditional
    WhereOpConversion,
    // Type conversion
    TypecastOpConversion,
    ReshapeOpConversion,
    // Tensor manipulation
    ConcatOpConversion,
    SliceStaticOpConversion
  >(converter, ctx);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertTTIRToSMTPass
    : public mlir::tt::impl::ConvertTTIRToSMTBase<ConvertTTIRToSMTPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addIllegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<smt::SMTDialect>();
    target.addLegalDialect<func::FuncDialect>();

    TypeConverter converter;
    addTTIRToSMTTypeConversions(converter);

    RewritePatternSet patterns(&getContext());
    populateTTIRToSMTConversionPatterns(converter, patterns);

    // Also convert func.func and func.return signatures.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    populateReturnOpTypeConversionPattern(patterns, converter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return converter.isLegal(op); });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToSMTPass() {
  return std::make_unique<ConvertTTIRToSMTPass>();
}

} // namespace mlir::tt
