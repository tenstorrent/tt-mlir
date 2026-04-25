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
  // tensor<1xiN> -> smt::BitVectorType(N)
  // tensor<KxiN> -> smt::ArrayType(bv<ceil_log2(K)>, bv<N>)
  converter.addConversion([](RankedTensorType t) -> std::optional<Type> {
    auto intTy = dyn_cast<IntegerType>(t.getElementType());
    if (!intTy)
      return std::nullopt;
    unsigned width = intTy.getWidth();
    int64_t numElems = t.getNumElements();
    if (numElems == 1)
      return smt::BitVectorType::get(t.getContext(), width);
    // Multi-element tensor -> SMT array
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

/// ttir.constant -> smt.bv.constant
struct ConstantOpConversion
    : OpConversionPattern<ttir::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    auto bvTy = dyn_cast<smt::BitVectorType>(resultTy);
    if (!bvTy)
      return failure(); // TODO: handle array constants

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return failure();

    APInt val = denseAttr.getSplatValue<APInt>();
    // Ensure the APInt width matches the BV width
    if (val.getBitWidth() != bvTy.getWidth())
      val = val.zextOrTrunc(bvTy.getWidth());

    rewriter.replaceOpWithNewOp<smt::BVConstantOp>(op, val);
    return success();
  }
};

/// ttir.eq -> smt.eq (produces bool) -> materialize to bv<1>
struct EqOpConversion : OpConversionPattern<ttir::EqualOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto boolResult = smt::EqOp::create(rewriter, op.getLoc(),
                                        adaptor.getLhs(), adaptor.getRhs());
    // Materialize bool -> bv<1>
    auto bv1Ty = smt::BitVectorType::get(op.getContext(), 1);
    auto one = smt::BVConstantOp::create(rewriter, op.getLoc(), 1, 1);
    auto zero = smt::BVConstantOp::create(rewriter, op.getLoc(), 0, 1);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, bv1Ty, boolResult, one, zero);
    return success();
  }
};

/// ttir.ne -> smt.distinct (produces bool) -> materialize to bv<1>
struct NeOpConversion : OpConversionPattern<ttir::NotEqualOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::NotEqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto boolResult = smt::DistinctOp::create(rewriter, op.getLoc(),
                                              adaptor.getLhs(), adaptor.getRhs());
    auto bv1Ty = smt::BitVectorType::get(op.getContext(), 1);
    auto one = smt::BVConstantOp::create(rewriter, op.getLoc(), 1, 1);
    auto zero = smt::BVConstantOp::create(rewriter, op.getLoc(), 0, 1);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, bv1Ty, boolResult, one, zero);
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

/// Generic comparison op conversion: ttir.lt/le/gt/ge -> smt.bv.cmp
template <typename SourceOp, bool IsLess, bool IsStrict>
struct CmpOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pred =
        getTTIRCmpPredicate(op.getLhs().getType(), IsLess, IsStrict);
    auto boolResult = smt::BVCmpOp::create(rewriter, op.getLoc(), pred,
                                           adaptor.getLhs(), adaptor.getRhs());
    // Materialize bool -> bv<1>
    auto bv1Ty = smt::BitVectorType::get(op.getContext(), 1);
    auto one = smt::BVConstantOp::create(rewriter, op.getLoc(), 1, 1);
    auto zero = smt::BVConstantOp::create(rewriter, op.getLoc(), 0, 1);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, bv1Ty, boolResult, one, zero);
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
    ConcatOpConversion
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
