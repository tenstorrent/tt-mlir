// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/SFPIToEmitC/SFPIToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/SFPI/IR/SFPI.h"
#include "ttmlir/Dialect/SFPI/IR/SFPIOps.h"

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::tt::sfpi;

namespace mlir::tt {

#define GEN_PASS_DEF_CONVERTSFPITOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

/// Converts SFPI types to EmitC types for GCC builtin translation
class SFPIToEmitCTypeConverter : public TypeConverter {
public:
  SFPIToEmitCTypeConverter() {
    // Convert SFPI 4x8 vectors to __rvtt_vec_t
    addConversion([](Type type) -> std::optional<Type> {
      if (auto vectorType = dyn_cast<VectorType>(type)) {
        if (vectorType.getShape().size() == 2 &&
            vectorType.getShape()[0] == 4 && vectorType.getShape()[1] == 8 &&
            (vectorType.getElementType().isF32() ||
             vectorType.getElementType().isInteger(32))) {
          // Convert to __rvtt_vec_t (represented as opaque type)
          auto context = type.getContext();
          return emitc::OpaqueType::get(context, "__rvtt_vec_t");
        }
      }
      return type; // Pass through other types unchanged
    });

    addConversion([](emitc::OpaqueType type) { return type; });
    addConversion([](IntegerType type) { return type; });
    addConversion([](FloatType type) { return type; });
    addConversion([](IndexType type) { return type; });

    // Add materialization patterns for unrealized conversion casts
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return nullptr;
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return nullptr;
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

//===----------------------------------------------------------------------===//
// Pattern Rewriters
//===----------------------------------------------------------------------===//

/// Base class for SFPI to EmitC conversion patterns
template <typename SFPIOpType>
class SFPIToEmitCOpConversionPattern : public OpConversionPattern<SFPIOpType> {
public:
  using OpConversionPattern<SFPIOpType>::OpConversionPattern;

protected:
  /// Create an EmitC call to a GCC builtin function
  emitc::CallOpaqueOp createBuiltinCall(ConversionPatternRewriter &rewriter,
                                        Location loc, StringRef builtinName,
                                        ArrayRef<Value> operands,
                                        Type resultType) const {
    return rewriter.create<emitc::CallOpaqueOp>(
        loc, resultType, rewriter.getStringAttr(builtinName),
        rewriter.getArrayAttr({}), nullptr, operands);
  }
};

/// Convert SFPI add operations to __builtin_rvtt_sfpadd
struct ConvertSFPIAddOp : public SFPIToEmitCOpConversionPattern<AddOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpadd",
                          {adaptor.getLhs(), adaptor.getRhs()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI mul operations to __builtin_rvtt_sfpmul
struct ConvertSFPIMulOp : public SFPIToEmitCOpConversionPattern<MulOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpmul",
                          {adaptor.getLhs(), adaptor.getRhs()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI mad operations to __builtin_rvtt_sfpmad
struct ConvertSFPIMadOp : public SFPIToEmitCOpConversionPattern<MadOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(MadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call = createBuiltinCall(
        rewriter, op.getLoc(), "__builtin_rvtt_sfpmad",
        {adaptor.getA(), adaptor.getB(), adaptor.getC()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI mov operations to __builtin_rvtt_sfpmov
struct ConvertSFPIMovOp : public SFPIToEmitCOpConversionPattern<MovOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(MovOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpmov",
                          {adaptor.getSrc()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI abs operations to __builtin_rvtt_sfpabs
struct ConvertSFPIAbsOp : public SFPIToEmitCOpConversionPattern<AbsOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpabs",
                          {adaptor.getSrc()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI and operations to __builtin_rvtt_sfpand
struct ConvertSFPIAndOp : public SFPIToEmitCOpConversionPattern<AndOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpand",
                          {adaptor.getLhs(), adaptor.getRhs()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI or operations to __builtin_rvtt_sfpor
struct ConvertSFPIOrOp : public SFPIToEmitCOpConversionPattern<OrOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpor",
                          {adaptor.getLhs(), adaptor.getRhs()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI xor operations to __builtin_rvtt_sfpxor
struct ConvertSFPIXorOp : public SFPIToEmitCOpConversionPattern<XorOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpxor",
                          {adaptor.getLhs(), adaptor.getRhs()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Convert SFPI not operations to __builtin_rvtt_sfpnot
struct ConvertSFPINotOp : public SFPIToEmitCOpConversionPattern<NotOp> {
  using SFPIToEmitCOpConversionPattern::SFPIToEmitCOpConversionPattern;

  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }

    auto call =
        createBuiltinCall(rewriter, op.getLoc(), "__builtin_rvtt_sfpnot",
                          {adaptor.getSrc()}, resultType);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

struct ConvertSFPIToEmitCPass
    : public ::mlir::tt::impl::ConvertSFPIToEmitCBase<ConvertSFPIToEmitCPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    SFPIToEmitCTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    // Target is legal if it doesn't contain SFPI operations
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<sfpi::SFPIDialect>();

    // Function dialect handling
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    // Populate conversion patterns
    populateSFPIToEmitCConversionPatterns(patterns, typeConverter);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//

void mlir::tt::populateSFPIToEmitCConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConvertSFPIAddOp, ConvertSFPIMulOp, ConvertSFPIMadOp,
               ConvertSFPIMovOp, ConvertSFPIAbsOp, ConvertSFPIAndOp,
               ConvertSFPIOrOp, ConvertSFPIXorOp, ConvertSFPINotOp>(
      typeConverter, patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tt::createConvertSFPIToEmitCPass() {
  return std::make_unique<ConvertSFPIToEmitCPass>();
}
