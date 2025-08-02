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
          return emitc::OpaqueType::get(context, "sfpi::__rvtt_vec_t");
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
template <typename SFPIOpType, typename SFPIBuiltinInfo>
class SFPIToEmitCOpConversionPattern : public OpConversionPattern<SFPIOpType> {
public:
  using OpConversionPattern<SFPIOpType>::OpConversionPattern;

protected:
  LogicalResult
  matchAndRewrite(SFPIOpType op, typename SFPIOpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = this->getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultType, rewriter.getStringAttr(SFPIBuiltinInfo::builtinName),
        nullptr, nullptr, adaptor.getOperands());
    return success();
  }
};

struct SFPIAddBuiltin {
  constexpr static char const* builtinName = "__builtin_rvtt_sfpadd";
};

struct SFPIMulBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpmul";
};

struct SFPIMadBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpmad";
};

struct SFPIMovBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpmov";
};

struct SFPIAbsBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpabs";
};

struct SFPIAndBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpand";
};

struct SFPIOrBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpor";
};

struct SFPIXorBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpxor";
};

struct SFPINotBuiltin {
  constexpr static const char *builtinName = "__builtin_rvtt_sfpnot";
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
  patterns.add<SFPIToEmitCOpConversionPattern<AddOp, SFPIAddBuiltin>,
               SFPIToEmitCOpConversionPattern<MulOp, SFPIMulBuiltin>,
               SFPIToEmitCOpConversionPattern<MadOp, SFPIMadBuiltin>,
               SFPIToEmitCOpConversionPattern<MovOp, SFPIMovBuiltin>,
               SFPIToEmitCOpConversionPattern<AbsOp, SFPIAbsBuiltin>,
               SFPIToEmitCOpConversionPattern<AndOp, SFPIAndBuiltin>,
               SFPIToEmitCOpConversionPattern<OrOp, SFPIOrBuiltin>,
               SFPIToEmitCOpConversionPattern<XorOp, SFPIXorBuiltin>,
               SFPIToEmitCOpConversionPattern<NotOp, SFPINotBuiltin>>(
      typeConverter, patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tt::createConvertSFPIToEmitCPass() {
  return std::make_unique<ConvertSFPIToEmitCPass>();
}
