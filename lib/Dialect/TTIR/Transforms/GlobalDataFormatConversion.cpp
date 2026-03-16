// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGLOBALDATAFORMATCONVERSION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

static std::optional<ttcore::DataType>
parseTargetFormat(llvm::StringRef format) {
  if (format == "f32") {
    return ttcore::DataType::Float32;
  }
  if (format == "bf16") {
    return ttcore::DataType::BFloat16;
  }
  if (format == "bfp_bf8") {
    return ttcore::DataType::BFP_BFloat8;
  }
  return std::nullopt;
}

// Normalizes unsupported 64-bit element types to their 32-bit equivalents
// (f64 -> f32, i64/si64/ui64 -> i32/si32/ui32). This ensures the IR only
// contains types that the hardware can represent.
struct UnsupportedTypeNormalizer : mlir::TypeConverter {
  UnsupportedTypeNormalizer() {
    addConversion([](mlir::RankedTensorType type)
                      -> std::optional<mlir::RankedTensorType> {
      Type elementType = type.getElementType();

      if (auto floatType = dyn_cast<FloatType>(elementType);
          floatType && floatType.getWidth() == 64) {
        return mlir::RankedTensorType::get(type.getShape(),
                                           Float32Type::get(type.getContext()),
                                           type.getEncoding());
      }

      if (auto intType = dyn_cast<IntegerType>(elementType);
          intType && intType.getWidth() == 64) {
        return mlir::RankedTensorType::get(
            type.getShape(),
            IntegerType::get(type.getContext(), 32, intType.getSignedness()),
            type.getEncoding());
      }

      return type;
    });

    auto materialize = [](mlir::OpBuilder &builder, mlir::Type type,
                          mlir::ValueRange inputs,
                          mlir::Location loc) -> mlir::Value {
      return builder.create<ttir::TypecastOp>(
          loc, mlir::cast<mlir::RankedTensorType>(type), inputs);
    };
    addSourceMaterialization(materialize);
    addTargetMaterialization(materialize);
  }
};

struct GlobalDataFormatBodyConverter : mlir::TypeConverter {
  GlobalDataFormatBodyConverter(ttcore::DataType targetDataType) {
    addConversion([targetDataType](mlir::RankedTensorType type)
                      -> std::optional<mlir::RankedTensorType> {
      mlir::Type elementType = type.getElementType();

      // Convert everything to the target data type
      Type newElementType =
          ttcore::dataTypeToElementType(type.getContext(), targetDataType);

      if (!newElementType || newElementType == elementType) {
        return type;
      }

      return mlir::RankedTensorType::get(type.getShape(), newElementType,
                                         type.getEncoding());
    });

    // Materialization function for automatic typecast insertion
    auto materializeFunc = [](mlir::OpBuilder &builder, mlir::Type type,
                              mlir::ValueRange inputs,
                              mlir::Location loc) -> mlir::Value {
      mlir::RankedTensorType rankedType =
          mlir::cast<mlir::RankedTensorType>(type);
      return builder.create<ttir::TypecastOp>(loc, rankedType, inputs);
    };

    addSourceMaterialization(materializeFunc); // Input conversions
    addTargetMaterialization(materializeFunc); // Output conversions
  }
};

class FuncBodyTypeCast : public mlir::ConversionPattern {
public:
  FuncBodyTypeCast(const mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Skip func.func and func.return — these are handled by the dedicated
    // MLIR function/return conversion patterns when doing signature conversion,
    // and don't need explicit handling in the body-only case.
    if (isa<func::FuncOp, func::ReturnOp>(op)) {
      return failure();
    }

    mlir::IRMapping mapping;
    mapping.map(op->getOperands(), operands);

    mlir::Operation *newOp = rewriter.clone(*op, mapping);

    llvm::SmallVector<Type> convertedTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                convertedTypes))) {
      return op->emitOpError("Failed to convert result types.");
    }

    rewriter.modifyOpInPlace(newOp, [&]() {
      for (auto [newResult, newType] :
           llvm::zip(newOp->getResults(), convertedTypes)) {
        newResult.setType(newType);
      }
    });

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Run a type conversion using the given converter, skipping ops marked with
// "preserveDataFormat". When convertFuncSignatures is true, function argument
// and return types are also converted; otherwise function signatures are
// preserved and typecasts are inserted at the boundaries.
static LogicalResult runBodyTypeConversion(mlir::Operation *root,
                                           mlir::TypeConverter &converter,
                                           mlir::MLIRContext &ctx,
                                           bool convertFuncSignatures) {
  mlir::ConversionTarget target(ctx);
  target.markUnknownOpDynamicallyLegal(
      [&converter, convertFuncSignatures](mlir::Operation *op) {
        if (isa<func::FuncOp>(op)) {
          if (!convertFuncSignatures) {
            return true;
          }
          auto funcOp = cast<func::FuncOp>(op);
          return converter.isSignatureLegal(funcOp.getFunctionType()) &&
                 converter.isLegal(&funcOp.getBody());
        }
        // func.return has no results, so check its operand types instead to
        // ensure the return values are converted to match the function
        // signature.
        if (isa<func::ReturnOp>(op)) {
          if (!convertFuncSignatures) {
            return true;
          }
          return llvm::all_of(op->getOperandTypes(), [&converter](Type type) {
            return converter.isLegal(type);
          });
        }
        if (op->hasAttr("preserveDataFormat")) {
          return true;
        }
        return llvm::all_of(op->getResultTypes(), [&converter](Type type) {
          return converter.isLegal(type);
        });
      });

  mlir::RewritePatternSet patterns(&ctx);
  patterns.add<FuncBodyTypeCast>(converter, &ctx);
  if (convertFuncSignatures) {
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
  }
  return mlir::applyFullConversion(root, target, std::move(patterns));
}

struct TTIRGlobalDataFormatConversion
    : public impl::TTIRGlobalDataFormatConversionBase<
          TTIRGlobalDataFormatConversion> {
  using impl::TTIRGlobalDataFormatConversionBase<
      TTIRGlobalDataFormatConversion>::TTIRGlobalDataFormatConversionBase;

  void runOnOperation() final {
    // Step 1: Always normalize unsupported 64-bit types (f64 -> f32,
    // i64 -> i32). This converts both function signatures and body types.
    UnsupportedTypeNormalizer normalizer;
    if (failed(runBodyTypeConversion(getOperation(), normalizer, getContext(),
                                     /*convertFuncSignatures=*/true))) {
      signalPassFailure();
      return;
    }

    // Step 2: Optionally convert all body types to a target format if
    // one was specified. Function signatures are preserved and
    // typecasts are inserted at boundaries.
    if (targetFormat.empty()) {
      return;
    }

    auto targetDataType = parseTargetFormat(targetFormat);
    if (!targetDataType) {
      getOperation()->emitError("Invalid target format '")
          << targetFormat << "'. Supported formats: f32, bf16, bfp_bf8";
      signalPassFailure();
      return;
    }

    GlobalDataFormatBodyConverter bodyConverter(*targetDataType);
    if (failed(runBodyTypeConversion(getOperation(), bodyConverter,
                                     getContext(),
                                     /*convertFuncSignatures=*/false))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
