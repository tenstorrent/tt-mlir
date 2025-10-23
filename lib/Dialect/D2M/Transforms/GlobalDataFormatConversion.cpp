// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGLOBALDATAFORMATCONVERSION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper function to parse target format string to DataType
static std::optional<ttcore::DataType>
parseTargetFormat(llvm::StringRef format) {
  if (format == "f32") {
    return ttcore::DataType::Float32;
  } else if (format == "bf16") {
    return ttcore::DataType::BFloat16;
  } else if (format == "bfp_f8") {
    return ttcore::DataType::BFP_Float8;
  }
  return std::nullopt;
}

// TypeConverter that converts all tensor types to the target format
struct GlobalDataFormatBodyConverter : mlir::TypeConverter {
  GlobalDataFormatBodyConverter(ttcore::DataType targetDataType) {
    addConversion([targetDataType](
                      mlir::RankedTensorType type) -> mlir::RankedTensorType {
      mlir::Type elementType = type.getElementType();

      // Convert everything to the target data type
      Type newElementType =
          ttcore::dataTypeToElementType(type.getContext(), targetDataType);

      if (!newElementType || newElementType == elementType) {
        return type;
      }

      return type.clone(newElementType);
    });

    // Materialization function for automatic typecast insertion
    auto materializeFunc = [](mlir::OpBuilder &builder, mlir::Type type,
                              mlir::ValueRange inputs,
                              mlir::Location loc) -> mlir::Value {
      mlir::RankedTensorType rankedType =
          mlir::cast<mlir::RankedTensorType>(type);
      return ttir::utils::createDPSOp<ttir::TypecastOp>(builder, loc,
                                                        rankedType, inputs);
    };

    addSourceMaterialization(materializeFunc); // Input conversions
    addTargetMaterialization(materializeFunc); // Output conversions
  }
};

// Pattern that converts operation types while preserving function signatures
class FuncBodyTypeCast : public mlir::ConversionPattern {
public:
  FuncBodyTypeCast(const mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Remap original operands to converted operands.
    mlir::IRMapping mapping;
    mapping.map(op->getOperands(), operands);

    // Clone the original operation with the new operands.
    mlir::Operation *newOp = rewriter.clone(*op, mapping);

    // Convert the result types using the type converter.
    llvm::SmallVector<Type> convertedTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                convertedTypes))) {
      return op->emitOpError("Failed to convert result types.");
    }

    // Update result types in-place on the new operation.
    rewriter.modifyOpInPlace(newOp, [&]() {
      for (auto [newResult, newType] :
           llvm::zip(newOp->getResults(), convertedTypes)) {
        newResult.setType(newType);
      }
    });

    // Replace the old op with the new one.
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct D2MGlobalDataFormatConversion
    : public impl::D2MGlobalDataFormatConversionBase<
          D2MGlobalDataFormatConversion> {
  using impl::D2MGlobalDataFormatConversionBase<
      D2MGlobalDataFormatConversion>::D2MGlobalDataFormatConversionBase;

  void runOnOperation() final {
    // Parse and validate the target format option
    auto targetDataType = parseTargetFormat(targetFormat);
    if (!targetDataType) {
      getOperation()->emitError("Invalid target format '")
          << targetFormat << "'. Supported formats: f32, bf16, bfp_f8";
      signalPassFailure();
      return;
    }

    GlobalDataFormatBodyConverter bodyConverter(*targetDataType);

    // Set up conversion target: preserve function signatures, convert
    // operations
    mlir::ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([&bodyConverter](mlir::Operation *op) {
      // Preserve function operations (their signatures)
      if (isa<func::FuncOp>(op)) {
        return true;
      }

      // Convert all other operations to use the target format
      return llvm::all_of(op->getResultTypes(), [&bodyConverter](Type type) {
        return bodyConverter.isLegal(type);
      });
    });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuncBodyTypeCast>(bodyConverter, &getContext());

    if (failed(mlir::applyFullConversion(getOperation(), target,
                                         std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
