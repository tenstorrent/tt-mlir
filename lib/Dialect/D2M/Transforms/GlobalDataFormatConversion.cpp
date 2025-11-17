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
      return ttir::utils::createDPSOp<ttir::TypecastOp>(builder, loc,
                                                        rankedType, inputs);
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

struct D2MGlobalDataFormatConversion
    : public impl::D2MGlobalDataFormatConversionBase<
          D2MGlobalDataFormatConversion> {
  using impl::D2MGlobalDataFormatConversionBase<
      D2MGlobalDataFormatConversion>::D2MGlobalDataFormatConversionBase;

  void runOnOperation() final {
    auto targetDataType = parseTargetFormat(targetFormat);
    if (!targetDataType) {
      getOperation()->emitError("Invalid target format '")
          << targetFormat << "'. Supported formats: f32, bf16, bfp_bf8";
      signalPassFailure();
      return;
    }

    GlobalDataFormatBodyConverter bodyConverter(*targetDataType);

    mlir::ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([&bodyConverter](mlir::Operation *op) {
      // Preserve function signatures
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
