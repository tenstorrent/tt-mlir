// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/QuantUtils.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <limits>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRQUANTDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Helper function to convert integer bit width to IntegerType.
static IntegerType getIntegerTypeFromBitWidth(MLIRContext *context,
                                              uint32_t bitWidth, Location loc) {
  switch (bitWidth) {
  case 8:
  case 16:
  case 32:
  case 64:
    return IntegerType::get(context, bitWidth, IntegerType::Signed);
  default:
    emitError(loc,
              "Invalid quantization bit width (must be 8, 16, 32, or 64). ");
    return nullptr;
  }
}

// Convert quantized type to use the target integer bit width.
static mlir::FailureOr<Type>
convertQuantizedType(quant::QuantizedType quantType, IntegerType targetIntType,
                     Location loc) {
  // Use the target integer type's min and max values.
  mlir::FailureOr<std::pair<int64_t, int64_t>> storageTypeMinMax =
      mlir::tt::ttir::utils::getStorageTypeMinMax(targetIntType, loc);
  if (mlir::failed(storageTypeMinMax)) {
    return mlir::failure();
  }
  auto [storageTypeMin, storageTypeMax] = *storageTypeMinMax;

  if (targetIntType.getWidth() <
      quantType.getStorageType().getIntOrFloatBitWidth()) {
    emitError(loc) << "Target integer type of width "
                   << targetIntType.getWidth()
                   << " is smaller than quantized type of width "
                   << quantType.getStorageType().getIntOrFloatBitWidth()
                   << ". Out of range.";
    return mlir::failure();
  }

  if (quant::UniformQuantizedType uniformType =
          dyn_cast<quant::UniformQuantizedType>(quantType)) {
    return quant::UniformQuantizedType::get(
        uniformType.getFlags(), targetIntType, uniformType.getExpressedType(),
        uniformType.getScale(), uniformType.getZeroPoint(), storageTypeMin,
        storageTypeMax);
  }
  if (quant::UniformQuantizedPerAxisType perAxisType =
          dyn_cast<quant::UniformQuantizedPerAxisType>(quantType)) {
    return quant::UniformQuantizedPerAxisType::get(
        perAxisType.getFlags(), targetIntType, perAxisType.getExpressedType(),
        perAxisType.getScales(), perAxisType.getZeroPoints(),
        perAxisType.getQuantizedDimension(), storageTypeMin, storageTypeMax);
  }
  return mlir::failure();
}

class QuantDataTypeConverter : public TypeConverter {
private:
  IntegerType targetIntType;
  Location loc;
  bool conversionFailed = false;

public:
  QuantDataTypeConverter(IntegerType targetIntType, Location loc)
      : targetIntType(targetIntType), loc(loc) {
    addConversion([](Type type) -> Type { return type; });

    addConversion([this](RankedTensorType type) -> std::optional<Type> {
      Type elementType = type.getElementType();
      auto quantElementType = mlir::dyn_cast<quant::QuantizedType>(elementType);
      if (!quantElementType) {
        return std::nullopt;
      }
      mlir::FailureOr<Type> result = convertQuantizedType(
          quantElementType, this->targetIntType, this->loc);
      if (mlir::failed(result)) {
        this->conversionFailed = true;
        return std::nullopt;
      }
      return RankedTensorType::get(type.getShape(), *result,
                                   type.getEncoding());
    });
  }

  bool hasConversionFailed() const { return conversionFailed; }
};

struct TTIRQuantDataTypeConversionPass
    : public impl::TTIRQuantDataTypeConversionPassBase<
          TTIRQuantDataTypeConversionPass> {
  using impl::TTIRQuantDataTypeConversionPassBase<
      TTIRQuantDataTypeConversionPass>::TTIRQuantDataTypeConversionPassBase;

  TTIRQuantDataTypeConversionPass(
      TTIRQuantDataTypeConversionPassOptions options) {
    targetBitWidth = options.targetBitWidth;
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();

    IntegerType targetIntType = getIntegerTypeFromBitWidth(
        context, targetBitWidth, getOperation()->getLoc());
    if (!targetIntType) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(context);
    QuantDataTypeConverter converter(targetIntType, getOperation()->getLoc());
    patterns.add<UniformTypeRewriter>(converter, context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        converter.hasConversionFailed()) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
