// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
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
    emitError(loc, "Invalid quantization bit width (must be 8, 16, 32, or 64). "
                   "Returning null IntegerType.");
    return nullptr;
  }
}

static std::optional<std::pair<int64_t, int64_t>>
getStorageTypeMinMax(IntegerType intType, Location loc) {
  switch (intType.getWidth()) {
  case 8:
    return std::make_pair(std::numeric_limits<int8_t>::min(),
                          std::numeric_limits<int8_t>::max());
  case 16:
    return std::make_pair(std::numeric_limits<int16_t>::min(),
                          std::numeric_limits<int16_t>::max());
  case 32:
    return std::make_pair(std::numeric_limits<int32_t>::min(),
                          std::numeric_limits<int32_t>::max());
  case 64:
    return std::make_pair(std::numeric_limits<int64_t>::min(),
                          std::numeric_limits<int64_t>::max());
  default:
    emitError(loc, "Invalid quantization bit width (must be 8, 16, 32, or 64). "
                   "Returning null bounds.");
    return std::nullopt;
  }
}

// Convert quantized type to use the target integer bit width.
static Type convertQuantizedType(quant::QuantizedType quantType,
                                 IntegerType targetIntType, Location loc,
                                 std::shared_ptr<bool> conversionFailed) {
  // Use the target integer type's min and max values.
  std::optional<std::pair<int64_t, int64_t>> storageTypeMinMax =
      getStorageTypeMinMax(targetIntType, loc);
  if (!storageTypeMinMax) {
    *conversionFailed = true;
    return nullptr;
  }
  int64_t storageTypeMin = storageTypeMinMax->first;
  int64_t storageTypeMax = storageTypeMinMax->second;

  if (targetIntType.getWidth() <
      quantType.getStorageType().getIntOrFloatBitWidth()) {
    emitError(loc, "Target integer type of width " +
                       std::to_string(targetIntType.getWidth()) +
                       " is smaller than quantized type of width " +
                       std::to_string(
                           quantType.getStorageType().getIntOrFloatBitWidth()) +
                       ". Out of range.");
    *conversionFailed = true;
    return nullptr;
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
  return quantType;
}

class QuantDataTypeConverter : public TypeConverter {
private:
  IntegerType targetIntType;
  Location loc;
  std::shared_ptr<bool> conversionFailed;

public:
  QuantDataTypeConverter(IntegerType targetIntType, Location loc,
                         std::shared_ptr<bool> conversionFailed)
      : targetIntType(targetIntType), loc(loc),
        conversionFailed(conversionFailed) {
    addConversion([](Type type) -> Type { return type; });

    addConversion([targetIntType, loc, conversionFailed](
                      RankedTensorType type) -> std::optional<Type> {
      Type elementType = type.getElementType();
      if (!mlir::isa<quant::QuantizedType>(elementType)) {
        return std::nullopt;
      }
      quant::QuantizedType quantElementType =
          mlir::cast<quant::QuantizedType>(elementType);
      Type newElementType = convertQuantizedType(
          quantElementType, targetIntType, loc, conversionFailed);
      if (!newElementType) {
        return std::nullopt;
      }
      if (newElementType == elementType) {
        return type;
      }
      return RankedTensorType::get(type.getShape(), newElementType,
                                   type.getEncoding());
    });

    addConversion([targetIntType, loc, conversionFailed](
                      quant::QuantizedType type) -> std::optional<Type> {
      return convertQuantizedType(type, targetIntType, loc, conversionFailed);
    });
  }
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
    std::shared_ptr<bool> conversionFailed = std::make_shared<bool>(false);
    QuantDataTypeConverter converter(targetIntType, getOperation()->getLoc(),
                                     conversionFailed);
    patterns.add<UniformTypeRewriter>(converter, context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        *conversionFailed) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TTIRDialect>();
    registry.insert<TTDialect>();
    registry.insert<quant::QuantDialect>();
  }
};

} // namespace

} // namespace mlir::tt::ttir
