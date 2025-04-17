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
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <limits>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRQUANTDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Helper function to convert integer bit width to IntegerType.
static IntegerType getIntegerTypeFromBitWidth(MLIRContext *context,
                                              uint32_t bitWidth) {
  // Default to int32 if 0 or invalid.
  switch (bitWidth) {
  case 8:
  case 16:
  case 32:
  case 64:
    return IntegerType::get(context, bitWidth, IntegerType::Signed);
  default:
    // Fallback to int32 if the bit width is not supported.
    return IntegerType::get(context, 32, IntegerType::Signed);
  }
}

static std::pair<int64_t, int64_t> getStorageTypeMinMax(IntegerType intType) {
  switch (intType.getWidth()) {
  case 8:
    return {std::numeric_limits<int8_t>::min(),
            std::numeric_limits<int8_t>::max()};
  case 16:
    return {std::numeric_limits<int16_t>::min(),
            std::numeric_limits<int16_t>::max()};
  case 32:
    return {std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max()};
  case 64:
    return {std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max()};
  default:
    // Fallback for unsupported widths.
    return {-(1LL << (intType.getWidth() - 1)),
            (1LL << (intType.getWidth() - 1)) - 1};
  }
}

// Convert quantized type to use the target integer bit width.
static Type convertQuantizedType(quant::QuantizedType quantType,
                                 IntegerType targetIntType) {
  // Use the target integer type's min and max values.
  std::pair<int64_t, int64_t> storageTypeMinMax =
      getStorageTypeMinMax(targetIntType);
  int64_t storageTypeMin = storageTypeMinMax.first;
  int64_t storageTypeMax = storageTypeMinMax.second;

  assert(targetIntType.getWidth() >=
             quantType.getStorageType().getIntOrFloatBitWidth() &&
         "Target integer type is smaller than quantized type. Out of range.");

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

public:
  QuantDataTypeConverter(IntegerType targetIntType)
      : targetIntType(targetIntType) {
    addConversion([](Type type) -> Type { return type; });

    addConversion(
        [targetIntType](RankedTensorType type) -> std::optional<Type> {
          Type elementType = type.getElementType();
          if (!isa<quant::QuantizedType>(elementType)) {
            return std::nullopt;
          }
          auto quantElementType = dyn_cast<quant::QuantizedType>(elementType);
          Type newElementType =
              convertQuantizedType(quantElementType, targetIntType);
          if (newElementType == elementType) {
            return type;
          }
          return RankedTensorType::get(type.getShape(), newElementType,
                                       type.getEncoding());
        });

    addConversion([targetIntType](quant::QuantizedType type) -> Type {
      return convertQuantizedType(type, targetIntType);
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

    IntegerType targetIntType =
        getIntegerTypeFromBitWidth(context, targetBitWidth);
    assert(targetIntType &&
           "Invalid quantization bit width (must be 8, 16, 32, or 64).");
    RewritePatternSet patterns(context);
    QuantDataTypeConverter converter(targetIntType);
    patterns.add<UniformTypeRewriter>(converter, context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TTIRDialect>();
    registry.insert<TTDialect>();
    registry.insert<quant::QuantDialect>();
  }
};

} // namespace

} // namespace mlir::tt::ttir
