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

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRQUANTDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Helper function to convert integer bit width string to integer type.
static Type getIntegerTypeFromString(MLIRContext *context, StringRef bitWidth) {
  // Default to int32 if empty.
  if (bitWidth.empty()) {
    return IntegerType::get(context, 32, IntegerType::Signed);
  }

  struct BitWidthMapping {
    StringRef name;
    unsigned width;
  };

  static const BitWidthMapping mappings[] = {
      {"int8", 8}, {"int16", 16}, {"int32", 32}, {"int64", 64}};

  for (const BitWidthMapping &mapping : mappings) {
    if (bitWidth == mapping.name) {
      return IntegerType::get(context, mapping.width, IntegerType::Signed);
    }
  }

  return nullptr;
}

static std::pair<int64_t, int64_t> getStorageTypeMinMax(IntegerType intType) {
  // Only need to return signed min and max.
  int64_t min = -(1LL << (intType.getWidth() - 1));
  int64_t max = (1LL << (intType.getWidth() - 1)) - 1;
  return {min, max};
}

// Convert quantized type to use the target integer bit width.
static Type convertQuantizedType(Type type, Type targetIntType) {
  if (quant::QuantizedType quantType = dyn_cast<quant::QuantizedType>(type)) {
    // Use the target integer type's min and max values.
    IntegerType intType = cast<IntegerType>(targetIntType);
    std::pair<int64_t, int64_t> storageTypeMinMax =
        getStorageTypeMinMax(intType);
    int64_t storageTypeMin = storageTypeMinMax.first;
    int64_t storageTypeMax = storageTypeMinMax.second;

    assert(intType.getWidth() >=
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
  }
  if (RankedTensorType tensorType = dyn_cast<RankedTensorType>(type)) {
    if (quant::QuantizedType elementType =
            dyn_cast<quant::QuantizedType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(),
          convertQuantizedType(elementType, targetIntType),
          tensorType.getEncoding());
    }
  }
  return type;
}

class QuantDataTypeConverter : public TypeConverter {
private:
  Type targetIntType;

public:
  QuantDataTypeConverter(Type targetIntType) : targetIntType(targetIntType) {
    addConversion([](Type type) -> Type { return type; });

    addConversion([targetIntType](RankedTensorType type) -> Type {
      Type elementType = type.getElementType();
      if (!isa<quant::QuantizedType>(elementType)) {
        return type;
      }
      Type newElementType = convertQuantizedType(elementType, targetIntType);
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

    Type targetIntType = getIntegerTypeFromString(context, targetBitWidth);
    assert(targetIntType &&
           ("Invalid target bit width: " + targetBitWidth).c_str());

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
