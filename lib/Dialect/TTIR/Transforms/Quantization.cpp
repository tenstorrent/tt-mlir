// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRQUANTDATATYPECONVERSIONPASS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

// Helper function to convert integer bit width string to integer type.
static Type getIntegerTypeFromString(MLIRContext *context, StringRef bitWidth) {
  if (bitWidth == "int8")
    return IntegerType::get(context, 8, IntegerType::Signed);
  if (bitWidth == "int16")
    return IntegerType::get(context, 16, IntegerType::Signed);
  if (bitWidth == "int32")
    return IntegerType::get(context, 32, IntegerType::Signed);
  if (bitWidth == "int64")
    return IntegerType::get(context, 64, IntegerType::Signed);

  // Default to int32.
  return IntegerType::get(context, 32, IntegerType::Signed);
}

static std::pair<int64_t, int64_t> getStorageTypeMinMax(IntegerType intType) {
  // Only need to return signed min and max.
  int64_t min = -(1LL << (intType.getWidth() - 1));
  int64_t max = (1LL << (intType.getWidth() - 1)) - 1;
  return {min, max};
}

// Convert quantized type to use the target integer bit width.
static Type convertQuantizedType(Type type, Type targetIntType) {
  if (auto quantType = mlir::dyn_cast<quant::QuantizedType>(type)) {
    // Use the target integer type's min and max values.
    std::pair<int64_t, int64_t> storageTypeMinMax =
        getStorageTypeMinMax(mlir::cast<IntegerType>(targetIntType));
    int64_t storageTypeMin = storageTypeMinMax.first;
    int64_t storageTypeMax = storageTypeMinMax.second;

    if (auto uniformType =
            mlir::dyn_cast<quant::UniformQuantizedType>(quantType)) {
      return quant::UniformQuantizedType::get(
          uniformType.getFlags(), targetIntType, uniformType.getExpressedType(),
          uniformType.getScale(), uniformType.getZeroPoint(), storageTypeMin,
          storageTypeMax);
    } else if (auto perAxisType =
                   mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(
                       quantType)) {
      return quant::UniformQuantizedPerAxisType::get(
          perAxisType.getFlags(), targetIntType, perAxisType.getExpressedType(),
          perAxisType.getScales(), perAxisType.getZeroPoints(),
          perAxisType.getQuantizedDimension(), storageTypeMin, storageTypeMax);
    }
  } else if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    if (auto elementType =
            mlir::dyn_cast<quant::QuantizedType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(),
          convertQuantizedType(elementType, targetIntType));
    }
  }
  return type;
}

class QuantOpTypeConverterBase {
protected:
  Type targetIntType;
  MLIRContext *context;

  QuantOpTypeConverterBase(MLIRContext *context, Type targetIntType)
      : targetIntType(targetIntType), context(context) {}

  // Helper to create a new empty tensor with the given type.
  Value createEmptyTensor(Location loc, Type tensorType,
                          PatternRewriter &rewriter) const {
    RankedTensorType rankedTensorType =
        mlir::cast<RankedTensorType>(tensorType);
    return rewriter.create<ttir::EmptyOp>(loc, rankedTensorType.getShape(),
                                          rankedTensorType.getElementType(),
                                          rankedTensorType.getEncoding());
  }

  // Helper to convert a type to use the target integer bit width.
  Type convertType(Type type) const {
    return convertQuantizedType(type, targetIntType);
  }
};

class QuantizeOpTypeConverter : public QuantOpTypeConverterBase,
                                public OpRewritePattern<ttir::QuantizeOp> {
public:
  QuantizeOpTypeConverter(MLIRContext *context, Type targetIntType)
      : QuantOpTypeConverterBase(context, targetIntType),
        OpRewritePattern<ttir::QuantizeOp>(context) {}

  LogicalResult matchAndRewrite(ttir::QuantizeOp op,
                                PatternRewriter &rewriter) const override {
    // Convert the result types.
    Type resultType = convertType(op.getResult().getType());
    Type outputType = convertType(op.getOutput().getType());

    if (resultType == op.getResult().getType())
      return failure();

    // Create a new empty tensor.
    Value newOutputOp = createEmptyTensor(op.getLoc(), outputType, rewriter);

    // Create a new quantize operation.
    ttir::QuantizeOp newOp = rewriter.create<ttir::QuantizeOp>(
        op.getLoc(), resultType, op.getInput(), newOutputOp);

    // Replace the old operation with the new one.
    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

class DequantizeOpTypeConverter : public QuantOpTypeConverterBase,
                                  public OpRewritePattern<ttir::DequantizeOp> {
public:
  DequantizeOpTypeConverter(MLIRContext *context, Type targetIntType)
      : QuantOpTypeConverterBase(context, targetIntType),
        OpRewritePattern<ttir::DequantizeOp>(context) {}

  LogicalResult matchAndRewrite(ttir::DequantizeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Type inputType = input.getType();
    Type convertedType = convertType(inputType);

    if (convertedType == inputType)
      return failure();

    ttir::DequantizeOp newOp = rewriter.create<ttir::DequantizeOp>(
        op.getLoc(), op.getResult().getType(), input, op.getOutput());

    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

class RequantizeOpTypeConverter : public QuantOpTypeConverterBase,
                                  public OpRewritePattern<ttir::RequantizeOp> {
public:
  RequantizeOpTypeConverter(MLIRContext *context, Type targetIntType)
      : QuantOpTypeConverterBase(context, targetIntType),
        OpRewritePattern<ttir::RequantizeOp>(context) {}

  LogicalResult matchAndRewrite(ttir::RequantizeOp op,
                                PatternRewriter &rewriter) const override {
    // Convert the types.
    Type resultType = convertType(op.getResult().getType());
    Type inputType = convertType(op.getInput().getType());
    Type outputType = convertType(op.getOutput().getType());

    if (resultType == op.getResult().getType() &&
        inputType == op.getInput().getType())
      return failure();

    // Create a new output tensor.
    Value newOutputOp = createEmptyTensor(op.getLoc(), outputType, rewriter);

    // Create a new requantize operation.
    ttir::RequantizeOp newOp = rewriter.create<ttir::RequantizeOp>(
        op.getLoc(), resultType, op.getInput(), newOutputOp);

    // Replace the old operation with the new one
    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

class TTIRQuantDataTypeConversionPass
    : public impl::TTIRQuantDataTypeConversionPassBase<
          TTIRQuantDataTypeConversionPass> {
public:
  TTIRQuantDataTypeConversionPass() = default;
  TTIRQuantDataTypeConversionPass(
      TTIRQuantDataTypeConversionPassOptions options) {
    targetBitWidth = options.targetBitWidth;
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    Type targetIntType = getIntegerTypeFromString(context, targetBitWidth);

    module.walk([&](func::FuncOp funcOp) {
      bool needsUpdate = false;
      SmallVector<Type, 4> newInputTypes;
      SmallVector<Type, 4> newResultTypes;

      for (Type inputType : funcOp.getFunctionType().getInputs()) {
        Type newType = convertQuantizedType(inputType, targetIntType);
        newInputTypes.push_back(newType);
        if (newType != inputType) {
          needsUpdate = true;
        }
      }

      for (Type resultType : funcOp.getFunctionType().getResults()) {
        Type newType = convertQuantizedType(resultType, targetIntType);
        newResultTypes.push_back(newType);
        if (newType != resultType) {
          needsUpdate = true;
        }
      }

      if (!needsUpdate) {
        return;
      }

      FunctionType newFuncType =
          FunctionType::get(context, newInputTypes, newResultTypes);

      funcOp.setType(newFuncType);

      if (!funcOp.empty()) {
        Block &entryBlock = funcOp.getBody().front();

        for (unsigned i = 0;
             i < newInputTypes.size() && i < entryBlock.getNumArguments();
             ++i) {
          BlockArgument arg = entryBlock.getArgument(i);
          if (arg.getType() != newInputTypes[i]) {
            arg.setType(newInputTypes[i]);
          }
        }
      }
    });

    RewritePatternSet patterns(context);
    patterns.add<QuantizeOpTypeConverter>(context, targetIntType);
    patterns.add<RequantizeOpTypeConverter>(context, targetIntType);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::quant::QuantDialect>();
  }
};

} // namespace mlir::tt::ttir
