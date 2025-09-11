// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_ELEMENTTYPENORMALIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class ElementTypeConverter : public TypeConverter {
public:
  ElementTypeConverter() {
    addConversion(
        [](mlir::RankedTensorType type) -> std::optional<RankedTensorType> {
          Type elementType = type.getElementType();

          // Skip quantized types - don't modify them.
          if (mlir::isa<quant::QuantizedType>(elementType)) {
            return type;
          }

          elementType =
              mlir::tt::ttcore::toTTMLIRSupportedDataType(elementType);
          if (!elementType) {
            return {};
          }

          return mlir::RankedTensorType::get(type.getShape(), elementType,
                                             type.getEncoding());
        });
  }
};

// TODO(milant): https://github.com/tenstorrent/tt-mlir/issues/2847
// We need more sophisticated rewriting of the constant op,
// since current logic doesn't rely on the type converter and
// doesn't support DenseResourceElementsAttr conversion.
class ConstantOpAttrRewriter
    : public mlir::OpRewritePattern<tt::ttir::ConstantOp> {
public:
  using mlir::OpRewritePattern<tt::ttir::ConstantOp>::OpRewritePattern;

  ConstantOpAttrRewriter(const mlir::TypeConverter &converter,
                         mlir::MLIRContext *ctx)
      : OpRewritePattern(ctx), converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(tt::ttir::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (auto newAttr = rebuildElementsAttr(op.getValue())) {
      if (newAttr == op.getValue()) {
        return failure();
      }

      rewriter.modifyOpInPlace(op, [&]() { op.setValueAttr(newAttr); });
      return success();
    }

    return rewriter.notifyMatchFailure(
        op, "ttir.constant op only supports int or float types.");
  }

private:
  mlir::TypeConverter converter;

  mlir::ElementsAttr rebuildElementsAttr(mlir::ElementsAttr attr) const {
    auto elementType = attr.getElementType();
    auto newType = mlir::cast<mlir::ShapedType>(
        converter.convertType(attr.getShapedType()));

    // Skip rewriting if type is already supported or if the attribute is
    // DenseResourceElementsAttr (conversion not supported yet).
    if (newType.getElementType() == elementType ||
        mlir::isa<DenseResourceElementsAttr>(attr)) {
      return attr;
    }

    if (mlir::isa<IntegerType>(elementType)) {
      return rebuildIntAttr(attr, newType);
    }
    if (mlir::isa<FloatType>(elementType)) {
      return rebuildFloatAttr(attr, newType);
    }

    return nullptr;
  }

  mlir::ElementsAttr rebuildIntAttr(mlir::ElementsAttr attr,
                                    mlir::ShapedType newType) const {
    const size_t bitWidth = attr.getElementType().getIntOrFloatBitWidth();

    // Represent booleans as bfloat16 APFloats
    if (bitWidth == 1) {
      llvm::SmallVector<mlir::APFloat> bf16Values;
      for (bool v : attr.getValues<bool>()) {
        bf16Values.emplace_back(mlir::APFloat::BFloat(), v);
      }

      return mlir::DenseElementsAttr::get(newType, bf16Values);
    }

    llvm::SmallVector<mlir::APInt> intValues;
    for (mlir::APInt v : attr.getValues<mlir::APInt>()) {
      intValues.push_back(bitWidth == 64 ? v.truncSSat(32) : v);
    }

    return mlir::DenseElementsAttr::get(newType, intValues);
  }

  mlir::ElementsAttr rebuildFloatAttr(mlir::ElementsAttr attr,
                                      mlir::ShapedType newType) const {
    const size_t bitWidth = attr.getElementType().getIntOrFloatBitWidth();
    llvm::SmallVector<mlir::APFloat> floatValues;

    if (bitWidth == 64) {
      // Convert f64 -> f32
      for (mlir::APFloat v : attr.getValues<mlir::APFloat>()) {
        float f = static_cast<float>(v.convertToDouble());
        floatValues.emplace_back(f);
      }
    } else {
      floatValues.assign(attr.getValues<mlir::APFloat>().begin(),
                         attr.getValues<mlir::APFloat>().end());
    }

    return mlir::DenseElementsAttr::get(newType, floatValues);
  }
};

// This pattern is currently used only for conversion between bf16 and bfp8.
// In future it will be extended to support other types as well.
//
// This pattern converts types of operations in the function body and
// inserts materialization operations at the begining of the function body
// and at the end of the function body just before the return operation.
// This way we ensure that input and output types of original function
// are preserved while the body is converted to use TTMLIR types.
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

  struct FuncBodyTypeConverter : mlir::TypeConverter {
    FuncBodyTypeConverter() {
      addConversion([](mlir::RankedTensorType type) -> mlir::RankedTensorType {
        mlir::Type elementType = type.getElementType();
        if (!mlir::isa<BFloat16Type>(elementType)) {
          // Allow other element types to pass through unchanged.
          // Explicitly skip conversion for int32 tensors.
          if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
            if (intType.getWidth() == 32) {
              return type;
            }
          }
          return type;
          //assert(mlir::isa<ttcore::TileType>(elementType) &&
          //       "Expected TileType for non-bfloat16 element type.");
          //assert(
          //    mlir::cast<ttcore::TileType>(elementType).getDataType() ==
          //        ttcore::DataType::BFP_BFloat8 &&
          //    "Expected BFP_BFloat8 TileType for non-bfloat16 element type."); 
        }

        return type.clone(ttcore::TileType::get(
            type.getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::DataType::BFP_BFloat8));
      });

      auto materializeFunc = [](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
        mlir::RankedTensorType rankedType =
            mlir::cast<mlir::RankedTensorType>(type);
        return ttir::utils::createDPSOp<ttir::TypecastOp>(builder, loc,
                                                          rankedType, inputs);
      };

      addSourceMaterialization(materializeFunc);
      addTargetMaterialization(materializeFunc);
    }
  };
};

using FuncBodyTypeConverter = FuncBodyTypeCast::FuncBodyTypeConverter;

struct ElementTypeNormalization
    : public impl::ElementTypeNormalizationBase<ElementTypeNormalization> {
  using impl::ElementTypeNormalizationBase<
      ElementTypeNormalization>::ElementTypeNormalizationBase;

  void runOnOperation() final {
    // Check that all types are supported by TTMLIR.
    if (!checkSupportedTypes()) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<UniformTypeRewriter>(converter, &getContext());
    patterns.add<ConstantOpAttrRewriter>(converter, &getContext());
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    if (enableBfp8Conversion) {
      mlir::ConversionTarget target(getContext());
      FuncBodyTypeConverter funcBodyConverter;
      target.markUnknownOpDynamicallyLegal(
          [&funcBodyConverter](mlir::Operation *op) {
            if (!isa<TTIRDialect>(op->getDialect())) {
              return true;
            }
            if (llvm::all_of(op->getResultTypes(),
                             [&funcBodyConverter](mlir::Type type) {
                               return funcBodyConverter.isLegal(type);
                             })) {
              return true;
            }
            return false;
          });

      mlir::RewritePatternSet patterns(&getContext());

      patterns.add<FuncBodyTypeCast>(funcBodyConverter, &getContext());
      if (failed(mlir::applyFullConversion(getOperation(), target,
                                           std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  ElementTypeConverter converter;

  bool isLegal(mlir::Type type) const {
    return converter.convertType(type) != nullptr;
  }

  bool isLegal(mlir::Operation *op) const {
    auto isTypeLegal = [this](mlir::Type type) { return isLegal(type); };

    // Special handling for function signature
    if (auto funcOp = mlir::dyn_cast<func::FuncOp>(op)) {
      return llvm::all_of(funcOp.getArgumentTypes(), isTypeLegal) &&
             llvm::all_of(funcOp.getResultTypes(), isTypeLegal);
    }

    // Default case: check operand and result types
    return llvm::all_of(op->getOperandTypes(), isTypeLegal) &&
           llvm::all_of(op->getResultTypes(), isTypeLegal);
  }

  // Check that all operations in module are using supported types.
  bool checkSupportedTypes() {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::WalkResult walkResult =
        moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (!isLegal(op)) {
            op->emitOpError("Unsupported type.");
            return mlir::WalkResult::interrupt();
          }

          return mlir::WalkResult::advance();
        });

    return !walkResult.wasInterrupted();
  }
};
} // namespace

} // namespace mlir::tt::ttir
