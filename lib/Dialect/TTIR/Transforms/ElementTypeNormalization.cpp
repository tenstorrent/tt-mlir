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

    // Match this pattern only if at least one result is bf16.
    bool hasBf16Result = llvm::any_of(op->getResultTypes(), [](mlir::Type t) {
      if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t)) {
        return mlir::isa<mlir::BFloat16Type>(rt.getElementType());
      }
      return false;
    });
    if (!hasBf16Result) {
      return rewriter.notifyMatchFailure(op, "no bf16 results"); // don’t touch
    }

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
        mlir::Type elType = type.getElementType();
        if (mlir::isa<mlir::BFloat16Type>(elType)) {
          // For bf16 tensors, change ONLY the element type to tile<bfp8>.
          return type.clone(ttcore::TileType::get(
              type.getContext(), ttcore::TileType::getDefaultShape(),
              ttcore::DataType::BFP_BFloat8));
        }
        // Pass-through for everything else (f32, i32, existing tile<bfp8>,
        // etc.)
        return type;
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

// This pattern converts weight operands of matmul and conv2d operations
// to bfp8_b type, while leaving activations in high precision.
// This is used for mixed precision training/inference where weights are
// in lower precision but activations remain in higher precision.
// NOTE: We only convert weight tensors, not bias tensors, since bias tensors
// are typically small and don't provide significant memory savings.
class WeightsTypeCast : public mlir::ConversionPattern {
public:
  WeightsTypeCast(const mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {

    // Helper to check if a weight tensor has float type (bf16 or f32)
    auto hasFloatType = [](mlir::Value weight) -> bool {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(weight.getType())) {
        mlir::Type elType = tensorType.getElementType();
        return mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType);
      }
      return false;
    };

    // Helper to convert a weight to bfp8
    auto convertWeightToBfp8 =
        [&](mlir::Value weight) -> std::optional<mlir::Value> {
      auto weightType =
          mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
      if (!weightType) {
        return std::nullopt;
      }

      mlir::Type elType = weightType.getElementType();
      // Convert float types (bf16 or f32) to bfp8_b
      if (mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
        // Create target type: tile<bfp8_b>
        mlir::Type bfp8Type = ttcore::TileType::get(
            op->getContext(), ttcore::TileType::getDefaultShape(),
            ttcore::DataType::BFP_BFloat8);
        mlir::RankedTensorType targetType = weightType.clone(bfp8Type);

        // Insert typecast operation
        return ttir::utils::createDPSOp<ttir::TypecastOp>(
            rewriter, op->getLoc(), targetType, weight);
      }
      return std::nullopt;
    };

    // Get weight operand using named accessors
    mlir::Value weightOperand;
    unsigned weightOperandIdx = 0;

    if (auto matmulOp = llvm::dyn_cast<ttir::MatmulOp>(op)) {
      weightOperand = matmulOp.getB();
      weightOperandIdx = 1;
    } else if (auto conv2dOp = llvm::dyn_cast<ttir::Conv2dOp>(op)) {
      weightOperand = conv2dOp.getWeight();
      weightOperandIdx = 1;
    } else if (auto convTranspose2dOp =
                   llvm::dyn_cast<ttir::ConvTranspose2dOp>(op)) {
      weightOperand = convTranspose2dOp.getWeight();
      weightOperandIdx = 1;
    } else if (auto convolutionOp = llvm::dyn_cast<ttir::ConvolutionOp>(op)) {
      weightOperand = convolutionOp.getWeight();
      weightOperandIdx = 1;
    } else if (auto dotGeneralOp = llvm::dyn_cast<ttir::DotGeneralOp>(op)) {
      weightOperand = dotGeneralOp.getRhs();
      weightOperandIdx = 1;
    } else if (auto linearOp = llvm::dyn_cast<ttir::LinearOp>(op)) {
      weightOperand = linearOp.getB();
      weightOperandIdx = 1;
    } else {
      return rewriter.notifyMatchFailure(
          op, "not a supported operation with weights");
    }

    // Check if weight has float type
    if (!hasFloatType(weightOperand)) {
      return rewriter.notifyMatchFailure(op, "weight is not float type");
    }

    // Convert the weight to bfp8
    auto convertedWeight = convertWeightToBfp8(weightOperand);
    if (!convertedWeight) {
      return rewriter.notifyMatchFailure(op, "failed to convert weight");
    }

    // Create new operands with converted weight
    llvm::SmallVector<mlir::Value> newOperands(op->getOperands());
    newOperands[weightOperandIdx] = *convertedWeight;

    // Clone the operation with new operands
    mlir::Operation *newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        newOperands, op->getResultTypes(), op->getAttrs());

    // Replace the old operation
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

  struct WeightsTypeConverter : mlir::TypeConverter {
    WeightsTypeConverter() {
      // Identity conversion - we don't change types, we insert casts
      addConversion([](mlir::Type type) { return type; });
    }
  };
};

using WeightsTypeConverter = WeightsTypeCast::WeightsTypeConverter;

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

    if (experimentalBfp8Weights) {
      mlir::ConversionTarget target(getContext());
      WeightsTypeConverter weightsConverter;

      // Mark all ops as legal by default
      target.markUnknownOpDynamicallyLegal(
          [](mlir::Operation *op) { return true; });

      // Helper lambda to check if a weight operand is already bfp8
      // This checks both:
      // 1. If the weight is produced by a typecast to bfp8
      // 2. If the weight tensor itself already has bfp8 element type
      auto isWeightBfp8 = [](mlir::Value weight) -> bool {
        auto weightType =
            mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
        if (!weightType) {
          return false;
        }

        // Check if the element type is already bfp8
        if (auto tileType =
                mlir::dyn_cast<ttcore::TileType>(weightType.getElementType())) {
          if (tileType.getDataType() == ttcore::DataType::BFP_BFloat8) {
            return true;
          }
        }

        // Check if the weight is produced by a typecast to bfp8
        if (auto typecastOp = weight.getDefiningOp<ttir::TypecastOp>()) {
          auto resultType =
              mlir::dyn_cast<mlir::RankedTensorType>(typecastOp.getType());
          if (resultType) {
            auto tileType =
                mlir::dyn_cast<ttcore::TileType>(resultType.getElementType());
            if (tileType &&
                tileType.getDataType() == ttcore::DataType::BFP_BFloat8) {
              return true;
            }
          }
        }

        return false;
      };

      // Mark matmul ops as illegal unless their weights already have bfp8
      target.addDynamicallyLegalOp<ttir::MatmulOp>(
          [&isWeightBfp8](ttir::MatmulOp op) {
            return isWeightBfp8(op.getB());
          });

      // Mark conv2d ops as illegal unless their weights already have bfp8
      target.addDynamicallyLegalOp<ttir::Conv2dOp>(
          [&isWeightBfp8](ttir::Conv2dOp op) {
            return isWeightBfp8(op.getWeight());
          });

      // Mark conv_transpose2d ops as illegal unless their weights already have
      // bfp8
      target.addDynamicallyLegalOp<ttir::ConvTranspose2dOp>(
          [&isWeightBfp8](ttir::ConvTranspose2dOp op) {
            return isWeightBfp8(op.getWeight());
          });

      // Mark convolution ops as illegal unless their weights already have bfp8
      target.addDynamicallyLegalOp<ttir::ConvolutionOp>(
          [&isWeightBfp8](ttir::ConvolutionOp op) {
            return isWeightBfp8(op.getWeight());
          });

      // Mark dot_general ops as illegal unless their weights already have bfp8
      target.addDynamicallyLegalOp<ttir::DotGeneralOp>(
          [&isWeightBfp8](ttir::DotGeneralOp op) {
            return isWeightBfp8(op.getRhs());
          });

      // Mark linear ops as illegal unless their weights already have bfp8
      target.addDynamicallyLegalOp<ttir::LinearOp>(
          [&isWeightBfp8](ttir::LinearOp op) {
            return isWeightBfp8(op.getB());
          });

      mlir::RewritePatternSet weightsPatterns(&getContext());
      weightsPatterns.add<WeightsTypeCast>(weightsConverter, &getContext());

      if (failed(mlir::applyPartialConversion(getOperation(), target,
                                              std::move(weightsPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    if (enableBfp8Conversion) {
      mlir::ConversionTarget target(getContext());
      FuncBodyTypeConverter funcBodyConverter;
      target.markUnknownOpDynamicallyLegal([](mlir::Operation *op) {
        // Non-TTIR ops are not policed by this pass → legal.
        if (!mlir::isa<TTIRDialect>(op->getDialect())) {
          return true;
        }
        // TTIR ops are illegal iff any result is a RankedTensorType with bf16
        // element type.
        bool hasBf16Result =
            llvm::any_of(op->getResultTypes(), [](mlir::Type t) {
              if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t)) {
                return mlir::isa<mlir::BFloat16Type>(rt.getElementType());
              }
              return false; // non-tensors are fine
            });
        // Legal if there are no bf16 results; illegal if there are.
        return !hasBf16Result;
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
