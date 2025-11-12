// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>

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
class WeightsTypeCast : public mlir::ConversionPattern {
public:
  WeightsTypeCast(const mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {

    // Only match matmul, conv2d, convolution, dot_general, and linear
    // operations
    bool isMatmul = llvm::isa<ttir::MatmulOp>(op);
    bool isConv2d = llvm::isa<ttir::Conv2dOp>(op);
    bool isConvolution = llvm::isa<ttir::ConvolutionOp>(op);
    bool isDotGeneral = llvm::isa<ttir::DotGeneralOp>(op);
    bool isLinear = llvm::isa<ttir::LinearOp>(op);
    if (!isMatmul && !isConv2d && !isConvolution && !isDotGeneral &&
        !isLinear) {
      return rewriter.notifyMatchFailure(
          op, "not a matmul, conv2d, convolution, dot_general, or linear op");
    }

    // Determine which operands are weights
    // For matmul: operand 1 (b) is typically the weight
    // For conv2d and convolution: operand 1 (weight) and optionally operand 2
    // (bias) are weights
    // For dot_general: operand 1 (rhs) is the weight
    // For linear: operand 1 (b) is the weight and optionally operand 2 (bias)
    llvm::SmallVector<unsigned> weightIndices;
    if (isMatmul) {
      weightIndices.push_back(1); // b operand
    } else if (isDotGeneral) {
      weightIndices.push_back(1); // rhs operand
    } else if (isLinear) {
      auto linearOp = llvm::cast<ttir::LinearOp>(op);
      weightIndices.push_back(1); // b operand
      // Check if bias exists using the named accessor
      if (linearOp.getBias()) {
        // Find the index of the bias operand
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          if (op->getOperand(i) == linearOp.getBias()) {
            weightIndices.push_back(i);
            break;
          }
        }
      }
    } else if (isConv2d) {
      auto conv2dOp = llvm::cast<ttir::Conv2dOp>(op);
      weightIndices.push_back(1); // weight operand
      // Check if bias exists using the named accessor
      if (conv2dOp.getBias()) {
        // Find the index of the bias operand
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          if (op->getOperand(i) == conv2dOp.getBias()) {
            weightIndices.push_back(i);
            break;
          }
        }
      }
    } else if (isConvolution) {
      auto convolutionOp = llvm::cast<ttir::ConvolutionOp>(op);
      weightIndices.push_back(1); // weight operand
      // Check if bias exists using the named accessor
      if (convolutionOp.getBias()) {
        // Find the index of the bias operand
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          if (op->getOperand(i) == convolutionOp.getBias()) {
            weightIndices.push_back(i);
            break;
          }
        }
      }
    }

    // Check if any weight operand has float type (bf16 or f32)
    bool hasFloatWeights = false;
    for (unsigned idx : weightIndices) {
      if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
              op->getOperand(idx).getType())) {
        mlir::Type elType = tensorType.getElementType();
        if (mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
          hasFloatWeights = true;
          break;
        }
      }
    }

    if (!hasFloatWeights) {
      return rewriter.notifyMatchFailure(op, "no float weight operands");
    }

    // Create new operands with converted weights
    llvm::SmallVector<mlir::Value> newOperands(op->getOperands());

    for (unsigned idx : weightIndices) {
      mlir::Value weight = op->getOperand(idx);
      auto weightType =
          mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
      if (!weightType) {
        continue;
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
        mlir::Value convertedWeight =
            ttir::utils::createDPSOp<ttir::TypecastOp>(rewriter, op->getLoc(),
                                                       targetType, weight);
        newOperands[idx] = convertedWeight;
      }
    }

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

      // Helper lambda to check if weight operands have bfp8 typecasts
      auto hasWeightsWithBfp8 = [](mlir::Operation *op,
                                   llvm::ArrayRef<unsigned> weightIndices) {
        for (unsigned idx : weightIndices) {
          if (idx >= op->getNumOperands()) {
            continue;
          }
          mlir::Value weight = op->getOperand(idx);
          // Check if the weight operand is produced by a typecast to bfp8
          if (auto typecastOp = weight.getDefiningOp<ttir::TypecastOp>()) {
            auto resultType =
                mlir::dyn_cast<mlir::RankedTensorType>(typecastOp.getType());
            if (resultType) {
              auto tileType =
                  mlir::dyn_cast<ttcore::TileType>(resultType.getElementType());
              if (tileType &&
                  tileType.getDataType() == ttcore::DataType::BFP_BFloat8) {
                return true; // Found a weight with bfp8 typecast
              }
            }
          }
        }
        return false;
      };

      // Mark matmul ops as illegal unless their weights already have bfp8
      // typecasts
      target.addDynamicallyLegalOp<ttir::MatmulOp>(
          [hasWeightsWithBfp8](ttir::MatmulOp op) {
            return hasWeightsWithBfp8(op.getOperation(), {1});
          });

      // Mark conv2d ops as illegal unless their weights already have bfp8
      // typecasts
      target.addDynamicallyLegalOp<ttir::Conv2dOp>(
          [hasWeightsWithBfp8](ttir::Conv2dOp op) {
            llvm::SmallVector<unsigned> weightIndices = {1};
            // Use named accessor to check for bias
            if (op.getBias()) {
              // Find the index of the bias operand
              for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                if (op->getOperand(i) == op.getBias()) {
                  weightIndices.push_back(i);
                  break;
                }
              }
            }
            return hasWeightsWithBfp8(op.getOperation(), weightIndices);
          });

      // Mark convolution ops as illegal unless their weights already have bfp8
      // typecasts
      target.addDynamicallyLegalOp<ttir::ConvolutionOp>(
          [hasWeightsWithBfp8](ttir::ConvolutionOp op) {
            llvm::SmallVector<unsigned> weightIndices = {1};
            // Use named accessor to check for bias
            if (op.getBias()) {
              // Find the index of the bias operand
              for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                if (op->getOperand(i) == op.getBias()) {
                  weightIndices.push_back(i);
                  break;
                }
              }
            }
            return hasWeightsWithBfp8(op.getOperation(), weightIndices);
          });

      // Mark dot_general ops as illegal unless their weights already have bfp8
      // typecasts
      target.addDynamicallyLegalOp<ttir::DotGeneralOp>(
          [hasWeightsWithBfp8](ttir::DotGeneralOp op) {
            return hasWeightsWithBfp8(op.getOperation(), {1});
          });

      // Mark linear ops as illegal unless their weights already have bfp8
      // typecasts
      target.addDynamicallyLegalOp<ttir::LinearOp>(
          [hasWeightsWithBfp8](ttir::LinearOp op) {
            llvm::SmallVector<unsigned> weightIndices = {1};
            // Use named accessor to check for bias
            if (op.getBias()) {
              // Find the index of the bias operand
              for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                if (op->getOperand(i) == op.getBias()) {
                  weightIndices.push_back(i);
                  break;
                }
              }
            }
            return hasWeightsWithBfp8(op.getOperation(), weightIndices);
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
