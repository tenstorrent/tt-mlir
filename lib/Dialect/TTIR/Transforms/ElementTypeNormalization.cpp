// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_ELEMENTTYPENORMALIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class ElementTypeConverter : public mlir::TypeConverter {
public:
  void init(bool enableBfp8Type) {
    addConversion([enableBfp8Type](mlir::RankedTensorType type)
                      -> std::optional<mlir::RankedTensorType> {
      Type elementType = type.getElementType();

      // Skip quantized types - don't modify them.
      if (mlir::isa<quant::QuantizedType>(elementType)) {
        return type;
      }

      if (mlir::isa<ttcore::BFloat8BType>(elementType)) {
        return type;
      }

      if (enableBfp8Type && mlir::isa<mlir::BFloat16Type>(elementType)) {
        return mlir::RankedTensorType::get(
            type.getShape(), ttcore::BFloat8BType::get(type.getContext()),
            type.getEncoding());
      }

      elementType = mlir::tt::ttcore::toTTMLIRSupportedDataType(elementType);
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
        mlir::isa<mlir::DenseResourceElementsAttr>(attr)) {
      return attr;
    }

    if (mlir::isa<mlir::IntegerType>(elementType)) {
      return rebuildIntAttr(attr, newType);
    }
    if (mlir::isa<mlir::FloatType>(elementType)) {
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

class ElementTypeNormalization
    : public impl::ElementTypeNormalizationBase<ElementTypeNormalization> {
  using impl::ElementTypeNormalizationBase<
      ElementTypeNormalization>::ElementTypeNormalizationBase;

public:
  void runOnOperation() final {
    // Command line options are parsed after the pass is created
    // so we can't initialize the converter in the constructor.
    converter.init(enableBfp8Type);

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
  }

private:
  bool isLegal(mlir::Type type) const {
    return converter.convertType(type) != nullptr;
  }

  bool isLegal(mlir::Operation *op) const {
    auto isTypeLegal = [this](mlir::Type type) { return isLegal(type); };

    // Special handling for function signature
    if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
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
        moduleOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
          if (!isLegal(op)) {
            op->emitOpError("Unsupported type.");
            return mlir::WalkResult::interrupt();
          }

          return mlir::WalkResult::advance();
        });

    return !walkResult.wasInterrupted();
  }

  ElementTypeConverter converter;
};
} // namespace

} // namespace mlir::tt::ttir
