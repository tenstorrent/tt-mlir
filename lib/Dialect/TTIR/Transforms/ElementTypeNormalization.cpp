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
class ElementTypeConverter : public TypeConverter {
public:
  ElementTypeConverter() {
    addConversion([](RankedTensorType type) -> std::optional<RankedTensorType> {
      Type elementType = type.getElementType();

      // Skip quantized types - don't modify them.
      if (isa<quant::QuantizedType>(elementType)) {
        return type;
      }

      elementType = mlir::tt::ttcore::toTTMLIRSupportedDataType(elementType);
      if (!elementType) {
        return {};
      }

      return RankedTensorType::get(type.getShape(), elementType,
                                   type.getEncoding());
    });
  }
};

// TODO(milant): https://github.com/tenstorrent/tt-mlir/issues/2847
// We need more sophisticated rewriting of the constant op,
// since current logic doesn't rely on the type converter and
// doesn't support DenseResourceElementsAttr conversion.
class ConstantOpAttrRewriter : public OpRewritePattern<tt::ttir::ConstantOp> {
public:
  using OpRewritePattern<tt::ttir::ConstantOp>::OpRewritePattern;

  ConstantOpAttrRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : OpRewritePattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(tt::ttir::ConstantOp op,
                                PatternRewriter &rewriter) const override {
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
  TypeConverter converter;

  mlir::ElementsAttr rebuildElementsAttr(mlir::ElementsAttr attr) const {
    auto elementType = attr.getElementType();
    auto newType = mlir::cast<mlir::ShapedType>(
        converter.convertType(attr.getShapedType()));

    // Skip rewriting if type is already supported or if the attribute is
    // DenseResourceElementsAttr (conversion not supported yet).
    if (newType.getElementType() == elementType ||
        isa<DenseResourceElementsAttr>(attr)) {
      return attr;
    }

    if (isa<IntegerType>(elementType)) {
      return rebuildIntAttr(attr, newType);
    }
    if (isa<FloatType>(elementType)) {
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

    RewritePatternSet patterns(&getContext());
    patterns.add<UniformTypeRewriter>(converter, &getContext());
    patterns.add<ConstantOpAttrRewriter>(converter, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

private:
  ElementTypeConverter converter;

  bool isLegal(Type type) const {
    return converter.convertType(type) != nullptr;
  }

  bool isLegal(Operation *op) const {
    auto isTypeLegal = [this](Type type) { return isLegal(type); };

    // Special handling for function signature
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      return llvm::all_of(funcOp.getArgumentTypes(), isTypeLegal) &&
             llvm::all_of(funcOp.getResultTypes(), isTypeLegal);
    }

    // Default case: check operand and result types
    return llvm::all_of(op->getOperandTypes(), isTypeLegal) &&
           llvm::all_of(op->getResultTypes(), isTypeLegal);
  }

  // Check that all operations in module are using supported types.
  bool checkSupportedTypes() {
    ModuleOp moduleOp = getOperation();
    WalkResult walkResult =
        moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (!isLegal(op)) {
            op->emitOpError("Unsupported type.");
            return WalkResult::interrupt();
          }

          return WalkResult::advance();
        });

    return !walkResult.wasInterrupted();
  }
};
} // namespace

} // namespace mlir::tt::ttir
