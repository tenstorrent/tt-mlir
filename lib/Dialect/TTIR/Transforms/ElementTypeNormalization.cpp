// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_ELEMENTTYPENORMALIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class ElementTypeConverter : public TypeConverter {
public:
  ElementTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> { return type; });
    addConversion(
        [](mlir::RankedTensorType type) -> std::optional<RankedTensorType> {
          Type elementType = type.getElementType();

          // Skip quantized types - don't modify them.
          if (mlir::isa<quant::QuantizedType>(elementType)) {
            return type;
          }

          // Preserve tensor<!ttcore.tile<…>>: lowering tile to scalar f32
          // breaks L1 sharded layouts and runtime CB sizing.
          if (mlir::isa<ttcore::TileType>(elementType)) {
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
    GreedyRewriteConfig config;
    config.enableFolding(false);
    patterns.add<UniformTypeRewriter>(converter, &getContext());
    patterns.add<ConstantOpAttrRewriter>(converter, &getContext());
    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns),
                                           config))) {
      signalPassFailure();
      return;
    }

    // After type normalization, update ttcore.local_shape annotations on
    // function results to match the new element types. ElementTypeConverter
    // may have downgraded f64→f32 or i64→i32, leaving stale local_shape
    // attrs that cause PJRT buffer size mismatches at runtime.
    mlir::ModuleOp moduleOp = getOperation();
    moduleOp.walk([](func::FuncOp funcOp) {
      MLIRContext *ctx = funcOp.getContext();
      for (unsigned i = 0; i < funcOp.getNumResults(); ++i) {
        auto resultType = mlir::cast<mlir::RankedTensorType>(
            funcOp.getResultTypes()[i]);
        auto resultAttrDict = funcOp.getResultAttrDict(i);
        if (!resultAttrDict)
          continue;

        auto localShapeAttrVal =
            resultAttrDict.get(mlir::tt::ttcore::LocalShapeAttr::name);
        if (!localShapeAttrVal)
          continue;

        auto localShapeAttr =
            mlir::cast<mlir::tt::ttcore::LocalShapeAttr>(localShapeAttrVal);
        auto localShapeType = localShapeAttr.getLocalShape();

        if (localShapeType.getElementType() == resultType.getElementType())
          continue;

        auto newLocalShape = mlir::RankedTensorType::get(
            localShapeType.getShape(), resultType.getElementType(),
            localShapeType.getEncoding());

        llvm::SmallVector<mlir::NamedAttribute> newAttrs(
            resultAttrDict.getValue());
        for (auto &attr : newAttrs) {
          if (attr.getName() == mlir::tt::ttcore::LocalShapeAttr::name) {
            attr = {attr.getName(),
                    mlir::tt::ttcore::LocalShapeAttr::get(ctx, newLocalShape)};
          }
        }
        funcOp.setResultAttrs(i,
                              mlir::DictionaryAttr::get(ctx, newAttrs));
      }
    });
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
