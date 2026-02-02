// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MRANKNORMALIZATION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Get the minimum rank that tensors should be normalized to.
static constexpr int64_t kMinRank = 2;

/// Promote a shape array to the minimum rank by prepending 1's.
static SmallVector<int32_t> promoteShape(ArrayRef<int32_t> shape) {
  int64_t currentRank = shape.size();
  if (currentRank >= kMinRank) {
    return SmallVector<int32_t>(shape);
  }

  int64_t numOnesToAdd = kMinRank - currentRank;
  SmallVector<int32_t> newShape(numOnesToAdd, 1);
  newShape.append(shape.begin(), shape.end());
  return newShape;
}

/// Promote a tensor type to the minimum rank by prepending 1's.
/// For example: tensor<32xf32> -> tensor<1x32xf32>
///              tensor<f32> -> tensor<1x1xf32>
static RankedTensorType promoteRank(RankedTensorType type) {
  int64_t currentRank = type.getRank();
  if (currentRank >= kMinRank) {
    return type;
  }

  int64_t numOnesToAdd = kMinRank - currentRank;
  SmallVector<int64_t> newShape(numOnesToAdd, 1);
  newShape.append(type.getShape().begin(), type.getShape().end());

  return RankedTensorType::get(newShape, type.getElementType(),
                               type.getEncoding());
}

/// Check if a type needs rank promotion (is a tensor with rank < kMinRank).
static bool needsRankPromotion(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    return tensorType.getRank() < kMinRank;
  }
  return false;
}

/// Promote a type if it's a tensor that needs rank promotion.
static Type promoteTypeIfNeeded(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    if (needsRankPromotion(tensorType)) {
      return promoteRank(tensorType);
    }
  }
  return type;
}

/// Update the value attribute of a constant op to match its promoted result
/// type (same data, new shape). Required because ttir.constant has
/// AllShapesMatch<["value", "result"]>.
static void updateConstantValueAttr(ttir::ConstantOp constantOp) {
  auto valueAttr = mlir::dyn_cast<DenseElementsAttr>(constantOp.getValue());
  if (!valueAttr) {
    return;
  }
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(constantOp.getResult().getType());
  if (!resultType) {
    return;
  }
  auto valueType = mlir::dyn_cast<RankedTensorType>(valueAttr.getType());
  if (!valueType || valueType.getShape() == resultType.getShape()) {
    return;
  }
  // Reshape the value to match the promoted result type (same raw data).
  auto newValue =
      DenseElementsAttr::getFromRawBuffer(resultType, valueAttr.getRawData());
  constantOp.setValueAttr(newValue);
}

/// Update the arange_dimension attribute when the result type is promoted from
/// 1D to 2D (we prepend a dimension, so the range dimension index shifts by 1).
static void updateArangeDimension(ttir::ArangeOp arangeOp) {
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(arangeOp.getResult().getType());
  if (!resultType || resultType.getRank() < kMinRank) {
    return;
  }
  // We only promoted from 1D to 2D; arange_dimension was 0, now must be 1.
  int64_t oldDim = arangeOp.getArangeDimension();
  if (oldDim == 0 && resultType.getRank() == kMinRank) {
    OpBuilder builder(arangeOp.getContext());
    arangeOp.setArangeDimensionAttr(builder.getI64IntegerAttr(1));
  }
}

/// Update the shape attribute of a reshape op to match its promoted result
/// type.
static void updateReshapeShapeAttr(ttir::ReshapeOp reshapeOp) {
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
  if (!resultType) {
    return;
  }

  // Get the current shape attribute.
  ArrayAttr shapeAttr = reshapeOp.getShapeAttr();
  SmallVector<int32_t> currentShape;
  for (Attribute attr : shapeAttr) {
    currentShape.push_back(mlir::cast<IntegerAttr>(attr).getInt());
  }

  // Check if the shape attribute rank matches the result type rank.
  if (static_cast<int64_t>(currentShape.size()) == resultType.getRank()) {
    return;
  }

  // Promote the shape attribute to match the result type rank.
  SmallVector<int32_t> newShape = promoteShape(currentShape);

  // Set the new shape attribute.
  OpBuilder builder(reshapeOp.getContext());
  reshapeOp.setShapeAttr(builder.getI32ArrayAttr(newShape));
}

class D2MRankNormalization
    : public impl::D2MRankNormalizationBase<D2MRankNormalization> {
public:
  using D2MRankNormalizationBase::D2MRankNormalizationBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      // Check if any argument or result types need promotion.
      bool needsPromotion = false;
      for (Type argType : funcOp.getArgumentTypes()) {
        if (needsRankPromotion(argType)) {
          needsPromotion = true;
          break;
        }
      }
      if (!needsPromotion) {
        for (Type resultType : funcOp.getResultTypes()) {
          if (needsRankPromotion(resultType)) {
            needsPromotion = true;
            break;
          }
        }
      }

      if (!needsPromotion) {
        return;
      }

      // For external functions, only update CPU-hoisted declarations.
      // Other external functions should be skipped.
      if (funcOp.isExternal()) {
        if (!ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
          return;
        }
        // Update the function signature for CPU-hoisted declarations.
        SmallVector<Type> newArgTypes;
        for (Type argType : funcOp.getArgumentTypes()) {
          newArgTypes.push_back(promoteTypeIfNeeded(argType));
        }

        SmallVector<Type> newResultTypes;
        for (Type resultType : funcOp.getResultTypes()) {
          newResultTypes.push_back(promoteTypeIfNeeded(resultType));
        }

        auto newFuncType =
            FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);
        funcOp.setType(newFuncType);
        return;
      }

      // Step 1: Update function argument types.
      Block &entryBlock = funcOp.getBody().front();
      for (BlockArgument arg : entryBlock.getArguments()) {
        Type newType = promoteTypeIfNeeded(arg.getType());
        if (newType != arg.getType()) {
          arg.setType(newType);
        }
      }

      // Step 2: Update all operations in the function body.
      // Walk through all ops and update their result types.
      funcOp.walk([&](Operation *op) {
        // Skip the function itself.
        if (mlir::isa<func::FuncOp>(op)) {
          return;
        }

        // Update result types.
        for (OpResult result : op->getResults()) {
          Type newType = promoteTypeIfNeeded(result.getType());
          if (newType != result.getType()) {
            result.setType(newType);
          }
        }

        // Handle ops with shape attributes that need to match their result
        // type.
        if (auto reshapeOp = mlir::dyn_cast<ttir::ReshapeOp>(op)) {
          updateReshapeShapeAttr(reshapeOp);
        }
        // Handle constant: value attribute shape must match promoted result.
        if (auto constantOp = mlir::dyn_cast<ttir::ConstantOp>(op)) {
          updateConstantValueAttr(constantOp);
        }
        // Handle arange: arange_dimension must refer to the range dim after
        // prepending a dimension (0 -> 1 when promoting 1D to 2D).
        if (auto arangeOp = mlir::dyn_cast<ttir::ArangeOp>(op)) {
          updateArangeDimension(arangeOp);
        }
      });

      // Step 3: Update function signature to match the new types.
      SmallVector<Type> newArgTypes;
      for (Type argType : funcOp.getArgumentTypes()) {
        newArgTypes.push_back(promoteTypeIfNeeded(argType));
      }

      SmallVector<Type> newResultTypes;
      for (Type resultType : funcOp.getResultTypes()) {
        newResultTypes.push_back(promoteTypeIfNeeded(resultType));
      }

      auto newFuncType =
          FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);
      funcOp.setType(newFuncType);
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
