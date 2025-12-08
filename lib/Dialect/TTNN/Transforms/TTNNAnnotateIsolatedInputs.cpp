// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

using namespace mlir;
using namespace mlir::tt::ttnn;

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNANNOTATEISOLATEDINPUTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttnn

namespace {

/// Extract shape from a tensor type as an array of integers
SmallVector<int64_t> extractShape(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return llvm::to_vector(tensorType.getShape());
  }
  return {};
}

/// Extract data type string from a tensor type
std::string extractDType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elementType = tensorType.getElementType();

    if (elementType.isBF16()) {
      return "bf16";
    } else if (elementType.isF32()) {
      return "f32";
    } else if (elementType.isF16()) {
      return "f16";
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      unsigned width = intType.getWidth();
      if (intType.isUnsigned()) {
        return "u" + std::to_string(width);
      } else {
        return "i" + std::to_string(width);
      }
    }
    // For other types, return string representation
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    elementType.print(os);
    return os.str();
  }
  return "unknown";
}

/// Extract layout encoding attribute from tensor type
Attribute extractLayout(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getEncoding();
  }
  return nullptr;
}

/// Find the target operation in the function (assumes single op in isolated func)
Operation* findTargetOp(func::FuncOp funcOp) {
  Operation *targetOp = nullptr;
  funcOp.walk([&](Operation *op) {
    // Skip terminator ops
    if (isa<func::ReturnOp>(op)) {
      return;
    }
    // Skip the function itself
    if (op == funcOp.getOperation()) {
      return;
    }
    // Skip GetDeviceOp
    if (isa<GetDeviceOp>(op)) {
      return;
    }
    // Find TTNN dialect operations
    if (op->getName().getDialect()->getNamespace() == "ttnn") {
      targetOp = op;
    }
  });
  return targetOp;
}

struct TTNNAnnotateIsolatedInputsPass
    : public ::mlir::tt::ttnn::impl::TTNNAnnotateIsolatedInputsBase<
          TTNNAnnotateIsolatedInputsPass> {
  using ::mlir::tt::ttnn::impl::TTNNAnnotateIsolatedInputsBase<
      TTNNAnnotateIsolatedInputsPass>::TTNNAnnotateIsolatedInputsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](func::FuncOp funcOp) {
      // Only process functions marked as isolated
      if (!funcOp->hasAttr("isolated")) {
        return;
      }

      // Find the target operation in the function body
      Operation *targetOp = findTargetOp(funcOp);
      if (!targetOp) {
        return;
      }

      // Extract operation name
      std::string opName = targetOp->getName().getStringRef().str();
      funcOp->setAttr("op_name", builder.getStringAttr(opName));

      // Collect input metadata from function arguments
      SmallVector<Attribute> inputShapes;
      SmallVector<Attribute> inputDTypes;
      SmallVector<Attribute> inputLayouts;

      for (BlockArgument arg : funcOp.getArguments()) {
        Type argType = arg.getType();

        // Extract shape
        auto shape = extractShape(argType);
        if (!shape.empty()) {
          SmallVector<Attribute> shapeAttrs;
          for (int64_t dim : shape) {
            shapeAttrs.push_back(builder.getI64IntegerAttr(dim));
          }
          inputShapes.push_back(builder.getArrayAttr(shapeAttrs));
        } else {
          // Non-tensor type (e.g., device) - add empty array
          inputShapes.push_back(builder.getArrayAttr({}));
        }

        // Extract dtype
        std::string dtype = extractDType(argType);
        inputDTypes.push_back(builder.getStringAttr(dtype));

        // Extract layout
        Attribute layout = extractLayout(argType);
        if (layout) {
          inputLayouts.push_back(layout);
        } else {
          // Non-tensor type - add unit attribute as placeholder
          inputLayouts.push_back(builder.getUnitAttr());
        }
      }

      // Set attributes on the function
      funcOp->setAttr("input_shapes", builder.getArrayAttr(inputShapes));
      funcOp->setAttr("input_dtypes", builder.getArrayAttr(inputDTypes));
      funcOp->setAttr("input_layouts", builder.getArrayAttr(inputLayouts));
    });
  }
};

} // namespace
