// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GspmdUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::stablehlo {
  #define GEN_PASS_DEF_ANNOTATETENSORMESHPASS
  #include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static Type addEncodingToType(OpBuilder &builder, Type type, Attribute encoding) {
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return mlir::RankedTensorType::get(tensorType.getShape(), tensorType.getElementType(), encoding);
  }

  return type;
}

class AnnotateTensorMeshPass
  : public impl::AnnotateTensorMeshPassBase<
        AnnotateTensorMeshPass> {
public:
  using impl::AnnotateTensorMeshPassBase<
      AnnotateTensorMeshPass>::AnnotateTensorMeshPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Get the shardy mesh op in the root module.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        shardy_utils::getMeshOps(rootModule);

    if (parsedMeshOps.size() == 0) {
      rootModule.emitError(
          "Pass requires a shardy mesh op to be present in the root module.\n");
      signalPassFailure();
      return;
    }

    if (parsedMeshOps.size() > 1) {
      rootModule.emitError("Pass currently only support a single shardy mesh "
                           "op in the module.\n");
      signalPassFailure();
      return;
    }

    globalMeshOp = parsedMeshOps[0];
    llvm::StringRef meshName = globalMeshOp.getSymName();
    
    // Since we only support a single mesh in the module, we iterate through all the tensors and apply the mesh as a tensor encoding.
    Attribute encodingAttr = mlir::tt::ttcore::TensorMeshAttr::get(context, meshName);
    rootModule.walk([&](Operation *op) {
      // Update operation result types.
      for (auto result : op->getResults()) {
        result.setType(addEncodingToType(builder, result.getType(), encodingAttr));
      }

      // Update operation operand types.
      for (auto operand : op->getOperands()) {
        operand.setType(addEncodingToType(builder, operand.getType(), encodingAttr));
      }

      // Update other attributes in operation attributes.
      for (mlir::NamedAttribute attr : op->getAttrs()) {
        Attribute value = attr.getValue();

        if (auto typeAttr =  mlir::dyn_cast<mlir::TypeAttr>(value)) {
          op->setAttr(attr.getName(), mlir::TypeAttr::get(addEncodingToType(builder, typeAttr.getValue(), encodingAttr)));
        }
        else if (auto elementsAttr = mlir::dyn_cast<mlir::ElementsAttr>(value)) {
          mlir::RankedTensorType oldType = mlir::dyn_cast<mlir::RankedTensorType>(elementsAttr.getType());
          mlir::RankedTensorType newType = mlir::RankedTensorType::get(oldType.getShape(), oldType.getElementType(), encodingAttr);
          op->setAttr(attr.getName(), elementsAttr.cloneWithNewType(newType));
        }
      }

      // If the operation is a function, update its argument and result types.
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        // Update argument types.
        SmallVector<mlir::Type> newInputs;
        for (BlockArgument arg : funcOp.getArguments()) {
          mlir::Type updated = addEncodingToType(builder, arg.getType(), encodingAttr);
          newInputs.push_back(updated);
          arg.setType(updated);
        }
      
        // Update result types.
        SmallVector<mlir::Type> newResults;
        for (mlir::Type resType : funcOp.getFunctionType().getResults()) {
          newResults.push_back(addEncodingToType(builder, resType, encodingAttr));
        }
      
        // Update function type.
        auto newFuncType = builder.getFunctionType(newInputs, newResults);
        funcOp.setType(newFuncType);
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
