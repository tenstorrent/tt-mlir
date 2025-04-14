// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H

#include "mlir/Analysis/TopologicalSortUtils.h"
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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#pragma clang diagnostic pop

#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::sdy_utils {

#if defined(TTMLIR_ENABLE_STABLEHLO) && (TTMLIR_ENABLE_STABLEHLO != 0)

// We used MapVector because we want the order to be preserved when inserting
// axes into the meshMap.
using MeshMap =
    llvm::MapVector</*axis_name*/ llvm::StringRef, /*axis_size*/ int64_t>;

// Get all the meshOps from the module.
inline llvm::SmallVector<mlir::sdy::MeshOp> getMeshOps(mlir::ModuleOp &module) {
  llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps;

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    if (mlir::sdy::MeshOp meshOp = mlir::dyn_cast<mlir::sdy::MeshOp>(op)) {
      parsedMeshOps.push_back(meshOp);
    }
  }

  return parsedMeshOps;
}

// Remove all meshOps from the module.
inline void removeMeshOps(mlir::ModuleOp &module) {
  for (mlir::Operation &op :
       llvm::make_early_inc_range(module.getBody()->getOperations())) {
    if (auto meshOp = llvm::dyn_cast<mlir::sdy::MeshOp>(op)) {
      meshOp.erase();
    }
  }
}

// Create a meshAttr from a meshMap helper function.
inline mlir::sdy::MeshAttr createMeshAttrFromMeshMap(MLIRContext *context,
                                                     MeshMap &meshMap) {
  llvm::SmallVector<mlir::sdy::MeshAxisAttr> sdyMeshAxisAttrs;

  for (const auto &entry : meshMap) {
    mlir::sdy::MeshAxisAttr sdyMeshAxisAttrTensor =
        mlir::sdy::MeshAxisAttr::get(context, entry.first, entry.second);
    sdyMeshAxisAttrs.push_back(sdyMeshAxisAttrTensor);
  }

  mlir::sdy::MeshAttr sdyMeshAttr =
      mlir::sdy::MeshAttr::get(context, sdyMeshAxisAttrs);
  return sdyMeshAttr;
}

// Create a meshMap from a meshAttr helper function.
inline MeshMap createMeshMapFromMeshAttr(mlir::sdy::MeshAttr meshAttr) {
  MeshMap meshMap;

  for (auto meshAxisAttr : meshAttr.getAxes()) {
    meshMap[meshAxisAttr.getName()] = meshAxisAttr.getSize();
  }

  return meshMap;
}

// Check if the module has any sdy tensor sharding annotations.
inline bool sdyAnnotationsExist(mlir::ModuleOp &module) {
  bool doSdyAnnotationsExist = false;

  for (auto &op : module.getBody()->getOperations()) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      // Check if sdy.sharding exists for any of the arguments.
      for (BlockArgument arg : funcOp.getBody().front().getArguments()) {
        if (auto currentArgAttrDict =
                funcOp.getArgAttrDict(arg.getArgNumber())) {
          if (currentArgAttrDict.contains(
                  mlir::sdy::TensorShardingAttr::name)) {
            doSdyAnnotationsExist = true;
          }
        }
      }

      // Check if sdy.sharding exists for any of the operations.
      funcOp.getBody().walk([&](mlir::Operation *op) {
        if (auto currentArgAttrDict = mlir::DictionaryAttr::get(
                module.getContext(), op->getAttrs())) {
          if (currentArgAttrDict.contains(
                  mlir::sdy::TensorShardingAttr::name)) {
            doSdyAnnotationsExist = true;
          }
        }
      });
    }
  }

  return doSdyAnnotationsExist;
}

// Check if the module has any gspmd annotations.
inline bool gspmdAnnotationsExist(mlir::ModuleOp &module) {
  bool doGspmdAnnotationsExist = false;

  for (auto &op : module.getBody()->getOperations()) {
    if (auto funcOp = mlir::dyn_cast<func::FuncOp>(op)) {
      // Check if any gspmd operations exist.
      funcOp.getBody().walk([&](mlir::Operation *op) {
        if (mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
          auto customCall = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
          auto callTarget = customCall.getCallTargetName();

          if (callTarget == "Sharding" ||
              callTarget == "SPMDFullToShardShape" ||
              callTarget == "SPMDShardToFullShape") {
            doGspmdAnnotationsExist = true;
          }
        }
      });
    }
  }

  return doGspmdAnnotationsExist;
}

// Create a new DictionaryAttr from an old DictionaryAttr with all sdy.sharding
// annotations removed.
inline mlir::DictionaryAttr
removeDictionaryAttrShardingAnnotation(MLIRContext *context,
                                       mlir::DictionaryAttr dictAttr) {
  llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

  if (dictAttr && dictAttr.contains(mlir::sdy::TensorShardingAttr::name)) {
    llvm::copy_if(dictAttr.getValue(), std::back_inserter(newArgAttrs),
                  [&](mlir::NamedAttribute currentArgAttr) {
                    return currentArgAttr.getName() !=
                           mlir::sdy::TensorShardingAttr::name;
                  });
  } else {
    newArgAttrs = SmallVector<mlir::NamedAttribute>(dictAttr.getValue());
  }

  return mlir::DictionaryAttr::get(context, newArgAttrs);
}

// Create a new DictionaryAttr (from an old DictionaryAttr if provided) and add
// a sdy.sharding annotation to it.
inline mlir::DictionaryAttr addDictionaryAttrShardingAnnotation(
    MLIRContext *context, mlir::sdy::TensorShardingAttr shardingAttr,
    std::optional<mlir::DictionaryAttr> dictAttr = std::nullopt) {
  llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

  if (dictAttr) {
    newArgAttrs =
        SmallVector<mlir::NamedAttribute>(dictAttr.value().getValue());
  }

  newArgAttrs.emplace_back(
      mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
      shardingAttr);
  return mlir::DictionaryAttr::get(context, newArgAttrs);
}

// Get a default sdy.sharding annotation (ie all dimensions are open and
// replicated).
mlir::sdy::TensorShardingAttr
getDefaultTensorShardingAttr(MLIRContext *context, llvm::StringRef meshName,
                             mlir::Type type) {
  mlir::RankedTensorType rankedTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(type);
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

  for (uint32_t i = 0; i < rankedTensorType.getShape().size(); i++) {
    dimShardings.push_back(
        mlir::sdy::DimensionShardingAttr::get(context, {}, true));
  }

  return mlir::sdy::TensorShardingAttr::get(context, meshName, dimShardings,
                                            {});
}

// Calculate the new sharded output based on the sdy tensor sharding attribute.
inline FailureOr<mlir::RankedTensorType>
populateShardedOutputType(mlir::sdy::MeshAttr meshAttr,
                          mlir::RankedTensorType oldType,
                          mlir::sdy::TensorShardingAttr tensorShardingAttr) {
  MeshMap meshMap = createMeshMapFromMeshAttr(meshAttr);

  llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
      tensorShardingAttr.getDimShardings();
  llvm::SmallVector<int64_t> newShape(oldType.getShape().begin(),
                                      oldType.getShape().end());

  for (uint32_t i = 0; i < newShape.size(); i++) {
    int64_t currIndexShape = newShape[i];
    llvm::ArrayRef<mlir::sdy::AxisRefAttr> shardingAxes =
        dimShardings[i].getAxes();

    for (auto shardingAxis : shardingAxes) {
      llvm::StringRef axisName = shardingAxis.getName();
      if (meshMap.find(axisName) == meshMap.end() ||
          currIndexShape % meshMap[axisName] != 0) {
        return failure();
      }

      currIndexShape = currIndexShape / meshMap[axisName];
    }
    newShape[i] = currIndexShape;
  }

  return RankedTensorType::get(newShape, oldType.getElementType());
};

#endif // #if defined(TTMLIR_ENABLE_STABLEHLO) && (TTMLIR_ENABLE_STABLEHLO != 0)

} // namespace mlir::tt::sdy_utils

#endif // TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H
