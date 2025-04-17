// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_STABLEHLO_SHARDYUTILS_H
#define TTMLIR_STABLEHLO_SHARDYUTILS_H

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

// Get number of sdy.mesh that exist in the module.
inline uint32_t getNumMeshes(mlir::ModuleOp &module) {
  uint32_t numMeshes = 0;

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    if (mlir::sdy::MeshOp meshOp = mlir::dyn_cast<mlir::sdy::MeshOp>(op)) {
      numMeshes++;
    }
  }

  return numMeshes;
}

// Get the sdy.mesh from the module.
inline mlir::sdy::MeshOp getMeshOp(mlir::ModuleOp &module) {
  mlir::sdy::MeshOp parsedMeshOp;

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    if (mlir::sdy::MeshOp meshOp = mlir::dyn_cast<mlir::sdy::MeshOp>(op)) {
      parsedMeshOp = meshOp;
    }
  }

  return parsedMeshOp;
}

// Create a meshAttr from a meshMap helper function.
inline mlir::sdy::MeshAttr createMeshAttrFromMeshMap(MLIRContext *context,
                                                     MeshMap meshMap) {
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

// Check if the module has any sdy sharding annotations.
inline bool sdyAnnotationsExist(mlir::ModuleOp &module) {
  bool doSdyAnnotationsExist = false;

  for (auto &op : module.getBody()->getOperations()) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      // Check if sdy.sharding exists for any of the arguments.
      for (BlockArgument arg : funcOp.getBody().front().getArguments()) {
        if (auto currentArgAttrDict =
                funcOp.getArgAttrDict(arg.getArgNumber())) {
          if (currentArgAttrDict && currentArgAttrDict.contains(
                                        mlir::sdy::TensorShardingAttr::name)) {
            doSdyAnnotationsExist = true;
          }
        }
      }

      // Check if sdy.sharding exists for any of the operations.
      funcOp.getBody().walk([&](mlir::Operation *op) {
        if (auto currentArgAttrDict = mlir::DictionaryAttr::get(
                module.getContext(), op->getAttrs())) {
          if (currentArgAttrDict && currentArgAttrDict.contains(
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
              callTarget == "SPMDFullToShardShape" || callTarget == "SPMDShardToFullShape") {
            doGspmdAnnotationsExist = true;
          }
        }
      });
    }
  }

  return doGspmdAnnotationsExist;
}

// Remove sdy.sharding annotations from DictionaryAttr
inline mlir::DictionaryAttr
removeDictionaryAttrShardingAnnotation(MLIRContext *context,
                                       mlir::DictionaryAttr &dictAttr) {
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

inline mlir::LogicalResult
getTensorShardingAttr(mlir::DictionaryAttr &dictAttr,
                      mlir::sdy::TensorShardingAttr &shardingAttr) {
  if (dictAttr && dictAttr.contains(mlir::sdy::TensorShardingAttr::name)) {
    shardingAttr = mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(
        dictAttr.get(mlir::sdy::TensorShardingAttr::name));
    return success();
  }

  llvm::errs()
      << "Could not find tensor sharding attribute in attribute dictionary.\n";
  return failure();
}

inline mlir::LogicalResult
getTensorShardingPerValueAttr(mlir::DictionaryAttr &dictAttr,
                      mlir::sdy::TensorShardingPerValueAttr &tensorShardingPerValueAttr) {
  if (dictAttr && dictAttr.contains(mlir::sdy::TensorShardingAttr::name)) {
    tensorShardingPerValueAttr = mlir::dyn_cast<mlir::sdy::TensorShardingPerValueAttr>(
        dictAttr.get(mlir::sdy::TensorShardingAttr::name));
    return success();
  }

  llvm::errs()
      << "Could not find tensor sharding per value attribute in attribute dictionary.\n";
  return failure();
}

inline mlir::LogicalResult
getOutShardingAttr(mlir::DictionaryAttr &dictAttr, mlir::sdy::TensorShardingAttr &shardingAttr) {
  if (dictAttr && dictAttr.contains("out_sharding")) {
    shardingAttr = mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(
        dictAttr.get("out_sharding"));
    return success();
  }

  llvm::errs()
      << "Could not find out sharding attribute in attribute dictionary.\n";
  return failure();
}

inline mlir::LogicalResult
getSlicingAxesAttr(mlir::DictionaryAttr &dictAttr, mlir::sdy::ListOfAxisRefListsAttr &slicingAxesAttr) {
  if (dictAttr && dictAttr.contains("slicing_axes")) {
    slicingAxesAttr = mlir::dyn_cast<mlir::sdy::ListOfAxisRefListsAttr>(
        dictAttr.get("slicing_axes"));
    return success();
  }

  llvm::errs() << "Could not find slicing axes in attribute dictionary.\n";
  return failure();
}

// Calculate the new sharded output based on the sdy slicing axes attribute
inline mlir::LogicalResult
populateShardedOutputType(mlir::sdy::MeshAttr meshAttr, mlir::RankedTensorType &newType, mlir::RankedTensorType &oldType, mlir::sdy::TensorShardingAttr &tensorShardingAttr, mlir::sdy::ListOfAxisRefListsAttr &slicingAxesAttr) {
  MeshMap meshMap = createMeshMapFromMeshAttr(meshAttr);
  
  llvm::ArrayRef<mlir::sdy::AxisRefListAttr> axisRefList = slicingAxesAttr.getValue();
  llvm::SmallVector<int64_t> newShape(oldType.getShape().begin(), oldType.getShape().end());

  // Loop through axisRefList and find which axis need to still get sharded in the output.
  for (auto shardingAxes : axisRefList) {
    for (auto shardingAxis : shardingAxes) {
      llvm::StringRef axisName = shardingAxis.getName();
      if (meshMap.find(axisName) == meshMap.end()) {
        llvm::errs() << "Could not find mesh axis in the current meshmap being "
                        "used by the function.\n";
        return failure();
      }

      // Find which tensor dimension this axis is on in tensorShardingAttr.
      llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings = tensorShardingAttr.getDimShardings();

      for (uint32_t i = 0; i < dimShardings.size(); i++) {
        llvm::ArrayRef<mlir::sdy::AxisRefAttr> dimShardingAxes = dimShardings[i].getAxes();

        for (auto dimShardAxis : dimShardingAxes) {
          if (axisName == dimShardAxis.getName()) {
            int64_t currIndexShape = newShape[i];

            if (currIndexShape % meshMap[axisName] != 0) {
              llvm::errs() << "Output shape does not divide evenly by it's sharded annotation.\n";
              return failure();
            }

            currIndexShape = currIndexShape / meshMap[axisName];
            newShape[i] = currIndexShape;
          }
        }
      }
    }
  }

  newType = RankedTensorType::get(newShape, oldType.getElementType());
  return success();
}

// Calculate the new sharded output based on the sdy tensor sharding attribute
inline mlir::LogicalResult
populateShardedOutputType(mlir::sdy::MeshAttr meshAttr,
                          mlir::RankedTensorType &newType,
                          mlir::RankedTensorType &oldType,
                          mlir::sdy::TensorShardingAttr &tensorShardingAttr) {
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
      if (meshMap.find(axisName) == meshMap.end()) {
        llvm::errs() << "Could not find mesh axis in the current meshmap being "
                        "used by the function.\n";
        return failure();
      }

      if (currIndexShape % meshMap[axisName] != 0) {
        llvm::errs() << "Output shape does not divide evenly by it's sharded annotation.\n";
        return failure();
      }
      currIndexShape = currIndexShape / meshMap[axisName];
    }
    newShape[i] = currIndexShape;
  }

  newType = RankedTensorType::get(newShape, oldType.getElementType());
  return success();
};

#endif // #if defined(TTMLIR_ENABLE_STABLEHLO) && (TTMLIR_ENABLE_STABLEHLO != 0)

} // namespace mlir::tt::sdy_utils

#endif // TTMLIR_STABLEHLO_SHARDYUTILS_H
