// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"

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

#ifdef TTMLIR_ENABLE_STABLEHLO

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
  for (auto &op : module.getBody()->getOperations()) {
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      // Check if sdy.sharding exists for any of the arguments.
      for (BlockArgument arg : funcOp.getBody().front().getArguments()) {
        if (auto currentArgAttrDict =
                funcOp.getArgAttrDict(arg.getArgNumber())) {
          if (currentArgAttrDict.contains(
                  mlir::sdy::TensorShardingAttr::name)) {
            return true;
          }
        }
      }

      // Check if sdy.sharding exists for any of the operations.
      mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
        if (auto currentArgAttrDict = mlir::DictionaryAttr::get(
                module.getContext(), op->getAttrs())) {
          if (currentArgAttrDict.contains(
                  mlir::sdy::TensorShardingAttr::name)) {
            return WalkResult::interrupt();
          }
        }

        return WalkResult::advance();
      });

      if (result.wasInterrupted()) {
        return true;
      }
    }
  }

  return false;
}

// Check if the module has any gspmd annotations.
inline bool gspmdAnnotationsExist(mlir::ModuleOp &module) {
  for (auto &op : module.getBody()->getOperations()) {
    if (auto funcOp = mlir::dyn_cast<func::FuncOp>(op)) {
      // Check if any gspmd operations exist.
      mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
        if (mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
          auto customCall = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
          auto callTarget = customCall.getCallTargetName();

          if (callTarget == "Sharding" ||
              callTarget == "SPMDFullToShardShape" ||
              callTarget == "SPMDShardToFullShape") {
            return WalkResult::interrupt();
          }
        }

        return WalkResult::advance();
      });

      if (result.wasInterrupted()) {
        return true;
      }
    }
  }

  return false;
}

// Create a new DictionaryAttr from an old DictionaryAttr with all sdy.sharding
// annotations removed.
inline mlir::DictionaryAttr
removeDictionaryAttrSdyShardingAnnotations(MLIRContext *context,
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
inline mlir::DictionaryAttr addDictionaryAttrSdyShardingAnnotation(
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
inline mlir::sdy::TensorShardingAttr
getDefaultTensorSdyShardingAttr(MLIRContext *context, llvm::StringRef meshName,
                                mlir::Type type) {
  mlir::RankedTensorType rankedTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(type);
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

  for (uint32_t i = 0; i < rankedTensorType.getShape().size(); i++) {
    dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
        context, {}, /*is_closed*/ false));
  }

  return mlir::sdy::TensorShardingAttr::get(context, meshName, dimShardings, {},
                                            {});
}

// Get the argument sharding attributes.
inline llvm::SmallVector<mlir::sdy::TensorShardingAttr>
getInShardingAttrs(MLIRContext *context, func::FuncOp &funcOp,
                   mlir::sdy::MeshOp &globalMeshOp) {
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> inShardingAttrs;

  for (auto arg : funcOp.getArguments()) {
    // Get the tensor sharding attribute from argument dictionary.
    mlir::sdy::TensorShardingAttr shardingAttr;
    if (auto argAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
      shardingAttr = mlir::dyn_cast_if_present<mlir::sdy::TensorShardingAttr>(
          argAttrDict.get(mlir::sdy::TensorShardingAttr::name));

      if (!shardingAttr) {
        funcOp.emitWarning("In function ")
            << funcOp.getName() << " argument #: " << arg.getArgNumber()
            << " is not annotated with sdy.sharding. Using default "
               "sharding annotation (ie all dimensions replicated).\n";
        shardingAttr = sdy_utils::getDefaultTensorSdyShardingAttr(
            context, globalMeshOp.getSymName(), arg.getType());

        // Set argument with it's default sdy.sharding attribute
        mlir::DictionaryAttr newDictAttr =
            sdy_utils::addDictionaryAttrSdyShardingAnnotation(
                context, shardingAttr, argAttrDict);
        funcOp.setArgAttrs(arg.getArgNumber(), newDictAttr);
      }
    } else {
      funcOp.emitWarning("In function ")
          << funcOp.getName() << " argument #: " << arg.getArgNumber()
          << " does not have an attributes dictionary. Using default "
             "sharding annotation (ie all dimensions replicated).\n";
      shardingAttr = sdy_utils::getDefaultTensorSdyShardingAttr(
          context, globalMeshOp.getSymName(), arg.getType());

      // Set argument with it's default sdy.sharding attribute
      mlir::DictionaryAttr newDictAttr =
          sdy_utils::addDictionaryAttrSdyShardingAnnotation(context,
                                                            shardingAttr);
      funcOp.setArgAttrs(arg.getArgNumber(), newDictAttr);
    }

    inShardingAttrs.push_back(shardingAttr);
  }

  return inShardingAttrs;
}

// Get the result sharding attributes.
inline llvm::SmallVector<mlir::sdy::TensorShardingAttr>
getOutShardingAttrs(MLIRContext *context, func::FuncOp &funcOp,
                    mlir::sdy::MeshOp &globalMeshOp) {
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> outShardingAttrs;
  mlir::FunctionType funcType = funcOp.getFunctionType();

  for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
    // Get the tensor sharding attribute from results dictionary.
    mlir::sdy::TensorShardingAttr shardingAttr;
    if (auto resultAttrDict =
            mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i))) {
      shardingAttr = mlir::dyn_cast_if_present<mlir::sdy::TensorShardingAttr>(
          resultAttrDict.get(mlir::sdy::TensorShardingAttr::name));

      if (!shardingAttr) {
        funcOp.emitWarning("In function ")
            << funcOp.getName() << " result #: " << i
            << " is not annotated with sdy.sharding. Using default "
               "sharding annotation (ie all dimensions replicated).\n";
        shardingAttr = sdy_utils::getDefaultTensorSdyShardingAttr(
            context, globalMeshOp.getSymName(), funcType.getResult(i));
      }
    } else {
      funcOp.emitWarning("In function ")
          << funcOp.getName() << " result #: " << i
          << " does not have an attributes dictionary. Using default "
             "sharding annotation (ie all dimensions replicated).\n";
      shardingAttr = sdy_utils::getDefaultTensorSdyShardingAttr(
          context, globalMeshOp.getSymName(), funcType.getResult(i));
    }

    outShardingAttrs.push_back(shardingAttr);
  }

  return outShardingAttrs;
}

// Calculate the updated shape based on the tensor sharding annotation.
inline FailureOr<int64_t>
calculateUpdatedShapeDim(mlir::sdy::MeshAttr meshAttr,
                         mlir::sdy::DimensionShardingAttr dimShardingAttr,
                         int64_t oldShapeDim) {
  int64_t updatedShapeDim = oldShapeDim;
  MeshMap meshMap = createMeshMapFromMeshAttr(meshAttr);
  llvm::ArrayRef<mlir::sdy::AxisRefAttr> shardingAxes =
      dimShardingAttr.getAxes();

  for (auto shardingAxis : shardingAxes) {
    llvm::StringRef axisName = shardingAxis.getName();
    if (meshMap.find(axisName) == meshMap.end() ||
        oldShapeDim % meshMap[axisName] != 0) {
      return failure();
    }

    updatedShapeDim = updatedShapeDim / meshMap[axisName];
  }

  return updatedShapeDim;
}

// Calculate the new sharded output based on the sdy tensor sharding attribute.
inline FailureOr<mlir::RankedTensorType>
populateShardedOutputType(mlir::sdy::MeshAttr meshAttr,
                          mlir::RankedTensorType oldType,
                          mlir::sdy::TensorShardingAttr tensorShardingAttr) {
  llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
      tensorShardingAttr.getDimShardings();
  llvm::SmallVector<int64_t> newShape(oldType.getShape().begin(),
                                      oldType.getShape().end());

  for (uint32_t i = 0; i < newShape.size(); i++) {
    FailureOr<int64_t> updatedShapeDim =
        calculateUpdatedShapeDim(meshAttr, dimShardings[i], newShape[i]);

    if (failed(updatedShapeDim)) {
      return failure();
    }

    newShape[i] = *updatedShapeDim;
  }

  return RankedTensorType::get(newShape, oldType.getElementType());
};

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::sdy_utils

#endif // TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H
