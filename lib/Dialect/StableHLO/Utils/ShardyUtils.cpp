// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/MeshShardingUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <numeric>

namespace mlir::tt::shardy_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Get all the meshOps from the module.
llvm::SmallVector<mlir::sdy::MeshOp> getMeshOps(mlir::ModuleOp &module) {
  llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps;

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    if (mlir::sdy::MeshOp meshOp = mlir::dyn_cast<mlir::sdy::MeshOp>(op)) {
      parsedMeshOps.push_back(meshOp);
    }
  }

  return parsedMeshOps;
}

// Remove all meshOps from the module.
void removeMeshOps(mlir::ModuleOp &module) {
  for (mlir::Operation &op :
       llvm::make_early_inc_range(module.getBody()->getOperations())) {
    if (auto meshOp = llvm::dyn_cast<mlir::sdy::MeshOp>(op)) {
      meshOp.erase();
    }
  }
}

// Create a meshAttr from a meshMap helper function.
mlir::sdy::MeshAttr createMeshAttrFromMeshMap(MLIRContext *context,
                                              shardy_utils::MeshMap &meshMap) {
  llvm::SmallVector<mlir::sdy::MeshAxisAttr> sdyMeshAxisAttrs;

  for (const auto &entry : meshMap) {
    mlir::sdy::MeshAxisAttr sdyMeshAxisAttrTensor =
        mlir::sdy::MeshAxisAttr::get(context, entry.first, entry.second);
    sdyMeshAxisAttrs.push_back(sdyMeshAxisAttrTensor);
  }

  return mlir::sdy::MeshAttr::get(context, sdyMeshAxisAttrs);
}

// Create a meshMap from a meshAttr helper function.
shardy_utils::MeshMap createMeshMapFromMeshAttr(mlir::sdy::MeshAttr meshAttr) {
  shardy_utils::MeshMap meshMap;

  for (auto meshAxisAttr : meshAttr.getAxes()) {
    meshMap[meshAxisAttr.getName()] = meshAxisAttr.getSize();
  }

  return meshMap;
}

// Get mesh shape from meshAttr.
llvm::SmallVector<int64_t>
getMeshShapeFromMeshAttr(mlir::sdy::MeshAttr meshAttr) {
  llvm::SmallVector<int64_t> meshShape;

  for (auto meshAxisAttr : meshAttr.getAxes()) {
    meshShape.push_back(meshAxisAttr.getSize());
  }

  return meshShape;
}

// Insert a new mesh into the module.
void addMeshToModule(mlir::ModuleOp &module, std::string meshName,
                     std::string firstAxisName, std::string secondAxisName,
                     int64_t firstAxisSize, int64_t secondAxisSize) {
  MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  shardy_utils::MeshMap meshMap;
  meshMap[firstAxisName] = firstAxisSize;
  meshMap[secondAxisName] = secondAxisSize;
  mlir::sdy::MeshAttr sdyMeshAttr =
      shardy_utils::createMeshAttrFromMeshMap(context, meshMap);
  builder.setInsertionPoint(&(module.getBody()->front()));
  builder.create<mlir::sdy::MeshOp>(
      builder.getUnknownLoc(), builder.getStringAttr(meshName), sdyMeshAttr);
}

// Check if the module has any sdy tensor sharding annotations.
bool sdyAnnotationsExist(mlir::ModuleOp &module) {
  for (auto &op : module.getBody()->getOperations()) {
    if (!mlir::isa<func::FuncOp>(op)) {
      continue;
    }

    func::FuncOp funcOp = mlir::cast<func::FuncOp>(op);
    // Check if sdy.sharding exists for any of the arguments.
    for (BlockArgument arg : funcOp.getBody().front().getArguments()) {
      if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
        if (currentArgAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
          return true;
        }
      }
    }

    // Check if sdy.sharding exists for any of the results.
    mlir::FunctionType funcType = funcOp.getFunctionType();
    for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
      if (auto resultAttrDict = mlir::DictionaryAttr::get(
              module.getContext(), funcOp.getResultAttrs(i))) {
        if (resultAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
          return true;
        }
      }
    }

    // Check if sdy.sharding exists for any of the operations.
    mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
      if (auto currentArgAttrDict =
              mlir::DictionaryAttr::get(module.getContext(), op->getAttrs())) {
        if (currentArgAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      return true;
    }
  }

  return false;
}

// Check if the module has any gspmd annotations.
bool gspmdAnnotationsExist(mlir::ModuleOp &module) {
  for (auto &op : module.getBody()->getOperations()) {
    if (!mlir::isa<func::FuncOp>(op)) {
      continue;
    }

    // Check if any gspmd operations exist.
    func::FuncOp funcOp = mlir::cast<func::FuncOp>(op);
    mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
        auto customCall = mlir::cast<mlir::stablehlo::CustomCallOp>(op);
        auto callTarget = customCall.getCallTargetName();

        if (callTarget == "Sharding" || callTarget == "SPMDFullToShardShape" ||
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

  return false;
}

// Check if the graph is solved.
bool isGraphSolved(mlir::ModuleOp &module) {
  for (auto &op : module.getBody()->getOperations()) {
    if (!mlir::isa<func::FuncOp>(op)) {
      continue;
    }

    func::FuncOp funcOp = mlir::cast<func::FuncOp>(op);
    mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::sdy::ManualComputationOp>(op)) {
        // If we find a manual computation op and it has its manual axes set, we
        // consider the graph solved. (todo: tapspatel) - we need to generalize
        // this to check for all manual computation ops, as we might have mixed
        // graphs.
        mlir::sdy::ManualComputationOp manualComputationOp =
            mlir::cast<mlir::sdy::ManualComputationOp>(op);
        if (!manualComputationOp.getManualAxes().empty()) {
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      return true;
    }
  }

  return false;
}

// Create a new DictionaryAttr from an old DictionaryAttr with all sdy.sharding
// annotations removed.
mlir::DictionaryAttr
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

// Remove all sdy tensor shardings from the module.
void removeSdyTensorShardings(MLIRContext *context, func::FuncOp &funcOp) {
  // Remove sharding annotations from arguments
  for (auto arg : funcOp.getArguments()) {
    if (auto argAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
      funcOp.setArgAttrs(
          arg.getArgNumber(),
          shardy_utils::removeDictionaryAttrSdyShardingAnnotations(
              context, argAttrDict));
    }
  }

  // Remove sharding annotations from results
  mlir::FunctionType funcType = funcOp.getFunctionType();
  for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
    if (auto resultAttrDict =
            mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i))) {
      funcOp.setResultAttrs(
          i, shardy_utils::removeDictionaryAttrSdyShardingAnnotations(
                 context, resultAttrDict));
    }
  }

  // Remove sharding annotations from operations
  funcOp.getBody().walk([&](mlir::Operation *op) {
    if (auto opAttrDict = op->getAttrDictionary()) {
      op->setAttrs(shardy_utils::removeDictionaryAttrSdyShardingAnnotations(
          context, opAttrDict));
    }
  });
}

// Create a new DictionaryAttr (from an old DictionaryAttr if provided) and add
// a sdy.sharding annotation to it.
mlir::DictionaryAttr addDictionaryAttrSdyShardingAnnotation(
    MLIRContext *context, mlir::sdy::TensorShardingAttr shardingAttr,
    std::optional<mlir::DictionaryAttr> dictAttr) {
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
getDefaultTensorSdyShardingAttr(MLIRContext *context, llvm::StringRef meshName,
                                mlir::Type type) {
  mlir::RankedTensorType rankedTensorType =
      mlir::cast<mlir::RankedTensorType>(type);
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

  for (uint32_t i = 0; i < rankedTensorType.getShape().size(); i++) {
    dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
        context, {}, /*is_closed*/ false));
  }

  return mlir::sdy::TensorShardingAttr::get(context, meshName, dimShardings, {},
                                            {});
}

// Get the argument sharding attributes.
llvm::SmallVector<mlir::sdy::TensorShardingAttr>
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
        shardingAttr = shardy_utils::getDefaultTensorSdyShardingAttr(
            context, globalMeshOp.getSymName(), arg.getType());

        // Set argument with it's default sdy.sharding attribute
        mlir::DictionaryAttr newDictAttr =
            shardy_utils::addDictionaryAttrSdyShardingAnnotation(
                context, shardingAttr, argAttrDict);
        funcOp.setArgAttrs(arg.getArgNumber(), newDictAttr);
      }
    } else {
      funcOp.emitWarning("In function ")
          << funcOp.getName() << " argument #: " << arg.getArgNumber()
          << " does not have an attributes dictionary. Using default "
             "sharding annotation (ie all dimensions replicated).\n";
      shardingAttr = shardy_utils::getDefaultTensorSdyShardingAttr(
          context, globalMeshOp.getSymName(), arg.getType());

      // Set argument with it's default sdy.sharding attribute
      mlir::DictionaryAttr newDictAttr =
          shardy_utils::addDictionaryAttrSdyShardingAnnotation(context,
                                                               shardingAttr);
      funcOp.setArgAttrs(arg.getArgNumber(), newDictAttr);
    }

    inShardingAttrs.push_back(shardingAttr);
  }

  return inShardingAttrs;
}

// Get the result sharding attributes.
llvm::SmallVector<mlir::sdy::TensorShardingAttr>
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
        shardingAttr = shardy_utils::getDefaultTensorSdyShardingAttr(
            context, globalMeshOp.getSymName(), funcType.getResult(i));
      }
    } else {
      funcOp.emitWarning("In function ")
          << funcOp.getName() << " result #: " << i
          << " does not have an attributes dictionary. Using default "
             "sharding annotation (ie all dimensions replicated).\n";
      shardingAttr = shardy_utils::getDefaultTensorSdyShardingAttr(
          context, globalMeshOp.getSymName(), funcType.getResult(i));
    }

    outShardingAttrs.push_back(shardingAttr);
  }

  return outShardingAttrs;
}

// Calculate the updated shape based on the tensor sharding annotation.
FailureOr<int64_t>
calculateUpdatedDim(mlir::sdy::MeshAttr meshAttr,
                    mlir::sdy::DimensionShardingAttr dimShardingAttr,
                    int64_t oldShapeDim) {
  int64_t updatedShapeDim = oldShapeDim;
  shardy_utils::MeshMap meshMap = createMeshMapFromMeshAttr(meshAttr);
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
FailureOr<mlir::RankedTensorType>
populateShardedOutputType(mlir::sdy::MeshAttr meshAttr,
                          mlir::RankedTensorType oldType,
                          mlir::sdy::TensorShardingAttr tensorShardingAttr) {
  llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
      tensorShardingAttr.getDimShardings();
  llvm::SmallVector<int64_t> newShape(oldType.getShape().begin(),
                                      oldType.getShape().end());

  for (uint32_t i = 0; i < newShape.size(); i++) {
    FailureOr<int64_t> updatedShapeDim =
        calculateUpdatedDim(meshAttr, dimShardings[i], newShape[i]);

    if (failed(updatedShapeDim)) {
      return failure();
    }

    newShape[i] = *updatedShapeDim;
  }

  return RankedTensorType::get(newShape, oldType.getElementType());
};

// Loop through all the tensorShardings and apply them to each output.
FailureOr<llvm::SmallVector<mlir::RankedTensorType>> getNewResultTypes(
    mlir::Operation *op, mlir::sdy::MeshOp &globalMeshOp,
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings) {
  llvm::SmallVector<mlir::RankedTensorType> newTypes;
  for (uint32_t i = 0; i < tensorShardings.size(); i++) {
    mlir::sdy::TensorShardingAttr tensorShardingAttr = tensorShardings[i];
    mlir::RankedTensorType oldType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(i).getType());
    FailureOr<mlir::RankedTensorType> newType = populateShardedOutputType(
        globalMeshOp.getMesh(), oldType, tensorShardingAttr);

    if (failed(newType)) {
      return mlir::failure();
    }

    newTypes.push_back(*newType);
  }

  return newTypes;
}

// Copy nested regions between srcOp and destOp
void copyNestedRegions(mlir::OpBuilder &builder, mlir::Operation *srcOp,
                       mlir::Operation *destOp) {
  for (unsigned i = 0; i < srcOp->getNumRegions(); i++) {
    auto &oldRegion = srcOp->getRegion(i);
    auto &newRegion = destOp->getRegion(i);

    // Figure out new operand mapping and apply to newly created op.
    mlir::IRMapping mapping;
    auto *oldBlock = &oldRegion.front();
    std::unique_ptr<mlir::Block> newBlock = std::make_unique<mlir::Block>();
    for (auto oldArg : oldBlock->getArguments()) {
      auto newArg = newBlock->addArgument(oldArg.getType(), oldArg.getLoc());
      mapping.map(oldArg, newArg);
    }

    for (auto &srcOp : *oldBlock) {
      builder.setInsertionPointToEnd(newBlock.get());
      builder.clone(srcOp, mapping);
    }
    newRegion.push_back(newBlock.release());
  }
}

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::shardy_utils
