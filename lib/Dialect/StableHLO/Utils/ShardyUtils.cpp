// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Error.h"

#include "stablehlo/dialect/StablehloOps.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <numeric>
#include <optional>

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
    meshMap[meshAxisAttr.getName().str()] = meshAxisAttr.getSize();
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

// Create a TTMeshAttr from a sdy::meshOp.
mlir::tt::ttcore::MeshAttr
createTTMeshAttrFromSdyMeshOp(mlir::sdy::MeshOp meshOp) {
  return mlir::tt::ttcore::MeshAttr::get(
      meshOp.getContext(),
      mlir::StringAttr::get(meshOp.getContext(), meshOp.getSymName()),
      getMeshShapeFromMeshAttr(meshOp.getMeshAttr()));
}

// Check if the module has any sdy tensor sharding annotations.
bool sdyAnnotationsExist(mlir::ModuleOp &module) {
  llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
      shardy_utils::getMeshOps(module);

  if (parsedMeshOps.size() > 0) {
    return true;
  }

  for (auto &op : module.getBody()->getOperations()) {
    if (mlir::isa<mlir::sdy::ManualComputationOp>(op)) {
      return true;
    }

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
        // consider the graph solved.
        // TODO (tapspatel) - we need to generalize this to check for all manual
        // computation ops, as we might have mixed graphs.
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
               "sharding annotation (ie all dimensions replicated)";
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
             "sharding annotation (ie all dimensions replicated)";
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
               "sharding annotation (ie all dimensions replicated)";
        shardingAttr = shardy_utils::getDefaultTensorSdyShardingAttr(
            context, globalMeshOp.getSymName(), funcType.getResult(i));
      }
    } else {
      funcOp.emitWarning("In function ")
          << funcOp.getName() << " result #: " << i
          << " does not have an attributes dictionary. Using default "
             "sharding annotation (ie all dimensions replicated)";
      shardingAttr = shardy_utils::getDefaultTensorSdyShardingAttr(
          context, globalMeshOp.getSymName(), funcType.getResult(i));
    }

    outShardingAttrs.push_back(shardingAttr);
  }

  return outShardingAttrs;
}

// Returns TensorShardingAttr for an operand's Value, checking (in order):
// 1) func.func entry-block arg attrs
// 2) sdy.manual_computation region arg per-value attrs
// 3) OpResult per-value attrs
// Falls back to full replicate if none found.
mlir::sdy::TensorShardingAttr
getOperandShardingAttr(const mlir::OpOperand &operand,
                       mlir::sdy::MeshOp globalMeshOp) {
  mlir::Value val = operand.get();

  mlir::sdy::TensorShardingAttr result =
      shardy_utils::getDefaultTensorSdyShardingAttr(
          val.getContext(), globalMeshOp.getSymName(), val.getType());

  if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    unsigned argNo = barg.getArgNumber();
    mlir::Operation *parentOp = barg.getOwner()->getParentOp();
    if (auto func = llvm::dyn_cast_or_null<mlir::func::FuncOp>(parentOp)) {
      auto inAttrs = getInShardingAttrs(func.getContext(), func, globalMeshOp);
      if (argNo < inAttrs.size()) {
        result = inAttrs[argNo];
      }
    } else if (auto mc = llvm::dyn_cast_or_null<mlir::sdy::ManualComputationOp>(
                   parentOp)) {
      if (auto spv = llvm::dyn_cast<mlir::sdy::TensorShardingPerValueAttr>(
              mc.getInShardings())) {
        auto shardings = spv.getShardings();
        if (argNo < shardings.size()) {
          result = shardings[argNo];
        }
      }
    }
  } else if (auto opRes = mlir::dyn_cast<mlir::OpResult>(val)) {
    mlir::Operation *owner = opRes.getOwner();
    unsigned resNo = opRes.getResultNumber();
    if (auto spv = owner->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
            mlir::sdy::TensorShardingAttr::name)) {
      auto shardings = spv.getShardings();
      if (!shardings.empty()) {
        unsigned i = (resNo < shardings.size()) ? resNo : 0u; // broadcast-safe
        result = shardings[i];
      }
    }
  }

  return result;
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
    if (meshMap.find(axisName.str()) == meshMap.end() ||
        oldShapeDim % meshMap[axisName.str()] != 0) {
      return failure();
    }

    updatedShapeDim = updatedShapeDim / meshMap[axisName.str()];
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

// Parse Shardy sharding attribute.
llvm::Expected<bool> parseSdySharding(mlir::sdy::TensorShardingAttr sdySharding,
                                      mlir::sdy::MeshAttr meshAttr,
                                      llvm::SmallVector<int64_t> &shardShape,
                                      llvm::SmallVector<int64_t> &shardDims,
                                      llvm::SmallVector<int64_t> &meshShape) {

  shardShape.assign(sdySharding.getRank(), 1);
  shardDims.assign(meshAttr.getAxes().size(), -1);

  meshShape.clear();
  llvm::SmallDenseMap<::llvm::StringRef, int64_t> axisPosition;
  for (auto [idx, meshAxisAttr] : llvm::enumerate(meshAttr.getAxes())) {
    axisPosition[meshAxisAttr.getName()] = idx;
    meshShape.push_back(meshAxisAttr.getSize());
  }

  if (!sdySharding.isFullyClosed()) {
    return llvm::createStringError(
        "Sharding with open dimension is currently not supported.");
  }

  // Iterate each dimSharding in TensorShardingAttr.
  for (auto [dimIdx, dimSharding] :
       llvm::enumerate(sdySharding.getDimShardings())) {
    for (auto [axisIdx, axes] : llvm::enumerate(dimSharding.getAxes())) {
      // Check if there is any subaxis sharding.
      if (auto subAxis = axes.getSubAxisInfo()) {
        return llvm::createStringError(
            "Sharding with subaxis partitioning is currently not supported.");
      }
      shardShape[dimIdx] *= axes.getSize(meshAttr);
      // Sharding makes sense when it is higher than 1.
      if (axes.getSize(meshAttr) > 1) {
        shardDims[axisPosition[axes.getName()]] = dimIdx;
      }
    }
  }

  return true;
}

// Generate default ShardyMeshSharding.
llvm::Expected<ShardyMeshSharding> ShardyMeshSharding::generateDefault() {
  return ShardyMeshSharding{
      mlir::tt::ttcore::MeshShardDirection::FullToShard,
      mlir::tt::ttcore::MeshShardType::Identity,
      /*shardShape=*/llvm::SmallVector<int64_t>{},
      /*shardDims=*/llvm::SmallVector<int64_t>{},
      /*meshShape=*/llvm::SmallVector<int64_t>{},
      /*deviceIds=*/llvm::SmallVector<int64_t>{},
      mlir::tt::ttcore::ShardStatus::Unsharded,
      sdy::MeshAttr(),
      mlir::sdy::TensorShardingAttr(),
  };
}

llvm::Expected<ShardyMeshSharding>
ShardyMeshSharding::generate(sdy::MeshAttr meshAttr,
                             sdy::TensorShardingAttr sdySharding,
                             mlir::tt::ttcore::ShardStatus shardStatus,
                             ttcore::MeshShardDirection shardDirection) {
  // Need to parse sdy sharding and fill out MeshSharding info.
  mlir::tt::ttcore::MeshShardType shardType =
      mlir::tt::ttcore::MeshShardType::Identity;
  llvm::SmallVector<int64_t> shardShape = {-1};
  llvm::SmallVector<int64_t> shardDims = {-1};
  llvm::SmallVector<int64_t> meshShape = {-1};
  llvm::SmallVector<int64_t> deviceIds = {-1};

  // Lambda function to set non-device shard types.
  auto setNonDevicesShardType =
      [&](ttcore::MeshShardType targetShardType) -> void {
    shardType = targetShardType;
    shardShape = llvm::SmallVector<int64_t>{1};
    shardDims = llvm::SmallVector<int64_t>{-1};
    meshShape = llvm::SmallVector<int64_t>{-1};
  };

  // Empty meshAttr indicates single device, so no need to convert.
  if (meshAttr.empty()) {
    meshShape.clear();
    return ShardyMeshSharding{shardDirection, shardType, shardShape,
                              shardDims,      meshShape, deviceIds,
                              shardStatus,    meshAttr,  sdySharding};
  }

  if (meshAttr.getAxes().empty()) {
    if (meshAttr.getDeviceIds().empty()) {
      // Set shard type to replicate.
      setNonDevicesShardType(mlir::tt::ttcore::MeshShardType::Replicate);
    } else {
      // Set shard type to maximal.
      setNonDevicesShardType(mlir::tt::ttcore::MeshShardType::Maximal);
      deviceIds = llvm::SmallVector<int64_t>(meshAttr.getDeviceIds());
    }
    return ShardyMeshSharding{shardDirection, shardType, shardShape,
                              shardDims,      meshShape, deviceIds,
                              shardStatus,    meshAttr,  sdySharding};
  }

  shardType = ttcore::MeshShardType::Devices;
  auto error =
      parseSdySharding(sdySharding, meshAttr, shardShape, shardDims, meshShape);
  if (auto e = error.takeError()) {
    return e;
  }

  // totalPartition is the total number of multi-chips such as 8 for t3k. Thus,
  // no overflow is expected with int64_t.
  int64_t totalPartition =
      std::accumulate(shardShape.begin(), shardShape.end(), int64_t{1},
                      std::multiplies<int64_t>());
  // No partition indicates replicate to all devices.
  if (totalPartition == 1) {
    setNonDevicesShardType(mlir::tt::ttcore::MeshShardType::Replicate);
  }

  // Check if the input is already pre-sharded. If it is, override shardType to
  // Identity.
  shardType = shardStatus == mlir::tt::ttcore::ShardStatus::Presharded
                  ? ttcore::MeshShardType::Identity
                  : shardType;

  return ShardyMeshSharding{shardDirection, shardType, shardShape,
                            shardDims,      meshShape, deviceIds,
                            shardStatus,    meshAttr,  sdySharding};
}

bool isFullyReplicatedTensor(mlir::sdy::TensorShardingAttr tsh) {
  for (auto dim : tsh.getDimShardings()) {
    if (!dim.getAxes().empty()) {
      return false;
    }
  }
  return true;
}

// Get all mesh names from a function, returning empty vector if none found.
llvm::SmallVector<std::string> getMeshNames(mlir::func::FuncOp &funcOp) {
  llvm::SmallVector<std::string> meshNames;
  if (auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>()) {
    llvm::SmallVector<mlir::sdy::MeshOp> meshOps = getMeshOps(moduleOp);
    for (auto meshOp : meshOps) {
      meshNames.push_back(meshOp.getSymName().str());
    }
  }
  return meshNames;
}

// Parse dimension shardings from a string representation.
llvm::SmallVector<mlir::sdy::DimensionShardingAttr>
parseDimensionShardings(const std::string &dimsContent,
                        mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
  size_t pos = 0;

  while (pos < dimsContent.length()) {
    size_t braceStart = dimsContent.find("{", pos);
    if (braceStart == std::string::npos) {
      break;
    }

    size_t braceEnd = dimsContent.find("}", braceStart);
    if (braceEnd == std::string::npos) {
      break;
    }

    std::string dimContent =
        dimsContent.substr(braceStart + 1, braceEnd - braceStart - 1);

    if (dimContent.empty()) {
      dimShardings.push_back(
          mlir::sdy::DimensionShardingAttr::get(context, {}, false));
    } else {
      llvm::SmallVector<mlir::sdy::AxisRefAttr> axisRefs;
      size_t axisPos = 0;
      while (axisPos < dimContent.length()) {
        size_t quoteStart = dimContent.find('"', axisPos);
        if (quoteStart == std::string::npos) {
          break;
        }

        size_t quoteEnd = dimContent.find('"', quoteStart + 1);
        if (quoteEnd == std::string::npos) {
          break;
        }

        std::string axisName =
            dimContent.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
        axisRefs.push_back(mlir::sdy::AxisRefAttr::get(context, axisName));
        axisPos = quoteEnd + 1;
      }

      dimShardings.push_back(
          mlir::sdy::DimensionShardingAttr::get(context, axisRefs, false));
    }

    pos = braceEnd + 1;
  }

  return dimShardings;
}

std::vector<std::string> parseItemList(const std::string &input) {
  std::vector<std::string> items;
  size_t pos = 0;

  // Skip leading whitespace
  while (pos < input.length() && std::isspace(input[pos])) {
    pos++;
  }

  while (pos < input.length()) {
    // Skip whitespace and commas
    while (pos < input.length() && 
           (std::isspace(input[pos]) || input[pos] == ',')) {
      pos++;
    }
    
    if (pos >= input.length()) {
      break;
    }

    // Check for opening angle bracket
    if (input[pos] != '<') {
      break; // Invalid format, skip this item
    }
    
    size_t itemStart = pos;
    pos++; // Skip opening angle bracket
    
    // Find the matching closing angle bracket
    int bracketCount = 1;
    while (pos < input.length() && bracketCount > 0) {
      if (input[pos] == '<') {
        bracketCount++;
      } else if (input[pos] == '>') {
        bracketCount--;
      }
      pos++;
    }
    
    if (bracketCount == 0) {
      // Found matching closing bracket
      std::string item = input.substr(itemStart, pos - itemStart);
      items.push_back(item);
    } else {
      // No matching closing bracket found, break
      break;
    }
  }

  return items;
}

std::optional<mlir::sdy::TensorShardingAttr> handleSingleSharding(
    const std::string &shardingValue,
    mlir::MLIRContext *context) {

  // Sharding string is "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"
  size_t atPos = shardingValue.find("@");
  size_t commaPos = shardingValue.find(",", atPos);
  std::string meshName = shardingValue.substr(atPos + 1, commaPos - atPos - 1);

  size_t startPos = shardingValue.find("[");
  size_t endPos = shardingValue.rfind("]");
  if (startPos != std::string::npos && endPos != std::string::npos) {
    std::string dimsContent =
        shardingValue.substr(startPos + 1, endPos - startPos - 1);

    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings =
        parseDimensionShardings(dimsContent, context);

    if (!dimShardings.empty()) {
      mlir::sdy::TensorShardingAttr sharding =
          mlir::sdy::TensorShardingAttr::get(context, meshName, dimShardings,
                                            {}, {});
      return sharding;
    }
  }
  return std::nullopt;
}

// Convert dictionary with frontend attributes to dictionary with sdy.sharding.
mlir::DictionaryAttr
convertXlaSdyToSdyDictionary(mlir::MLIRContext *context,
                             mlir::DictionaryAttr currentArgAttrDict) {
  if (!currentArgAttrDict) {
    return mlir::DictionaryAttr::get(context, {});
  }

  if (!currentArgAttrDict.contains(gspmd_utils::kFrontendAttributesAttr)) {
    return currentArgAttrDict;
  }

  mlir::Attribute frontendAttrs =
      currentArgAttrDict.get(gspmd_utils::kFrontendAttributesAttr);
  mlir::DictionaryAttr dictAttr =
      mlir::dyn_cast<mlir::DictionaryAttr>(frontendAttrs);
  if (!dictAttr || !dictAttr.contains(sharding_utils::kXlaSdyShardingAttr)) {
    return currentArgAttrDict;
  }

  mlir::StringAttr shardingStr = mlir::dyn_cast<mlir::StringAttr>(
      dictAttr.get(sharding_utils::kXlaSdyShardingAttr));
  if (!shardingStr) {
    return currentArgAttrDict;
  }

  std::string shardingValue = shardingStr.getValue().str();
  llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
  llvm::copy_if(currentArgAttrDict.getValue(), std::back_inserter(newArgAttrs),
                [&](mlir::NamedAttribute attr) {
                  return attr.getName() !=
                             gspmd_utils::kFrontendAttributesAttr &&
                         attr.getName() != gspmd_utils::kXlaShardingAttr;
                });

  // Find mesh name from the sharding string.
  // Example sharding string: "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"
  // Check for sharding_per_value
  if (shardingValue.find("sharding_per_value") != std::string::npos) {
    // Sharding string is "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
    size_t startPos = shardingValue.find("[");
    size_t endPos = shardingValue.rfind("]");
    std::string shardingList = shardingValue.substr(startPos + 1, endPos - startPos - 1);
    std::vector<std::string> shardingListItems = parseItemList(shardingList);
    llvm::SmallVector<mlir::sdy::TensorShardingAttr> shardingListAttrs;
    for (const std::string &shardingItem : shardingListItems) {
      std::optional<mlir::sdy::TensorShardingAttr> maybeSharding = handleSingleSharding(shardingItem, context);
      if (maybeSharding) {
        shardingListAttrs.push_back(maybeSharding.value());
      }
    }
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(context,
                              mlir::sdy::TensorShardingAttr::name),
        mlir::sdy::TensorShardingPerValueAttr::get(context, shardingListAttrs));
  } else {
    std::optional<mlir::sdy::TensorShardingAttr> maybeSharding = handleSingleSharding(shardingValue, context);
    if (maybeSharding) {
      newArgAttrs.emplace_back(
          mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
          maybeSharding.value());
    }
  }

  return mlir::DictionaryAttr::get(context, newArgAttrs);
}

// Convert all function arguments from frontend attributes format to SDY format.
mlir::LogicalResult convertFrontendAttributesToSDY(mlir::ModuleOp &rootModule,
                                                   mlir::MLIRContext *context) {
  rootModule.walk([&](func::FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      funcOp.setArgAttrs(
          arg.getArgNumber(),
          shardy_utils::convertXlaSdyToSdyDictionary(
              context, funcOp.getArgAttrDict(arg.getArgNumber())));
    }
    // Walk through the funcOp as well
    funcOp->walk([&](Operation *op) {
      mlir::DictionaryAttr newAttrDict =
          shardy_utils::convertXlaSdyToSdyDictionary(
              context, op->getAttrDictionary());
      op->setAttrs(newAttrDict);
    });
  });

  mlir::DictionaryAttr moduleAttrs = rootModule->getAttrDictionary();
  llvm::SmallVector<mlir::NamedAttribute> newModuleAttrs;
  llvm::copy_if(moduleAttrs.getValue(), std::back_inserter(newModuleAttrs),
                [&](mlir::NamedAttribute currentArgAttr) {
                  return currentArgAttr.getName() !=
                         gspmd_utils::kFrontendAttributesAttr;
                });
  rootModule->setAttrs(mlir::DictionaryAttr::get(context, newModuleAttrs));

  return mlir::success();
}

// Convert all stablehlo.custom_call @Sharding ops to sdy.sharding_constraint
// ops.
mlir::LogicalResult
convertCustomCallToShardingConstraint(mlir::ModuleOp &rootModule,
                                      mlir::MLIRContext *context,
                                      mlir::OpBuilder &builder) {
  rootModule.walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
      return;
    }

    // Check call target name to see if it's the one we are interested in.
    mlir::stablehlo::CustomCallOp customCallOp =
        mlir::cast<mlir::stablehlo::CustomCallOp>(op);
    auto callTargetName = customCallOp.getCallTargetNameAttr();
    if (callTargetName != gspmd_utils::kShardingCustomCallTargetName) {
      return;
    }

    mlir::DictionaryAttr newAttrDict =
        shardy_utils::convertXlaSdyToSdyDictionary(
            context, customCallOp->getAttrDictionary());
    mlir::Attribute sdyShardingAttr =
        newAttrDict.get(mlir::sdy::TensorShardingAttr::name);
    mlir::sdy::TensorShardingAttr tensorShardingAttr =
        mlir::cast<mlir::sdy::TensorShardingAttr>(sdyShardingAttr);

    // Create sdy.sharding_constraint op and replace it in place of custom
    // call
    builder.setInsertionPointAfter(customCallOp);
    auto shardingConstraintOp = builder.create<mlir::sdy::ShardingConstraintOp>(
        customCallOp->getLoc(), customCallOp.getResult(0).getType(),
        customCallOp.getOperand(0), tensorShardingAttr);
    customCallOp.getResult(0).replaceAllUsesWith(
        shardingConstraintOp.getResult());
  });

  return mlir::success();
}

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::shardy_utils
