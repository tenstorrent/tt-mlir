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
// If createIfMissing is true, create default sharding attributes (replicated
// on) for any arguments that do not have sdy.sharding annotations.
llvm::SmallVector<mlir::sdy::TensorShardingAttr>
getInShardingAttrs(MLIRContext *context, func::FuncOp &funcOp,
                   mlir::sdy::MeshOp &globalMeshOp, bool createIfMissing) {
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> inShardingAttrs;

  auto createDefaultShardingAttr = [&](uint32_t argNum) {
    if (!createIfMissing) {
      return mlir::sdy::TensorShardingAttr();
    }
    funcOp.emitWarning("In function ")
        << funcOp.getName() << " argument #: " << argNum
        << " is not annotated with sdy.sharding. Using default "
           "sharding annotation (ie all dimensions replicated)";
    mlir::sdy::TensorShardingAttr shardingAttr =
        shardy_utils::getDefaultTensorSdyShardingAttr(
            context, globalMeshOp.getSymName(),
            funcOp.getArgument(argNum).getType());

    // Set argument with it's default sdy.sharding attribute
    mlir::DictionaryAttr newDictAttr =
        shardy_utils::addDictionaryAttrSdyShardingAnnotation(
            context, shardingAttr,
            funcOp.getArgAttrDict(funcOp.getArgument(argNum).getArgNumber()));
    funcOp.setArgAttrs(argNum, newDictAttr);

    return shardingAttr;
  };

  for (auto arg : funcOp.getArguments()) {
    // Get the tensor sharding attribute from argument dictionary.
    mlir::sdy::TensorShardingAttr shardingAttr;
    auto argNumber = arg.getArgNumber();
    if (auto argAttrDict = funcOp.getArgAttrDict(argNumber)) {
      shardingAttr = mlir::dyn_cast_if_present<mlir::sdy::TensorShardingAttr>(
          argAttrDict.get(mlir::sdy::TensorShardingAttr::name));

      if (!shardingAttr) {
        shardingAttr = createDefaultShardingAttr(argNumber);
      }
    } else {
      shardingAttr = createDefaultShardingAttr(argNumber);
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

bool isShardedModule(mlir::ModuleOp &module) {
  llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
      shardy_utils::getMeshOps(module);
  if (parsedMeshOps.empty()) {
    return false;
  }
  auto globalMeshOp = parsedMeshOps[0];
  for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
    for (auto inSharding :
         getInShardingAttrs(funcOp.getContext(), funcOp, globalMeshOp, false)) {
      if (!inSharding) {
        continue;
      }
      if (!isFullyReplicatedTensor(inSharding)) {
        return true;
      }
    }

    for (auto outSharding :
         getOutShardingAttrs(funcOp.getContext(), funcOp, globalMeshOp)) {
      if (!isFullyReplicatedTensor(outSharding)) {
        return true;
      }
    }
  }
  return false;
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

    // Parse priority if present (e.g., "}p0" or "}p1")
    std::optional<int64_t> priority = std::nullopt;
    size_t afterBrace = braceEnd + 1;
    if (afterBrace < dimsContent.length() && dimsContent[afterBrace] == 'p') {
      size_t numStart = afterBrace + 1;
      size_t numEnd = numStart;
      while (numEnd < dimsContent.length() &&
             std::isdigit(dimsContent[numEnd])) {
        ++numEnd;
      }
      if (numEnd > numStart) {
        priority = std::stoll(dimsContent.substr(numStart, numEnd - numStart));
      }
    }

    if (dimContent.empty()) {
      // "#sdy.sharding<@mesh, [{}]>" or "{?}" (open, empty)
      dimShardings.push_back(
          mlir::sdy::DimensionShardingAttr::get(context, {}, true, priority));
    } else {
      llvm::SmallVector<mlir::sdy::AxisRefAttr> axisRefs;
      size_t axisPos = 0;
      bool isClosed = dimContent.find("?") == std::string::npos;
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

      dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
          context, axisRefs, isClosed, priority));
    }

    pos = braceEnd + 1;
  }

  return dimShardings;
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

    mlir::sdy::TensorShardingAttr sharding = mlir::sdy::TensorShardingAttr::get(
        context, meshName, dimShardings, {}, {});
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
        sharding);
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

// Replace mesh_idx_N placeholders with actual axis names from mesh.
// e.g., "mesh_idx_0" -> "x", "mesh_idx_1" -> "y" or "mesh_idx_0" -> "_axis_0",
// "mesh_idx_1" -> "_axis_1"
static std::string replaceMeshIdxPlaceholders(
    const std::string &shardingStr,
    const llvm::SmallVector<mlir::sdy::MeshAxisAttr> &meshAxes) {
  std::string result = shardingStr;
  constexpr llvm::StringLiteral kMeshIdxPrefix = "mesh_idx_";

  for (size_t i = 0; i < meshAxes.size(); ++i) {
    std::string placeholder = (kMeshIdxPrefix + llvm::Twine(i)).str();
    std::string axisName = meshAxes[i].getName().str();

    // Replace all occurrences of placeholder with actual axis name
    size_t pos = 0;
    while ((pos = result.find(placeholder, pos)) != std::string::npos) {
      result.replace(pos, placeholder.length(), axisName);
      pos += axisName.length();
    }
  }
  return result;
}

// Convert all stablehlo.custom_call @Sharding, @tt.sharding_constraint, and
// @xla.sdy.FuncResultSharding ops to sdy.sharding_constraint ops.
mlir::LogicalResult
convertCustomCallToShardingConstraint(mlir::ModuleOp &rootModule,
                                      mlir::MLIRContext *context,
                                      mlir::OpBuilder &builder) {
  // Get mesh axes for placeholder replacement.
  llvm::SmallVector<mlir::sdy::MeshAxisAttr> meshAxes;
  llvm::SmallVector<mlir::sdy::MeshOp> meshOps = getMeshOps(rootModule);
  if (!meshOps.empty()) {
    meshAxes = llvm::to_vector(meshOps[0].getMeshAttr().getAxes());
  }

  llvm::SmallVector<mlir::stablehlo::CustomCallOp> opsToErase;
  bool hasFuncResultSharding = false;

  rootModule.walk([&](mlir::stablehlo::CustomCallOp customCallOp) {
    auto callTargetName = customCallOp.getCallTargetNameAttr();
    // Handle @Sharding (from xs.mark_sharding), @tt.sharding_constraint
    // (from custom ops), and @xla.sdy.FuncResultSharding (result shardings).
    if (callTargetName != gspmd_utils::kShardingCustomCallTargetName &&
        callTargetName != sharding_utils::kTTShardingConstraintTargetName &&
        callTargetName != shardy_utils::kFuncResultShardingTargetName) {
      return;
    }

    // For tt.sharding_constraint, replace mesh_idx_N placeholders with actual
    // axis names before parsing.
    mlir::DictionaryAttr attrDict = customCallOp->getAttrDictionary();
    if (callTargetName == sharding_utils::kTTShardingConstraintTargetName &&
        !meshAxes.empty()) {
      if (auto frontendAttrs = attrDict.getAs<mlir::DictionaryAttr>(
              gspmd_utils::kFrontendAttributesAttr)) {
        if (auto sdyShardingStr = frontendAttrs.getAs<mlir::StringAttr>(
                sharding_utils::kXlaSdyShardingAttr)) {
          std::string replacedStr = replaceMeshIdxPlaceholders(
              sdyShardingStr.getValue().str(), meshAxes);
          // Create new frontend_attributes with replaced sharding string.
          llvm::SmallVector<mlir::NamedAttribute> newFrontendAttrs;
          for (auto attr : frontendAttrs) {
            if (attr.getName() == sharding_utils::kXlaSdyShardingAttr) {
              newFrontendAttrs.push_back(mlir::NamedAttribute(
                  attr.getName(), mlir::StringAttr::get(context, replacedStr)));
            } else {
              newFrontendAttrs.push_back(attr);
            }
          }
          // Create new attrDict with updated frontend_attributes.
          llvm::SmallVector<mlir::NamedAttribute> newAttrs;
          for (auto attr : attrDict) {
            if (attr.getName() == gspmd_utils::kFrontendAttributesAttr) {
              newAttrs.push_back(mlir::NamedAttribute(
                  attr.getName(),
                  mlir::DictionaryAttr::get(context, newFrontendAttrs)));
            } else {
              newAttrs.push_back(attr);
            }
          }
          attrDict = mlir::DictionaryAttr::get(context, newAttrs);
        }
      }
    }

    mlir::DictionaryAttr newAttrDict =
        shardy_utils::convertXlaSdyToSdyDictionary(context, attrDict);
    mlir::Attribute sdyShardingAttr =
        newAttrDict.get(mlir::sdy::TensorShardingAttr::name);

    // If no sharding attribute found, skip this op - it uses a different
    // sharding format (e.g., mhlo.sharding) that is handled by AnalyzeMeshPass.
    if (!sdyShardingAttr) {
      return;
    }

    mlir::sdy::TensorShardingAttr tensorShardingAttr =
        mlir::cast<mlir::sdy::TensorShardingAttr>(sdyShardingAttr);

    // Create sdy.sharding_constraint op and replace it in place of custom call.
    builder.setInsertionPointAfter(customCallOp);
    auto shardingConstraintOp = builder.create<mlir::sdy::ShardingConstraintOp>(
        customCallOp->getLoc(), customCallOp.getResult(0).getType(),
        customCallOp.getOperand(0), tensorShardingAttr);
    customCallOp.getResult(0).replaceAllUsesWith(
        shardingConstraintOp.getResult());

    // Only erase FuncResultSharding ops, others are needed by AnalyzeMeshPass.
    if (callTargetName == shardy_utils::kFuncResultShardingTargetName) {
      hasFuncResultSharding = true;
      opsToErase.push_back(customCallOp);
    }
  });

  for (auto op : opsToErase) {
    op.erase();
  }

  // Remove mhlo.sharding from function results to allow
  // WrapUnderManualComputation to wrap the function body (gspmdAnnotationsExist
  // checks for mhlo.sharding).
  if (hasFuncResultSharding) {
    rootModule.walk([&](func::FuncOp funcOp) {
      for (unsigned i = 0; i < funcOp.getNumResults(); i++) {
        funcOp.removeResultAttr(
            i, mlir::StringAttr::get(context, gspmd_utils::kXlaShardingAttr));
      }
    });
  }

  return mlir::success();
}

std::optional<mlir::DenseElementsAttr> tryGetPeriodicShardSlice(
    mlir::DenseElementsAttr globalAttr, mlir::RankedTensorType localType,
    mlir::sdy::TensorShardingAttr sharding, mlir::sdy::MeshOp meshOp) {

  auto oldType = mlir::cast<mlir::RankedTensorType>(globalAttr.getType());
  auto globalShape = oldType.getShape();

  // 1. Identify Sharding Dimension & Count
  int64_t shardingDim = -1;
  int64_t numShards = 1;

  // We assume single-axis sharding for this optimization
  llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
      sharding.getDimShardings();

  for (auto [dimIdx, dimSharding] : llvm::enumerate(dimShardings)) {
    auto axisRefs = dimSharding.getAxes();
    if (!axisRefs.empty()) {
      shardingDim = dimIdx;
      auto axisName = axisRefs[0].getName();

      for (auto meshAxis : meshOp.getMesh().getAxes()) {
        if (meshAxis.getName() == axisName) {
          numShards = meshAxis.getSize();
          break;
        }
      }
      break;
    }
  }

  // If no valid sharding found, we cannot proceed
  if (shardingDim == -1 || numShards <= 1) {
    return std::nullopt;
  }

  // 2. Pre-calculate Dimensions and Strides
  int64_t totalDimSize = globalShape[shardingDim];
  int64_t shardDimSize = totalDimSize / numShards;

  int64_t innerBlockSize = 1;
  for (int64_t i = shardingDim + 1;
       i < static_cast<int64_t>(globalShape.size()); ++i) {
    innerBlockSize *= globalShape[i];
  }

  int64_t outerCount = 1;
  for (int64_t i = 0; i < shardingDim; ++i) {
    outerCount *= globalShape[i];
  }

  // 3. Verify Periodicity
  // We check if Shard K (k=1..N) contains the same data as Shard 0
  auto globalValues = globalAttr.getValues<mlir::Attribute>();
  int64_t shardStep = shardDimSize * innerBlockSize;
  int64_t rowSize = totalDimSize * innerBlockSize;

  for (int64_t out = 0; out < outerCount; ++out) {
    int64_t rowStart = out * rowSize;

    for (int64_t k = 1; k < numShards; ++k) {
      // Compare Shard 0 vs Shard K block by block
      for (int64_t s = 0; s < shardDimSize; ++s) {
        for (int64_t inner = 0; inner < innerBlockSize; ++inner) {
          int64_t offset0 = rowStart + (s * innerBlockSize) + inner;
          int64_t offsetK = offset0 + (k * shardStep);

          if (globalValues[offset0] != globalValues[offsetK]) {
            return std::nullopt; // Verification failed: Data is different
          }
        }
      }
    }
  }

  // 4. Construct New Attribute (Slice 0)
  // Since verification passed, we copy only the data for Shard 0
  std::vector<mlir::Attribute> newValues;
  newValues.reserve(localType.getNumElements());

  for (int64_t out = 0; out < outerCount; ++out) {
    int64_t rowStart = out * rowSize;

    // Only iterate over the first shard (s < shardDimSize)
    for (int64_t s = 0; s < shardDimSize; ++s) {
      for (int64_t inner = 0; inner < innerBlockSize; ++inner) {
        int64_t idx = rowStart + (s * innerBlockSize) + inner;
        newValues.push_back(globalValues[idx]);
      }
    }
  }

  return mlir::DenseElementsAttr::get(localType, newValues);
}

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::shardy_utils
