// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
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

// Reads sharding for a manual_computation region argument.
// Returns nullptr if not available.
mlir::sdy::TensorShardingAttr
getShardingForManualComputationArg(mlir::BlockArgument barg) {
  mlir::sdy::TensorShardingAttr result = nullptr;

  mlir::Operation *parent = barg.getOwner()->getParentOp();
  auto mc = llvm::dyn_cast_or_null<mlir::sdy::ManualComputationOp>(parent);
  if (mc) {
    auto spv = mlir::dyn_cast<mlir::sdy::TensorShardingPerValueAttr>(
        mc.getInShardings());
    if (spv) {
      auto shardings = spv.getShardings(); // ArrayRef<TensorShardingAttr>
      unsigned idx = barg.getArgNumber();
      if (idx < shardings.size()) {
        result = shardings[idx];
      }
    }
  }

  return result;
}

// Reads sharding for an OpResult via op-level attributes.
// Prefers sdy.sharding_per_value, then falls back to sdy.sharding.
// Returns nullptr if not available.
mlir::sdy::TensorShardingAttr getShardingForOpResult(mlir::OpResult res) {
  mlir::sdy::TensorShardingAttr result = nullptr;

  mlir::Operation *owner = res.getOwner();
  unsigned resNo = res.getResultNumber();

  auto spv = owner->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
      "sdy.sharding_per_value");
  if (spv) {
    auto shardings = spv.getShardings(); // ArrayRef<TensorShardingAttr>
    if (!shardings.empty()) {
      unsigned i = (resNo < shardings.size()) ? resNo : 0u; // broadcast-safe
      result = shardings[i];
    }
  }

  if (!result) {
    auto ts =
        owner->getAttrOfType<mlir::sdy::TensorShardingAttr>("sdy.sharding");
    if (ts) {
      result = ts;
    }
  }

  return result;
}

// Returns the TensorShardingAttr for a given Value, if available.
//
// This function checks for sharding annotations in the following order:
//   - sdy.manual_computation region arguments (via
//   getShardingForManualComputationArg)
//   - func.func entry-block arguments (via getInShardingAttrs)
//   - OpResults (using per-value or op-level sharding attributes)
//
// If no sharding annotation is found, returns a default "full replicate"
// sharding attribute.
mlir::sdy::TensorShardingAttr getShardingAttr(mlir::Value v,
                                              mlir::sdy::MeshOp globalMeshOp) {
  mlir::sdy::TensorShardingAttr result = nullptr;

  if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    // manual_computation region argument
    result = getShardingForManualComputationArg(barg);

    // func.func entry-block argument (only if still unresolved)
    if (!result) {
      mlir::Operation *parent = barg.getOwner()->getParentOp();
      if (parent) {
        auto funcOp = parent->getParentOfType<mlir::func::FuncOp>();
        if (funcOp && parent == funcOp.getOperation()) {
          auto inAttrs =
              getInShardingAttrs(funcOp.getContext(), funcOp, globalMeshOp);
          unsigned i = barg.getArgNumber();
          if (i < inAttrs.size()) {
            result = inAttrs[i];
          }
        }
      }
    }
  } else if (auto res = mlir::dyn_cast<mlir::OpResult>(v)) {
    result = getShardingForOpResult(res);
  }
  // If still unresolved, fall back to "full replicate"
  if (!result) {
    result = shardy_utils::getDefaultTensorSdyShardingAttr(
        v.getContext(), globalMeshOp.getSymName(), v.getType());
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

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::shardy_utils
