// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_ANALYZEMESHPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// This parent class defines how arguments should be analyzed and how the
// argument dictionary should be updated based on the analysis.
// TODO (tapspatel): Currently we only support batch parallel in this pass.
// We can extend this to other types of parallelism which will require some
// more analysis.
// https://github.com/tenstorrent/tt-mlir/issues/3289
class ArgumentAnalysis {
public:
  ArgumentAnalysis(int64_t largestRank) : largestRank(largestRank) {}
  virtual mlir::LogicalResult processArgument(BlockArgument *arg,
                                              func::FuncOp &funcOp) = 0;
  virtual mlir::DictionaryAttr getUpdatedArgumentDictionaryAttr(
      mlir::MLIRContext *context, func::FuncOp &funcOp, BlockArgument *arg) = 0;
  virtual ~ArgumentAnalysis() = default;

public:
  int64_t largestRank;
};

// This class is used to analyze arguments if no shard hints are provided (no tt
// arguments are given). It will shard inputs based on batch parallelism.
class BatchParallelismArgumentAnalysis : public ArgumentAnalysis {
public:
  // TODO (tapspatel): Need to generalize largest rank such that batch dim
  // doesn't always have to be with tensors of rank 4.
  // https://github.com/tenstorrent/tt-mlir/issues/3292
  // For example, if we have a 3D tensor with batch dim as 0, we currently don't
  // parallelize across it and will assume all dimensions are replicated. Only
  // if the tensor is 4D, we will parallelize across the batch dim. eg. add: [
  // 32, 256, 256 ] + [ 32, 256, 256 ] where '32' is the batch dim.
  BatchParallelismArgumentAnalysis() : ArgumentAnalysis(4) {}

  // Add an argument to the BatchParallelismArgumentAnalysis and update
  // largestRank.
  mlir::LogicalResult processArgument(BlockArgument *arg,
                                      func::FuncOp &funcOp) override {
    mlir::RankedTensorType tensorType =
        mlir::cast<mlir::RankedTensorType>(arg->getType());
    this->largestRank = std::max(this->largestRank, tensorType.getRank());
    return mlir::success();
  }

  // Get the updated argument dictionary with new sdy.sharding annotations based
  // on how the argument should be sharded.
  mlir::DictionaryAttr
  getUpdatedArgumentDictionaryAttr(mlir::MLIRContext *context,
                                   func::FuncOp &funcOp,
                                   BlockArgument *arg) override {
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

    // Copy the current dictionary if it exists for the op.
    if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber())) {
      newArgAttrs =
          SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
    }

    // Determine sdy.sharding annotation to add to this argument based on
    // largest rank seen.
    mlir::RankedTensorType argType =
        mlir::cast<mlir::RankedTensorType>(arg->getType());
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    if (argType.getRank() == this->largestRank && argType.getShape()[0] != 1) {
      mlir::sdy::AxisRefAttr axisAttr =
          mlir::sdy::AxisRefAttr::get(context, "batch");
      mlir::sdy::DimensionShardingAttr dimShardingAttr =
          mlir::sdy::DimensionShardingAttr::get(context, {axisAttr},
                                                /*is_closed*/ false);
      dimShardings.push_back(dimShardingAttr);

      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank() - 1, full);
    } else {
      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank(), full);
    }

    // Add the shardy sharding attribute to the argument.
    mlir::sdy::TensorShardingAttr sharding = mlir::sdy::TensorShardingAttr::get(
        context, "mesh", dimShardings, {}, {});
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
        sharding);
    return mlir::DictionaryAttr::get(context, newArgAttrs);
  }
};

// This class is used to analyze arguments if shard hints are provided (tt
// argument annotations). It will shard inputs based on batch parallelism.
class BatchParallelismTTArgumentAnalysis : public ArgumentAnalysis {
public:
  BatchParallelismTTArgumentAnalysis() : ArgumentAnalysis(0) {}

  mlir::LogicalResult processArgument(BlockArgument *arg,
                                      func::FuncOp &funcOp) override {
    auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber());

    if (!currentArgAttrDict) {
      funcOp.emitError("In function ")
          << funcOp.getName() << " argument #: " << arg->getArgNumber()
          << " does not have an argument dictionary. This is required in order "
             "to do analysis on ttir.name tensor annotations";
      return mlir::failure();
    }

    if (!currentArgAttrDict.contains(
            mlir::tt::ttcore::ArgumentTypeAttr::name)) {
      funcOp.emitError("In function ")
          << funcOp.getName() << " argument #: " << arg->getArgNumber()
          << " is not annotated with ttir.name tensor annotations";
      return mlir::failure();
    }

    mlir::tt::ttcore::ArgumentTypeAttr argumentTypeAttr =
        mlir::cast<mlir::tt::ttcore::ArgumentTypeAttr>(
            currentArgAttrDict.get(mlir::tt::ttcore::ArgumentTypeAttr::name));
    mlir::tt::ttcore::ArgumentType argTypeValue = argumentTypeAttr.getValue();
    if (argTypeValue == mlir::tt::ttcore::ArgumentType::Input) {
      mlir::RankedTensorType tensorType =
          mlir::cast<mlir::RankedTensorType>(arg->getType());
      this->largestRank = std::max(this->largestRank, tensorType.getRank());
    }

    return mlir::success();
  }

  mlir::DictionaryAttr
  getUpdatedArgumentDictionaryAttr(mlir::MLIRContext *context,
                                   func::FuncOp &funcOp,
                                   BlockArgument *arg) override {
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
    mlir::tt::ttcore::ArgumentType argTypeValue =
        mlir::tt::ttcore::ArgumentType::Default;

    // Copy the current dictionary if it exists for the op.
    if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber())) {
      mlir::tt::ttcore::ArgumentTypeAttr argumentTypeAttr =
          mlir::cast<mlir::tt::ttcore::ArgumentTypeAttr>(
              currentArgAttrDict.get(mlir::tt::ttcore::ArgumentTypeAttr::name));
      argTypeValue = argumentTypeAttr.getValue();
      newArgAttrs =
          SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
    }

    // Determine sdy.sharding annotation to add to this argument based on tt
    // argument type. If it's an input, we annotate it. Otherwise, we keep it
    // open for shardy propogation to fill it in if required.
    mlir::RankedTensorType argType =
        mlir::cast<mlir::RankedTensorType>(arg->getType());
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    if (argType.getRank() == this->largestRank &&
        argTypeValue == mlir::tt::ttcore::ArgumentType::Input) {
      mlir::sdy::AxisRefAttr axisAttr =
          mlir::sdy::AxisRefAttr::get(context, "batch");
      mlir::sdy::DimensionShardingAttr dimShardingAttr =
          mlir::sdy::DimensionShardingAttr::get(context, {axisAttr},
                                                /*is_closed*/ false);
      dimShardings.push_back(dimShardingAttr);

      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank() - 1, full);
    } else {
      mlir::sdy::DimensionShardingAttr full =
          mlir::sdy::DimensionShardingAttr::get(context, {},
                                                /*is_closed*/ false);
      dimShardings.append(argType.getRank(), full);
    }

    // Add the shardy sharding attribute to the argument
    mlir::sdy::TensorShardingAttr sharding = mlir::sdy::TensorShardingAttr::get(
        context, "mesh", dimShardings, {}, {});
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
        sharding);

    return mlir::DictionaryAttr::get(context, newArgAttrs);
  }
};

// Parse axis definitions from SDY mesh string format.
static std::vector<std::pair<std::string, int64_t>>
parseAxisDefinitions(const std::string &axesContent) {
  std::vector<std::pair<std::string, int64_t>> axes;
  size_t pos = 0;

  while (pos < axesContent.length()) {
    while (pos < axesContent.length() &&
           (axesContent[pos] == ' ' || axesContent[pos] == ',')) {
      pos++;
    }
    if (pos >= axesContent.length()) {
      break;
    }

    size_t quoteStart = axesContent.find('"', pos);
    if (quoteStart == std::string::npos) {
      break;
    }

    size_t quoteEnd = axesContent.find('"', quoteStart + 1);
    if (quoteEnd == std::string::npos) {
      break;
    }

    std::string axisName =
        axesContent.substr(quoteStart + 1, quoteEnd - quoteStart - 1);

    size_t equalPos = axesContent.find("=", quoteEnd + 1);
    if (equalPos == std::string::npos) {
      break;
    }

    size_t numStart = equalPos + 1;
    size_t numEnd = axesContent.find_first_of(",] ", numStart);
    if (numEnd == std::string::npos) {
      numEnd = axesContent.length();
    }

    std::string sizeStr = axesContent.substr(numStart, numEnd - numStart);
    while (!sizeStr.empty() &&
           (sizeStr.back() == ' ' || sizeStr.back() == '\t')) {
      sizeStr.pop_back();
    }

    if (!sizeStr.empty()) {
      int64_t size = std::stoi(sizeStr);
      axes.push_back({axisName, size});
    }

    pos = numEnd;
  }

  return axes;
}

// Parse mesh information from mhlo.frontend_attributes and create sdy.mesh.
static mlir::LogicalResult
parseMeshFromFrontendAttributes(mlir::ModuleOp &rootModule,
                                mlir::MLIRContext *context) {
  auto moduleAttrs = rootModule->getAttrDictionary();
  if (!moduleAttrs.contains("mhlo.frontend_attributes")) {
    return mlir::success();
  }

  auto frontendAttrs = moduleAttrs.get("mhlo.frontend_attributes");
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(frontendAttrs);
  if (!dictAttr || !dictAttr.contains("xla.sdy.meshes")) {
    return mlir::success();
  }

  auto meshesStr =
      mlir::dyn_cast<mlir::StringAttr>(dictAttr.get("xla.sdy.meshes"));
  if (!meshesStr) {
    return mlir::success();
  }

  std::string meshStr = meshesStr.getValue().str();
  std::string meshName = "mesh";

  size_t startPos = meshStr.find("<[");
  size_t endPos = meshStr.find("]>");
  if (startPos == std::string::npos || endPos == std::string::npos) {
    return mlir::success();
  }

  std::string axesContent = meshStr.substr(startPos + 2, endPos - startPos - 2);
  std::vector<std::pair<std::string, int64_t>> axes =
      parseAxisDefinitions(axesContent);

  if (axes.size() == 1) {
    shardy_utils::addMeshToModule(rootModule, meshName,
                                  axes[0].first + "_updated", axes[0].first, 1,
                                  axes[0].second);
  } else if (axes.size() == 2) {
    shardy_utils::addMeshToModule(rootModule, meshName, axes[0].first,
                                  axes[1].first, axes[0].second,
                                  axes[1].second);
  } else {
    rootModule.emitError(
        "Unsupported mesh configuration: only 1D and 2D meshes are supported");
    return mlir::failure();
  }

  return mlir::success();
}

// Parse dimension shardings from SDY sharding string format.
static llvm::SmallVector<mlir::sdy::DimensionShardingAttr>
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

// Convert function argument from mhlo.frontend_attributes to sdy.sharding.
static mlir::LogicalResult convertArgumentSharding(mlir::func::FuncOp &funcOp,
                                                   mlir::BlockArgument &arg,
                                                   mlir::MLIRContext *context) {
  auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber());
  if (!currentArgAttrDict ||
      !currentArgAttrDict.contains("mhlo.frontend_attributes")) {
    return mlir::success();
  }

  auto frontendAttrs = currentArgAttrDict.get("mhlo.frontend_attributes");
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(frontendAttrs);
  if (!dictAttr || !dictAttr.contains("xla.sdy.sharding")) {
    return mlir::success();
  }

  auto shardingStr =
      mlir::dyn_cast<mlir::StringAttr>(dictAttr.get("xla.sdy.sharding"));
  if (!shardingStr) {
    return mlir::success();
  }

  std::string shardingValue = shardingStr.getValue().str();

  llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
  for (auto attr : currentArgAttrDict) {
    if (attr.getName() != "mhlo.frontend_attributes" &&
        attr.getName() != "mhlo.sharding") {
      newArgAttrs.push_back(attr);
    }
  }

  std::string meshName = "mesh";

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
      newArgAttrs.emplace_back(
          mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
          sharding);
    }
  }

  funcOp.setArgAttrs(arg.getArgNumber(),
                     mlir::DictionaryAttr::get(context, newArgAttrs));
  return mlir::success();
}

// Convert all function arguments from frontend attributes format to SDY format.
static mlir::LogicalResult
convertFrontendAttributesToSDY(mlir::ModuleOp &rootModule,
                               mlir::MLIRContext *context) {
  mlir::WalkResult result = rootModule.walk([&](func::FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      if (mlir::failed(convertArgumentSharding(funcOp, arg, context))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return mlir::failure();
  }

  auto moduleAttrs = rootModule->getAttrDictionary();
  llvm::SmallVector<mlir::NamedAttribute> newModuleAttrs;
  for (auto attr : moduleAttrs) {
    if (attr.getName() != "mhlo.frontend_attributes") {
      newModuleAttrs.push_back(attr);
    }
  }
  rootModule->setAttrs(mlir::DictionaryAttr::get(context, newModuleAttrs));

  return mlir::success();
}

// Check if tt argument annotations exist in the module.
static bool ttAnnotationsExist(mlir::ModuleOp &rootModule) {
  mlir::WalkResult result = rootModule.walk([&](func::FuncOp funcOp) {
    // Check if ttir.name exists for any of the arguments.
    for (BlockArgument arg : funcOp.getArguments()) {
      if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
        if (currentArgAttrDict.contains(ttcore::ArgumentTypeAttr::name)) {
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

// This pass support 2 types of graphs, single chip and multi chip.
// Multi chip graphs will have either shardy or GSPMD annotations. In this case,
// we will analyze the graph, extract the mesh and determine the validity of the
// mesh. For shardy graphs, we will also determine whether the graph is solved
// or not. If so, we will remove all sdy tensor shardings from the arguments and
// results. Single chip graphs will not have any annotations. If the user
// enabled automatic argument analysis, we will analyze the arguments and
// determine how they should be sharded based on the mesh shape provided by the
// user. Otherwise, we will add a 1x1 mesh to signify this module is a single
// chip graph.
class AnalyzeMeshPass : public impl::AnalyzeMeshPassBase<AnalyzeMeshPass> {
public:
  using impl::AnalyzeMeshPassBase<AnalyzeMeshPass>::AnalyzeMeshPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    bool gspmdAnnotationsExist = gspmd_utils::gspmdAnnotationsExist(rootModule);
    bool sdyAnnotationsExist = shardy_utils::sdyAnnotationsExist(rootModule);

    // If shardy annotations exist, check for validity of the mesh. Shardy will
    // already populate a sdy.meshOp.
    if (sdyAnnotationsExist) {
      // Get the shardy mesh op in the root module.
      llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
          shardy_utils::getMeshOps(rootModule);

      if (parsedMeshOps.size() > 1) {
        rootModule.emitError("TTMLIR compiler only supports single mesh");
        signalPassFailure();
        return;
      }

      llvm::SmallVector<int64_t> originalMeshShape =
          shardy_utils::getMeshShapeFromMeshAttr(
              parsedMeshOps[0].getMeshAttr());

      if (originalMeshShape.size() > 2) {
        rootModule.emitError("Currently, only 1D and 2D meshes are supported");
        signalPassFailure();
        return;
      }

      // If the mesh is not 2D (or if the mesh is empty), we need to update the
      // mesh to either insert a 1x1 mesh (if the mesh is empty), or bump up a
      // 1D mesh to 1xn.
      llvm::SmallVector<int64_t> newMeshShape = originalMeshShape;
      if (newMeshShape.size() == 0) {
        newMeshShape = {1, 1};
      } else if (newMeshShape.size() == 1) {
        newMeshShape = {1, newMeshShape[0]};
      }

      // Check for validity of the mesh.
      if (failed(sharding_utils::checkValidMesh(newMeshShape))) {
        rootModule.emitError("Mesh is not valid");
        signalPassFailure();
        return;
      }

      // If mesh shape was changed, insert the new mesh shape.
      if (originalMeshShape != newMeshShape) {
        // Get the old mesh axis name. We will make the new dim0 "1" axis name
        // from the old mesh axis name. This is to ensure that we don't have
        // duplicate axis names in the mesh.
        std::string existingAxisName = "default";
        for (auto meshAxisAttr : parsedMeshOps[0].getMeshAttr().getAxes()) {
          existingAxisName = meshAxisAttr.getName().str();
        }
        std::string newAxisName = existingAxisName + "_updated";

        // Create the new mesh.
        std::string meshOpName = parsedMeshOps[0].getSymName().str();
        shardy_utils::removeMeshOps(rootModule);
        shardy_utils::addMeshToModule(rootModule, meshOpName, newAxisName,
                                      existingAxisName, newMeshShape[0],
                                      newMeshShape[1]);
      }

      // Check if the graph is already solved by shardy. If so, we will remove
      // all sdy tensor shardings from the arguments and results.
      if (shardy_utils::isGraphSolved(rootModule)) {
        rootModule.walk([&](func::FuncOp funcOp) {
          shardy_utils::removeSdyTensorShardings(context, funcOp);
        });
      }

      return;
    }

    // If gspmd annotations exist, analyze the graph and determine the mesh
    // shape. Then we can add it to the module as a sdy.meshOp.
    if (gspmdAnnotationsExist) {
      // Check if we have frontend_attributes with sdy information
      bool hasFrontendSdyAttrs = false;
      rootModule.walk([&](func::FuncOp funcOp) {
        for (BlockArgument arg : funcOp.getArguments()) {
          if (auto currentArgAttrDict =
                  funcOp.getArgAttrDict(arg.getArgNumber())) {
            if (currentArgAttrDict.contains("mhlo.frontend_attributes")) {
              auto frontendAttrs =
                  currentArgAttrDict.get("mhlo.frontend_attributes");
              if (auto dictAttr =
                      mlir::dyn_cast<mlir::DictionaryAttr>(frontendAttrs)) {
                if (dictAttr.contains("xla.sdy.sharding")) {
                  hasFrontendSdyAttrs = true;
                  return WalkResult::interrupt();
                }
              }
            }
          }
        }
        return WalkResult::advance();
      });

      if (hasFrontendSdyAttrs) {
        // Handle frontend_attributes conversion
        // Parse mesh information from module attributes and create sdy.mesh
        if (mlir::failed(
                parseMeshFromFrontendAttributes(rootModule, context))) {
          signalPassFailure();
          return;
        }

        // Convert function arguments from frontend_attributes to sdy.sharding
        if (mlir::failed(convertFrontendAttributesToSDY(rootModule, context))) {
          signalPassFailure();
          return;
        }

        return;
      }

      llvm::Expected<llvm::SmallVector<llvm::SmallVector<int64_t>>>
          parsedMeshes = gspmd_utils::parseMeshesFromGspmdModule(rootModule);

      if (auto err = parsedMeshes.takeError()) {
        rootModule.emitError("Could not query meshes from gspmd module");
        signalPassFailure();
        return;
      }

      // Create the new mesh and add it to the module.
      llvm::SmallVector<int64_t> meshShape = (*parsedMeshes)[0];
      shardy_utils::addMeshToModule(rootModule, "mesh", "x", "y", meshShape[0],
                                    meshShape[1]);
      return;
    }

    // If the user did not enable automatic argument analysis, we will not do
    // any analysis and just add a 1x1 mesh to the module.
    if (!automaticArgAnalysis) {
      // Create a new 1x1 mesh.
      shardy_utils::addMeshToModule(rootModule, "mesh", "x", "y",
                                    /*firstAxisSize=*/1, /*secondAxisSize=*/1);
      return;
    }

    // If we are here, it means we are doing automatic argument analysis.
    llvm::ArrayRef<int64_t> meshShapeRef = *meshShape;
    bool ttArgAnnotationsExist =
        mlir::tt::stablehlo::ttAnnotationsExist(rootModule);

    // Remove any sdy meshOps that may exist and insert our own to run analysis
    // on.
    shardy_utils::removeMeshOps(rootModule);

    // Check if user provided a mesh.
    if (meshShapeRef.size() == 0) {
      rootModule.emitError("User did not provide a mesh");
      signalPassFailure();
      return;
    }

    // Generate new meshOp based on user provided mesh.
    if (meshShapeRef.size() != 2 || meshShapeRef[0] != 1) {
      rootModule.emitError(
          "Currently, automatic arg analysis only supports 2d mesh "
          "shape and mesh shape dim0 must be 1");
      signalPassFailure();
      return;
    }

    shardy_utils::addMeshToModule(rootModule, "mesh", "default", "batch",
                                  meshShapeRef[0], meshShapeRef[1]);
    // Iterate through all arguments in the module and analyze them to
    // determine how they should be sharded.
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (!mlir::isa<func::FuncOp>(op)) {
        continue;
      }

      // Get appropriate analysis manager.
      func::FuncOp funcOp = mlir::cast<func::FuncOp>(op);
      std::unique_ptr<ArgumentAnalysis> analysis;
      if (ttArgAnnotationsExist) {
        analysis = std::make_unique<BatchParallelismTTArgumentAnalysis>();
      } else if (automaticArgAnalysis) {
        analysis = std::make_unique<BatchParallelismArgumentAnalysis>();
      }

      Block &entryBlock = funcOp.getBody().front();

      // Process each argument and add it to the analysis manager.
      for (BlockArgument arg : entryBlock.getArguments()) {
        if (mlir::failed(analysis->processArgument(&arg, funcOp))) {
          funcOp.emitError("Failed to process arguments in function");
          signalPassFailure();
          return;
        }
      }

      // Once we processed all the elements, we want to iterate through all
      // the arguments again and update it's attributes to add the shardy
      // tensor sharding attribute.
      for (BlockArgument arg : entryBlock.getArguments()) {
        funcOp.setArgAttrs(
            arg.getArgNumber(),
            analysis->getUpdatedArgumentDictionaryAttr(context, funcOp, &arg));
      }
    }
  }
};

} // namespace mlir::tt::stablehlo
