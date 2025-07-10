// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Transforms/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_APPLYSHARDYARGUMENTANNOTATIONSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// This parent class defines how arguments should be analyzed and how the
// argument dictionary should be updated based on the analysis.
// todo: (tapspatel) Currently we only support batch parallel in this pass.
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
  // todo: (tapspatel) Need to generalize largest rank such that batch dim
  // doesn't always have to be with tensors of rank 4.
  // https://github.com/tenstorrent/tt-mlir/issues/3292
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
             "to do analysis on ttir.name tensor annotations.\n";
      return mlir::failure();
    }

    if (!currentArgAttrDict.contains(
            mlir::tt::ttcore::ArgumentTypeAttr::name)) {
      funcOp.emitError("In function ")
          << funcOp.getName() << " argument #: " << arg->getArgNumber()
          << " is not annotated with ttir.name tensor annotations.\n";
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

// Check if tt argument annotations exist in the module.
static bool ttAnnotationsExist(mlir::ModuleOp &rootModule) {
  mlir::WalkResult result = rootModule.walk([&](func::FuncOp funcOp) {
    // Check if ttir.name exists for any of the arguments.
    for (BlockArgument arg : funcOp.getBody().front().getArguments()) {
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

class ApplyShardyArgumentAnnotationsPass
    : public impl::ApplyShardyArgumentAnnotationsPassBase<
          ApplyShardyArgumentAnnotationsPass> {
public:
  using impl::ApplyShardyArgumentAnnotationsPassBase<
      ApplyShardyArgumentAnnotationsPass>::
      ApplyShardyArgumentAnnotationsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Check if the graph is annotated with sharding attributes. If annotated,
    // get the meshOp. If not annotated, insert custom meshOp and sharding
    // annotations.
    llvm::ArrayRef<int64_t> meshShapeRef = *meshShape;

    // Determine whether any existing sharding annotations exist.
    bool gspmdAnnotationsExist = sdy_utils::gspmdAnnotationsExist(rootModule);
    bool sdyAnnotationsExist = sdy_utils::sdyAnnotationsExist(rootModule);
    bool ttArgAnnotationsExist =
        mlir::tt::stablehlo::ttAnnotationsExist(rootModule);

    // If no annotations exist, this is a single chip graph. We do not perform any analysis on it and set a 1x1 mesh.
    if (!sdyAnnotationsExist && !gspmdAnnotationsExist &&
        !ttArgAnnotationsExist && !automaticArgAnalysis) {
      std::string meshName = "mesh";
      sdy_utils::MeshMap meshMap;
      meshMap["x"] = 1;
      meshMap["y"] = 1;
      mlir::sdy::MeshAttr sdyMeshAttr =
          sdy_utils::createMeshAttrFromMeshMap(context, meshMap);
      builder.setInsertionPoint(&(rootModule.getBody()->front()));
      builder.create<mlir::sdy::MeshOp>(builder.getUnknownLoc(),
                                        builder.getStringAttr(meshName),
                                        sdyMeshAttr);
      return;
    }

    // GSPMD annotations.
    if (gspmdAnnotationsExist) {
      rootModule.emitWarning(
          "Applying shardy argument annotations pass does not "
          "support GSPMD annotated module for now.\n");
      return;
    }

    // If shardy annotations exist, we have 2 situations to consider, solved
    // graphs and partially solved graphs. For partially solved graphs, we do
    // nothing. For solved graphs, we want to remove any sdy annotations from
    // the arguments to prevent further analysis.
    if (sdyAnnotationsExist) {
      // Get the shardy mesh op in the root module.
      llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
          sdy_utils::getMeshOps(rootModule);

      if (parsedMeshOps.size() == 0) {
        rootModule.emitError(
            "Shardy annotations exist in the module but there is "
            "no sdy.meshOp. If annotating with sdy, you must "
            "provide a meshOp symbol.\n");
        signalPassFailure();
        return;
      }

      if (parsedMeshOps.size() > 1) {
        rootModule.emitError(
            "Shardy automatic parallelization pass only works on 1 "
            "meshOp for now.\n");
        signalPassFailure();
        return;
      }

      if (meshShapeRef.size() != 0) {
        rootModule.emitWarning(
            "User provided mesh but mesh already exists "
            "in mlir module. Using existing mesh in module.\n");
      }

      // If mesh shape is 1D, we bump it up to 2D with dim0 as 1. This is because TTNN only supports 2D meshes.
      mlir::sdy::MeshAttr parsedMeshAttr = parsedMeshOps[0].getMesh();
      sdy_utils::MeshMap parsedMeshMap = sdy_utils::createMeshMapFromMeshAttr(parsedMeshAttr);
      if (parsedMeshMap.size() == 1) {
        sdy_utils::MeshMap meshMapNew;
        meshMapNew["default"] = 1;
        for (const auto &pair : parsedMeshMap) {
          meshMapNew.insert(pair);
        }
        mlir::sdy::MeshAttr newMeshAttr = sdy_utils::createMeshAttrFromMeshMap(context, meshMapNew);
        sdy_utils::removeMeshOps(rootModule);
        builder.setInsertionPoint(&(rootModule.getBody()->front()));
        builder.create<mlir::sdy::MeshOp>(builder.getUnknownLoc(), builder.getStringAttr(parsedMeshOps[0].getSymName()), newMeshAttr);
      }

      // Check if manual computation op exists. If it does, this is a solved
      // graph and we remove all sdy annotations.
      if (sdy_utils::doesManualComputationOpExist(rootModule)) {
        rootModule.walk([&](func::FuncOp funcOp) {
          sdy_utils::removeSdyTensorShardings(context, funcOp);
        });
      }

      return;
    }

    // If graph is not annotated with sdy tensor shardings, determine how the
    // arguments should be sharded and insert sdy.sharding annotations and
    // sdy.meshOp. Use tt argument annotations if they exist as a guide.
    if (automaticArgAnalysis || ttArgAnnotationsExist) {
      // Remove any sdy meshOps that exists and insert our own to run analysis
      // on.
      sdy_utils::removeMeshOps(rootModule);

      // Check if user provided a mesh.
      if (meshShapeRef.size() == 0) {
        rootModule.emitError("User did not provide a mesh.\n");
        signalPassFailure();
        return;
      }

      // Generate new meshOp based on user provided mesh.
      if (meshShapeRef.size() != 2 || meshShapeRef[0] != 1) {
        rootModule.emitError(
            "Currently, shardy automatic parallel pass only supports 2d mesh "
            "shape and mesh shape dim0 must be 1.\n");
        signalPassFailure();
        return;
      }

      std::string meshName = "mesh";
      sdy_utils::MeshMap meshMap;
      meshMap["default"] = meshShapeRef[0];
      meshMap["batch"] = meshShapeRef[1];
      mlir::sdy::MeshAttr sdyMeshAttr =
          sdy_utils::createMeshAttrFromMeshMap(context, meshMap);
      builder.setInsertionPoint(&(rootModule.getBody()->front()));
      builder.create<mlir::sdy::MeshOp>(builder.getUnknownLoc(),
                                        builder.getStringAttr(meshName),
                                        sdyMeshAttr);

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
            funcOp.emitError("Failed to process arguments in function.\n");
            signalPassFailure();
            return;
          }
        }

        // Once we processed all the elements, we want to iterate through all
        // the arguments again and update it's attributes to add the shardy
        // tensor sharding attribute.
        for (BlockArgument arg : entryBlock.getArguments()) {
          funcOp.setArgAttrs(arg.getArgNumber(),
                             analysis->getUpdatedArgumentDictionaryAttr(
                                 context, funcOp, &arg));
        }
      }

      return;
    }
  }
};

} // namespace mlir::tt::stablehlo
