// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

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

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyUtils.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_SHARDYAUTOMATICPARALLELIZATION
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

/*
StableHLO graph can be
a. no annotation
b. gspmd annotations
c. sdy annotations

Algorithm:
1. Check if the graph is annotated with sharding attributes.
  a. If it's annotated with gspmd, throw pass error since this pass doesn't
support that yet (need to add a conversion from gspmd into sdy). b. If it's
annotated with sdy, we get the meshOp from the module. c. If it's not annotated,
we insert custom meshOp and sharding annotations for batch parallelization.
2. Run sdy sharding propogation pass.
3. Wrap all operations under a sdy.manual_computationOp.
4. Run topological sort on the graph and update all shapes with a new shape
based on their sharding annotation determine by the sdy sharding propagation
pass.
5. Remove all sdy annotations since analysis is complete.
6. Remove any dead operations that are no longer needed.
7. Close tensor shardings and drop replicated axes.
*/

// This class is used to analyze arguments if no shard hints are provided.
class ArgumentAnalysis {
public:
  // todo: (tapspatel) Need to generalize largest rank such that batch dim
  // doesn't always have to be with tensors of rank 4.
  ArgumentAnalysis() : largestRank(4) {}

  // Add an argument to the argumentAnalysis and update largestRank.
  void processArgument(BlockArgument *arg) {
    mlir::RankedTensorType tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(arg->getType());
    this->largestRank = std::max(this->largestRank, tensorType.getRank());
  }

  // Get the updated argument dictionary with new sdy.sharding annotations based
  // on how the argument should be sharded.
  mlir::DictionaryAttr
  getUpdatedArgumentDictionaryAttr(mlir::MLIRContext *context,
                                   func::FuncOp &funcOp, BlockArgument *arg) {
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

    // Copy the current dictionary if it exists for the op.
    if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber())) {
      if (currentArgAttrDict) {
        newArgAttrs =
            SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
      }
    }

    // Determine sdy.sharding annotation to add to this argument based on
    // largest rank seen.
    mlir::RankedTensorType argType =
        mlir::dyn_cast<mlir::RankedTensorType>(arg->getType());
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    // todo: (tapspatel) Currently we only support batch parallel in this pass.
    // We can extend this to other types of parallelism which will require some
    // more analysis.
    if (argType.getRank() == this->largestRank && argType.getShape()[0] != 1) {
      mlir::sdy::AxisRefAttr axisAttr =
          mlir::sdy::AxisRefAttr::get(context, "batch");
      mlir::sdy::DimensionShardingAttr dimShardingAttr =
          mlir::sdy::DimensionShardingAttr::get(context, {axisAttr}, true);
      dimShardings.push_back(dimShardingAttr);

      for (uint32_t i = 0; i < argType.getRank() - 1; i++) {
        dimShardings.push_back(
            mlir::sdy::DimensionShardingAttr::get(context, {}, true));
      }
    } else {
      for (uint32_t i = 0; i < argType.getRank(); i++) {
        dimShardings.push_back(
            mlir::sdy::DimensionShardingAttr::get(context, {}, true));
      }
    }

    // Add the shardy sharding attribute to the argument
    mlir::sdy::TensorShardingAttr sharding =
        mlir::sdy::TensorShardingAttr::get(context, "mesh", dimShardings, {});
    newArgAttrs.emplace_back(
        mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name),
        sharding);
    return mlir::DictionaryAttr::get(context, newArgAttrs);
  }

private:
  int64_t largestRank;
};

class ShardyAutomaticParallelization
    : public impl::ShardyAutomaticParallelizationBase<
          ShardyAutomaticParallelization> {
public:
  using impl::ShardyAutomaticParallelizationBase<
      ShardyAutomaticParallelization>::ShardyAutomaticParallelizationBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);
    auto pm = PassManager::on<ModuleOp>(context);

    // Inline all operations to make analysis easier.
    pm.addPass(mlir::createInlinerPass());
    if (failed(pm.run(rootModule))) {
      mlir::emitWarning(builder.getUnknownLoc(), "Failed to run inliner pass.");
      signalPassFailure();
      return;
    }
    pm.clear();

    // Check if the graph is annotated with sharding attributes. If annotated,
    // get the meshOp. If not annotated, insert custom meshOp and sharding
    // annotations.
    mlir::sdy::MeshOp globalMeshOp;
    llvm::ArrayRef<int64_t> meshShapeRef = *meshShape;
    bool userProvidedMesh = meshShapeRef.size() != 0 ? true : false;

    // Determine whether any existing sharding annotations exist.
    bool gspmdAnnotationsExist = sdy_utils::gspmdAnnotationsExist(rootModule);
    bool sdyAnnotationsExist = sdy_utils::sdyAnnotationsExist(rootModule);

    if (gspmdAnnotationsExist) {
      mlir::emitWarning(builder.getUnknownLoc(),
                        "Shardy automatic parallelization pass does not "
                        "support GSPMD annotated module.\n");
      signalPassFailure();
      return;
    }

    if (sdyAnnotationsExist) {
      // Get the shardy mesh op in the root module.
      uint32_t numMeshes = sdy_utils::getNumMeshes(rootModule);

      if (numMeshes == 0) {
        mlir::emitError(builder.getUnknownLoc(),
                        "Shardy annotations exist in the module but there is "
                        "no sdy.meshOp. If annotating with sdy, you must "
                        "provide a meshOp symbol.\n");
        signalPassFailure();
        return;
      } else if (numMeshes > 1) {
        mlir::emitError(builder.getUnknownLoc(),
                        "Shardy automatic parallelization pass only works on 1 "
                        "meshOp for now.\n");
        signalPassFailure();
        return;
      }

      if (userProvidedMesh) {
        mlir::emitWarning(builder.getUnknownLoc(),
                          "Warning: user provided mesh but mesh already exists "
                          "in mlir module. Using existing mesh in module.\n");
      }

      globalMeshOp = sdy_utils::getMeshOp(rootModule);
    }
    // If graph is not annotated with sdy shardings, determine how the arguments
    // should be sharded and insert sdy.sharding annotations and sdy.meshOp.
    else {
      // Generate new meshOp based on user provided mesh.
      if (meshShapeRef.size() != 2 || meshShapeRef[0] != 1) {
        mlir::emitError(
            builder.getUnknownLoc(),
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
      globalMeshOp = builder.create<mlir::sdy::MeshOp>(
          builder.getUnknownLoc(), builder.getStringAttr(meshName),
          sdyMeshAttr);

      for (auto &op : rootModule.getBody()->getOperations()) {
        if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
          Block &entryBlock = funcOp.getBody().front();
          ArgumentAnalysis argAnalysis;

          // Process each argument and add it to the analysis manager.
          for (BlockArgument arg : entryBlock.getArguments()) {
            argAnalysis.processArgument(&arg);
          }

          // Once we processed all the elements, we want to iterate through all
          // the arguments again and update it's attributes to add the shardy
          // sharding attribute.
          for (BlockArgument arg : entryBlock.getArguments()) {
            funcOp.setArgAttrs(arg.getArgNumber(),
                               argAnalysis.getUpdatedArgumentDictionaryAttr(
                                   context, funcOp, &arg));
          }
        }
      }
    }

    // Run shardy tensor sharding propagation pass.
    pm.addPass(mlir::sdy::createAggressivePropagationPass());
    if (failed(pm.run(rootModule))) {
      mlir::emitError(builder.getUnknownLoc(),
                      "Failed to run aggressive propoagation pass from sdy.\n");
      signalPassFailure();
      return;
    }
    pm.clear();

    // Sdy does not have support for a custom call for FullToShard or
    // ShardToFull like gspmd. Therefore, wrap all operations in each function
    // under a sdy.manual_computation op. This is required to enable the
    // conversion from sdy into ttir.
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        mlir::FunctionType funcType = funcOp.getFunctionType();

        // Get operands and in_shardings of current function.
        llvm::SmallVector<mlir::Value> operands;
        llvm::SmallVector<mlir::sdy::TensorShardingAttr> inShardingAttrs;

        for (auto arg : funcOp.getArguments()) {
          operands.push_back(arg);

          // Get the tensor sharding attribute from argument dictionary.
          if (auto argAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
            mlir::sdy::TensorShardingAttr shardingAttr;
            if (failed(sdy_utils::getTensorShardingAttr(argAttrDict,
                                                        shardingAttr))) {
              mlir::emitError(builder.getUnknownLoc(),
                              "Argument is not annotated with sdy.sharding.\n");
              signalPassFailure();
              return;
            }

            inShardingAttrs.push_back(shardingAttr);
          } else {
            mlir::emitError(
                builder.getUnknownLoc(),
                "Argument does not have an attributes dictionary.\n");
            signalPassFailure();
            return;
          }
        }
        mlir::sdy::TensorShardingPerValueAttr inShardings =
            mlir::sdy::TensorShardingPerValueAttr::get(context,
                                                       inShardingAttrs);

        // Get results and out_shardings of current function.
        llvm::SmallVector<mlir::Type> results;
        llvm::SmallVector<mlir::sdy::TensorShardingAttr> outShardingAttrs;

        for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
          results.push_back(funcType.getResult(i));

          // Get the tensor sharding attribute from results dictionary.
          if (auto resultAttrDict = mlir::DictionaryAttr::get(
                  context, funcOp.getResultAttrs(i))) {
            mlir::sdy::TensorShardingAttr shardingAttr;
            if (failed(sdy_utils::getTensorShardingAttr(resultAttrDict,
                                                        shardingAttr))) {
              mlir::emitError(builder.getUnknownLoc(),
                              "Result is not annotated with sdy.sharding.\n");
              signalPassFailure();
              return;
            }

            outShardingAttrs.push_back(shardingAttr);
          } else {
            mlir::emitError(builder.getUnknownLoc(),
                            "Result does not have an attributes dictionary.\n");
            signalPassFailure();
            return;
          }
        }
        mlir::sdy::TensorShardingPerValueAttr outShardings =
            mlir::sdy::TensorShardingPerValueAttr::get(context,
                                                       outShardingAttrs);

        // Get manual axes from meshOp since this pass currently only supports 1
        // mesh.
        llvm::SmallVector<mlir::StringAttr> manualAxes;
        llvm::ArrayRef<mlir::sdy::MeshAxisAttr> meshAxesAttr =
            globalMeshOp.getMesh().getAxes();

        for (auto meshAxisAttr : meshAxesAttr) {
          manualAxes.push_back(
              mlir::StringAttr::get(context, meshAxisAttr.getName()));
        }

        // Create sdy.manual_computation op
        mlir::Block &entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);
        mlir::sdy::ManualComputationOp manualComputationOp =
            builder.create<mlir::sdy::ManualComputationOp>(
                builder.getUnknownLoc(), results, operands, inShardings,
                outShardings, manualAxes);

        // Determine the argumentTypes and argumentLocations that need to get
        // added to the new region in manualComputationOp.
        llvm::SmallVector<mlir::Type> argumentTypes;
        llvm::SmallVector<mlir::Location> argumentLocations;

        for (auto arg : funcOp.getArguments()) {
          // All arguments must be annotated with sdy.sharding attribute at this
          // point.
          mlir::DictionaryAttr argAttrDict =
              funcOp.getArgAttrDict(arg.getArgNumber());
          mlir::sdy::TensorShardingAttr tensorShardingAttr =
              mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(
                  argAttrDict.get(mlir::sdy::TensorShardingAttr::name));
          mlir::RankedTensorType oldType =
              mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
          mlir::RankedTensorType newType;

          if (failed(sdy_utils::populateShardedOutputType(
                  globalMeshOp.getMesh(), newType, oldType,
                  tensorShardingAttr))) {
            mlir::emitError(builder.getUnknownLoc(),
                            "Could not apply propagated tensor shardings to "
                            "tensor dimensions.");
            signalPassFailure();
            return;
          }
          argumentTypes.push_back(newType);
          argumentLocations.push_back(arg.getLoc());
        }

        // Create a new block and new region in manualComputationOp since it is
        // a region based op.
        mlir::Block &sourceBlock = funcOp.getBody().front();
        mlir::Region &targetRegion = manualComputationOp->getRegion(0);
        targetRegion.push_back(
            new mlir::Block()); // Dynamically created block memory moves
                                // ownership into region.
        mlir::Block &targetBlock = targetRegion.front();
        targetRegion.front().addArguments(argumentTypes, argumentLocations);

        // Migrate all the ops currently in funcOp into the manualComputationOp
        // region. This will also copy func.ReturnOp.
        mlir::Block::iterator it = ++mlir::Block::iterator(manualComputationOp);
        targetBlock.getOperations().splice(targetBlock.end(),
                                           sourceBlock.getOperations(), it,
                                           sourceBlock.end());

        // Create a new func.ReturnOp in the original func.funcOp that takes the
        // manualComputationOp as it's operand.
        builder.setInsertionPointAfter(manualComputationOp);
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                             manualComputationOp->getResult(0));

        // Update old arguments with new arguments inside of the
        // manualComputationBlock.
        for (uint32_t i = 0; i < sourceBlock.getArguments().size(); i++) {
          auto oldArg = sourceBlock.getArgument(i);
          auto newArg = targetBlock.getArgument(i);

          for (auto &targetOp : targetBlock.getOperations()) {
            for (auto &operand : targetOp.getOpOperands()) {
              if (operand.get() == oldArg) {
                operand.set(newArg);
              }
            }
          }
        }

        // Update all func.return ops in manualComputationOp with sdy.return op.
        // This is because the manualComputationOp requires a sdy.returnOp.
        for (Operation &op :
             llvm::make_early_inc_range(targetBlock.getOperations())) {
          if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
            builder.setInsertionPoint(returnOp);
            builder.create<mlir::sdy::ReturnOp>(builder.getUnknownLoc(),
                                                returnOp->getOperand(0));
            returnOp->erase();
          }
        }
      }
    }

    // Analysis the graph and cut all the shapes of each operation according to
    // their sdy sharding attribute.
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        llvm::SetVector<mlir::Operation *> opSet;
        funcOp.getBody().walk([&](mlir::Operation *op) { opSet.insert(op); });
        llvm::SetVector<mlir::Operation *> sortedOpSet =
            mlir::topologicalSort(opSet);

        for (mlir::Operation *op : sortedOpSet) {
          for (auto namedAttr : op->getAttrs()) {
            if (namedAttr.getName() == mlir::sdy::TensorShardingAttr::name) {
              mlir::sdy::TensorShardingPerValueAttr tensorShardingPerValueAttr =
                  mlir::dyn_cast<mlir::sdy::TensorShardingPerValueAttr>(
                      op->getAttr(mlir::sdy::TensorShardingAttr::name));
              llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings =
                  tensorShardingPerValueAttr.getShardings();

              // Loop through all the tensorShardings and apply them to each
              // output
              for (uint32_t i = 0; i < tensorShardings.size(); i++) {
                mlir::sdy::TensorShardingAttr tensorShardingAttr =
                    tensorShardings[i];
                mlir::RankedTensorType oldType =
                    mlir::dyn_cast<mlir::RankedTensorType>(
                        op->getResult(i).getType());
                mlir::RankedTensorType newType;

                if (failed(sdy_utils::populateShardedOutputType(
                        globalMeshOp.getMesh(), newType, oldType,
                        tensorShardingAttr))) {
                  mlir::emitError(builder.getUnknownLoc(),
                                  "Could not apply propagated tensor shardings "
                                  "to tensor dimensions.\n");
                  signalPassFailure();
                  return;
                }

                // Create new operation state.
                mlir::OperationState state(op->getLoc(), op->getName());
                state.types.push_back(newType);
                state.operands.append(op->operand_begin(), op->operand_end());
                state.attributes = op->getAttrs();

                // Replace current operation with newly created operation with
                // updated tensor shardings.
                mlir::Operation *newOp = mlir::Operation::create(state);
                op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
                builder.setInsertionPoint(op);
                builder.insert(newOp);
              }
            }
          }
        }
      }
    }

    // Remove all sdy tensor sharding annotations since all the analysis is
    // complete
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        // Remove sharding annotations from arguments
        for (auto arg : funcOp.getArguments()) {
          if (auto argAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
            funcOp.setArgAttrs(
                arg.getArgNumber(),
                sdy_utils::removeDictionaryAttrShardingAnnotation(context,
                                                                  argAttrDict));
          }
        }

        // Remove sharding annotations from results
        mlir::FunctionType funcType = funcOp.getFunctionType();
        for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
          if (auto resultAttrDict = mlir::DictionaryAttr::get(
                  context, funcOp.getResultAttrs(i))) {
            funcOp.setResultAttrs(
                i, sdy_utils::removeDictionaryAttrShardingAnnotation(
                       context, resultAttrDict));
          }
        }

        // Remove sharding annotations from operations
        funcOp.getBody().walk([&](mlir::Operation *op) {
          if (auto opAttrDict = op->getAttrDictionary()) {
            op->setAttrs(sdy_utils::removeDictionaryAttrShardingAnnotation(
                context, opAttrDict));
          }
        });
      }
    }

    // Iterate through the entire graph and remove any operations that are no
    // longer being used.
    SmallVector<Operation *> opsToDelete;
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        funcOp.walk([&](Operation *currOp) {
          if (isa<func::FuncOp>(currOp) || isa<func::ReturnOp>(currOp) ||
              isa<mlir::sdy::ReturnOp>(currOp)) {
            return;
          }

          if (currOp->use_empty()) {
            opsToDelete.push_back(currOp);
          }
        });
      }
    }

    // Safely delete all unused operations
    for (Operation *op : opsToDelete) {
      op->erase();
    }

    // Close tensor shardings and drop replicated axes since we no longer need
    // to do analysis on the graph.
    pm.addPass(mlir::sdy::createCloseShardingsPass());
    if (failed(pm.run(rootModule))) {
      mlir::emitError(builder.getUnknownLoc(),
                      "Failed to run close shardings pass from sdy.\n");
      signalPassFailure();
      return;
    }
    pm.clear();
  }
};
} // namespace mlir::tt::stablehlo
