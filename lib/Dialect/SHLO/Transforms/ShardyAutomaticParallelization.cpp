// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
 //
 // SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/SHLO/Transforms/Passes.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Builders.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#pragma clang diagnostic pop

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::shlo {
#define GEN_PASS_DEF_SHARDYAUTOMATICPARALLELIZATION
#include "ttmlir/Dialect/SHLO/Transforms/Passes.h.inc"

/*
Algorithm:
SHLO graph can be
a. no annotation
b. GSPMD
c. shardy

[a, c]
1. check if sdy mesh exists
  - if exists, use that
  - if doesn't exist, add custom sdy mesh from user prompt and use that
2. check if any sdy.sharding exists for the arguments or sdy.sharding_per_value exists
  - if exists, we don't make any modifications
  - if doesn't exist, we will loop through arguments and annotate batch dimensions based on tensor rank
3. run shardy passes
  - aggressive propogation pass
4. analysis the graph and add sdy.manual_computation
  - need to cut all shapes depending on their annotated sharding value
  - wrap everything in a manual computation block
*/

// Helper function to calculate new output of the tensor.
mlir::LogicalResult computeNewType(mlir::OpBuilder& builder, mlir::sdy::MeshAttr meshAttr, mlir::RankedTensorType& newType, mlir::RankedTensorType& oldType, mlir::sdy::TensorShardingAttr& tensorShardingAttr) {
  // Build meshMap to use for calculating new output shapes of each tensor
  llvm::DenseMap<llvm::StringRef, int64_t> meshMap;
  llvm::ArrayRef<mlir::sdy::MeshAxisAttr> meshAxesAttr = meshAttr.getAxes();

  for (auto meshAxisAttr : meshAxesAttr) {
    meshMap[meshAxisAttr.getName()] = meshAxisAttr.getSize();
  }
  
  llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings = tensorShardingAttr.getDimShardings();
  llvm::SmallVector<int64_t> newShape(oldType.getShape().begin(), oldType.getShape().end());

  for (uint32_t i = 0; i < newShape.size(); i++) {
    int64_t currIndexShape = newShape[i];
    ::llvm::ArrayRef<mlir::sdy::AxisRefAttr> shardingAxes = dimShardings[i].getAxes();

    for (auto shardingAxis : shardingAxes) {
      // Check if the axis you want to shard on exists in the mesh.
      llvm::StringRef axisName = shardingAxis.getName();
      if (meshMap.find(axisName) == meshMap.end()) {
        mlir::emitWarning(builder.getUnknownLoc(), "could not find mesh axis in the current meshmap being used by the function");
        return failure();
      }

      // Check if the tensor dimension is divisible by the mesh axis.
      if (currIndexShape % meshMap[axisName] != 0) {
        mlir::emitWarning(builder.getUnknownLoc(), "could not find mesh axis in the current mesh map being used by the function");
        return failure();
      }

      currIndexShape = currIndexShape / meshMap[axisName];
    }

    // Update shape with new calculated shape at current index.
    newShape[i] = currIndexShape;
  }

  newType = RankedTensorType::get(newShape, oldType.getElementType());
  return success();
};

class ArgumentAnalysis {
public:
  // todo: (tapspatel) need to generalize largest rank such that batch dim doesn't always have to be with tensors of rank 4
  ArgumentAnalysis() : largestRank(4) {}

  void processArgument(BlockArgument* arg) {
    argumentCache.push_back(arg);
    mlir::RankedTensorType tensorType = mlir::dyn_cast<mlir::RankedTensorType>(arg->getType());
    this->largestRank = std::max(this->largestRank, tensorType.getRank());
  }

  int64_t getLargestRank() {
    return this->largestRank;
  }

  bool argumentsHaveSameRank() {
    if (this->argumentCache.empty()) {
      return true;
    }

    // Use the first element as a reference and check if all other elements are equal to it
    const BlockArgument* firstValue = this->argumentCache.front();
    auto firstRank = mlir::dyn_cast<mlir::RankedTensorType>(firstValue->getType()).getRank();

    return llvm::all_of(this->argumentCache, [&](const BlockArgument* currValue) {
        auto currRank = mlir::dyn_cast<mlir::RankedTensorType>(currValue->getType()).getRank();
        return currRank == firstRank;
    });
  }

  llvm::SmallVector<mlir::NamedAttribute> getArgumentDictionaryAttr(mlir::MLIRContext* context, func::FuncOp& funcOp, BlockArgument* arg) {
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

    // Copy the current dictionary if it exists for the op
    if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber())) {
      if (currentArgAttrDict) {
        newArgAttrs = SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
      }
    }

    mlir::RankedTensorType tensorType = mlir::dyn_cast<mlir::RankedTensorType>(arg->getType());
    llvm::ArrayRef<int64_t> shape = tensorType.getShape();
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    if (tensorType.getRank() == this->largestRank && shape[0] != 1) {
      mlir::sdy::AxisRefAttr axisAttr = mlir::sdy::AxisRefAttr::get(context, "batch");
      mlir::sdy::DimensionShardingAttr dimShardingAttr = mlir::sdy::DimensionShardingAttr::get(context, {axisAttr}, true);
      dimShardings.push_back(dimShardingAttr);

      for(uint32_t i = 0; i < tensorType.getRank() - 1; i++) {
        dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(context, {}, true));
      }
    }
    else {
      for(uint32_t i = 0; i < tensorType.getRank(); i++) {
        dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(context, {}, true));
      }
    }

    // Add the shardy sharding attribute to the argument
    mlir::sdy::TensorShardingAttr sharding =  mlir::sdy::TensorShardingAttr::get(context, "mesh", dimShardings, {});
    newArgAttrs.emplace_back(mlir::StringAttr::get(context, mlir::sdy::TensorShardingAttr::name), sharding);

    return newArgAttrs;
  }

private:
  llvm::SmallVector<BlockArgument*> argumentCache;
  int64_t largestRank;
};

class ShardyAutomaticParallelization : public impl::ShardyAutomaticParallelizationBase<ShardyAutomaticParallelization> {
public:
  using impl::ShardyAutomaticParallelizationBase<ShardyAutomaticParallelization>::ShardyAutomaticParallelizationBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext* context = rootModule.getContext();
    mlir::OpBuilder builder(context);
    auto pm = PassManager::on<ModuleOp>(context);

    // Inline all operations to make analysis easier
    pm.addPass(mlir::createInlinerPass());
    if (failed(pm.run(rootModule))) {
      signalPassFailure();
    }
    pm.clear();

    // Check if sdy mesh exists in the module.
    mlir::sdy::MeshOp globalMeshOp;
    bool sdyMeshExists = false;
    for (mlir::Operation &op : rootModule.getBody()->getOperations()) {
      if(mlir::sdy::MeshOp meshOp = mlir::dyn_cast<mlir::sdy::MeshOp>(op)) {
        sdyMeshExists = true;
        globalMeshOp = meshOp;
        break;
      }
    }

    // If sdy mesh doesn't exist, we want to add it based on user option.
    llvm::ArrayRef<int64_t> meshShapeRef = *meshShape;
    bool userProvidedMesh = meshShapeRef.size() != 0 ? true : false;
    if (sdyMeshExists && userProvidedMesh) {
      mlir::emitWarning(builder.getUnknownLoc(), "Warning: user provided mesh but mesh already exists in mlir module. Using existing mesh.");
    }
    else {
      if (meshShapeRef.size() != 2 || meshShapeRef[0] != 1) {
        mlir::emitWarning(builder.getUnknownLoc(), "Currently, shardy automatic parallel pass only supports 2d mesh shape and mesh shape dim0 must be 1.");
        return;
      }

      // Annotate module with shardy mesh
      std::string meshName = "mesh";
      llvm::DenseMap</*mesh_axis_name*/llvm::StringRef, /*mesh_axis_value*/int64_t> meshMap;
      meshMap["default"] = meshShapeRef[0];
      meshMap["batch"] = meshShapeRef[1];
      mlir::sdy::MeshAxisAttr sdyMeshAxisAttrTensor = mlir::sdy::MeshAxisAttr::get(context, "default", meshMap["default"]);
      mlir::sdy::MeshAxisAttr sdyMeshAxisAttrBatch = mlir::sdy::MeshAxisAttr::get(context, "batch", meshMap["batch"]);
      llvm::SmallVector<mlir::sdy::MeshAxisAttr> sdyMeshAxisAttrs = {sdyMeshAxisAttrTensor, sdyMeshAxisAttrBatch};
      mlir::sdy::MeshAttr sdyMeshAttr = mlir::sdy::MeshAttr::get(context, sdyMeshAxisAttrs);
      builder.setInsertionPoint(&(rootModule.getBody()->front()));
      globalMeshOp = builder.create<mlir::sdy::MeshOp>(builder.getUnknownLoc(), builder.getStringAttr(meshName), sdyMeshAttr);
    }

    // Check if the graph is annotated with sharding attributes
    bool isGraphAnnotatedWithShardings = false;

    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
         Block &entryBlock = funcOp.getBody().front();

        // Check if sdy.sharding exists for any of the arguments
        for (BlockArgument arg : entryBlock.getArguments()) {
          if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
            if (currentArgAttrDict && currentArgAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
              isGraphAnnotatedWithShardings = true;
              break;
            }
          }
        }

        // Check if sdy.sharding_per_value exists for any of the operations
        for (mlir::Operation &op : entryBlock.getOperations()) {
          llvm::ArrayRef<mlir::NamedAttribute> currentOpAttrDict = op.getAttrs();
          for (auto namedAttr : currentOpAttrDict) {
            if (namedAttr.getName() == mlir::sdy::TensorShardingAttr::name) {
              isGraphAnnotatedWithShardings = true;
              break;
            }
          }
        }

        // If graph is not annotated with shardings, we want to insert our own
        if (!isGraphAnnotatedWithShardings) {
          ArgumentAnalysis argAnalysis;

          // Process each argument and add it to the analysis manager
          for (BlockArgument arg : entryBlock.getArguments()) {
            argAnalysis.processArgument(&arg);
          }

          // Once we processed all the elements, we want to iterate through all the arguments again and update it's attributes to add the shardy sharding attribute
          for (BlockArgument arg : entryBlock.getArguments()) {
            llvm::SmallVector<mlir::NamedAttribute> newArgAttrs = argAnalysis.getArgumentDictionaryAttr(context, funcOp, &arg);
            funcOp.setArgAttrs(arg.getArgNumber(), mlir::DictionaryAttr::get(context, newArgAttrs));
          }
        }
      }
    }
    
    // Run shardy tensor sharding propagation pass.
    pm.addPass(mlir::sdy::createAggressivePropagationPass());

    if (failed(pm.run(rootModule))) {
      signalPassFailure();
    }

    // Wrap all operations in each function under a sdy.manual_computation op
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        mlir::FunctionType funcType = funcOp.getFunctionType();

        // Get results and out_shardings of current function (assume all results have shardings annotated)
        llvm::SmallVector<mlir::Type> results;
        llvm::SmallVector<mlir::sdy::TensorShardingAttr> outShardingAttrs;

        for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
          results.push_back(funcType.getResult(i));
          mlir::DictionaryAttr resultAttrs = mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i));
          if (resultAttrs && resultAttrs.contains(mlir::sdy::TensorShardingAttr::name)) {
            mlir::sdy::TensorShardingAttr shardingAttr = mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(resultAttrs.get(mlir::sdy::TensorShardingAttr::name));
            outShardingAttrs.push_back(shardingAttr);
          }
          else {
            mlir::emitError(builder.getUnknownLoc(), "Result is not annotated with sdy.sharding.");
            signalPassFailure();
          }
        }

        // Get operands and in_shardings of current function (assume all arguments have shardings annotated)
        llvm::SmallVector<mlir::Value> operands;
        llvm::SmallVector<mlir::sdy::TensorShardingAttr> inShardingAttrs;

        for (auto arg : funcOp.getArguments()) {
          operands.push_back(arg);

          if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
            if (currentArgAttrDict && currentArgAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
              mlir::sdy::TensorShardingAttr shardingAttr = mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(currentArgAttrDict.get(mlir::sdy::TensorShardingAttr::name));
              inShardingAttrs.push_back(shardingAttr);
            }
            else {
              mlir::emitError(builder.getUnknownLoc(), "Argument is not annotated with sdy.sharding.");
              signalPassFailure();
            }
          }
          else {
            mlir::emitError(builder.getUnknownLoc(), "Argument does not have an attributes dictionary.");
            signalPassFailure();
          }
        }

        // Get manual axes
        llvm::SmallVector<mlir::StringAttr> manualAxes;
        llvm::ArrayRef<mlir::sdy::MeshAxisAttr> meshAxesAttr = globalMeshOp.getMesh().getAxes();

        for (auto meshAxisAttr : meshAxesAttr) {
          manualAxes.push_back(mlir::StringAttr::get(context, meshAxisAttr.getName()));
        }

        // Create sdy.manual_computation op
        mlir::sdy::TensorShardingPerValueAttr inShardings = mlir::sdy::TensorShardingPerValueAttr::get(context, inShardingAttrs);
        mlir::sdy::TensorShardingPerValueAttr outShardings = mlir::sdy::TensorShardingPerValueAttr::get(context, outShardingAttrs);
        mlir::Block &entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);
        mlir::sdy::ManualComputationOp manualComputationOp = builder.create<mlir::sdy::ManualComputationOp>(builder.getUnknownLoc(), results, operands, inShardings, outShardings, manualAxes);

        // fancy addition of all the ops
        mlir::Block &sourceBlock = funcOp.getBody().front();
        mlir::Region &targetRegion = manualComputationOp->getRegion(0);
        targetRegion.push_back(new mlir::Block());
        mlir::Block &targetBlock = targetRegion.front();

        // Add arguments to region
        llvm::SmallVector<mlir::Type> argumentTypes;
        llvm::SmallVector<mlir::Location> argumentLocations;

        for (auto arg : funcOp.getArguments()) {
          mlir::RankedTensorType oldType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
          mlir::RankedTensorType newType;
          auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber());
          mlir::sdy::TensorShardingAttr tensorShardingAttr = mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(currentArgAttrDict.get(mlir::sdy::TensorShardingAttr::name));

          if(failed(computeNewType(builder, globalMeshOp.getMesh(), newType, oldType, tensorShardingAttr))) {
            mlir::emitError(builder.getUnknownLoc(), "Could not apply propagated tensor shardings to tensor dimensions.");
            signalPassFailure();
          }
          argumentTypes.push_back(newType);
          argumentLocations.push_back(arg.getLoc());
        }

        targetRegion.front().addArguments(argumentTypes, argumentLocations);
        mlir::Block::iterator it = ++mlir::Block::iterator(manualComputationOp);

        targetBlock.getOperations().splice(
          targetBlock.end(),
          sourceBlock.getOperations(),
          it,
          sourceBlock.end()
        );

        builder.setInsertionPointAfter(manualComputationOp);
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), manualComputationOp->getResult(0));

        // Update old arguments with new arguments
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

        // Update all func.return ops in manualComputationOp with sdy.return op
        for (Operation &op : llvm::make_early_inc_range(targetBlock.getOperations())) {
          if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
            builder.setInsertionPoint(returnOp);
            builder.create<mlir::sdy::ReturnOp>(builder.getUnknownLoc(), returnOp->getOperand(0));
            returnOp->erase();
          }
        }
      }
    }

    // Analysis the graph and cut all the shapes of each operation according to their sdy sharding attribute
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        llvm::SetVector<mlir::Operation*> opSet;
        funcOp.getBody().walk([&](mlir::Operation *op) {opSet.insert(op);});
        llvm::SetVector<mlir::Operation*> sortedOpSet = mlir::topologicalSort(opSet);

        for(mlir::Operation* op : sortedOpSet) {
          for (auto namedAttr : op->getAttrs()) {
            if (namedAttr.getName() == mlir::sdy::TensorShardingAttr::name) {
              // (todo) tapspatel: relax one output per operation constraint
              // Currently, we assume there is only 1 output per operation
              auto tensorShardingPerValueAttr = mlir::dyn_cast<mlir::sdy::TensorShardingPerValueAttr>(op->getAttr(mlir::sdy::TensorShardingAttr::name));
              mlir::sdy::TensorShardingAttr tensorShardingAttr = tensorShardingPerValueAttr.getShardings()[0];
              mlir::RankedTensorType oldType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
              mlir::RankedTensorType newType;

              if(failed(computeNewType(builder, globalMeshOp.getMesh(), newType, oldType, tensorShardingAttr))) {
                mlir::emitError(builder.getUnknownLoc(), "Could not apply propagated tensor shardings to tensor dimensions.");
                signalPassFailure();
              }

              mlir::OperationState state(op->getLoc(), op->getName());
              state.types.push_back(newType);
              state.operands.append(op->operand_begin(), op->operand_end());
              state.attributes = op->getAttrs();
              mlir::Operation *newOp = mlir::Operation::create(state);
              op->getResult(0).replaceAllUsesWith(newOp->getResult(0));

              builder.setInsertionPoint(op);
              builder.insert(newOp);
            }
          }
        }
      }
    }

    // Remove all sdy tensor sharding annotations since all the analysis is complete
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        // Remove sharding annotations from arguments
        for (auto arg : funcOp.getArguments()) {
          llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

          if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
            if (currentArgAttrDict && currentArgAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
              llvm::copy_if(currentArgAttrDict.getValue(), std::back_inserter(newArgAttrs),
                  [&](mlir::NamedAttribute currentArgAttr) {
                    return currentArgAttr.getName() != mlir::sdy::TensorShardingAttr::name;
                  });
            } else {
              newArgAttrs = SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
            }
            funcOp.setArgAttrs(arg.getArgNumber(), mlir::DictionaryAttr::get(context, newArgAttrs));
          }
        }

        // Remove sharding annotations from results
        mlir::FunctionType funcType = funcOp.getFunctionType();

        for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
          llvm::SmallVector<mlir::NamedAttribute> newResultAttrs;

          if (auto resultAttrs = mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i))) {
            if (resultAttrs && resultAttrs.contains(mlir::sdy::TensorShardingAttr::name)) {
              llvm::copy_if(resultAttrs.getValue(), std::back_inserter(newResultAttrs),
                  [&](mlir::NamedAttribute currentArgAttr) {
                    return currentArgAttr.getName() != mlir::sdy::TensorShardingAttr::name;
                  });
            } else {
              newResultAttrs = SmallVector<mlir::NamedAttribute>(resultAttrs.getValue());
            }
            funcOp.setResultAttrs(i, mlir::DictionaryAttr::get(context, newResultAttrs));
          }
        }

        // Remove sharding annotations from operations
        funcOp.getBody().walk([&](mlir::Operation *op) {
          llvm::SmallVector<mlir::NamedAttribute> newopAttrs;
          if (auto opAttrs = op->getAttrDictionary()) {
            if (opAttrs && opAttrs.contains(mlir::sdy::TensorShardingAttr::name)) {
              llvm::copy_if(opAttrs.getValue(), std::back_inserter(newopAttrs),
                  [&](mlir::NamedAttribute currentArgAttr) {
                    return currentArgAttr.getName() != mlir::sdy::TensorShardingAttr::name;
                  });
            } else {
              newopAttrs = SmallVector<mlir::NamedAttribute>(opAttrs.getValue());
            }
            op->setAttrs(mlir::DictionaryAttr::get(context, newopAttrs));
          }
        });
      }
    }

    // Iterate through the entire graph and remove any operations that are no longer being used.
    SmallVector<Operation *> opsToDelete;
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
          funcOp.walk([&](Operation *currOp) {
            if (isa<func::FuncOp>(currOp) || isa<func::ReturnOp>(currOp) || isa<mlir::sdy::ReturnOp>(currOp)) {
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
  }
};
}
