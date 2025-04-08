// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "mlir/Dialect/Traits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include <iostream>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#pragma clang diagnostic pop

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRAUTOMATICDATAPARALLELIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class ArgumentAnalysisManager {
public:
  // todo: (tapspatel) need to generalize largest rank such that batch dim doesn't always have to be with tensors of rank 4
  ArgumentAnalysisManager() : largestRank(4) {}

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

    // If the argument already has a shardy annotation, we will remove it in favor of this pass
    if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg->getArgNumber())) {
      if (!currentArgAttrDict && currentArgAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
        llvm::copy_if(currentArgAttrDict.getValue(), std::back_inserter(newArgAttrs),
            [&](mlir::NamedAttribute currentArgAttr) {
              return currentArgAttr.getName() != mlir::sdy::TensorShardingAttr::name;
            });
      } else {
        newArgAttrs = SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
      }
    }

    // If the argument does not contain shardy annotation, we will add it depending on what type of argument it is
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


class TTIRAutomaticDataParallelization : public impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization> {
public:
  using impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization>::TTIRAutomaticDataParallelizationBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext* context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    llvm::ArrayRef<int64_t> meshShapeRef = *meshShape;
    if (meshShapeRef.size() != 2 || meshShapeRef[0] != 1) {
      mlir::emitWarning(builder.getUnknownLoc(), "automatic data parallel pass only supports 2d mesh shape and mesh shape dim0 must be 1");
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
    builder.create<mlir::sdy::MeshOp>(builder.getUnknownLoc(), builder.getStringAttr(meshName), sdyMeshAttr);

    /*
    Loop through all functions and arguments and determine their shardy sharding annotation
    1. Find the largest rank out of all the arguments in the function
    2. For each arg
      a. If it's rank == largest rank, and it's dim[0] != 1, we will annotate dim[0] with the batch axis
      b. Else, we will not annotate any of it's dimensions
    */
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        ArgumentAnalysisManager argAnalysisManager;

        Block &entryBlock = funcOp.getBody().front();
        // Process each argument and add it to the analysis manager
        for (BlockArgument arg : entryBlock.getArguments()) {
          argAnalysisManager.processArgument(&arg);
        }

        // Once we processed all the elements, we want to iterate through all the arguments again and update it's attributes to add the shardy sharding attribute
        for (BlockArgument arg : entryBlock.getArguments()) {
          llvm::SmallVector<mlir::NamedAttribute> newArgAttrs = argAnalysisManager.getArgumentDictionaryAttr(context, funcOp, &arg);
          funcOp.setArgAttrs(arg.getArgNumber(), mlir::DictionaryAttr::get(context, newArgAttrs));
        }
      }
    }

    // Run shardy tensor sharding propagation pass
    auto pm = PassManager::on<ModuleOp>(context);
    pm.addPass(mlir::sdy::createAggressivePropagationPass());

    if (failed(pm.run(rootModule))) {
      signalPassFailure();
    }

    /*
    // Now, we want to insert mesh shard operations for the arguments and the outputs of each function in the module.
    // We iterate through all the arguments in each function. 
    // For each argument, we will extract it's sharding from shardy attribute annotation and shard the argument accordingly. 
    // We propagate the new shapes to all its consumers
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {

        // Helper function to calculate new output of the tensor.
        auto computeNewType = [&meshMap, &builder](mlir::RankedTensorType& outputType, mlir::RankedTensorType& oldType, mlir::sdy::TensorShardingAttr& tensorShardingAttr) -> mlir::LogicalResult {
          ::llvm::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings = tensorShardingAttr.getDimShardings();
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
        
          outputType = RankedTensorType::get(newShape, oldType.getElementType());
          return success();
        };


        Block &entryBlock = funcOp.getBody().front();
        // Parse arguments and add relevant mesh shard operations.
        for (BlockArgument arg : entryBlock.getArguments()) {
          Operation *firstUser = nullptr;
          for (auto &use : arg.getUses()) {
            Operation *userOp = use.getOwner();
            if (!firstUser || userOp->isBeforeInBlock(firstUser)) {
              firstUser = userOp;
            }
          }
          assert(firstUser && "argument does not have a user, this is an ill-formed graph");

          DictionaryAttr attrDict = funcOp.getArgAttrDict(arg.getArgNumber());
          auto tensorShardingAttr = dyn_cast<mlir::sdy::TensorShardingAttr>(attrDict.get(mlir::sdy::TensorShardingAttr::name));
          auto loc = arg.getLoc();
          mlir::RankedTensorType oldType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
          mlir::RankedTensorType outputType;

          // Insert mesh shard ops on the calculated output shapes
          builder.setInsertionPoint(firstUser);
          if(failed(computeNewType(outputType, oldType, tensorShardingAttr))) {
            mlir::emitWarning(builder.getUnknownLoc(), "could not apply propagated tensor shardings to tensor dimensions");
            return;
          }

          mlir::tt::MeshShardType meshShardType;
          llvm::SmallVector<int64_t> shardShape;
          llvm::SmallVector<int64_t> shardDims;

          if (oldType == outputType) {
            meshShardType = mlir::tt::MeshShardType::Replicate;
            shardShape = {1};
            shardDims = {1};
          }
          else {
            meshShardType = mlir::tt::MeshShardType::Devices;
            shardShape = {meshShapeRef[1], 1, 1, 1},
            shardDims = {-1, 0};
          }

          ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());
          ttir::MeshShardOp meshShardOp = builder.create<ttir::MeshShardOp>(
            loc, 
            outputType, 
            arg, 
            emptyOp, 
            meshShardType,
            mlir::tt::MeshShardDirection::FullToShard, 
            shardShape,
            shardDims
          );

          for (auto &use : arg.getUses()) {
            Operation *userOp = use.getOwner();
        
            if (userOp != meshShardOp) {
              userOp->replaceUsesOfWith(arg, meshShardOp);
            } 
          }
        }

        // Parse outputs and add relevant mesh shard operations.
        for (auto &op : entryBlock) {
          if (func::ReturnOp returnOp = llvm::dyn_cast<func::ReturnOp>(&op)) {
            auto returnTensors = returnOp.getOperands();

            for (auto [idx, tensor] : llvm::enumerate(returnTensors)) {
              DictionaryAttr attrDict = funcOp.getResultAttrDict(idx);
              auto tensorShardingAttr = dyn_cast<mlir::sdy::TensorShardingAttr>(attrDict.get(mlir::sdy::TensorShardingAttr::name));
              mlir::Operation* producerOp = tensor.getDefiningOp();
              auto loc = tensor.getLoc();
              mlir::RankedTensorType oldType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
              mlir::RankedTensorType outputType;

              // Insert mesh shard ops on the calculated output shapes
              builder.setInsertionPoint(returnOp);
              if(failed(computeNewType(outputType, oldType, tensorShardingAttr))) {
                mlir::emitWarning(builder.getUnknownLoc(), "could not apply propagated tensor shardings to tensor dimensions");
                return;
              }

              mlir::tt::MeshShardType meshShardType;
              llvm::SmallVector<int64_t> shardShape;
              llvm::SmallVector<int64_t> shardDims;

              if (oldType == outputType) {
                meshShardType = mlir::tt::MeshShardType::Replicate;
                shardShape = {1};
                shardDims = {1};
              }
              else {
                meshShardType = mlir::tt::MeshShardType::Devices;
                shardShape = {meshShapeRef[1], 1, 1, 1},
                shardDims = {-1, 0};
              }

              ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());
              ttir::MeshShardOp meshShardOp = builder.create<ttir::MeshShardOp>(
                loc, 
                outputType, 
                producerOp->getResult(0), 
                emptyOp, 
                meshShardType,
                mlir::tt::MeshShardDirection::FullToShard, 
                shardShape,
                shardDims
              );

              for (auto &use : tensor.getUses()) {
                Operation *userOp = use.getOwner();
            
                if (userOp != meshShardOp) {
                  userOp->replaceUsesOfWith(tensor, meshShardOp);
                } 
              }
            }
          }
        }
      }
    }

    // Now, we want to sort the graph topologically and process each op individually
    // For each op, we are going to get it's output shape, get the op output tensor sharding and update the output shape with its tensor sharding
    // Then, propagate the new shape to all children ops
    llvm::SetVector<Operation*> opSet;
    funcOp.getBody().walk([&](Operation *op) {opSet.insert(op);});
    llvm::SetVector<Operation*> sortedOpSet = mlir::topologicalSort(opSet);

    for(Operation* op : opSet) {
      // Mesh shard output shape calculation was already finished in the previous step, thus we can skip it.
      if(ttir::MeshShardOp meshShardOp = mlir::dyn_cast<ttir::MeshShardOp>(op)) {
        continue;
      }

      auto tensorShardingAttr = mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(op.getAttr(mlir::sdy::TensorShardingAttr::name));
      
    }
    */
  }
};
} // namespace mlir::tt::ttir
