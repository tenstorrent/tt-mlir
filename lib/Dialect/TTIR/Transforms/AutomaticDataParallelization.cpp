// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "llvm/ADT/SmallSet.h"  // check
#include "llvm/ADT/SmallString.h" // check
#include "llvm/ADT/SmallVector.h" // check

#include "mlir/IR/MLIRContext.h" // Include the MLIR context
#include "mlir/IR/Operation.h" // Include the operation definition
#include <iostream>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRAUTOMATICDATAPARALLELIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

/*
todo: 
- support arguments of different ranks
*/

class TTIRAutomaticDataParallelization : public impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization> {
public:
  using impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization>::TTIRAutomaticDataParallelizationBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext* context = rootModule.getContext();
    mlir::PatternRewriter rewriter(context);
    mlir::OpBuilder builder(context);

    ArrayRef<int64_t> meshShapeRef = *meshShape;

    // Currently we only allow linear meshes (1x2, 1x8). Meshes like the following are not supported (2x4, 8x4)
    assert(meshShapeRef[0] == 1 && "mesh shape dim0 must be 1");

    // Iterate through all the functions in the mlir graph
    // For each function, iterate through all the arguments
    // If all the arguments have a ArgumentTypeAttr annotated, then we don't need to make any modifications
    // If at least 1 argument is not annotated, then we will change the annotations for all arguments
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        auto areAllArgumentsAnnotated = [&funcOp](auto arguments) -> bool {
          for (BlockArgument arg : arguments) {
            DictionaryAttr attrDict = funcOp.getArgAttrDict(arg.getArgNumber());
            if (!attrDict || !attrDict.contains(ArgumentTypeAttr::name)) {
              return false;
            }
          }

          return true;
        };

        auto areAllArgumentsSameRank = [](auto arguments) -> bool {
          int64_t rank = 0;
          for (BlockArgument arg : arguments) {
            mlir::RankedTensorType tensorType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());

            if (rank == 0) {
              rank = tensorType.getRank();
            }
            else if (rank != tensorType.getRank()) {
              return false;
            }
          }
          return true;
        };

        auto getArgumentDictionaryAttr = [context, &funcOp](auto arg) -> llvm::SmallVector<mlir::NamedAttribute> {
          llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;

          if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
            if (!currentArgAttrDict && currentArgAttrDict.contains(ArgumentTypeAttr::name)) {
              llvm::copy_if(currentArgAttrDict.getValue(), std::back_inserter(newArgAttrs),
                  [&](mlir::NamedAttribute currentArgAttr) {
                    return currentArgAttr.getName() != ArgumentTypeAttr::name;
                  });
            } else {
              newArgAttrs = SmallVector<mlir::NamedAttribute>(currentArgAttrDict.getValue());
            }
          }

          // Determine the type of argument
          mlir::RankedTensorType tensorType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
          llvm::ArrayRef<int64_t> shape = tensorType.getShape();

          if (shape[0] == 1) {
            newArgAttrs.emplace_back(
              mlir::StringAttr::get(context, ArgumentTypeAttr::name),
              ArgumentTypeAttr::get(context, ArgumentType::Parameter));
          }
          else {
            newArgAttrs.emplace_back(
              mlir::StringAttr::get(context, ArgumentTypeAttr::name),
              ArgumentTypeAttr::get(context, ArgumentType::Input));
          }
          
          return newArgAttrs;
        };

        // If the arguments are not all annotated, we need to determine their ArgumentType and annotate them
        Block &entryBlock = funcOp.getBody().front();
        if(!areAllArgumentsAnnotated(entryBlock.getArguments())) {
          // If the arguments don't have the same rank, we need to apply additional logic. For now, it's not supported.
          if(!areAllArgumentsSameRank(entryBlock.getArguments())) {
            assert(false && "automatic data parallelism not supported to different ranked arguments");
          }

          // Iterate through all arguments, for each arg that has '1' in it's batch dim, annotate it as weight. for each arg that has non '1' annotate it as input
          for (BlockArgument arg : entryBlock.getArguments()) {
            llvm::SmallVector<mlir::NamedAttribute> newArgAttrs = getArgumentDictionaryAttr(arg);
            funcOp.setArgAttrs(arg.getArgNumber(), mlir::DictionaryAttr::get(context, newArgAttrs));
          }
        }
      }
    }

    // Iterate through the entire graph
    // For each arg, we will determine whether to shard or replicate it based on the ArgumentType
    // If it's an input, we shard. If it's a parameter or constant, we replicate
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        Block &entryBlock = funcOp.getBody().front();

        // Parse arguments and add relevant mesh shard operations.
        for (BlockArgument arg : entryBlock.getArguments()) {
          if (arg.use_empty()) {
              continue;
          }

          Operation *firstUser = nullptr;
          for (auto &use : arg.getUses()) {
            Operation *userOp = use.getOwner();
            if (!firstUser || userOp->isBeforeInBlock(firstUser)) {
              firstUser = userOp;
            }
          }
          assert(firstUser && "argument does not have a user, this is an ill-formed graph");

          DictionaryAttr attrDict = funcOp.getArgAttrDict(arg.getArgNumber());
          auto argumentTypeAttr = dyn_cast<ArgumentTypeAttr>(attrDict.get(ArgumentTypeAttr::name));
          auto loc = arg.getLoc();
          mlir::RankedTensorType outputType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
          builder.setInsertionPoint(firstUser);

          // For arguments that are not inputs, we want to replicate them on device.
          if (argumentTypeAttr.getValue() != ArgumentType::Input) {
            ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());
            ttir::MeshShardOp meshShardOp = builder.create<ttir::MeshShardOp>(
              loc, 
              outputType, 
              arg, 
              emptyOp, 
              mlir::tt::MeshShardType::Replicate,
              mlir::tt::MeshShardDirection::FullToShard, 
              /*shard_shape*/ llvm::SmallVector<int64_t>{1}, 
              /*shard_dims*/ llvm::SmallVector<int64_t>{-1}
            );

            for (auto &use : arg.getUses()) {
              Operation *userOp = use.getOwner();
          
              if (userOp != meshShardOp) {
                userOp->replaceUsesOfWith(arg, meshShardOp);
              } 
            }
          }
          // For inputs, we want to split the batch dimension of the input along each device.
          else {
            llvm::SmallVector<int64_t> newShape(outputType.getShape().begin(), outputType.getShape().end());
            newShape[0] = newShape[0] / meshShapeRef[1];
            mlir::RankedTensorType newOutputType = RankedTensorType::get(newShape, outputType.getElementType());
            ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, newOutputType.getShape(), newOutputType.getElementType());
            ttir::MeshShardOp meshShardOp = builder.create<ttir::MeshShardOp>(
              loc, 
              newOutputType, 
              arg, 
              emptyOp, 
              mlir::tt::MeshShardType::Devices,
              mlir::tt::MeshShardDirection::FullToShard, 
              /*shard_shape*/ llvm::SmallVector<int64_t>{meshShapeRef[1], 1, 1, 1},
              /*shard_dims*/ llvm::SmallVector<int64_t>{-1, 0}
            );

            for (auto &use : arg.getUses()) {
              Operation *userOp = use.getOwner();
          
              if (userOp != meshShardOp) {
                userOp->replaceUsesOfWith(arg, meshShardOp);
              } 
            }
          }
        }
      
        // Parse outputs and add in mesh shard operations to combine split tensor across all devices on their batch dimension.
        for (auto &op : entryBlock) {
          if (func::ReturnOp returnOp = llvm::dyn_cast<func::ReturnOp>(&op)) {
            auto returnTensors = returnOp.getOperands();

            assert(returnTensors.size() == 1 && "Return op function can only have 1 return value");

            for (auto tensor : returnTensors) {
              mlir::Operation* producerOp = tensor.getDefiningOp();
              builder.setInsertionPoint(returnOp);
              auto loc = tensor.getLoc();
              mlir::RankedTensorType outputType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
              ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());
              ttir::MeshShardOp meshShardOp = builder.create<ttir::MeshShardOp>(
                loc,
                outputType,
                producerOp->getResult(0),
                emptyOp,
                mlir::tt::MeshShardType::Devices,
                mlir::tt::MeshShardDirection::ShardToFull,
                /*shard_shape*/ llvm::SmallVector<int64_t>{meshShapeRef[1], 1, 1, 1},
                /*shard_dims*/ llvm::SmallVector<int64_t>{-1, 0}
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

        // Once we have promoted all the new mesh shard shapes, we want to do a topological sort on the entire graph, and do shape inference on all operations.
        // Run a topological sort on the graph
        // For each op that is not mesh_shard op, we will do shape inference on it
        // Propogate new shape down the graph
        llvm::SetVector<Operation*> opSet;
        funcOp.getBody().walk([&](Operation *op) {
            opSet.insert(op);
        });
        llvm::SetVector<Operation*> sortedOpSet = mlir::topologicalSort(opSet);

        // Iterate through all sorted operations and do shape inference
        for(Operation* op : opSet) {
          // Mesh shard shape inference was already finished in the previous step
          if(ttir::MeshShardOp meshShardOp = llvm::dyn_cast<ttir::MeshShardOp>(op)) {
            continue;
          }

          if(ttir::AddOp addOp = llvm::dyn_cast<ttir::AddOp>(op)) {
            builder.setInsertionPoint(addOp);

            // Acquire paramters to infer return shape and type of add op
            auto loc = addOp.getLoc();
            llvm::SmallVector<mlir::Value> operands = {addOp->mlir::Operation::getOperand(0), addOp->mlir::Operation::getOperand(1)};
            ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
            mlir::DictionaryAttr dictionaryAttr = DictionaryAttr::get(context, addOp->getAttrs());
            if(mlir::failed(AddOp::inferReturnTypes(context, loc, operands, dictionaryAttr, OpaqueProperties(nullptr), RegionRange(), inferredReturnTypes))) {
              assert(false && "could not infer output shape of op");
            }

            // Create empty output op
            mlir::RankedTensorType newOutputType = mlir::dyn_cast<mlir::RankedTensorType>(inferredReturnTypes[0]);
            ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, newOutputType.getShape(), newOutputType.getElementType());
            operands.push_back(emptyOp);
            ttir::AddOp newAddOp = builder.create<ttir::AddOp>(loc, operands, addOp->getAttrs());

            // Replace all users of the old add op with this new add op
            for (unsigned i = 0; i < addOp.getNumResults(); ++i) {
              addOp.getResult(i).replaceAllUsesWith(newAddOp.getResult(i));
            }

            addOp->erase();
          }
          else if(ttir::ReluOp reluOp = llvm::dyn_cast<ttir::ReluOp>(op)) {
            builder.setInsertionPoint(reluOp);

            // Acquire paramters to infer return shape and type of add op
            auto loc = reluOp.getLoc();
            llvm::SmallVector<mlir::Value> operands = {reluOp->mlir::Operation::getOperand(0)};
            ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
            mlir::DictionaryAttr dictionaryAttr = DictionaryAttr::get(context, reluOp->getAttrs());
            if(mlir::failed(ReluOp::inferReturnTypes(context, loc, operands, dictionaryAttr, OpaqueProperties(nullptr), RegionRange(), inferredReturnTypes))) {
              assert(false && "could not infer output shape of op");
            }

            // Create empty output op
            mlir::RankedTensorType newOutputType = mlir::dyn_cast<mlir::RankedTensorType>(inferredReturnTypes[0]);
            ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, newOutputType.getShape(), newOutputType.getElementType());
            operands.push_back(emptyOp);
            ttir::ReluOp newReluOp = builder.create<ttir::ReluOp>(loc, operands, reluOp->getAttrs());

            // Replace all users of the old add op with this new add op
            for (unsigned i = 0; i < reluOp.getNumResults(); ++i) {
              reluOp.getResult(i).replaceAllUsesWith(newReluOp.getResult(i));
            }

            reluOp->erase();
          }
          else if(ttir::MatmulOp matmulOp = llvm::dyn_cast<ttir::MatmulOp>(op)) {
            builder.setInsertionPoint(matmulOp);

            // Acquire paramters to infer return shape and type of add op
            auto loc = matmulOp.getLoc();
            llvm::SmallVector<mlir::Value> operands = {matmulOp->mlir::Operation::getOperand(0), matmulOp->mlir::Operation::getOperand(1)};
            ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
            mlir::DictionaryAttr dictionaryAttr = DictionaryAttr::get(context, matmulOp->getAttrs());
            if(mlir::failed(MatmulOp::inferReturnTypes(context, loc, operands, dictionaryAttr, OpaqueProperties(nullptr), RegionRange(), inferredReturnTypes))) {
              assert(false && "could not infer output shape of op");
            }

            // Create empty output op
            mlir::RankedTensorType newOutputType = mlir::dyn_cast<mlir::RankedTensorType>(inferredReturnTypes[0]);
            ttir::EmptyOp emptyOp = builder.create<ttir::EmptyOp>(loc, newOutputType.getShape(), newOutputType.getElementType());
            operands.push_back(emptyOp);
            ttir::MatmulOp newMatmulOp = builder.create<ttir::MatmulOp>(loc, operands, matmulOp->getAttrs());

            // Replace all users of the old add op with this new add op
            matmulOp.getResult().replaceAllUsesWith(newMatmulOp.getResult());
            matmulOp->erase();
          }
        }
      }
    }

    // Iterate through the entire graph
    // Remove any operations that are no longer being used (should just be emptyops since we inserted new ops in its place)
    SmallVector<Operation *> opsToDelete;
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
          funcOp.walk([&](Operation *currOp) {
            if (isa<func::FuncOp>(currOp) || isa<func::ReturnOp>(currOp)) {
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
} // namespace mlir::tt::ttir
