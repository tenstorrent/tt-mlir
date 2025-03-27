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

class TTIRAutomaticDataParallelization : public impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization> {
public:
  using impl::TTIRAutomaticDataParallelizationBase<TTIRAutomaticDataParallelization>::TTIRAutomaticDataParallelizationBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    mlir::PatternRewriter rewriter(rootModule.getContext());
    mlir::OpBuilder builder(rootModule.getContext());

    ArrayRef<int64_t> meshShapeRef = *meshShape;
    assert(meshShapeRef[0] == 1 && "mesh shape dim0 must be 0");

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
          
          builder.setInsertionPoint(firstUser);
          DictionaryAttr attrDict = funcOp.getArgAttrDict(arg.getArgNumber());
          auto argumentTypeAttr = dyn_cast<ArgumentTypeAttr>(attrDict.get(ArgumentTypeAttr::name));
          
          auto loc = arg.getLoc();
          mlir::RankedTensorType outputType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());

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
        for (auto &op : entryBlock) {
          if (ttir::MeshShardOp meshShardOp = llvm::dyn_cast<ttir::MeshShardOp>(&op)) {
            auto opResult = meshShardOp.getResult();

            for (auto resultOperation : opResult.getUsers()) {
              // We want to infer the new shape of this operation based on it's input shapes.
              if (isa<ttir::MatmulOp>(resultOperation)) {
                mlir::RankedTensorType rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(opResult.getType());
                mlir::RankedTensorType newRankedTensorType = mlir::RankedTensorType::get(rankedTensorType.getShape(), rankedTensorType.getElementType());
              }
            }
          }
        }
      }
    }
  }
};
} // namespace mlir::tt::ttir
