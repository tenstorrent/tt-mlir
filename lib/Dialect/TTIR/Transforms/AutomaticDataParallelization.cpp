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
    OpBuilder builder(rootModule.getContext());

    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
        Block &entryBlock = funcOp.getBody().front();

        // parse arguments and add relevant mesh shard operations
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

          if (firstUser) {
            builder.setInsertionPoint(firstUser);
            auto loc = arg.getLoc();
            auto outputType = dyn_cast<RankedTensorType>(arg.getType());
            auto outputTensor = builder.create<tensor::EmptyOp>(
              loc, 
              outputType.getShape(), 
              outputType.getElementType());
            auto newResult = builder.create<ttir::MeshShardOp>(
              loc, 
              outputType, 
              arg, 
              outputTensor, 
              mlir::tt::MeshShardType::Replicate,
              mlir::tt::MeshShardDirection::FullToShard, 
              /*shard_shape*/ llvm::SmallVector<int64_t>{1}, 
              /*shard_dims*/ llvm::SmallVector<int64_t>{-1});

            for (auto &use : arg.getUses()) {
              Operation *userOp = use.getOwner();

              if (userOp != newResult) {
                userOp->replaceUsesOfWith(arg, newResult);
              } 
            }
          }
        }

        // parse outputs and add relevant mesh shard operations
        for (auto &op : entryBlock) {
          if (auto returnOp = llvm::dyn_cast<func::ReturnOp>(&op)) {
            auto returnTensors = returnOp.getOperands();

            // need to assert returnTensors is of size 1

            for (auto tensor : returnTensors) {
              auto producerOp = tensor.getDefiningOp();

              builder.setInsertionPoint(returnOp);
              auto loc = tensor.getLoc();
              auto outputType = dyn_cast<RankedTensorType>(tensor.getType());
              auto outputTensor = builder.create<tensor::EmptyOp>(
                loc, 
                outputType.getShape(), 
                outputType.getElementType());
              [[ maybe_unused ]] auto newResult = builder.create<ttir::MeshShardOp>(
                loc, 
                outputType, 
                producerOp->getResult(0), 
                outputTensor, 
                mlir::tt::MeshShardType::Replicate,
                mlir::tt::MeshShardDirection::ShardToFull, 
                /*shard_shape*/ llvm::SmallVector<int64_t>{1}, 
                /*shard_dims*/ llvm::SmallVector<int64_t>{-1});

                for (auto &use : tensor.getUses()) {
                  Operation *userOp = use.getOwner();
    
                  if (userOp != newResult) {
                    userOp->replaceUsesOfWith(tensor, newResult);
                  } 
                }
            }
          }
        }
      }
    }
  }
};
} // namespace mlir::tt::ttir

/*
- for each argument, figure out what type it is
  - input, parameter, constant
- for parameter + constant
  - we are going to replicate this tensor across all devices
- for input
  - we are going to split it's batch dimension across all devices
  - we need to calculate it's new output shape based on the split batch
  - we need to propogate this new shape to all places that use this input
*/
