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
    mlir::sdy::MeshAxisAttr sdyMeshAxisAttrTensor = mlir::sdy::MeshAxisAttr::get(context, "default", meshShapeRef[0]);
    mlir::sdy::MeshAxisAttr sdyMeshAxisAttrBatch = mlir::sdy::MeshAxisAttr::get(context, "batch", meshShapeRef[1]);
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
  }
};
} // namespace mlir::tt::ttir
