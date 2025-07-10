// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/MeshShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
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

#include <queue>

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_APPLYARGUMENTSHARDSTATUSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static mlir::LogicalResult updateShardStatus(MLIRContext *context,
                                             mlir::OpBuilder &builder,
                                             func::FuncOp &funcOp,
                                             bool gspmdAnnotationsExist,
                                             bool shardyAnnotationsExist) {
  std::string annotationsKeyWord =
      shardyAnnotationsExist ? std::string(mlir::sdy::TensorShardingAttr::name)
                             : "mhlo.sharding";

  // Iterate through all the arguments and determine whether they are
  // pre-sharded or not.
  for (auto arg : funcOp.getArguments()) {
    mlir::DictionaryAttr argAttrDict =
        funcOp.getArgAttrDict(arg.getArgNumber());
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
    mlir::tt::ttcore::ShardStatus shardStatus =
        mlir::tt::ttcore::ShardStatus::Unsharded;

    // If the argument contains a gspmd.sharding annotations, it is already
    // pre-sharded.
    if (argAttrDict) {
      newArgAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(argAttrDict.getValue());

      if (argAttrDict.contains(annotationsKeyWord)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Sharded;
      }
    }

    newArgAttrs.push_back(
        {mlir::tt::ttcore::ShardStatusAttr::name,
         mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus)});
    funcOp.setArgAttrs(arg.getArgNumber(),
                       mlir::DictionaryAttr::get(context, newArgAttrs));
  }

  // Iterate through all the results and determine whether they are
  // pre-sharded or not.
  mlir::FunctionType funcType = funcOp.getFunctionType();
  for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
    mlir::DictionaryAttr resultAttrDict =
        mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i));
    llvm::SmallVector<mlir::NamedAttribute> newResultAttrs;
    mlir::tt::ttcore::ShardStatus shardStatus =
        mlir::tt::ttcore::ShardStatus::Unsharded;

    if (resultAttrDict) {
      newResultAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(resultAttrDict.getValue());

      if (resultAttrDict.contains(annotationsKeyWord)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Sharded;
      }
    }

    newResultAttrs.push_back(
        {mlir::tt::ttcore::ShardStatusAttr::name,
         mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus)});
    funcOp.setResultAttrs(i,
                          mlir::DictionaryAttr::get(context, newResultAttrs));
  }

  return mlir::success();
}

// Traverse the graph starting from the startNode.
// We are looking for the first @Sharding op that is a user of the startNode
// and annotate it with the shard status of the startNode.
// We use a queue to perform a breadth-first search (BFS) traversal of the
// graph. We also use a set to keep track of visited nodes to avoid cycles.
// If we find a @Sharding op, we annotate it with the shard status of the
// startNode and stop the traversal.
bool traverseGraphAndUpdate(MLIRContext *context, mlir::Value startNode,
                            mlir::tt::ttcore::ShardStatusAttr shardStatusAttr,
                            bool isStartNodeArgument) {
  // Setup graph traversal structures.
  llvm::DenseSet<mlir::Value> visited;
  std::queue<mlir::Value> worklist;
  worklist.push(startNode);
  visited.insert(startNode);
  bool found = false;

  // Iterate through the worklist until we find a @Sharding op or exhaust the
  // graph.
  while (!worklist.empty() && !found) {
    Value current = worklist.front();
    worklist.pop();

    // Depending on whether the startNode is an argument or a result,
    // we need to check its users or defining operation to iterate over.
    llvm::SmallVector<mlir::Operation *> opsToCheck;
    if (isStartNodeArgument) {
      opsToCheck = llvm::SmallVector<mlir::Operation *>(current.getUsers());
    } else {
      opsToCheck =
          llvm::SmallVector<mlir::Operation *>{current.getDefiningOp()};
    }

    for (Operation *childOp : opsToCheck) {
      if (auto customCall =
              mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(childOp)) {
        auto callTargetName = customCall.getCallTargetNameAttr();
        if (callTargetName ==
            mlir::tt::sharding_utils::kShardingCustomCallTargetName) {
          llvm::SmallVector<mlir::NamedAttribute> newResultAttrs(
              customCall->getAttrDictionary().getValue());
          newResultAttrs.push_back(
              {mlir::tt::ttcore::ShardStatusAttr::name, shardStatusAttr});
          customCall->setAttrs(
              mlir::DictionaryAttr::get(context, newResultAttrs));
          found = true;
          break;
        }
      }

      // Enqueue this op's results for further traversal
      if (isStartNodeArgument) {
        for (auto childNode : childOp->getResults()) {
          if (visited.insert(childNode).second) {
            worklist.push(childNode);
          }
        }
      } else {
        for (auto childNode : childOp->getOperands()) {
          if (visited.insert(childNode).second) {
            worklist.push(childNode);
          }
        }
      }
    }
  }

  return found;
}

// Generic graph traversal algorithm with argument as the starting node. Stop
// when the first @Sharding op is found and annotate it with the shard status of
// the argument.
static mlir::LogicalResult propagateGSPMDShardStatus(MLIRContext *context,
                                                     func::FuncOp &funcOp) {
  // Iterate through all the arguments of the function and annotate the first
  // @Sharding op found with the shard status of the argument.
  const auto shardStatusName = mlir::tt::ttcore::ShardStatusAttr::name;
  for (auto arg : funcOp.getArguments()) {
    // Get the current shard status of the argument.
    mlir::DictionaryAttr argAttrDict =
        funcOp.getArgAttrDict(arg.getArgNumber());
    auto attr = argAttrDict.get(shardStatusName);
    mlir::tt::ttcore::ShardStatusAttr shardStatusAttr =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::ShardStatusAttr>(attr);

    if (!traverseGraphAndUpdate(context, arg, shardStatusAttr,
                                /*isStartNodeArgument=*/true)) {
      funcOp.emitWarning("In function ")
          << funcOp.getName() << " argument #: " << arg.getArgNumber()
          << " does not have a downstream @Sharding call. Ill formed graph.\n";
      return mlir::failure();
    }
  }

  // Iterate through all the results of the function and annotate the first
  // @Sharding op found with the shard status of the result.
  mlir::FunctionType funcType = funcOp.getFunctionType();
  auto resultVals = funcOp.getBody().front().getTerminator()->getOperands();
  for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
    // Get the current shard status of the result.
    mlir::DictionaryAttr resultAttrDict =
        mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i));
    auto attr = resultAttrDict.get(shardStatusName);
    mlir::tt::ttcore::ShardStatusAttr shardStatusAttr =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::ShardStatusAttr>(attr);

    if (!traverseGraphAndUpdate(context, resultVals[i], shardStatusAttr,
                                /*isStartNodeArgument=*/false)) {
      funcOp.emitWarning("In function ")
          << funcOp.getName() << " result #: " << i
          << " does not have a upstream @Sharding call. Ill formed graph.\n";
      return mlir::failure();
    }
  }

  return mlir::success();
}

class ApplyArgumentShardStatusPass
    : public impl::ApplyArgumentShardStatusPassBase<
          ApplyArgumentShardStatusPass> {
public:
  using impl::ApplyArgumentShardStatusPassBase<
      ApplyArgumentShardStatusPass>::ApplyArgumentShardStatusPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Check what types of annotations exist in the module. If non exist, we
    // do nothing.
    bool gspmdAnnotationsExist =
        shardy_utils::gspmdAnnotationsExist(rootModule);
    bool sdyAnnotationsExist = shardy_utils::sdyAnnotationsExist(rootModule);

    if (!gspmdAnnotationsExist && !sdyAnnotationsExist) {
      rootModule.emitWarning("No sharding annotations found in the module. "
                           "Skipping pass.\n");
      return;
    }

    // Loop through each argument. For each argument, check if it has a sdy
    // sharding annotations. If it does, it is pre-sharded. If it does not, it
    // is unsharded.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(mlir::tt::stablehlo::updateShardStatus(
              context, builder, funcOp, gspmdAnnotationsExist,
              sdyAnnotationsExist))) {
        rootModule.emitError("Could not update shard status.\n");
        signalPassFailure();
        return;
      }
    });

    // If gspmd annotations exist, we want to update each @Sharding op with it's
    // pre-sharded or unsharded status. This is to help with the conversion from
    // shlo -> ttir where we don't have to backtrack the sharding annotations to
    // determine the shard status.
    if (gspmdAnnotationsExist) {
      rootModule.walk([&](func::FuncOp funcOp) {
        if (failed(mlir::tt::stablehlo::propagateGSPMDShardStatus(context,
                                                                  funcOp))) {
          rootModule.emitError("Could not propagate shard status.\n");
          signalPassFailure();
          return;
        }
      });
    }
  }
};

} // namespace mlir::tt::stablehlo
