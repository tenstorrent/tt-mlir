// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GspmdUtils.h"
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

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_APPLYARGUMENTSHARDSTATUSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static mlir::LogicalResult updateShardStatus(MLIRContext *context,
                                             mlir::OpBuilder &builder,
                                             func::FuncOp &funcOp,
                                             bool gspmdAnnotationsExist,
                                             bool shardyAnnotationsExist) {
  std::string annotationsKeyWord =
      shardyAnnotationsExist
          ? std::string(mlir::sdy::TensorShardingAttr::name)
          : std::string(mlir::tt::gspmd_utils::kXlaShardingAttr);

  // Iterate through all the arguments and determine whether they are
  // pre-sharded or not.
  for (auto arg : funcOp.getArguments()) {
    mlir::DictionaryAttr argAttrDict =
        funcOp.getArgAttrDict(arg.getArgNumber());
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
    mlir::tt::ttcore::ShardStatus shardStatus =
        mlir::tt::ttcore::ShardStatus::Unsharded;

    // If the argument contains a gspmd.sharding or sdy.sharding annotation, it
    // is already pre-sharded.
    if (argAttrDict) {
      newArgAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(argAttrDict.getValue());

      if (argAttrDict.contains(annotationsKeyWord)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;
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

    // If the result contains a gspmd.sharding or sdy.sharding annotation, it is
    // already pre-sharded.
    if (resultAttrDict) {
      newResultAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(resultAttrDict.getValue());

      if (resultAttrDict.contains(annotationsKeyWord)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;
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
  }
};

} // namespace mlir::tt::stablehlo
