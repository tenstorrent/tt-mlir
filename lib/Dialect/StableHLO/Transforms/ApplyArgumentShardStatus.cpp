// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_APPLYARGUMENTSHARDSTATUSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static mlir::LogicalResult updateShardStatus(MLIRContext *context,
                                             mlir::OpBuilder &builder,
                                             func::FuncOp &funcOp,
                                             bool gspmdAnnotationsExist,
                                             bool shardyAnnotationsExist) {
  llvm::StringRef annotationsKeyWord =
      shardyAnnotationsExist ? mlir::sdy::TensorShardingAttr::name
                             : mlir::tt::gspmd_utils::kXlaShardingAttr;

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

    mlir::NamedAttribute shardStatusNamedAttr = {
        mlir::tt::ttcore::ShardStatusAttr::name,
        mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus)};
    newArgAttrs.push_back(shardStatusNamedAttr);
    funcOp.setArgAttrs(arg.getArgNumber(),
                       mlir::DictionaryAttr::get(context, newArgAttrs));

    // Update the shard status for the @Sharding custom call if it exists for
    // this arg.
    if (!shardyAnnotationsExist) {
      gspmd_utils::updateShardStatusForArgument(context, arg,
                                                shardStatusNamedAttr);
    }
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

    mlir::NamedAttribute shardStatusNamedAttr = {
        mlir::tt::ttcore::ShardStatusAttr::name,
        mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus)};
    newResultAttrs.push_back(shardStatusNamedAttr);
    funcOp.setResultAttrs(i,
                          mlir::DictionaryAttr::get(context, newResultAttrs));

    // Update the shard status for the @Sharding custom call if it exists for
    // this result.
    if (!shardyAnnotationsExist) {
      gspmd_utils::updateShardStatusForResult(context, funcOp, i,
                                              shardStatusNamedAttr);
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
    bool gspmdAnnotationsExist = gspmd_utils::gspmdAnnotationsExist(rootModule);
    bool sdyAnnotationsExist = shardy_utils::sdyAnnotationsExist(rootModule);

    // Loop through each argument. For each argument, check if it has a sdy
    // sharding annotations. If it does, it is pre-sharded. If it does not, it
    // is unsharded.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(mlir::tt::stablehlo::updateShardStatus(
              context, builder, funcOp, gspmdAnnotationsExist,
              sdyAnnotationsExist))) {
        rootModule.emitError("Failed to update shard status");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
