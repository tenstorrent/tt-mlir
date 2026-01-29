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

static mlir::LogicalResult
updateShardStatus(MLIRContext *context, mlir::ModuleOp &module,
                  mlir::OpBuilder &builder, func::FuncOp &funcOp,
                  bool gspmdAnnotationsExist, bool shardyAnnotationsExist) {
  // Iterate through all the arguments and determine whether they are
  // pre-sharded or not.
  for (auto arg : funcOp.getArguments()) {
    mlir::DictionaryAttr argAttrDict =
        funcOp.getArgAttrDict(arg.getArgNumber());
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
    mlir::tt::ttcore::ShardStatus shardStatus =
        mlir::tt::ttcore::ShardStatus::Unsharded;
    FailureOr<mlir::RankedTensorType> newType =
        mlir::cast<mlir::RankedTensorType>(arg.getType());

    // If the argument contains a gspmd.sharding or sdy.sharding annotation, it
    // is already pre-sharded.
    if (argAttrDict) {
      newArgAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(argAttrDict.getValue());

      if (argAttrDict.contains(mlir::tt::gspmd_utils::kXlaShardingAttr)) {
        module.emitError("GSPMD presharding annotations are not supported.");
      }

      if (argAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;
        auto meshOp = shardy_utils::getMeshOps(module)[0];
        auto global_shape = mlir::cast<mlir::RankedTensorType>(arg.getType());
        mlir::sdy::TensorShardingAttr tensorShardingAttr =
            mlir::cast<mlir::sdy::TensorShardingAttr>(
                argAttrDict.get(mlir::sdy::TensorShardingAttr::name));
        newType = mlir::tt::shardy_utils::populateShardedOutputType(
            meshOp.getMesh(), global_shape, tensorShardingAttr);
      }
    }

    mlir::NamedAttribute runtimeTensorShardingNamedAttr = {
        mlir::tt::ttcore::RuntimeTensorShardingAttr::name,
        mlir::tt::ttcore::RuntimeTensorShardingAttr::get(
            context,
            mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus),
            newType.value())};
    newArgAttrs.push_back(runtimeTensorShardingNamedAttr);
    funcOp.setArgAttrs(arg.getArgNumber(),
                       mlir::DictionaryAttr::get(context, newArgAttrs));

    // Update the shard status for the @Sharding custom call if it exists for
    // this arg.
    if (!shardyAnnotationsExist) {
      gspmd_utils::updateShardStatusForArgument(context, arg,
                                                runtimeTensorShardingNamedAttr);
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
    FailureOr<mlir::RankedTensorType> newType =
        mlir::cast<mlir::RankedTensorType>(funcType.getResult(i));

    // If the result contains a gspmd.sharding or sdy.sharding annotation, it is
    // already pre-sharded.
    if (resultAttrDict) {
      newResultAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(resultAttrDict.getValue());

      if (resultAttrDict.contains(mlir::tt::gspmd_utils::kXlaShardingAttr)) {
        module.emitError("GSPMD presharding annotations are not supported.");
      }

      if (resultAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;
        auto meshOp = shardy_utils::getMeshOps(module)[0];
        auto global_shape =
            mlir::cast<mlir::RankedTensorType>(funcType.getResult(i));
        mlir::sdy::TensorShardingAttr tensorShardingAttr =
            mlir::cast<mlir::sdy::TensorShardingAttr>(
                resultAttrDict.get(mlir::sdy::TensorShardingAttr::name));
        newType = mlir::tt::shardy_utils::populateShardedOutputType(
            meshOp.getMesh(), global_shape, tensorShardingAttr);
      }
    }

    mlir::NamedAttribute runtimeTensorShardingNamedAttr = {
        mlir::tt::ttcore::RuntimeTensorShardingAttr::name,
        mlir::tt::ttcore::RuntimeTensorShardingAttr::get(
            context,
            mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus),
            newType.value())};
    newResultAttrs.push_back(runtimeTensorShardingNamedAttr);
    funcOp.setResultAttrs(i,
                          mlir::DictionaryAttr::get(context, newResultAttrs));

    // Update the shard status for the @Sharding custom call if it exists for
    // this result.
    if (!shardyAnnotationsExist) {
      gspmd_utils::updateShardStatusForResult(context, funcOp, i,
                                              runtimeTensorShardingNamedAttr);
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
              context, rootModule, builder, funcOp, gspmdAnnotationsExist,
              sdyAnnotationsExist))) {
        rootModule.emitError("Failed to update shard status");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
