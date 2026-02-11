// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Error.h"

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

      // We want to find the @Sharding custom call followed by the
      // @SPMDFullToShardShape custom call to determine the sharded type. The
      // pattern looks like: %0 = stablehlo.custom_call @Sharding(%arg0)
      // {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<1024x1024xf32>) ->
      // tensor<1024x1024xf32> %1 = stablehlo.custom_call
      // @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} :
      // (tensor<1024x1024xf32>) -> tensor<128x1024xf32>
      if (argAttrDict.contains(mlir::tt::gspmd_utils::kXlaShardingAttr)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;

        // Check if its replicated, mhlo.sharding = "{replicated}"}
        auto shardingStrAttr = mlir::cast<mlir::StringAttr>(
            argAttrDict.get(mlir::tt::gspmd_utils::kXlaShardingAttr));
        if (shardingStrAttr &&
            shardingStrAttr.getValue().contains("replicated")) {
          newType = mlir::cast<mlir::RankedTensorType>(arg.getType());
        } else {
          mlir::stablehlo::CustomCallOp shardingOp;
          for (auto *user : arg.getUsers()) {
            if (!mlir::isa<mlir::stablehlo::CustomCallOp>(user)) {
              continue;
            }
            mlir::stablehlo::CustomCallOp customCallOp =
                mlir::cast<mlir::stablehlo::CustomCallOp>(user);
            if (customCallOp.getCallTargetName() ==
                mlir::tt::gspmd_utils::kShardingCustomCallTargetName) {
              shardingOp = customCallOp;
              break;
            }
          }

          if (!shardingOp) {
            return module.emitError(
                "GSPMD presharded argument missing @Sharding custom call.");
          }

          mlir::stablehlo::CustomCallOp fullToShardOp;
          for (auto *user : shardingOp.getResult(0).getUsers()) {
            if (!mlir::isa<mlir::stablehlo::CustomCallOp>(user)) {
              continue;
            }
            mlir::stablehlo::CustomCallOp customCallOp =
                mlir::cast<mlir::stablehlo::CustomCallOp>(user);
            if (customCallOp.getCallTargetName() ==
                mlir::tt::gspmd_utils::kSPMDFullToShardShapeCallTargetName) {
              fullToShardOp = customCallOp;
              break;
            }
          }

          if (!fullToShardOp) {
            return module.emitError("GSPMD presharded argument missing "
                                    "@SPMDFullToShardShape custom call.");
          }
          newType = mlir::cast<mlir::RankedTensorType>(
              fullToShardOp.getResult(0).getType());
        }
      } else if (argAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
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
        shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;

        // Check if its replicated, mhlo.sharding = "{replicated}"}
        auto shardingStrAttr = mlir::cast<mlir::StringAttr>(
            resultAttrDict.get(mlir::tt::gspmd_utils::kXlaShardingAttr));
        if (shardingStrAttr &&
            shardingStrAttr.getValue().contains("replicated")) {
          newType = mlir::cast<mlir::RankedTensorType>(funcType.getResult(i));
        } else {
          // We want to backtrack from the function return to find the
          // @SPMDShardToFullShape custom call and extract its input type. This
          // input shape is its local device shape. The pattern looks like: %3 =
          // stablehlo.custom_call @SPMDShardToFullShape(%2) {mhlo.sharding =
          // "{devices=[8,1]<=[8]}"} : (tensor<128x1024xf32>) ->
          // tensor<1024x1024xf32> return %3 : tensor<1024x1024xf32>
          Value retVal = funcOp.getBody().back().getTerminator()->getOperand(i);

          auto shardToFullOp =
              retVal.getDefiningOp<mlir::stablehlo::CustomCallOp>();
          if (!shardToFullOp ||
              shardToFullOp.getCallTargetName() !=
                  mlir::tt::gspmd_utils::kSPMDShardToFullShapeCallTargetName) {
            return module.emitError("GSPMD presharded result missing "
                                    "@SPMDShardToFullShape custom call.");
          }

          newType = mlir::cast<mlir::RankedTensorType>(
              shardToFullOp.getOperand(0).getType());
        }
      } else if (resultAttrDict.contains(mlir::sdy::TensorShardingAttr::name)) {
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
