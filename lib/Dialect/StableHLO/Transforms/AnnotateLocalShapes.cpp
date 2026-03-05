// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
#define GEN_PASS_DEF_ANNOTATELOCALSHAPESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static mlir::LogicalResult annotateArgumentLocalShapes(MLIRContext *context,
                                                       mlir::ModuleOp &module,
                                                       mlir::OpBuilder &builder,
                                                       func::FuncOp &funcOp) {
  // Iterate though all arguments.
  for (auto arg : funcOp.getArguments()) {
    mlir::DictionaryAttr argAttrDict =
        funcOp.getArgAttrDict(arg.getArgNumber());
    FailureOr<mlir::RankedTensorType> newType =
        mlir::cast<mlir::RankedTensorType>(arg.getType());
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs =
        llvm::SmallVector<mlir::NamedAttribute>();

    if (argAttrDict) {
      newArgAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(argAttrDict.getValue());

      if (argAttrDict.contains(mlir::tt::ttcore::LocalShapeAttr::name)) {
        return module.emitError("Argument already has local shape annotation, "
                                "cannot annotate local shape");
      }

      // Check if argument is presharded.
      if (argAttrDict.contains(mlir::tt::ttcore::ShardStatusAttr::name)) {
        mlir::tt::ttcore::ShardStatusAttr shardStatusAttr =
            mlir::cast<mlir::tt::ttcore::ShardStatusAttr>(
                argAttrDict.get(mlir::tt::ttcore::ShardStatusAttr::name));
        if (shardStatusAttr.getValue() ==
            mlir::tt::ttcore::ShardStatus::Presharded) {
          // We want to find the @Sharding custom call followed by the
          // @SPMDFullToShardShape custom call to determine the sharded type.
          // The pattern looks like: %0 = stablehlo.custom_call @Sharding(%arg0)
          // {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<1024x1024xf32>)
          // -> tensor<1024x1024xf32> %1 = stablehlo.custom_call
          // @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} :
          // (tensor<1024x1024xf32>) -> tensor<128x1024xf32>

          if (argAttrDict.contains(mlir::tt::gspmd_utils::kXlaShardingAttr)) {
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
                    mlir::tt::gspmd_utils::
                        kSPMDFullToShardShapeCallTargetName) {
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
          } else if (argAttrDict.contains(
                         mlir::sdy::TensorShardingAttr::name)) {
            auto meshOp = shardy_utils::getMeshOps(module)[0];
            auto global_shape =
                mlir::cast<mlir::RankedTensorType>(arg.getType());
            mlir::sdy::TensorShardingAttr tensorShardingAttr =
                mlir::cast<mlir::sdy::TensorShardingAttr>(
                    argAttrDict.get(mlir::sdy::TensorShardingAttr::name));
            newType = mlir::tt::shardy_utils::populateShardedOutputType(
                meshOp.getMesh(), global_shape, tensorShardingAttr);
          }
        }
      }
    }

    mlir::NamedAttribute localShapeNamedAttr = {
        mlir::tt::ttcore::LocalShapeAttr::name,
        mlir::tt::ttcore::LocalShapeAttr::get(context, newType.value())};
    newArgAttrs.push_back(localShapeNamedAttr);
    funcOp.setArgAttrs(arg.getArgNumber(),
                       mlir::DictionaryAttr::get(context, newArgAttrs));
  }

  return mlir::success();
}

static mlir::LogicalResult annotateResultLocalShapes(MLIRContext *context,
                                                     mlir::ModuleOp &module,
                                                     mlir::OpBuilder &builder,
                                                     func::FuncOp &funcOp) {
  mlir::FunctionType funcType = funcOp.getFunctionType();
  for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
    mlir::DictionaryAttr resultAttrDict =
        mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i));
    llvm::SmallVector<mlir::NamedAttribute> newResultAttrs =
        llvm::SmallVector<mlir::NamedAttribute>();
    FailureOr<mlir::RankedTensorType> newType =
        mlir::cast<mlir::RankedTensorType>(funcType.getResult(i));

    if (resultAttrDict) {
      newResultAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(resultAttrDict.getValue());

      if (resultAttrDict.contains(mlir::tt::ttcore::LocalShapeAttr::name)) {
        return module.emitError("Result already has local shape annotation, "
                                "cannot annotate local shape");
      }

      // Check if result is presharded.
      if (resultAttrDict.contains(mlir::tt::ttcore::ShardStatusAttr::name)) {
        mlir::tt::ttcore::ShardStatusAttr shardStatusAttr =
            mlir::cast<mlir::tt::ttcore::ShardStatusAttr>(
                resultAttrDict.get(mlir::tt::ttcore::ShardStatusAttr::name));
        if (shardStatusAttr.getValue() ==
            mlir::tt::ttcore::ShardStatus::Presharded) {
          if (resultAttrDict.contains(
                  mlir::tt::gspmd_utils::kXlaShardingAttr)) {

            // Check if its replicated, mhlo.sharding = "{replicated}"}
            auto shardingStrAttr = mlir::cast<mlir::StringAttr>(
                resultAttrDict.get(mlir::tt::gspmd_utils::kXlaShardingAttr));
            if (shardingStrAttr &&
                shardingStrAttr.getValue().contains("replicated")) {
              newType =
                  mlir::cast<mlir::RankedTensorType>(funcType.getResult(i));
            } else {
              // We want to backtrack from the function return to find the
              // @SPMDShardToFullShape custom call and extract its input type.
              // This input shape is its local device shape. The pattern looks
              // like: %3 = stablehlo.custom_call @SPMDShardToFullShape(%2)
              // {mhlo.sharding =
              // "{devices=[8,1]<=[8]}"} : (tensor<128x1024xf32>) ->
              // tensor<1024x1024xf32> return %3 : tensor<1024x1024xf32>
              Value retVal =
                  funcOp.getBody().back().getTerminator()->getOperand(i);

              auto shardToFullOp =
                  retVal.getDefiningOp<mlir::stablehlo::CustomCallOp>();
              if (!shardToFullOp ||
                  shardToFullOp.getCallTargetName() !=
                      mlir::tt::gspmd_utils::
                          kSPMDShardToFullShapeCallTargetName) {
                return module.emitError("GSPMD presharded result missing "
                                        "@SPMDShardToFullShape custom call.");
              }

              newType = mlir::cast<mlir::RankedTensorType>(
                  shardToFullOp.getOperand(0).getType());
            }
          } else if (resultAttrDict.contains(
                         mlir::sdy::TensorShardingAttr::name)) {
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
      }
    }

    mlir::NamedAttribute localShapeNamedAttr = {
        mlir::tt::ttcore::LocalShapeAttr::name,
        mlir::tt::ttcore::LocalShapeAttr::get(context, newType.value())};
    newResultAttrs.push_back(localShapeNamedAttr);
    funcOp.setResultAttrs(i,
                          mlir::DictionaryAttr::get(context, newResultAttrs));
  }

  return mlir::success();
}

class AnnotateLocalShapesPass
    : public impl::AnnotateLocalShapesPassBase<AnnotateLocalShapesPass> {
public:
  using impl::AnnotateLocalShapesPassBase<
      AnnotateLocalShapesPass>::AnnotateLocalShapesPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Loop through all functions and annotate arguments with its local shape.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(mlir::tt::stablehlo::annotateArgumentLocalShapes(
              context, rootModule, builder, funcOp))) {
        rootModule.emitError("Failed to annotate local shapes for arguments.");
        signalPassFailure();
        return;
      }
    });

    // Loop through all functions and annotate results with its local shape.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(mlir::tt::stablehlo::annotateResultLocalShapes(
              context, rootModule, builder, funcOp))) {
        rootModule.emitError("Failed to annotate local shapes for results.");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
