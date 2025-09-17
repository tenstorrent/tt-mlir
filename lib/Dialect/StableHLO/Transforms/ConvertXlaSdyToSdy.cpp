// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_CONVERTXLASDYTOSDYPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class ConvertXlaSdyToSdyPass
    : public impl::ConvertXlaSdyToSdyPassBase<ConvertXlaSdyToSdyPass> {
public:
  using impl::ConvertXlaSdyToSdyPassBase<
      ConvertXlaSdyToSdyPass>::ConvertXlaSdyToSdyPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Check if we have frontend_attributes with sdy information
    bool hasFrontendSdyAttrs =
        gspmd_utils::hasFrontendSdyAttributes(rootModule);

    if (hasFrontendSdyAttrs) {
      // Handle frontend_attributes conversion
      // Parse mesh information from module attributes and create sdy.mesh
      if (mlir::failed(gspmd_utils::parseMeshFromFrontendAttributes(rootModule,
                                                                    context))) {
        signalPassFailure();
        return;
      }

      // Convert function arguments from frontend_attributes to sdy.sharding
      if (mlir::failed(shardy_utils::convertFrontendAttributesToSDY(rootModule,
                                                                    context))) {
        signalPassFailure();
        return;
      }

      // Walk through all operations and convert stablehlo.custom_call @Sharding
      // ops to sdy.sharding_constraint ops
      rootModule.walk([&](mlir::Operation *op) {
        if (!mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
          return;
        }

        // Check call target name to see if it's the one we are interested in.
        mlir::stablehlo::CustomCallOp customCallOp =
            mlir::cast<mlir::stablehlo::CustomCallOp>(op);
        auto callTargetName = customCallOp.getCallTargetNameAttr();
        if (callTargetName != gspmd_utils::kShardingCustomCallTargetName) {
          return;
        }

        mlir::DictionaryAttr newAttrDict =
            shardy_utils::convertXlaSdyToSdyDictionary(
                context, customCallOp->getAttrDictionary());
        mlir::Attribute sdyShardingAttr =
            newAttrDict.get(mlir::sdy::TensorShardingAttr::name);
        mlir::sdy::TensorShardingAttr tensorShardingAttr =
            mlir::cast<mlir::sdy::TensorShardingAttr>(sdyShardingAttr);

        // Create sdy.sharding_constraint op and replace it in place of custom
        // call
        builder.setInsertionPointAfter(customCallOp);
        auto shardingConstraintOp =
            builder.create<mlir::sdy::ShardingConstraintOp>(
                customCallOp->getLoc(), customCallOp.getResult(0).getType(),
                customCallOp.getOperand(0), tensorShardingAttr);
        customCallOp.getResult(0).replaceAllUsesWith(
            shardingConstraintOp.getResult());
      });
    }
  }
};
} // namespace mlir::tt::stablehlo
