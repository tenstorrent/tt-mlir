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

    // Check if we have frontend_attributes with sdy information.
    bool hasFrontendSdyAttrs =
        gspmd_utils::hasFrontendSdyAttributes(rootModule);

    if (hasFrontendSdyAttrs) {
      llvm::outs() << "[HET DEBUG] ConvertXlaSdyToSdyPass: hasFrontendSdyAttrs\n";
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
      if (mlir::failed(shardy_utils::convertCustomCallToShardingConstraint(
              rootModule, context, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace mlir::tt::stablehlo
