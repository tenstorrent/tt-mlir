// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES_H
#define TTMLIR_CONVERSION_PASSES_H

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "ttmlir/Conversion/ArithToStableHLO/ArithToStableHLO.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#endif
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Conversion/TosaToTTIR/TosaToTTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace mlir::tt {

#define GEN_PASS_REGISTRATION
#include "ttmlir/Conversion/Passes.h.inc"

struct MLIRModuleLogger {
  mlir::MLIRContext *context;
  std::vector<std::pair<std::string, std::string>> moduleCache;

  void attachContext(mlir::MLIRContext *ctx,
                     std::vector<std::string> passNamesToCache = {}) {
    context = ctx;

    context->registerActionHandler(
        [this, passNamesToCache](llvm::function_ref<void()> transform,
                                 const mlir::tracing::Action &action) {
          if (mlir::isa<mlir::PassExecutionAction>(action)) {
            auto passAction = mlir::cast<mlir::PassExecutionAction>(action);
            // A Pass action has occured, need to store the previous module
            // before transform is completed.
            std::string passName = passAction.getPass().getName().str();

            if (passNamesToCache.empty() or
                std::find(passNamesToCache.begin(), passNamesToCache.end(),
                          passName) != passNamesToCache.end()) {

              std::string outString;
              llvm::raw_string_ostream os(outString);
              mlir::OpPrintingFlags flags;
              flags.enableDebugInfo();
              passAction.getOp()->print(os, flags);
              os.flush();

              this->moduleCache.emplace_back(passName, outString);
            }
          }
          transform(); // Run the transformation pass.
        });
  }
};

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_PASSES_H
