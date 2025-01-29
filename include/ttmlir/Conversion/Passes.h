// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES_H
#define TTMLIR_CONVERSION_PASSES_H

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "ttmlir/Conversion/ArithToStableHLO/ArithToStableHLO.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#endif
#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"
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
                     std::vector<std::string> passNamesToCache = {},
                     std::vector<std::pair<std::string, std::string>>
                         *passedModuleCache = nullptr) {
    context = ctx;

    context->registerActionHandler([this, passNamesToCache, passedModuleCache](
                                       llvm::function_ref<void()> transform,
                                       const mlir::tracing::Action &action) {
      // Also might make sense to store the _FIRST_ module. Or the module before
      // it was sent through the pipeline.

      if (passedModuleCache != nullptr and passedModuleCache->empty()) {
        // In Python Env so we have to add it ot the passedCache
        std::string passName = "PRE-PIPELINE", outString;
        llvm::raw_string_ostream os(outString);
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo();
        action.getContextIRUnits()[0].print(os, flags);
        os.flush();
        passedModuleCache->emplace_back(passName, outString);
      } else if (passedModuleCache == nullptr and moduleCache.empty()) {
        // Add it to the current Cache.
        std::string passName = "PRE-PIPELINE", outString;
        llvm::raw_string_ostream os(outString);
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo();
        action.getContextIRUnits()[0].print(os, flags);
        os.flush();
        moduleCache.emplace_back(passName, outString);
      }

      // Might make more sense to hold the module after a transformation has
      // occured.
      transform(); // Run the transformation pass.

      // Now save the module if it should be Cached.
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
      } else if (action.getTag() ==
                 "pass-execution") { // Tag will always be pass-execution but
                                     // unable to cast
        // This block was made considering that PassActions are weirdly not
        // registered when run through python We can String parse the printed
        // PassAction to determine the passName, The Op will be part of the
        // IRUnits, and we can extract it

        // The printed OP looks like:
        // `pass-execution` running `TTNNDeallocate` on Operation
        // `builtin.module` So we can filter for the `s and get the PassName in
        // between these. There will always only be 1R Unit and it is the
        // ModuleOp.

        std::string passOutput, passName = "";
        llvm::raw_string_ostream passOut(passOutput);
        action.print(passOut);
        passOut.flush();

        int backTickCount = 0;
        const int BACKTICK_BEFORE_PASS_NAME = 3, BACKTICK_AFTER_PASS_NAME = 4;
        for (const auto &c : passOutput) {
          if (c == '`') {
            backTickCount++;
          }

          if (backTickCount ==
              BACKTICK_BEFORE_PASS_NAME) { // This is the specific backTickCount
                                           // that
                                           // prefixes the passName
            passName += c;
          } else if (backTickCount >=
                     BACKTICK_AFTER_PASS_NAME) { // Specific count after
                                                 // passName
                                                 // is complete.
            break;
          }
        }

        // Now save the ModuleOp from the IRUnits, for PassExecution there will
        // always be only 1 IR unit.
        if (passNamesToCache.empty() or
            std::find(passNamesToCache.begin(), passNamesToCache.end(),
                      passName) != passNamesToCache.end()) {
          std::string outString;
          llvm::raw_string_ostream os(outString);
          mlir::OpPrintingFlags flags;
          flags.enableDebugInfo();
          action.getContextIRUnits()[0].print(os, flags);
          os.flush();

          // Python passes do not maintain the sufficient context to actually
          // update moduleCache, one has to be passed You can pass this in
          // Python using the ModuleLog class in the `passes` module. See
          // python/Passes.cpp for usage.
          if (passedModuleCache != nullptr) {
            passedModuleCache->emplace_back(passName, outString);
          }
        }
      }
    });
  }
};

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_PASSES_H
