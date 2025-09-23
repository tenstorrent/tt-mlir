// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES_H
#define TTMLIR_CONVERSION_PASSES_H

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "ttmlir/Conversion/ArithToStableHLO/ArithToStableHLO.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOLegalizeComposite.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#endif
#include "ttmlir/Conversion/SFPIToEmitC/SFPIToEmitC.h"
#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Conversion/TTIRToTTIRGeneric/TTIRToTTIRGeneric.h"
#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"
#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"
#include "ttmlir/Conversion/TosaToTTIR/TosaToTTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include <string>
#include <unordered_set>

namespace mlir::tt {

#define GEN_PASS_REGISTRATION
#include "ttmlir/Conversion/Passes.h.inc"

struct MLIRModuleLogger {
  mlir::MLIRContext *context;
  std::vector<std::pair<std::string, std::string>> moduleCache;

  // Environment variable configuration
  struct Config {
    bool dumpEnabled = false;
    std::string dumpDir = "";
    std::unordered_set<std::string> specificPasses;
    bool dumpDialectCreation = false;
    bool preserveDebugInfo = true;

    // Parse environment variables and populate config
    static Config fromEnvironment();
  };

  void attachContext(mlir::MLIRContext *ctx,
                     std::vector<std::string> passNamesToCache = {});

  // Enhanced version with environment variable support
  void attachContextWithDumping(mlir::MLIRContext *ctx);

  // Dump IR at dialect creation
  static void dumpDialectCreation(const std::string &dialectName,
                                  mlir::MLIRContext *ctx);

  // Global utility to set up IR dumping for any PassManager/MLIRContext
  static void enableGlobalIRDumping(mlir::MLIRContext *ctx);

  // Utility to check if IR dumping should be enabled
  static bool shouldEnableIRDumping();

  // Utility to set up IR dumping for a PassManager
  static void setupIRDumping(mlir::PassManager &pm);

private:
  Config config;
  std::string getOutputFilename(const std::string &passName,
                                const std::string &stage = "") const;
  void dumpIRToFile(const std::string &irContent,
                    const std::string &filename) const;
};

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_PASSES_H
