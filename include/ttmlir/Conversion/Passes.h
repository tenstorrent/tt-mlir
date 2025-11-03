// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES_H
#define TTMLIR_CONVERSION_PASSES_H

#include "ttmlir/Conversion/ArithToD2MTileOps/ArithToD2MTileOps.h"
#ifdef TTMLIR_ENABLE_STABLEHLO
#include "ttmlir/Conversion/ArithToStableHLO/ArithToStableHLO.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOLegalizeComposite.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#endif
#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"
#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h"
#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"
#include "ttmlir/Conversion/MathToD2MTileOps/MathToD2MTileOps.h"
#include "ttmlir/Conversion/SFPIToEmitC/SFPIToEmitC.h"
#include "ttmlir/Conversion/TTIRToD2M/TTIRToD2M.h"
#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
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
    enum class DumpMode {
      Disabled,
      PerPass,
      PerDialect
    };

    DumpMode dumpMode = DumpMode::Disabled;
    std::string dumpDir = "";

    // Parse environment variables and populate config
    static Config fromEnvironment();
  };

  void attachContext(mlir::MLIRContext *ctx,
                     std::vector<std::string> passNamesToCache = {});

  // Enhanced version with environment variable support
  void attachContextWithDumping(mlir::MLIRContext *ctx, 
                                const std::string &modelName = "unknown",
                                const std::string &pipelineName = "unknown");

  // Dump IR at dialect creation
  static void dumpDialectCreation(const std::string &dialectName,
                                  mlir::MLIRContext *ctx);

  // Check if IR dumping should be enabled via environment variables
  static bool shouldEnableIRDumping();

  // Finalize IR dumping (for PerDialect mode)
  void finalizeDumping();

  // Destructor to ensure final IR is dumped
  ~MLIRModuleLogger();

private:
  Config config;
  std::string modelName = "unknown";
  std::string pipelineName = "unknown";
  int totalPassCount = 0;
  
  // For PerDialect mode: store the last IR to dump at the end
  std::string lastIRContent;
  bool hasFinalIR = false;
  
  std::string getOutputFilename(const std::string &passName,
                                const std::string &stage = "") const;
  void dumpIRToFile(const std::string &irContent,
                    const std::string &filename) const;
  void setModelName(const std::string &name);
  void setPipelineName(const std::string &name);
  std::string extractModelNameFromLocation(mlir::Operation *op) const;
  static std::string sanitizeFilename(const std::string &name);
};

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_PASSES_H
