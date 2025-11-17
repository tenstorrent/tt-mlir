// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SUPPORT_TTPRINTIRINSTRUMENTATION_H
#define TTMLIR_SUPPORT_TTPRINTIRINSTRUMENTATION_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include <atomic>
#include <fstream>
#include <string>

namespace mlir::tt {

/// IR dumping instrumentation that dumps MLIR at various stages.
/// Supports Pipeline, Pass, and Transformation level dumping.
class TTPrintIRInstrumentation : public PassInstrumentation {
public:
  enum class DumpLevel {
    Once,          // Dump only once at the very end (top-level only)
    Pipeline,      // Dump at pipeline boundaries (all nesting levels)
    Pass,          // Dump at pipeline + pass boundaries
    Transformation // Dump at pipeline + pass + transformation actions
  };

  struct TTPrintIRInstrumentationOptions {
    std::string outputDir =
        "~/explorer"; // Default path for tt-explorer integration
    DumpLevel level = DumpLevel::Transformation;
    bool debug = true;
    bool dumpInitial = false;   // Dump initial IR before any passes run
    std::string modelName = ""; // Empty means extract from operation location
    std::string pipelineName = ""; // Optional pipeline name for organization
  };

  TTPrintIRInstrumentation(TTPrintIRInstrumentationOptions options);
  ~TTPrintIRInstrumentation() override;

  //===--------------------------------------------------------------------===//
  // Configuration and Setup
  //===--------------------------------------------------------------------===//

  void attachActionHandler(mlir::MLIRContext *ctx);
  void setModelName(const std::string &name);

  //===--------------------------------------------------------------------===//
  // Pipeline Instrumentation Hooks
  //===--------------------------------------------------------------------===//

  void runBeforePipeline(std::optional<OperationName> name,
                         const PipelineParentInfo &parentInfo) override;
  void runAfterPipeline(std::optional<OperationName> name,
                        const PipelineParentInfo &parentInfo) override;

  //===--------------------------------------------------------------------===//
  // Pass Instrumentation Hooks
  //===--------------------------------------------------------------------===//

  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

  //===--------------------------------------------------------------------===//
  // Core IR Dumping Logic
  //===--------------------------------------------------------------------===//

  // Dump IR to a file (overloaded for Operation* or string)
  void dumpIR(mlir::Operation *op, const std::string &name,
              const std::string &source = "unknown");
  void dumpIR(const std::string &irString, const std::string &name);

  // Extract model name from operation location metadata
  std::string extractModelNameFromLocation(mlir::Operation *op) const;

  //===--------------------------------------------------------------------===//
  // File and Path Management
  //===--------------------------------------------------------------------===//

  // Sanitize a string for use as a filename
  std::string sanitizeFilename(const std::string &name) const;

  // Generate the full output filename for a dump
  std::string getOutputFilename(const std::string &name) const;

  // Initialize the dump counter (resets to 0)
  void initializeDumpCounter();

  //===--------------------------------------------------------------------===//
  // Member Variables
  //===--------------------------------------------------------------------===//

  std::atomic<int> dumpCounter_; ///< Counter for naming dump files
  std::string outputDir_;        ///< Base output directory path
  std::string modelName_;        ///< Name of the model being processed
  std::string pipelineName_;     ///< Optional pipeline identifier
  DumpLevel level_;              ///< Level of instrumentation detail
  bool dumpInitial_;   ///< Whether to dump initial IR before any passes
  bool dumpedInitial_; ///< Flag to ensure we only dump initial IR once

  // Pipeline-level dumping state
  std::vector<std::string>
      pipelineIRStack_; ///< Stack of IR strings for nested pipelines
  int currentDepth_;    ///< Current pipeline nesting depth (0 = top-level)
};

//===--------------------------------------------------------------------===//
// Convenience Functions
//===--------------------------------------------------------------------===//

/// Convenience function for adding TTPrintIRInstrumentation to a PassManager
void addTTPrintIRInstrumentation(
    PassManager &pm,
    TTPrintIRInstrumentation::TTPrintIRInstrumentationOptions options = {});

} // namespace mlir::tt

#endif // TTMLIR_SUPPORT_TTPRINTIRINSTRUMENTATION_H
