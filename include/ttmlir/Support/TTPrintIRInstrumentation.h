// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SUPPORT_TTPRINTIRINSTRUMENTATION_H
#define TTMLIR_SUPPORT_TTPRINTIRINSTRUMENTATION_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include <string>
#include <vector>

namespace mlir::tt {

/// IR dumping instrumentation that dumps MLIR at various stages.
/// Supports Pipeline, Pass, and Transformation level dumping.
class TTPrintIRInstrumentation : public PassInstrumentation {
public:
  enum class DumpLevel {
    Once,          // Dump only once at the very end (top-level only)
    Pipeline,      // Dump at pipeline boundaries (includes Once level)
    Pass,          // Dump after each pass
    Transformation // Dump at transformation actions (includes Pass level)
  };

  struct TTPrintIRInstrumentationOptions {
    std::string outputDir = "~/explorer";
    DumpLevel level = DumpLevel::Transformation;
    bool dumpInitial = false;
    bool onlyDumpOnChanges = true;
    std::string modelName = "";
    std::string pipelineName = "";
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

  void runBeforePipeline(std::optional<OperationName>,
                         const PipelineParentInfo &) override;
  void runAfterPipeline(std::optional<OperationName> name,
                        const PipelineParentInfo &parentInfo) override;

  //===--------------------------------------------------------------------===//
  // Pass Instrumentation Hooks
  //===--------------------------------------------------------------------===//

  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;

  //===--------------------------------------------------------------------===//
  // Core IR Dumping Logic
  //===--------------------------------------------------------------------===//

  void dumpIR(mlir::Operation *op, const std::string &name);
  void dumpIR(const std::string &irString, const std::string &name);
  std::string extractModelNameFromLocation(mlir::Operation *op) const;

  //===--------------------------------------------------------------------===//
  // File and Path Management
  //===--------------------------------------------------------------------===//

  std::string sanitizeFilename(const std::string &name) const;
  std::string getOutputFilename(const std::string &name) const;
  void initializeDumpCounter();

  //===--------------------------------------------------------------------===//
  // Member Variables
  //===--------------------------------------------------------------------===//

  int dumpCounter_;
  std::string outputDir_;
  std::string modelName_;
  std::string pipelineName_;
  DumpLevel level_;
  bool dumpInitial_;
  bool dumpedInitial_;
  bool onlyDumpOnChanges_;
  std::string lastDumpedIR_;
  std::vector<std::string> pipelineIRStack_;
  int currentDepth_;
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
