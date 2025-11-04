// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SUPPORT_POCINSTRUMENTATION_H
#define TTMLIR_SUPPORT_POCINSTRUMENTATION_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Transforms/DialectConversion.h"
#include <atomic>
#include <fstream>
#include <mutex>
#include <string>

namespace mlir::tt {

/// IR dumping instrumentation that dumps MLIR at various stages.
/// Supports Pipeline, Pass, and Transformation level dumping.
class POCInstrumentation : public PassInstrumentation {
public:
  enum class DumpLevel {
    Pipeline,       // Dump only at pipeline boundaries
    Pass,          // Dump at pipeline + pass boundaries
    Transformation // Dump at pipeline + pass + transformation actions
  };

  enum class ActionMode {
    Overwrite,  // Clear directory, start from 0
    Append     // Continue from max index + 1
  };

  POCInstrumentation(const std::string &outputDir = "./poc_ir_dumps", 
                    DumpLevel level = DumpLevel::Transformation,
                    ActionMode actionMode = ActionMode::Overwrite,
                    bool debug = false);
  ~POCInstrumentation() override;

  // Set up action handler with the MLIR context from PassManager
  void attachActionHandler(mlir::MLIRContext *ctx);
  
  // Set model name (extracted from operation location)
  void setModelName(const std::string &name);

  // Pipeline hooks
  void runBeforePipeline(std::optional<OperationName> name,
                        const PipelineParentInfo &parentInfo) override;
  void runAfterPipeline(std::optional<OperationName> name,
                       const PipelineParentInfo &parentInfo) override;

  // Pass hooks
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

  // Analysis hooks
  void runBeforeAnalysis(StringRef name, TypeID id, Operation *op) override;
  void runAfterAnalysis(StringRef name, TypeID id, Operation *op) override;

private:
  void dumpIR(mlir::Operation *op, const std::string &name);
  std::string extractModelNameFromLocation(mlir::Operation *op) const;
  std::string sanitizeFilename(const std::string &name) const;
  std::string getOutputFilename(const std::string &name) const;
  std::string getTargetDirectory() const;
  int detectNextIndex(const std::string &targetDir) const;
  void clearDirectory(const std::string &targetDir) const;

  std::atomic<int> dumpCounter_;
  std::string outputDir_;
  std::string modelName_;
  std::mutex fileMutex_;
  DumpLevel level_;
  ActionMode actionMode_;
  bool debug_;
};

} // namespace mlir::tt

#endif // TTMLIR_SUPPORT_POCINSTRUMENTATION_H
