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

/// POC instrumentation that logs counter increments to a file.
/// This is unrelated to MLIRModuleLogger and demonstrates pass instrumentation.
class POCInstrumentation : public PassInstrumentation {
public:
  enum class DumpLevel {
    Pipeline,       // Log only pipeline events
    Pass,          // Log pipeline + pass events
    Transformation // Log everything (pipeline + pass + analysis)
  };

  POCInstrumentation(const std::string &outputFile = "counter_log.txt", 
                    DumpLevel level = DumpLevel::Transformation,
                    bool debug = false);
  ~POCInstrumentation() override;

  // Set up action handler with the MLIR context from PassManager
  void attachActionHandler(mlir::MLIRContext *ctx);

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
  void logIncrement(const std::string &reason, const std::string &details = "");

  std::atomic<int> counter_;
  std::string outputFile_;
  std::mutex fileMutex_;
  DumpLevel level_;
  bool debug_;
};

} // namespace mlir::tt

#endif // TTMLIR_SUPPORT_POCINSTRUMENTATION_H
