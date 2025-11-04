// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/POCInstrumentation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <iostream>

namespace mlir::tt {

POCInstrumentation::POCInstrumentation(const std::string &outputFile, 
                                       DumpLevel level, bool debug)
    : counter_(0), outputFile_(outputFile), level_(level), debug_(debug) {
  // Clear the output file on construction
  std::ofstream file(outputFile_, std::ios::trunc);
  if (file.is_open()) {
    file << "# POC Counter Log\n";
    file << "# Counter increments during MLIR pass execution\n";
    file << "# Dump Level: ";
    switch (level_) {
      case DumpLevel::Pipeline:
        file << "Pipeline\n";
        break;
      case DumpLevel::Pass:
        file << "Pass\n";
        break;
      case DumpLevel::Transformation:
        file << "Transformation\n";
        break;
    }
    file << "# Format: counter_value reason details\n\n";
    file.close();
  }

  if (debug_) {
    std::cout << "POCInstrumentation: Constructor called, output file: " << outputFile_ << std::endl;
  }
}

POCInstrumentation::~POCInstrumentation() {
  // Final summary
  std::lock_guard<std::mutex> lock(fileMutex_);
  std::ofstream file(outputFile_, std::ios::app);
  if (file.is_open()) {
    file << "\n# Final counter value: " << counter_.load() << "\n";
    file.close();
  }
}

void POCInstrumentation::attachActionHandler(mlir::MLIRContext *ctx) {
  if (!ctx) {
    llvm::errs() << "POCInstrumentation: Cannot attach action handler - null context\n";
    return;
  }

  if (debug_) {
    llvm::outs() << "POCInstrumentation: Registering action handler with context\n";
  }

  // Register an action handler that increments counter on every action
  ctx->registerActionHandler([this](llvm::function_ref<void()> transform,
                                    const mlir::tracing::Action &action) {
    // Log the action before executing the transform
    std::string actionTag = action.getTag().str();
    std::string details = "Action: " + actionTag;
    
    // Try to get operation name if available
    if (!action.getContextIRUnits().empty()) {
      // Check if the IRUnit is an Operation before casting
      if (auto *op = llvm::dyn_cast_if_present<mlir::Operation*>(action.getContextIRUnits()[0])) {
        details += ", Op: " + op->getName().getStringRef().str();
      }
    }
    
    logIncrement("ACTION", details);
    
    // Execute the actual transform
    transform();
  });
}

void POCInstrumentation::runBeforePipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  std::string details = name ? name->getStringRef().str() : "any";
  logIncrement("PIPELINE_START", "Operation: " + details);
}

void POCInstrumentation::runAfterPipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  std::string details = name ? name->getStringRef().str() : "any";
  logIncrement("PIPELINE_END", "Operation: " + details);
}

void POCInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  // Only log if level is Pass or Transformation
  if (level_ == DumpLevel::Pipeline)
    return;
    
  std::string details = "Pass: " + pass->getName().str() +
                       ", Op: " + op->getName().getStringRef().str();
  logIncrement("PASS_BEFORE", details);
}

void POCInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  // Only log if level is Pass or Transformation
  if (level_ == DumpLevel::Pipeline)
    return;
    
  std::string details = "Pass: " + pass->getName().str() +
                       ", Op: " + op->getName().getStringRef().str();
  logIncrement("PASS_AFTER", details);
}

void POCInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  // Only log if level is Pass or Transformation
  if (level_ == DumpLevel::Pipeline)
    return;
    
  std::string details = "Pass: " + pass->getName().str() +
                       ", Op: " + op->getName().getStringRef().str();
  logIncrement("PASS_FAILED", details);
}

void POCInstrumentation::runBeforeAnalysis(StringRef name, TypeID id,
                                          Operation *op) {
  // Only log if level is Transformation
  if (level_ != DumpLevel::Transformation)
    return;
    
  std::string details = "Analysis: " + name.str() +
                       ", Op: " + op->getName().getStringRef().str();
  logIncrement("ANALYSIS_BEFORE", details);
}

void POCInstrumentation::runAfterAnalysis(StringRef name, TypeID id,
                                         Operation *op) {
  // Only log if level is Transformation
  if (level_ != DumpLevel::Transformation)
    return;
    
  std::string details = "Analysis: " + name.str() +
                       ", Op: " + op->getName().getStringRef().str();
  logIncrement("ANALYSIS_AFTER", details);
}

void POCInstrumentation::logIncrement(const std::string &reason,
                                     const std::string &details) {
  int currentValue = counter_.fetch_add(1, std::memory_order_relaxed) + 1;

  std::lock_guard<std::mutex> lock(fileMutex_);
  std::ofstream file(outputFile_, std::ios::app);
  if (file.is_open()) {
    file << currentValue << " " << reason;
    if (!details.empty()) {
      file << " - " << details;
    }
    file << "\n";
    file.close();
  } else {
    // Fallback to stderr if file write fails
    std::cerr << "POCInstrumentation: Failed to write to " << outputFile_
              << std::endl;
  }
}

} // namespace mlir::tt
