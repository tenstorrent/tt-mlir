// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/TTPrintIRInstrumentation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace mlir::tt {

TTPrintIRInstrumentation::TTPrintIRInstrumentation(
    TTPrintIRInstrumentationOptions options)
    : dumpCounter_(0), pipelineName_(options.pipelineName),
      level_(options.level), currentDepth_(0) {
  // Set model name - use provided name or default to "unknown"
  modelName_ = options.modelName.empty() ? "unknown" : options.modelName;

  // Expand ~ to home directory using LLVM utility
  llvm::SmallVector<char, 256> expandedPath;
  llvm::sys::fs::expand_tilde(options.outputDir, expandedPath);
  outputDir_ = std::string(expandedPath.begin(), expandedPath.end());

  // Create output directory
  std::filesystem::create_directories(outputDir_);

  // Initialize counter if model name was provided explicitly
  if (!options.modelName.empty()) {
    initializeDumpCounter();
  }
}

TTPrintIRInstrumentation::~TTPrintIRInstrumentation() {
  // For Pipeline level, dump any remaining IR at top-level (depth 0)
  if (level_ == DumpLevel::Pipeline && currentDepth_ == 0 &&
      !pipelineIRStack_.empty() && !pipelineIRStack_[0].empty()) {
    std::string filename = "depth0_pipeline";
    writeIRStringToFile(pipelineIRStack_[0], filename);
  }
}

void TTPrintIRInstrumentation::attachActionHandler(mlir::MLIRContext *ctx) {
  if (!ctx) {
    llvm::errs() << "TTPrintIRInstrumentation: Cannot attach action handler - "
                    "null context\n";
    return;
  }

  // Register an action handler to dump IR on PassExecutionAction
  ctx->registerActionHandler([this](llvm::function_ref<void()> transform,
                                    const mlir::tracing::Action &action) {
    // Execute the transform first
    transform();

    // Only dump IR for PassExecutionAction and only if level is Transformation
    if (level_ == DumpLevel::Transformation &&
        mlir::isa<mlir::PassExecutionAction>(action)) {
      auto passAction = mlir::cast<mlir::PassExecutionAction>(action);
      mlir::Operation *op = passAction.getOp();

      if (op) {
        // Extract model name on first pass
        if (modelName_ == "unknown") {
          std::string extractedName = extractModelNameFromLocation(op);
          if (extractedName != "unknown") {
            setModelName(extractedName);
          }
        }

        std::string passName = passAction.getPass().getName().str();
        dumpIR(op, passName, "transformation_action");
      }
    }
  });
}

void TTPrintIRInstrumentation::initializeDumpCounter() { dumpCounter_ = 0; }

void TTPrintIRInstrumentation::setModelName(const std::string &name) {
  modelName_ = name;
  // Initialize counter when model name is first set
  initializeDumpCounter();
}

void TTPrintIRInstrumentation::runBeforePipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  if (level_ != DumpLevel::Pipeline) {
    return;
  }

  currentDepth_++;
}

void TTPrintIRInstrumentation::runAfterPipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  if (level_ != DumpLevel::Pipeline) {
    return;
  }

  // Dump the accumulated IR if we have any
  if (!pipelineIRStack_.empty() && !pipelineIRStack_.back().empty()) {
    // Generate filename with depth and operation name
    std::string opName =
        name.has_value() ? name->getStringRef().str() : "pipeline";
    std::string filename =
        "depth" + std::to_string(currentDepth_) + "_" + opName;

    // Write stored IR string to file
    writeIRStringToFile(pipelineIRStack_.back(), filename);
  }

  // Pop this pipeline level
  if (!pipelineIRStack_.empty()) {
    pipelineIRStack_.pop_back();
  }
  currentDepth_--;
}

void TTPrintIRInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  // Pass dumps handled in runAfterPass
}

void TTPrintIRInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  // For Pipeline level, capture IR as string
  if (level_ == DumpLevel::Pipeline) {
    if (op) {
      std::string irString;
      llvm::raw_string_ostream os(irString);
      mlir::OpPrintingFlags flags;
      flags.enableDebugInfo();
      op->print(os, flags);
      os.flush();

      // Store at current depth - for top-level (depth 0), push to create stack
      // entry
      if (currentDepth_ < static_cast<int>(pipelineIRStack_.size())) {
        // Overwrite existing entry at this depth
        pipelineIRStack_[currentDepth_] = std::move(irString);
      } else {
        pipelineIRStack_.push_back(std::move(irString));
      }
    }
    return;
  }

  if (level_ == DumpLevel::Pass) {

    if (!op) {
      return;
    }

    // Extract model name on first pass
    if (modelName_ == "unknown") {
      std::string extractedName = extractModelNameFromLocation(op);
      if (extractedName != "unknown") {
        setModelName(extractedName);
      }
    }

    std::string passName = pass->getName().str();
    dumpIR(op, passName, "pass_instrumentation");
  }
}

void TTPrintIRInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  // Don't dump on failed passes
}

void TTPrintIRInstrumentation::runBeforeAnalysis(StringRef name, TypeID id,
                                                 Operation *op) {
  // Analysis tracking not needed for IR dumps
}

void TTPrintIRInstrumentation::runAfterAnalysis(StringRef name, TypeID id,
                                                Operation *op) {
  // Analysis tracking not needed for IR dumps
}

void TTPrintIRInstrumentation::writeIRStringToFile(const std::string &irString,
                                                   const std::string &name) {
  // Get output filename
  std::string filename = getOutputFilename(name);

  // Create directory if needed
  std::filesystem::path filePath(filename);
  std::filesystem::create_directories(filePath.parent_path());

  // Write IR string to file
  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec);
  if (!ec) {
    file << irString;
    file.close();
  } else {
    llvm::errs() << "TTPrintIRInstrumentation: Failed to open file " << filename
                 << ": " << ec.message() << "\n";
  }

  dumpCounter_++;
}

void TTPrintIRInstrumentation::dumpIR(mlir::Operation *op,
                                      const std::string &name,
                                      const std::string &source) {
  if (!op) {
    return;
  }

  // Get output filename
  std::string filename = getOutputFilename(name);

  // Create directory if needed
  std::filesystem::path filePath(filename);
  std::filesystem::create_directories(filePath.parent_path());

  // Dump IR to file
  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec);
  if (!ec) {
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo();
    op->print(file, flags);
    file.close();

  } else {
    llvm::errs() << "TTPrintIRInstrumentation: Failed to open file " << filename
                 << ": " << ec.message() << "\n";
  }

  dumpCounter_++;
}

std::string TTPrintIRInstrumentation::extractModelNameFromLocation(
    mlir::Operation *op) const {
  if (!op) {
    return "unknown";
  }

  mlir::Location loc = op->getLoc();

  // Try to extract filename from FileLineColLoc
  // Examples from real test files:
  //   #loc8 =
  //   loc("/proj_sw/user_dev/sdjukic/tt-xla-repo/tt-xla/third_party/tt_forge_models/mnist/image_classification/jax/mlp/model_implementation.py":16:12
  //   to :39)
  //   -> "model_implementation" (extracts filename without path/extension)
  //
  //   %arg0: tensor<128xf32> ... loc("variables['params']['Dense_0']['bias']")
  //   -> "variables['params']['Dense_0']['bias']" (no path separators,
  //   unchanged)
  //
  //   #loc = loc("ResNetForImageClassification")
  //   -> "ResNetForImageClassification" (module name, unchanged)
  if (mlir::isa<mlir::FileLineColLoc>(loc)) {
    mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
    llvm::StringRef filename = fileLoc.getFilename();
    if (!filename.empty()) {
      // Extract just the filename without path and extension
      std::string filenameStr = filename.str();
      size_t lastSlash = filenameStr.find_last_of("/\\");
      if (lastSlash != std::string::npos) {
        filenameStr = filenameStr.substr(lastSlash + 1);
      }
      size_t lastDot = filenameStr.find_last_of(".");
      if (lastDot != std::string::npos) {
        filenameStr = filenameStr.substr(0, lastDot);
      }
      return filenameStr;
    }
  }

  // Try to extract from FusedLoc (multiple locations fused together)
  // Example from real test files:
  //   #loc42 = loc(callsite(#loc33 at #loc34))
  //   where #loc33 and #loc34 reference different source locations
  //   -> returns filename from first FileLineColLoc found in the fused location
  //   list
  //
  //   loc(callsite(#loc74 at #loc75)) creates a fused location from two
  //   sub-locations
  if (mlir::isa<mlir::FusedLoc>(loc)) {
    mlir::FusedLoc fusedLoc = mlir::cast<mlir::FusedLoc>(loc);
    for (mlir::Location subLoc : fusedLoc.getLocations()) {
      if (mlir::isa<mlir::FileLineColLoc>(subLoc)) {
        mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(subLoc);
        llvm::StringRef filename = fileLoc.getFilename();
        if (!filename.empty()) {
          // Extract just the filename without path and extension
          std::string filenameStr = filename.str();
          size_t lastSlash = filenameStr.find_last_of("/\\");
          if (lastSlash != std::string::npos) {
            filenameStr = filenameStr.substr(lastSlash + 1);
          }
          size_t lastDot = filenameStr.find_last_of(".");
          if (lastDot != std::string::npos) {
            filenameStr = filenameStr.substr(0, lastDot);
          }
          return filenameStr;
        }
      }
    }
  }

  return "unknown";
}

std::string
TTPrintIRInstrumentation::sanitizeFilename(const std::string &name) const {
  std::string result = name;
  std::replace_if(
      result.begin(), result.end(),
      [](char c) { return !std::isalnum(c) && c != '_' && c != '-'; }, '_');
  return result;
}

std::string
TTPrintIRInstrumentation::getOutputFilename(const std::string &name) const {
  std::string safeName = sanitizeFilename(name);
  std::string safeModelName = sanitizeFilename(modelName_);
  std::string safePipelineName = sanitizeFilename(pipelineName_);

  // Format: <outputDir>/<modelName>/<pipelineName>/<counter>_<name>.mlir
  std::string filename =
      std::to_string(dumpCounter_.load()) + "_" + safeName + ".mlir";
  std::string subdirPath =
      outputDir_ + "/" + safeModelName + "/" + safePipelineName;

  return subdirPath + "/" + filename;
}

std::string TTPrintIRInstrumentation::getTargetDirectory() const {
  std::string safeModelName = sanitizeFilename(modelName_);
  std::string safePipelineName = sanitizeFilename(pipelineName_);
  return outputDir_ + "/" + safeModelName + "/" + safePipelineName;
}

void TTPrintIRInstrumentation::clearDirectory(
    const std::string &targetDir) const {
  if (std::filesystem::exists(targetDir)) {
    std::filesystem::remove_all(targetDir);
  }
  std::filesystem::create_directories(targetDir);
}

//===--------------------------------------------------------------------===//
// Convenience Functions
//===--------------------------------------------------------------------===//

void addTTPrintIRInstrumentation(
    PassManager &pm,
    TTPrintIRInstrumentation::TTPrintIRInstrumentationOptions options) {
  auto instrumentation = std::make_unique<TTPrintIRInstrumentation>(options);
  // Attach action handler to enable transformation-level tracing
  instrumentation->attachActionHandler(pm.getContext());
  pm.addInstrumentation(std::move(instrumentation));
}

} // namespace mlir::tt
