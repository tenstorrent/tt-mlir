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

namespace {
// Helper function to expand and create output directory
std::string expandAndCreateOutputDir(const std::string &outputDir) {
  llvm::SmallVector<char, 256> expandedPath;
  llvm::sys::fs::expand_tilde(outputDir, expandedPath);
  std::string result(expandedPath.begin(), expandedPath.end());
  std::filesystem::create_directories(result);
  return result;
}
} // namespace

TTPrintIRInstrumentation::TTPrintIRInstrumentation(
    TTPrintIRInstrumentationOptions options)
    : dumpCounter_(0), outputDir_(expandAndCreateOutputDir(options.outputDir)),
      modelName_(options.modelName.empty() ? "unknown" : options.modelName),
      pipelineName_(options.pipelineName), level_(options.level),
      dumpInitial_(options.dumpInitial), dumpedInitial_(false),
      onlyDumpOnChanges_(options.onlyDumpOnChanges), currentDepth_(0) {
  // Initialize counter if model name was provided explicitly
  if (!options.modelName.empty()) {
    initializeDumpCounter();
  }

  // Initialize pipeline IR stack for relevant levels
  if (level_ == DumpLevel::Once || level_ == DumpLevel::Pipeline) {
    pipelineIRStack_.resize(1);
  }
}

TTPrintIRInstrumentation::~TTPrintIRInstrumentation() {
  // For Once or Pipeline level, dump any remaining IR at top-level (depth 0)
  if ((level_ == DumpLevel::Once || level_ == DumpLevel::Pipeline) &&
      currentDepth_ == 0 && !pipelineIRStack_.empty() &&
      !pipelineIRStack_[0].empty()) {
    std::string filename =
        "after_" + (pipelineName_.empty() ? "pipeline" : pipelineName_);
    dumpIR(pipelineIRStack_[0], filename);
  }
}

void TTPrintIRInstrumentation::attachActionHandler(mlir::MLIRContext *ctx) {
  if (!ctx) {
    llvm::errs() << "TTPrintIRInstrumentation: Cannot attach action handler - "
                    "null context\n";
    return;
  }

  // Register an action handler to dump IR on transformation actions
  ctx->registerActionHandler([this](llvm::function_ref<void()> transform,
                                    const mlir::tracing::Action &action) {
    // Execute the transform first
    transform();

    // Only process if level is Transformation
    if (level_ != DumpLevel::Transformation) {
      return;
    }

    std::string actionTag = action.getTag().str();

    // Handle PassExecutionAction - dumps IR after pass completes
    if (mlir::isa<mlir::PassExecutionAction>(action)) {
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
        dumpIR(op, passName + "_after", "pass_execution");
      }
    }
    // Handle GreedyPatternRewriteIteration - dumps IR after each iteration
    else if (actionTag == "GreedyPatternRewriteIteration") {
      auto irUnits = action.getContextIRUnits();

      if (!irUnits.empty()) {
        mlir::Operation *op = nullptr;

        // Extract operation from IRUnit
        if (auto *opPtr =
                llvm::dyn_cast_if_present<mlir::Operation *>(irUnits[0])) {
          op = opPtr;
        } else if (auto *region =
                       llvm::dyn_cast_if_present<mlir::Region *>(irUnits[0])) {
          op = region->getParentOp();
        } else if (auto *block =
                       llvm::dyn_cast_if_present<mlir::Block *>(irUnits[0])) {
          op = block->getParentOp();
        }

        if (op) {
          if (modelName_ == "unknown") {
            std::string extractedName = extractModelNameFromLocation(op);
            if (extractedName != "unknown") {
              setModelName(extractedName);
            }
          }

          // Extract iteration number from action.print() output
          std::string actionStr;
          llvm::raw_string_ostream os(actionStr);
          action.print(os);
          os.flush();

          // actionStr format: "GreedyPatternRewriteIteration(N)"
          std::string iterNum = "0";
          size_t openParen = actionStr.find('(');
          size_t closeParen = actionStr.find(')');
          if (openParen != std::string::npos &&
              closeParen != std::string::npos) {
            iterNum =
                actionStr.substr(openParen + 1, closeParen - openParen - 1);
          }

          std::string opName =
              sanitizeFilename(op->getName().getStringRef().str());
          std::string filename = actionTag + "_iter" + iterNum + "_" + opName;
          dumpIR(op, filename, "greedy_iteration");
        }
      }
    }
    // Handle apply-pattern if it exists
    else if (actionTag == "apply-pattern") {
      auto irUnits = action.getContextIRUnits();

      if (!irUnits.empty()) {
        mlir::Operation *op = nullptr;

        // Extract operation from IRUnit
        if (auto *opPtr =
                llvm::dyn_cast_if_present<mlir::Operation *>(irUnits[0])) {
          op = opPtr;
        } else if (auto *region =
                       llvm::dyn_cast_if_present<mlir::Region *>(irUnits[0])) {
          op = region->getParentOp();
        } else if (auto *block =
                       llvm::dyn_cast_if_present<mlir::Block *>(irUnits[0])) {
          op = block->getParentOp();
        }

        if (op) {
          if (modelName_ == "unknown") {
            std::string extractedName = extractModelNameFromLocation(op);
            if (extractedName != "unknown") {
              setModelName(extractedName);
            }
          }

          std::string opName =
              sanitizeFilename(op->getName().getStringRef().str());
          std::string filename = actionTag + "_" + opName;
          dumpIR(op, filename, "apply_pattern");
        }
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

  std::string opName =
      name.has_value() ? name->getStringRef().str() : "pipeline";
  std::string parentPassName = parentInfo.parentPass
                                   ? parentInfo.parentPass->getName().str()
                                   : "unknown";

  // Dump the accumulated IR if we have any
  if (!pipelineIRStack_.empty() && !pipelineIRStack_.back().empty()) {
    // Generate filename: {sanitized_parent_pass}_{operation_name}_pipeline
    std::string sanitizedParentPass = sanitizeFilename(parentPassName);
    std::string sanitizedOpName = sanitizeFilename(opName);
    std::string filename =
        sanitizedParentPass + "_" + sanitizedOpName + "_pipeline";

    // Write stored IR string to file
    dumpIR(pipelineIRStack_.back(), filename);
  }

  // Pop this pipeline level
  if (!pipelineIRStack_.empty()) {
    pipelineIRStack_.pop_back();
  }
  currentDepth_--;
}

void TTPrintIRInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (!pass) {
    return;
  }

  // Dump initial IR if requested and not already dumped
  if (dumpInitial_ && !dumpedInitial_ && op) {
    dumpedInitial_ = true;
    dumpIR(op, "initial", "initial_dump");
  }
  if (dumpCounter_ == 0) {
    dumpCounter_ = 1;
  }

  // Pass dumps handled in runAfterPass
}

void TTPrintIRInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  if (!pass || !op) {
    return;
  }

  // For Once or Pipeline level, capture IR as string
  if (level_ == DumpLevel::Once || level_ == DumpLevel::Pipeline) {
    // For Once level, only capture at depth 0 (top-level)
    if (level_ == DumpLevel::Once && currentDepth_ != 0) {
      return;
    }

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

  // Pass level includes Pass and Transformation (Transformation includes Pass)
  if (level_ == DumpLevel::Pass || level_ == DumpLevel::Transformation) {

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
  if (!pass) {
    return;
  }
  // Don't dump on failed passes
}

void TTPrintIRInstrumentation::dumpIR(mlir::Operation *op,
                                      const std::string &name,
                                      const std::string &source) {
  if (!op) {
    return;
  }

  // Convert operation to string and call string overload
  std::string irString;
  llvm::raw_string_ostream os(irString);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  op->print(os, flags);
  os.flush();

  dumpIR(irString, name);
}

void TTPrintIRInstrumentation::dumpIR(const std::string &irString,
                                      const std::string &name) {
  // Check if IR has changed since last dump
  if (onlyDumpOnChanges_ && irString == lastDumpedIR_) {
    // IR hasn't changed, skip dumping
    return;
  }

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
    // Update last dumped IR
    lastDumpedIR_ = irString;
  } else {
    llvm::errs() << "TTPrintIRInstrumentation: Failed to open file " << filename
                 << ": " << ec.message() << "\n";
  }

  dumpCounter_++;
}

// Helper function to extract clean filename from a path string
std::string extractFilename(llvm::StringRef filename) {
  if (filename.empty()) {
    return "unknown";
  }

  std::string filenameStr = filename.str();

  // Extract just the filename without path and extension
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

std::string TTPrintIRInstrumentation::extractModelNameFromLocation(
    mlir::Operation *op) const {
  if (!op) {
    return "unknown";
  }

  mlir::Location loc = op->getLoc();

  // Try to extract filename from FileLineColLoc
  if (mlir::isa<mlir::FileLineColLoc>(loc)) {
    mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
    return extractFilename(fileLoc.getFilename());
  }

  // Try to extract from FusedLoc (multiple locations fused together)
  if (mlir::isa<mlir::FusedLoc>(loc)) {
    mlir::FusedLoc fusedLoc = mlir::cast<mlir::FusedLoc>(loc);
    for (mlir::Location subLoc : fusedLoc.getLocations()) {
      if (mlir::isa<mlir::FileLineColLoc>(subLoc)) {
        mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(subLoc);
        std::string result = extractFilename(fileLoc.getFilename());
        if (result != "unknown") {
          return result;
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
