// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/POCInstrumentation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace mlir::tt {

POCInstrumentation::POCInstrumentation(const std::string &outputDir, 
                                       DumpLevel level,
                                       ActionMode actionMode,
                                       bool debug)
    : dumpCounter_(0), outputDir_(outputDir), modelName_("unknown"), 
      level_(level), actionMode_(actionMode), debug_(debug) {
  // Create output directory
  std::filesystem::create_directories(outputDir_);

  if (debug_) {
    llvm::outs() << "POCInstrumentation: Constructor called, output dir: " 
                 << outputDir_ << ", action mode: " 
                 << (actionMode_ == ActionMode::Overwrite ? "Overwrite" : "Append") << "\n";
  }
}

POCInstrumentation::~POCInstrumentation() {
  if (debug_) {
    llvm::outs() << "POCInstrumentation: Destructor called, total dumps: " 
                 << dumpCounter_.load() << "\n";
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

  // Register an action handler to dump IR on PassExecutionAction
  ctx->registerActionHandler([this](llvm::function_ref<void()> transform,
                                    const mlir::tracing::Action &action) {
    // Execute the transform first
    transform();
    
    // Only dump IR for PassExecutionAction and only if level is Transformation
    if (level_ == DumpLevel::Transformation && mlir::isa<mlir::PassExecutionAction>(action)) {
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
        dumpIR(op, passName);
      }
    }
  });
}

void POCInstrumentation::setModelName(const std::string &name) {
  modelName_ = name;
  
  // Initialize counter based on action mode when model name is first set
  std::string targetDir = getTargetDirectory();
  if (actionMode_ == ActionMode::Overwrite) {
    clearDirectory(targetDir);
    dumpCounter_ = 0;
  } else {  // Append mode
    int maxIndex = detectNextIndex(targetDir);
    dumpCounter_ = (maxIndex >= 0) ? maxIndex + 1 : 0;
  }
}

void POCInstrumentation::runBeforePipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  // Pipeline dumps happen at boundaries for all levels
}

void POCInstrumentation::runAfterPipeline(
    std::optional<OperationName> name, const PipelineParentInfo &parentInfo) {
  // Pipeline dumps happen at boundaries for all levels
}

void POCInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  // Pass dumps handled in runAfterPass
}

void POCInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  // Only dump if level is Pass or higher (not Pipeline-only)
  if (level_ == DumpLevel::Pipeline)
    return;
    
  if (!op)
    return;
    
  // Extract model name on first pass
  if (modelName_ == "unknown") {
    std::string extractedName = extractModelNameFromLocation(op);
    if (extractedName != "unknown") {
      setModelName(extractedName);
    }
  }
  
  std::string passName = pass->getName().str();
  dumpIR(op, passName);
}

void POCInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  // Don't dump on failed passes
}

void POCInstrumentation::runBeforeAnalysis(StringRef name, TypeID id,
                                          Operation *op) {
  // Analysis tracking not needed for IR dumps
}

void POCInstrumentation::runAfterAnalysis(StringRef name, TypeID id,
                                         Operation *op) {
  // Analysis tracking not needed for IR dumps
}

void POCInstrumentation::dumpIR(mlir::Operation *op, const std::string &name) {
  if (!op)
    return;
    
  std::lock_guard<std::mutex> lock(fileMutex_);
  
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

    if (debug_) {
      llvm::outs() << "POCInstrumentation: Dumped IR to " << filename << "\n";
    }
  } else {
    llvm::errs() << "POCInstrumentation: Failed to open file " << filename << ": " << ec.message() << "\n";
  }
  
  dumpCounter_++;
}

std::string POCInstrumentation::extractModelNameFromLocation(mlir::Operation *op) const {
  if (!op) {
    return "unknown";
  }

  mlir::Location loc = op->getLoc();
  
  // Try to extract filename from FileLineColLoc
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
  
  // Try to extract from FusedLoc
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

std::string POCInstrumentation::sanitizeFilename(const std::string &name) const {
  std::string result = name;
  std::replace_if(result.begin(), result.end(),
                  [](char c) { return !std::isalnum(c) && c != '_' && c != '-'; },
                  '_');
  return result;
}

std::string POCInstrumentation::getOutputFilename(const std::string &name) const {
  std::string safeName = sanitizeFilename(name);
  std::string safeModelName = sanitizeFilename(modelName_);
  
  // Format: <outputDir>/<modelName>/<counter>_<name>.mlir
  std::string filename = std::to_string(dumpCounter_.load()) + "_" + safeName + ".mlir";
  std::string subdirPath = outputDir_ + "/" + safeModelName;
  
  return subdirPath + "/" + filename;
}

std::string POCInstrumentation::getTargetDirectory() const {
  std::string safeModelName = sanitizeFilename(modelName_);
  return outputDir_ + "/" + safeModelName;
}

int POCInstrumentation::detectNextIndex(const std::string &targetDir) const {
  if (!std::filesystem::exists(targetDir)) {
    return -1;  // Directory doesn't exist, start from 0
  }

  int maxIndex = -1;
  for (const auto& entry : std::filesystem::directory_iterator(targetDir)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      // Look for pattern: <number>_<anything>.mlir
      if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".mlir") {
        size_t underscorePos = filename.find('_');
        if (underscorePos != std::string::npos) {
          std::string indexStr = filename.substr(0, underscorePos);
          int index;
          if (sscanf(indexStr.c_str(), "%d", &index) == 1) {
            if (index > maxIndex) {
              maxIndex = index;
            }
          }
        }
      }
    }
  }
  return maxIndex;
}

void POCInstrumentation::clearDirectory(const std::string &targetDir) const {
  if (std::filesystem::exists(targetDir)) {
    std::filesystem::remove_all(targetDir);
  }
  std::filesystem::create_directories(targetDir);
}

} // namespace mlir::tt
