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

namespace mlir::tt {

namespace {
std::string expandAndCreateOutputDir(const std::string &outputDir) {
  llvm::SmallVector<char, 256> expandedPath;
  llvm::sys::fs::expand_tilde(outputDir, expandedPath);
  std::string result(expandedPath.begin(), expandedPath.end());
  std::filesystem::create_directories(result);
  return result;
}

mlir::OpPrintingFlags getStandardPrintingFlags() {
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(true);
  flags.elideLargeElementsAttrs(16);
  flags.elideLargeResourceString(64);
  return flags;
}

mlir::Operation *extractOperationFromIRUnit(const mlir::IRUnit &unit) {
  if (auto *opPtr = llvm::dyn_cast_if_present<mlir::Operation *>(unit)) {
    return opPtr;
  }
  if (auto *region = llvm::dyn_cast_if_present<mlir::Region *>(unit)) {
    return region->getParentOp();
  }
  if (auto *block = llvm::dyn_cast_if_present<mlir::Block *>(unit)) {
    return block->getParentOp();
  }
  return nullptr;
}
} // namespace

TTPrintIRInstrumentation::TTPrintIRInstrumentation(
    TTPrintIRInstrumentationOptions options)
    : dumpCounter_(0), outputDir_(expandAndCreateOutputDir(options.outputDir)),
      modelName_(options.modelName.empty() ? "unknown" : options.modelName),
      pipelineName_(options.pipelineName), level_(options.level),
      dumpInitial_(options.dumpInitial), dumpedInitial_(false),
      onlyDumpOnChanges_(options.onlyDumpOnChanges), currentDepth_(0) {
  if (!options.modelName.empty()) {
    initializeDumpCounter();
  }
  if (level_ == DumpLevel::Once || level_ == DumpLevel::Pipeline) {
    pipelineIRStack_.resize(1);
  }
}

TTPrintIRInstrumentation::~TTPrintIRInstrumentation() {
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

  ctx->registerActionHandler([this](llvm::function_ref<void()> transform,
                                    const mlir::tracing::Action &action) {
    transform();
    if (level_ != DumpLevel::Transformation) {
      return;
    }

    std::string actionTag = action.getTag().str();

    if (mlir::isa<mlir::PassExecutionAction>(action)) {
      auto passAction = mlir::cast<mlir::PassExecutionAction>(action);
      mlir::Operation *op = passAction.getOp();
      if (op) {
        std::string passName = passAction.getPass().getName().str();
        dumpIR(op, passName + "_after");
      }
    } else if (actionTag == "GreedyPatternRewriteIteration") {
      auto irUnits = action.getContextIRUnits();
      if (!irUnits.empty()) {
        mlir::Operation *op = extractOperationFromIRUnit(irUnits[0]);
        if (op) {
          std::string actionStr;
          llvm::raw_string_ostream os(actionStr);
          action.print(os);
          os.flush();
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
          dumpIR(op, filename);
        }
      }
    } else if (actionTag == "apply-pattern") {
      auto irUnits = action.getContextIRUnits();
      if (!irUnits.empty()) {
        mlir::Operation *op = extractOperationFromIRUnit(irUnits[0]);
        if (op) {
          std::string opName =
              sanitizeFilename(op->getName().getStringRef().str());
          std::string filename = actionTag + "_" + opName;
          dumpIR(op, filename);
        }
      }
    }
  });
}

void TTPrintIRInstrumentation::initializeDumpCounter() { dumpCounter_ = 0; }

void TTPrintIRInstrumentation::setModelName(const std::string &name) {
  modelName_ = name;
  initializeDumpCounter();
}

void TTPrintIRInstrumentation::runBeforePipeline(std::optional<OperationName>,
                                                 const PipelineParentInfo &) {
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
  if (!pipelineIRStack_.empty() && !pipelineIRStack_.back().empty()) {
    std::string sanitizedParentPass = sanitizeFilename(parentPassName);
    std::string sanitizedOpName = sanitizeFilename(opName);
    std::string filename =
        sanitizedParentPass + "_" + sanitizedOpName + "_pipeline";
    dumpIR(pipelineIRStack_.back(), filename);
  }
  if (!pipelineIRStack_.empty()) {
    pipelineIRStack_.pop_back();
  }
  currentDepth_--;
}

void TTPrintIRInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (!pass) {
    return;
  }
  if (modelName_ == "unknown" && op) {
    std::string extractedName = extractModelNameFromLocation(op);
    if (extractedName != "unknown") {
      setModelName(extractedName);
    }
  }
  if (dumpInitial_ && !dumpedInitial_ && op) {
    dumpedInitial_ = true;
    dumpIR(op, "initial");
  }
  if (dumpCounter_ == 0) {
    dumpCounter_ = 1;
  }
}

void TTPrintIRInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  if (!pass || !op) {
    return;
  }
  if (level_ == DumpLevel::Once || level_ == DumpLevel::Pipeline) {
    if (level_ == DumpLevel::Once && currentDepth_ != 0) {
      return;
    }
    std::string irString;
    llvm::raw_string_ostream os(irString);
    op->print(os, getStandardPrintingFlags());
    os.flush();
    if (currentDepth_ < static_cast<int>(pipelineIRStack_.size())) {
      pipelineIRStack_[currentDepth_] = std::move(irString);
    } else {
      pipelineIRStack_.push_back(std::move(irString));
    }
    return;
  }
  if (level_ == DumpLevel::Pass || level_ == DumpLevel::Transformation) {
    std::string passName = pass->getName().str();
    dumpIR(op, passName);
  }
}

void TTPrintIRInstrumentation::dumpIR(mlir::Operation *op,
                                      const std::string &name) {
  if (!op) {
    return;
  }
  std::string irString;
  llvm::raw_string_ostream os(irString);
  op->print(os, getStandardPrintingFlags());
  os.flush();
  dumpIR(irString, name);
}

void TTPrintIRInstrumentation::dumpIR(const std::string &irString,
                                      const std::string &name) {
  if (onlyDumpOnChanges_ && irString == lastDumpedIR_) {
    return;
  }
  std::string filename = getOutputFilename(name);
  std::filesystem::path filePath(filename);
  std::filesystem::create_directories(filePath.parent_path());
  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec);
  if (!ec) {
    file << irString;
    file.close();
    lastDumpedIR_ = irString;
  } else {
    llvm::errs() << "TTPrintIRInstrumentation: Failed to open file " << filename
                 << ": " << ec.message() << "\n";
  }
  dumpCounter_++;
}

std::string extractFilename(llvm::StringRef filename) {
  if (filename.empty()) {
    return "unknown";
  }
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

std::string TTPrintIRInstrumentation::extractModelNameFromLocation(
    mlir::Operation *op) const {
  if (!op) {
    return "unknown";
  }
  mlir::Location loc = op->getLoc();
  if (mlir::isa<mlir::FileLineColLoc>(loc)) {
    mlir::FileLineColLoc fileLoc = mlir::cast<mlir::FileLineColLoc>(loc);
    return extractFilename(fileLoc.getFilename());
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

  std::string filename =
      std::to_string(dumpCounter_) + "_" + safeName + ".mlir";
  std::string subdirPath =
      outputDir_ + "/" + safeModelName + "/" + safePipelineName;

  return subdirPath + "/" + filename;
}

void addTTPrintIRInstrumentation(
    PassManager &pm,
    TTPrintIRInstrumentation::TTPrintIRInstrumentationOptions options) {
  auto instrumentation = std::make_unique<TTPrintIRInstrumentation>(options);
  instrumentation->attachActionHandler(pm.getContext());
  pm.addInstrumentation(std::move(instrumentation));
}

} // namespace mlir::tt
