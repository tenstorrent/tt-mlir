// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/TTGraphTelemetryInstrumentation.h"

#include "ttmlir/Telemetry/GraphTypes.h"
#include "ttmlir/Telemetry/IRSerializer.h"
#include "ttmlir/Telemetry/TelemetryVisitor.h"
#include "ttmlir/Telemetry/TelemetryWriter.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <chrono>
#include <filesystem>

// Defined in tools/graph-telemetry/visitors/RegisterVisitors.cpp.
namespace mlir::tt::telemetry {
void registerTTMLIRVisitors(TelemetryVisitorRegistry &registry);
} // namespace mlir::tt::telemetry

namespace mlir::tt {

namespace {

/// Expand ~ and ensure the output directory exists.
std::string expandAndCreateOutputDir(const std::string &outputDir) {
  llvm::SmallVector<char, 256> expandedPath;
  llvm::sys::fs::expand_tilde(outputDir, expandedPath);
  std::string result(expandedPath.begin(), expandedPath.end());
  std::filesystem::create_directories(result);
  return result;
}

/// Extract a bare filename (no extension) from a source path.
std::string extractFilename(llvm::StringRef filename) {
  if (filename.empty()) {
    return "unknown";
  }
  std::string s = filename.str();
  size_t lastSlash = s.find_last_of("/\\");
  if (lastSlash != std::string::npos) {
    s = s.substr(lastSlash + 1);
  }
  size_t lastDot = s.find_last_of('.');
  if (lastDot != std::string::npos) {
    s = s.substr(0, lastDot);
  }
  return s;
}

/// Replace characters that are unsafe in a file name with '_'.
std::string sanitizeForFilename(llvm::StringRef name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
                   c == '-' || c == '_')
                      ? c
                      : '_');
  }
  return out;
}

/// Write pre-rendered MLIR `text` into
/// "<outputDir>/<graphId>/<index>_<tag>.mlir" and return that path relative to
/// outputDir (empty on failure), so a viewer can resolve it against the
/// telemetry directory.
std::string writeSnapshotMLIR(llvm::StringRef text,
                              const std::string &outputDir,
                              const std::string &graphId, int index,
                              llvm::StringRef tag) {
  std::string relPath = graphId + "/" + std::to_string(index) + "_" +
                        sanitizeForFilename(tag) + ".mlir";
  std::filesystem::create_directories(outputDir + "/" + graphId);
  std::string absPath = outputDir + "/" + relPath;
  std::error_code ec;
  llvm::raw_fd_ostream os(absPath, ec);
  if (ec) {
    llvm::errs() << "TTGraphTelemetry: failed to write " << absPath << ": "
                 << ec.message() << "\n";
    return "";
  }
  os << text;
  return relPath;
}

} // namespace

//===----------------------------------------------------------------------===//
// TTGraphTelemetryInstrumentation
//===----------------------------------------------------------------------===//

TTGraphTelemetryInstrumentation::TTGraphTelemetryInstrumentation(
    TTGraphTelemetrySession *session, Stage stage)
    : session_(session), stage_(std::move(stage)) {}

TTGraphTelemetryInstrumentation::~TTGraphTelemetryInstrumentation() {
  // Emit the final IR once the stage's pass manager has finished. The module
  // outlives the pass manager (and thus this instrumentation), so serializing
  // lastOp_ here captures the true end-of-pipeline state and gets the last
  // snapshot index.
  if (!stage_.finalTag.empty() && lastOp_) {
    emitSnapshot(lastOp_, stage_.finalTag);
  }
}

void TTGraphTelemetryInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (!pass || !op || initialCaptured_) {
    return;
  }
  initialCaptured_ = true;

  // Resolve the model name from the IR once, if the session doesn't have one.
  if (!session_->modelNameResolved_ && session_->options_.modelName.empty()) {
    std::string name = resolveModelName(op);
    if (name != "unknown") {
      session_->options_.modelName = name;
      session_->modelNameResolved_ = true;
    }
  }

  if (!stage_.initialTag.empty()) {
    emitSnapshot(op, stage_.initialTag);
  }
}

void TTGraphTelemetryInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  if (!pass || !op) {
    return;
  }

  // Track the root module for the final snapshot.
  Operation *root = op;
  while (root->getParentOp()) {
    root = root->getParentOp();
  }
  lastOp_ = root;

  if (stage_.targetPasses.empty()) {
    return;
  }
  llvm::StringRef passName = pass->getName();
  bool matches = llvm::any_of(stage_.targetPasses, [&](const std::string &t) {
    return passName.contains(t);
  });
  if (!matches) {
    return;
  }

  // Only the first snapshot per matching pass name.
  std::string name = passName.str();
  if (!capturedPasses_.insert(name).second) {
    return;
  }
  emitSnapshot(op, name);
}

void TTGraphTelemetryInstrumentation::emitSnapshot(Operation *op,
                                                   llvm::StringRef tag) {
  int index = session_->nextIndex();

  // Print the IR once, capturing each op's line in the output, so serialized
  // op nodes can be deep-linked to their line in the dashboard's MLIR view.
  mlir::AsmState::LocationMap lineMap;
  std::string mlirText;
  {
    llvm::raw_string_ostream os(mlirText);
    mlir::AsmState state(op, mlir::OpPrintingFlags(), &lineMap);
    op->print(os, state);
  }

  auto snapshot = telemetry::serialize(op, session_->registry(), tag.str(),
                                       tag.str(), index, &lineMap);
  // Skip empty snapshots (a pass ran on a scope with no serializable ops).
  if (snapshot.ops.empty()) {
    return;
  }
  // Persist the snapshot's MLIR text alongside the JSON for source references.
  snapshot.mlirPath = writeSnapshotMLIR(mlirText, session_->options_.outputDir,
                                        session_->options_.graphId, index, tag);
  session_->writer().addSnapshot(std::move(snapshot));
}

std::string
TTGraphTelemetryInstrumentation::resolveModelName(Operation *op) const {
  if (!op) {
    return "unknown";
  }
  Operation *root = op;
  while (root->getParentOp()) {
    root = root->getParentOp();
  }
  // Prefer the module's symbol name (e.g. @jit_add), then a file location.
  if (auto symName = root->getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName());
      symName && !symName.getValue().empty()) {
    return symName.getValue().str();
  }
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(root->getLoc())) {
    return extractFilename(fileLoc.getFilename());
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// TTGraphTelemetrySession
//===----------------------------------------------------------------------===//

TTGraphTelemetrySession::TTGraphTelemetrySession(
    TTGraphTelemetryOptions options, std::string sourcePath, int graphIndex)
    : options_(std::move(options)) {
  options_.outputDir = expandAndCreateOutputDir(options_.outputDir);
  if (options_.graphId.empty()) {
    options_.graphId = telemetry::generateUUID();
  }

  registry_ = std::make_unique<telemetry::TelemetryVisitorRegistry>();
  telemetry::registerTTMLIRVisitors(*registry_);

  telemetry::GraphMeta graph;
  graph.graphId = options_.graphId;
  graph.modelName = options_.modelName.empty() ? "unknown" : options_.modelName;
  graph.testName = options_.testName;
  graph.createdAt = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
  writer_ = std::make_unique<telemetry::TelemetryWriter>(
      std::move(graph), std::move(sourcePath), graphIndex);
}

TTGraphTelemetrySession::~TTGraphTelemetrySession() { flush(); }

void TTGraphTelemetrySession::instrument(
    PassManager &pm, TTGraphTelemetryInstrumentation::Stage stage) {
  pm.addInstrumentation(std::make_unique<TTGraphTelemetryInstrumentation>(
      this, std::move(stage)));
}

void TTGraphTelemetrySession::flush() {
  if (flushed_) {
    return;
  }
  flushed_ = true;

  if (modelNameResolved_) {
    writer_->setModelName(options_.modelName);
  }

  std::string outputPath =
      options_.outputDir + "/" + options_.graphId + ".json";
  if (failed(writer_->writeJSON(outputPath))) {
    llvm::errs() << "TTGraphTelemetrySession: failed to write " << outputPath
                 << "\n";
  }
}

} // namespace mlir::tt
