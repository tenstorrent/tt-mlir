// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SUPPORT_TTGRAPHTELEMETRYINSTRUMENTATION_H
#define TTMLIR_SUPPORT_TTGRAPHTELEMETRYINSTRUMENTATION_H

#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

// Forward declarations for telemetry types that only need pointers.
namespace mlir::tt::telemetry {
class TelemetryVisitorRegistry;
class TelemetryWriter;
} // namespace mlir::tt::telemetry

namespace mlir::tt {

/// Graph-level metadata and output location for a telemetry session. The
/// emitter records compilation facts; run-level CI and GitHub provenance is
/// stamped later by the push tool.
struct TTGraphTelemetryOptions {
  /// Directory where the telemetry JSON file is written.
  std::string outputDir;

  /// Unique identifier for this graph. If empty, a UUID is generated.
  std::string graphId;

  /// Name of the model being compiled. Extracted from the IR location if empty.
  std::string modelName;

  /// The test/invocation that triggered this compile (e.g. a pytest node id).
  /// Compilation context only the frontend knows at emit time; optional.
  std::string testName;
};

class TTGraphTelemetrySession;

/// PassInstrumentation that serializes the IR graph into a shared telemetry
/// session at configured points. One instance is attached per PassManager; all
/// instances of a session write into one JSON with a monotonic snapshot index,
/// so a compilation spanning several PassManagers (e.g. tt-xla's SHLO,
/// SHLO->TTIR and TTIR->TTNN pipelines) produces a single graph.
class TTGraphTelemetryInstrumentation : public PassInstrumentation {
public:
  /// What to capture for one instrumented pipeline stage.
  struct Stage {
    /// Capture the input IR before the first pass (empty = skip).
    std::string initialTag;
    /// Capture the IR after the stage's pipeline finishes (empty = skip).
    std::string finalTag;
    /// Snapshot after the first pass whose name contains any of these
    /// substrings (empty = no per-pass snapshots).
    std::vector<std::string> targetPasses;
  };

  TTGraphTelemetryInstrumentation(TTGraphTelemetrySession *session,
                                  Stage stage);
  ~TTGraphTelemetryInstrumentation() override;

  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;

private:
  /// Serialize `op` into the session as a snapshot with the next shared index.
  void emitSnapshot(Operation *op, llvm::StringRef tag);

  /// Resolve a model name from the top-level operation (symbol name, then
  /// file location).
  std::string resolveModelName(Operation *op) const;

  TTGraphTelemetrySession *session_;
  Stage stage_;

  bool initialCaptured_ = false;
  Operation *lastOp_ = nullptr;
  std::set<std::string> capturedPasses_;
};

/// Accumulates telemetry snapshots from the PassManagers of one compilation
/// (one Graph in the schema) and writes a single JSON file. The caller owns the
/// session, keeps it alive across every PassManager it instruments, and calls
/// flush() when done.
class TTGraphTelemetrySession {
public:
  TTGraphTelemetrySession(TTGraphTelemetryOptions options,
                          std::string sourcePath = "", int graphIndex = 0);
  ~TTGraphTelemetrySession();

  /// Attach an instrumentation configured by `stage` to `pm`.
  void instrument(PassManager &pm,
                  TTGraphTelemetryInstrumentation::Stage stage);

  /// Write the accumulated snapshots to JSON. Idempotent.
  void flush();

private:
  friend class TTGraphTelemetryInstrumentation;

  int nextIndex() { return snapshotIndex_++; }
  telemetry::TelemetryVisitorRegistry &registry() { return *registry_; }
  telemetry::TelemetryWriter &writer() { return *writer_; }

  TTGraphTelemetryOptions options_;
  std::unique_ptr<telemetry::TelemetryVisitorRegistry> registry_;
  std::unique_ptr<telemetry::TelemetryWriter> writer_;
  int snapshotIndex_ = 0;
  bool modelNameResolved_ = false;
  bool flushed_ = false;
};

} // namespace mlir::tt

#endif // TTMLIR_SUPPORT_TTGRAPHTELEMETRYINSTRUMENTATION_H
