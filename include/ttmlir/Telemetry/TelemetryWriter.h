// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_GRAPH_TELEMETRY_TELEMETRYWRITER_H
#define TOOLS_GRAPH_TELEMETRY_TELEMETRYWRITER_H

#include "ttmlir/Telemetry/GraphTypes.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::tt::telemetry {

/// Accumulates telemetry data and writes it out as JSON.
///
/// Usage:
///   TelemetryWriter writer(graphMeta, "model.mlir", 0);
///   writer.addSnapshot(std::move(snapshot));
///   writer.writeJSON("/tmp/telemetry/output.json");
class TelemetryWriter {
  TelemetryFile file;

public:
  explicit TelemetryWriter(GraphMeta graph, std::string sourcePath = "",
                           int graphIndex = 0);

  /// Add a snapshot to this graph.
  void addSnapshot(SnapshotData snapshot);

  /// Override the graph's model name (e.g. once resolved from the IR location).
  void setModelName(llvm::StringRef modelName);

  /// Write the accumulated TelemetryFile to JSON at the given path.
  mlir::LogicalResult writeJSON(llvm::StringRef outputPath);
};

} // namespace mlir::tt::telemetry

#endif // TOOLS_GRAPH_TELEMETRY_TELEMETRYWRITER_H
