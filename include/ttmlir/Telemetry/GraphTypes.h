// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_GRAPH_TELEMETRY_GRAPHTYPES_H
#define TOOLS_GRAPH_TELEMETRY_GRAPHTYPES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::tt::telemetry {

/// Generate a UUID v4 string (random-based).
std::string generateUUID();

/// Compilation-level metadata. A graph is one compilation of one model.
/// `testName` is compilation context -- the test/invocation that triggered this
/// compile, which only the frontend knows at emit time. CI and GitHub
/// provenance (run, branch, commit, workflow) is run-level and stamped later by
/// the push tool, not here.
struct GraphMeta {
  std::string graphId;
  std::string modelName;
  std::string testName;
  int64_t createdAt = 0; // unix millis
};

/// A single operation. Modules and functions are ops too -- the schema is a
/// uniform reflection of MLIR's op/region/block structure, with no node type
/// dedicated to any particular op. Generic interface-derived fields (symName,
/// isFunc, ...) come from upstream MLIR interfaces, never from dialect names.
struct OpData {
  std::string id;       // UUID
  std::string parentId; // UUID of enclosing op (empty for the root)
  int regionIdx = 0;    // position of this op within its parent
  int blockIdx = 0;
  int order = 0; // index within the parent block (program order)

  std::string opName;
  std::string dialect;
  std::string location;
  // 1-based line of this op in the snapshot's printed MLIR (0 if unknown), so a
  // node can be deep-linked to its line in the dashboard's MLIR view.
  int mlirLine = 0;

  // Generic, interface-derived fields (no dialect-specific knowledge).
  std::string symName; // SymbolOpInterface symbol name, if any
  bool isFunc = false; // implements FunctionOpInterface
  int numArgs = 0;     // function argument count
  int numResults = 0;  // function result count
  bool isTerminator = false;

  // Open-ended attribute bag. Every attribute is captured here (dense elements
  // are elided to a summary). Visitors add decomposed keys to the same bag.
  llvm::StringMap<std::string> attrs;
};

struct BlockArgData {
  std::string id; // UUID
  std::string parentId;
  int regionIdx = 0;
  int blockIdx = 0;
  int argIdx = 0;
};

/// A first-class SSA value (op result or block argument). Shape/dtype/rank are
/// typed top-level fields; everything else (layout, encoding, ...) lands in the
/// queryable attribute bag.
struct TensorData {
  std::string id; // UUID
  std::string producerId;
  std::string producerType; // "op" or "block_arg"
  int resultIdx = 0;
  llvm::SmallVector<int64_t> shape;
  std::string dtype;
  int rank = 0;
  llvm::StringMap<std::string> attrs;
};

struct EdgeData {
  std::string tensorId;
  std::string consumerId;
  int operandIdx = 0;
};

/// Call-site -> callee op edge (resolved via the symbol table).
struct CallEdge {
  std::string callerId;
  std::string calleeId;
};

/// Symbol-using op -> referenced symbol op edge (e.g. an attr holding @sym).
struct RefEdge {
  std::string userId;
  std::string targetId;
  std::string attrName;
};

struct SnapshotData {
  std::string snapshotId; // UUID
  std::string tag;
  std::string passName;
  int snapshotIndex = 0;
  int64_t timestampUs = 0;
  // Path to the snapshot's printed MLIR text, relative to the telemetry output
  // directory. Empty if the dump could not be written.
  std::string mlirPath;
  std::vector<OpData> ops;
  std::vector<BlockArgData> blockArgs;
  std::vector<TensorData> tensors;
  std::vector<EdgeData> edges;
  std::vector<CallEdge> calls;
  std::vector<RefEdge> refs;
};

struct TelemetryFile {
  int version = 2;
  GraphMeta graph;
  std::string sourcePath;
  int graphIndex = 0;
  std::vector<SnapshotData> snapshots;
};

} // namespace mlir::tt::telemetry

#endif // TOOLS_GRAPH_TELEMETRY_GRAPHTYPES_H
