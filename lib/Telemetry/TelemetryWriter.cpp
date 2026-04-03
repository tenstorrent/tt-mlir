// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Telemetry/TelemetryWriter.h"
#include "ttmlir/Telemetry/GraphTypes.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <random>

using namespace mlir::tt::telemetry;

//===----------------------------------------------------------------------===//
// UUID generation
//===----------------------------------------------------------------------===//

std::string mlir::tt::telemetry::generateUUID() {
  static thread_local std::mt19937_64 gen(std::random_device{}());
  static thread_local std::uniform_int_distribution<uint64_t> dist;

  uint64_t high = dist(gen);
  uint64_t low = dist(gen);

  // Set version to 4 (bits 12-15 of high).
  high = (high & ~(0xFULL << 12)) | (0x4ULL << 12);
  // Set variant to RFC 4122 (bits 62-63 of low).
  low = (low & ~(0x3ULL << 62)) | (0x2ULL << 62);

  // Format: 8-4-4-4-12
  char buf[37];
  snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
           static_cast<uint32_t>(high >> 32),
           static_cast<uint16_t>((high >> 16) & 0xFFFF),
           static_cast<uint16_t>(high & 0xFFFF),
           static_cast<uint16_t>(low >> 48),
           static_cast<unsigned long long>(low & 0xFFFFFFFFFFFFULL));
  return std::string(buf);
}

//===----------------------------------------------------------------------===//
// JSON serialization helpers
//===----------------------------------------------------------------------===//

namespace {

/// Convert a StringMap to a JSON object.
llvm::json::Object stringMapToJSON(const llvm::StringMap<std::string> &map) {
  llvm::json::Object obj;
  for (const auto &entry : map) {
    obj[entry.getKey()] = entry.getValue();
  }
  return obj;
}

/// Convert a SmallVector<int64_t> to a JSON array.
llvm::json::Array shapeToJSON(const llvm::SmallVector<int64_t> &shape) {
  llvm::json::Array arr;
  for (int64_t dim : shape) {
    arr.push_back(dim);
  }
  return arr;
}

llvm::json::Object opToJSON(const OpData &op) {
  llvm::json::Object obj;
  obj["id"] = op.id;
  obj["parentId"] = op.parentId;
  obj["regionIdx"] = op.regionIdx;
  obj["blockIdx"] = op.blockIdx;
  obj["order"] = op.order;
  obj["opName"] = op.opName;
  obj["dialect"] = op.dialect;
  obj["location"] = op.location;
  obj["mlirLine"] = op.mlirLine;
  obj["symName"] = op.symName;
  obj["isFunc"] = op.isFunc;
  obj["numArgs"] = op.numArgs;
  obj["numResults"] = op.numResults;
  obj["isTerminator"] = op.isTerminator;
  obj["attrs"] = stringMapToJSON(op.attrs);
  return obj;
}

llvm::json::Object blockArgToJSON(const BlockArgData &ba) {
  llvm::json::Object obj;
  obj["id"] = ba.id;
  obj["parentId"] = ba.parentId;
  obj["regionIdx"] = ba.regionIdx;
  obj["blockIdx"] = ba.blockIdx;
  obj["argIdx"] = ba.argIdx;
  return obj;
}

llvm::json::Object tensorToJSON(const TensorData &tensor) {
  llvm::json::Object obj;
  obj["id"] = tensor.id;
  obj["producerId"] = tensor.producerId;
  obj["producerType"] = tensor.producerType;
  obj["resultIdx"] = tensor.resultIdx;
  obj["shape"] = shapeToJSON(tensor.shape);
  obj["dtype"] = tensor.dtype;
  obj["rank"] = tensor.rank;
  obj["attrs"] = stringMapToJSON(tensor.attrs);
  return obj;
}

llvm::json::Object edgeToJSON(const EdgeData &edge) {
  llvm::json::Object obj;
  obj["tensorId"] = edge.tensorId;
  obj["consumerId"] = edge.consumerId;
  obj["operandIdx"] = edge.operandIdx;
  return obj;
}

llvm::json::Object callToJSON(const CallEdge &call) {
  llvm::json::Object obj;
  obj["callerId"] = call.callerId;
  obj["calleeId"] = call.calleeId;
  return obj;
}

llvm::json::Object refToJSON(const RefEdge &ref) {
  llvm::json::Object obj;
  obj["userId"] = ref.userId;
  obj["targetId"] = ref.targetId;
  obj["attrName"] = ref.attrName;
  return obj;
}

llvm::json::Object snapshotToJSON(const SnapshotData &snap) {
  llvm::json::Object obj;
  obj["snapshotId"] = snap.snapshotId;
  obj["tag"] = snap.tag;
  obj["passName"] = snap.passName;
  obj["snapshotIndex"] = snap.snapshotIndex;
  obj["timestampUs"] = snap.timestampUs;
  obj["mlirPath"] = snap.mlirPath;

  llvm::json::Array ops;
  for (const auto &o : snap.ops) {
    ops.push_back(opToJSON(o));
  }
  obj["ops"] = std::move(ops);

  llvm::json::Array blockArgs;
  for (const auto &ba : snap.blockArgs) {
    blockArgs.push_back(blockArgToJSON(ba));
  }
  obj["blockArgs"] = std::move(blockArgs);

  llvm::json::Array tensors;
  for (const auto &t : snap.tensors) {
    tensors.push_back(tensorToJSON(t));
  }
  obj["tensors"] = std::move(tensors);

  llvm::json::Array edges;
  for (const auto &e : snap.edges) {
    edges.push_back(edgeToJSON(e));
  }
  obj["edges"] = std::move(edges);

  llvm::json::Array calls;
  for (const auto &c : snap.calls) {
    calls.push_back(callToJSON(c));
  }
  obj["calls"] = std::move(calls);

  llvm::json::Array refs;
  for (const auto &r : snap.refs) {
    refs.push_back(refToJSON(r));
  }
  obj["refs"] = std::move(refs);

  return obj;
}

llvm::json::Object graphToJSON(const GraphMeta &graph) {
  llvm::json::Object obj;
  obj["graphId"] = graph.graphId;
  obj["modelName"] = graph.modelName;
  obj["testName"] = graph.testName;
  obj["createdAt"] = graph.createdAt;
  return obj;
}

} // namespace

//===----------------------------------------------------------------------===//
// TelemetryWriter
//===----------------------------------------------------------------------===//

TelemetryWriter::TelemetryWriter(GraphMeta graph, std::string sourcePath,
                                 int graphIndex) {
  file.graph = std::move(graph);
  file.sourcePath = std::move(sourcePath);
  file.graphIndex = graphIndex;
}

void TelemetryWriter::addSnapshot(SnapshotData snapshot) {
  file.snapshots.push_back(std::move(snapshot));
}

void TelemetryWriter::setModelName(llvm::StringRef modelName) {
  file.graph.modelName = modelName.str();
}

mlir::LogicalResult TelemetryWriter::writeJSON(llvm::StringRef outputPath) {
  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec);
  if (ec) {
    llvm::errs() << "Failed to open output file '" << outputPath
                 << "': " << ec.message() << "\n";
    return mlir::failure();
  }

  llvm::json::Object root;
  root["version"] = file.version;
  root["graph"] = graphToJSON(file.graph);
  root["sourcePath"] = file.sourcePath;
  root["graphIndex"] = file.graphIndex;

  llvm::json::Array snapshots;
  for (const auto &s : file.snapshots) {
    snapshots.push_back(snapshotToJSON(s));
  }
  root["snapshots"] = std::move(snapshots);

  llvm::json::Value jsonValue(std::move(root));
  llvm::json::OStream jsonOS(os, /*IndentSize=*/2);
  jsonOS.value(jsonValue);
  os << "\n";

  return mlir::success();
}
