// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Telemetry/TelemetryVisitor.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::tt::telemetry;

namespace {

/// Format a shape array as a string like "32x64".
std::string printShape(llvm::ArrayRef<int64_t> shape) {
  std::string result;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      result += "x";
    }
    result += std::to_string(shape[i]);
  }
  return result;
}

/// Decompose TTNNLayoutAttr encoding on tensor types into queryable fields.
///
/// Extracts from the layout attribute:
/// - buffer_type: DRAM, L1, SystemMemory, etc.
/// - tensor_memory_layout: interleaved, height_sharded, etc.
/// - grid: the core grid shape
/// - shard_shape: the per-shard shape from the memref
/// - is_tiled: whether the tensor uses tile layout
class TTNNLayoutVisitor : public TelemetryVisitor {
public:
  void visitValue(Value value, TensorData &data) override {
    // Call base to extract shape/dtype/rank.
    TelemetryVisitor::visitValue(value, data);

    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return;
    }

    auto layout =
        dyn_cast_or_null<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
    if (!layout) {
      return;
    }

    // Buffer type derived from memref memory space.
    data.attrs["buffer_type"] =
        std::string(stringifyBufferType(layout.getBufferType()));

    // Tensor memory layout (optional).
    if (auto memLayout = layout.getMemLayout()) {
      data.attrs["tensor_memory_layout"] =
          std::string(stringifyTensorMemoryLayout(memLayout.getValue()));
    }

    // Grid shape.
    data.attrs["grid"] = printShape(layout.getGridShape());

    // Shard shape from memref.
    MemRefType memref = layout.getMemref();
    data.attrs["shard_shape"] = printShape(memref.getShape());

    // Whether the tensor is tiled.
    data.attrs["is_tiled"] = layout.isTiled() ? "true" : "false";
  }
};

} // namespace

namespace mlir::tt::telemetry {
std::unique_ptr<TelemetryVisitor> createTTNNLayoutVisitor() {
  return std::make_unique<TTNNLayoutVisitor>();
}
} // namespace mlir::tt::telemetry
