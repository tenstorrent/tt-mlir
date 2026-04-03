// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Telemetry/TelemetryVisitor.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::tt::telemetry;

namespace {

/// Decompose MemoryConfigAttr into queryable sub-fields.
///
/// When a MemoryConfigAttr is encountered on an op, instead of storing the
/// raw printed representation, this visitor decomposes it into:
/// - <name>.buffer_type: the buffer type (dram, l1, system_memory, etc.)
/// - <name>.tensor_memory_layout: the memory layout (interleaved, sharded,
/// etc.)
///
/// Returns false to suppress the default stringification of this attr.
class MemoryConfigVisitor : public TelemetryVisitor {
public:
  bool visitAttr(StringRef name, Attribute attr,
                 llvm::StringMap<std::string> &attrs) override {
    if (auto mc = dyn_cast<ttnn::MemoryConfigAttr>(attr)) {
      std::string prefix = name.str() + ".";
      attrs[prefix + "buffer_type"] =
          std::string(stringifyBufferType(mc.getBufferType().getValue()));
      if (auto tml = mc.getTensorMemoryLayout()) {
        attrs[prefix + "tensor_memory_layout"] =
            std::string(stringifyTensorMemoryLayout(tml.getValue()));
      }
      return false; // handled, don't also stringify the raw attr
    }
    return true; // default: keep as string
  }
};

} // namespace

namespace mlir::tt::telemetry {
std::unique_ptr<TelemetryVisitor> createMemoryConfigVisitor() {
  return std::make_unique<MemoryConfigVisitor>();
}
} // namespace mlir::tt::telemetry
