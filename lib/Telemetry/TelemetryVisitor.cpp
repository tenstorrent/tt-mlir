// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Telemetry/TelemetryVisitor.h"
#include "ttmlir/Telemetry/GraphTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tt::telemetry;

//===----------------------------------------------------------------------===//
// TelemetryVisitor base implementation
//===----------------------------------------------------------------------===//

void TelemetryVisitor::visitValue(Value value, TensorData &data) {
  // Guard: only extract base fields once (multiple visitors chain visitValue).
  if (!data.shape.empty()) {
    return;
  }
  Type type = value.getType();
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    if (shapedType.hasRank()) {
      data.rank = shapedType.getRank();
      for (int64_t dim : shapedType.getShape()) {
        data.shape.push_back(dim);
      }
    }
    Type elementType = shapedType.getElementType();
    std::string dtypeStr;
    llvm::raw_string_ostream os(dtypeStr);
    elementType.print(os);
    data.dtype = dtypeStr;
  }
}

//===----------------------------------------------------------------------===//
// TelemetryVisitorRegistry
//===----------------------------------------------------------------------===//

void TelemetryVisitorRegistry::add(std::unique_ptr<TelemetryVisitor> v) {
  visitors.push_back(std::move(v));
}

bool TelemetryVisitorRegistry::visitOp(Operation *op, OpData &data) {
  for (auto &visitor : visitors) {
    if (!visitor->visitOp(op, data)) {
      return false;
    }
  }
  return true;
}

void TelemetryVisitorRegistry::visitValue(Value value, TensorData &data) {
  for (auto &visitor : visitors) {
    visitor->visitValue(value, data);
  }
}

bool TelemetryVisitorRegistry::visitAttr(StringRef name, Attribute attr,
                                         llvm::StringMap<std::string> &attrs) {
  for (auto &visitor : visitors) {
    if (!visitor->visitAttr(name, attr, attrs)) {
      return false;
    }
  }
  return true;
}
