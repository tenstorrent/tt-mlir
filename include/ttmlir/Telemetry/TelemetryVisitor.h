// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_GRAPH_TELEMETRY_TELEMETRYVISITOR_H
#define TOOLS_GRAPH_TELEMETRY_TELEMETRYVISITOR_H

#include "ttmlir/Telemetry/GraphTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir::tt::telemetry {

/// Base visitor class for telemetry serialization.
///
/// Concrete visitors override these methods to customize what gets serialized
/// and how. The default implementations provide baseline behavior:
/// - visitOp: returns true (serialize everything)
/// - visitValue: extracts shape/dtype/rank from ShapedType
/// - visitAttr: returns true (stringify all attrs)
class TelemetryVisitor {
public:
  virtual ~TelemetryVisitor() = default;

  /// Per-op: populate attrs. Return false to skip this op.
  virtual bool visitOp(Operation *op, OpData &data) { return true; }

  /// Per-value (SSA result or block arg): extract encoding from type.
  /// Base impl extracts shape/dtype/rank from ShapedType.
  virtual void visitValue(Value value, TensorData &data);

  /// Per-attribute on an op: return false to suppress default stringification.
  virtual bool visitAttr(StringRef name, Attribute attr,
                         llvm::StringMap<std::string> &attrs) {
    return true;
  }
};

/// Registry that chains multiple visitors in registration order.
///
/// Chaining semantics:
/// - visitOp: first visitor returning false short-circuits (op skipped).
/// - visitValue: all visitors run in order, each can augment data.
/// - visitAttr: first visitor returning false stops the chain (attr handled).
class TelemetryVisitorRegistry {
  llvm::SmallVector<std::unique_ptr<TelemetryVisitor>> visitors;

public:
  /// Add a visitor to the registry. Visitors are called in registration order.
  void add(std::unique_ptr<TelemetryVisitor> v);

  /// Chain visitOp across all visitors. Returns false if any visitor returns
  /// false (short-circuits).
  bool visitOp(Operation *op, OpData &data);

  /// Chain visitValue across all visitors. All visitors run in order.
  void visitValue(Value value, TensorData &data);

  /// Chain visitAttr across all visitors. First visitor returning false
  /// stops the chain.
  bool visitAttr(StringRef name, Attribute attr,
                 llvm::StringMap<std::string> &attrs);
};

} // namespace mlir::tt::telemetry

#endif // TOOLS_GRAPH_TELEMETRY_TELEMETRYVISITOR_H
