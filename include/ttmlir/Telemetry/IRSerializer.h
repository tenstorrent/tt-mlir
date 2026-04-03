// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_GRAPH_TELEMETRY_IRSERIALIZER_H
#define TOOLS_GRAPH_TELEMETRY_IRSERIALIZER_H

#include "ttmlir/Telemetry/GraphTypes.h"
#include "ttmlir/Telemetry/TelemetryVisitor.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"

#include <string>

namespace mlir::tt::telemetry {

/// Recursively walk an Operation* (typically a ModuleOp) and produce a
/// SnapshotData. The structure is a uniform reflection of MLIR: every op --
/// modules and functions included -- becomes one OpData, nested via parentId
/// and region/block/order indices. The registry's visitors control what gets
/// serialized and how (and may prune an op together with its subtree).
///
/// For each op: emit OpData (with generic interface-derived fields), a tensor
/// per result, an edge per resolvable operand, block-arg tensors for each
/// nested block, and -- after the walk -- call/symbol-reference edges.
///
/// `lineMap`, if given, is an AsmState location map captured while printing the
/// same root op; each op's printed-MLIR line is recorded into OpData.mlirLine.
SnapshotData serialize(Operation *rootOp, TelemetryVisitorRegistry &registry,
                       const std::string &tag, const std::string &passName,
                       int snapshotIndex,
                       const mlir::AsmState::LocationMap *lineMap = nullptr);

} // namespace mlir::tt::telemetry

#endif // TOOLS_GRAPH_TELEMETRY_IRSERIALIZER_H
