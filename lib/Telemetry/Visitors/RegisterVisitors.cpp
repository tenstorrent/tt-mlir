// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Telemetry/TelemetryVisitor.h"

#include <memory>

namespace mlir::tt::telemetry {

// Factory functions defined in their respective translation units.
std::unique_ptr<TelemetryVisitor> createTTNNLayoutVisitor();
std::unique_ptr<TelemetryVisitor> createMemoryConfigVisitor();

/// Register all tt-mlir specific telemetry visitors.
///
/// 1. TTNNLayoutVisitor -- decompose tensor encodings into the attr bag.
/// 2. MemoryConfigVisitor -- decompose MemoryConfigAttr.
///
/// No ops are filtered out: const-eval functions are retained so that call
/// edges (func.call, ttcore.load_cached) resolve to real callee nodes.
void registerTTMLIRVisitors(TelemetryVisitorRegistry &registry) {
  registry.add(createTTNNLayoutVisitor());
  registry.add(createMemoryConfigVisitor());
}

} // namespace mlir::tt::telemetry
