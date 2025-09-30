// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_CBPRODUCERCONSUMER_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_CBPRODUCERCONSUMER_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {

// Default here is mostly for constructed testing purposes. i.e. Hand written
// LIT tests that don't have d2m.yield or d2m.await ops.
enum ThreadCBOrientation {
  Producer,
  Consumer,
  ProducerConsumer,
  Default,
};
struct CBProducerConsumer {
  CBProducerConsumer(Operation *op);

  ThreadCBOrientation get(Value cb) const {
    auto match = threadCBOrientationMap.find(cb);
    if (match == threadCBOrientationMap.end()) {
      return Default;
    }
    return match->second;
  }

  llvm::DenseMap<Value, ThreadCBOrientation> threadCBOrientationMap;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_CBPRODUCERCONSUMER_H
