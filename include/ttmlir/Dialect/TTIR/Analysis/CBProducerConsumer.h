// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_CBPRODUCERCONSUMER_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_CBPRODUCERCONSUMER_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::ttir {

enum ThreadCBOrientation {
  Producer,
  Consumer,
  ProducerConsumer,
};
struct CBProducerConsumer {
  CBProducerConsumer(Operation *op);

  ThreadCBOrientation get(Value cb) const {
    auto match = threadCBOrientationMap.find(cb);
    assert(match != threadCBOrientationMap.end() &&
           "CB orientation mapping not found.");
    return match->second;
  }

  llvm::DenseMap<Value, ThreadCBOrientation> threadCBOrientationMap;
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_CBPRODUCERCONSUMER_H
