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

struct CBProducerConsumer {
  CBProducerConsumer(Operation *op);

  ttir::ThreadCBMapping get(mlir::StringRef threadRef, Value cb) const {
    auto match = threadCBMappingMap.find(std::make_pair(threadRef, cb));
    assert(match != threadCBMappingMap.end() &&
           "CB producer consumer mapping not found.");
    return match->second;
  }

  llvm::DenseMap<std::pair<mlir::StringRef, Value>, ttir::ThreadCBMapping>
      threadCBMappingMap;
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_CBPRODUCERCONSUMER_H
