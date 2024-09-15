// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_EDGE_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_EDGE_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt::ttir {
struct Edge {
  Operation *producerOp = nullptr;
  Operation *consumerOp = nullptr;
  size_t operandIndex = 0;

  Edge(Operation *producerOp, Operation *consumerOp, size_t operandIndex)
      : producerOp(producerOp), consumerOp(consumerOp),
        operandIndex(operandIndex) {}

  bool operator==(const Edge &other) const {
    return producerOp == other.producerOp && consumerOp == other.consumerOp &&
           operandIndex == other.operandIndex;
  }
};

} // namespace mlir::tt::ttir

namespace std {
template <> struct hash<mlir::tt::ttir::Edge> {
  size_t operator()(const mlir::tt::ttir::Edge &edge) const noexcept {
    llvm::hash_code code = llvm::hash_value(edge.operandIndex);
    code = llvm::hash_combine(code, edge.producerOp);
    code = llvm::hash_combine(code, edge.consumerOp);
    return code;
  }
};
} // namespace std

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_EDGE_H
