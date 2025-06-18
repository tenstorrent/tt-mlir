// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_EDGE_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_EDGE_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::ttnn {
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

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Edge &edge) {
  auto producerName =
      edge.producerOp ? edge.producerOp->getName().getStringRef() : "nullptr";
  auto consumerName =
      edge.consumerOp ? edge.consumerOp->getName().getStringRef() : "nullptr";
  os << "Edge(" << producerName << " (" << edge.producerOp << " @ "
     << (edge.producerOp ? edge.producerOp->getLoc()
                         : mlir::UnknownLoc::get(edge.consumerOp->getContext()))
     << "), " << consumerName << " (" << edge.consumerOp << " @ "
     << (edge.consumerOp ? edge.consumerOp->getLoc()
                         : mlir::UnknownLoc::get(edge.consumerOp->getContext()))
     << "), " << edge.operandIndex << ")";
  return os;
}

} // namespace mlir::tt::ttnn

namespace std {
template <>
struct hash<mlir::tt::ttnn::Edge> {
  size_t operator()(const mlir::tt::ttnn::Edge &edge) const noexcept {
    llvm::hash_code code = llvm::hash_value(edge.operandIndex);
    code = llvm::hash_combine(code, edge.producerOp);
    code = llvm::hash_combine(code, edge.consumerOp);
    return code;
  }
};
} // namespace std

namespace llvm {
template <>
struct DenseMapInfo<mlir::tt::ttnn::Edge> {
  static constexpr size_t EmptyIndex = std::numeric_limits<size_t>::max();
  static constexpr size_t TombstoneIndex = EmptyIndex - 1;

  static mlir::tt::ttnn::Edge getEmptyKey() {
    return mlir::tt::ttnn::Edge(nullptr, nullptr, EmptyIndex);
  }

  static mlir::tt::ttnn::Edge getTombstoneKey() {
    return mlir::tt::ttnn::Edge(nullptr, nullptr, TombstoneIndex);
  }

  static bool isEqual(const mlir::tt::ttnn::Edge &lhs,
                      const mlir::tt::ttnn::Edge &rhs) {
    return lhs == rhs;
  }

  static unsigned getHashValue(const mlir::tt::ttnn::Edge &val) {
    return hash_combine(
        hash_combine(hash_value(val.operandIndex), hash_value(val.producerOp)),
        hash_value(val.consumerOp));
  }

  // Verify at compile time that empty and tombstone keys are different
  static_assert(EmptyIndex != TombstoneIndex,
                "Empty and tombstone keys must be different");
};
} // namespace llvm

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_EDGE_H
