// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTPROPAGATIONTYPES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTPROPAGATIONTYPES_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>

namespace mlir::tt::ttnn {

/// An input candidate for one operand of an op.
struct InputCandidate {
  TTNNLayoutAttr layout;
  size_t producerCandidateIndex = 0;
  bool isReshard = false;
};

/// Info bundle for an in-place (zero-result) op.
struct InplaceOpInfo {
  Operation *op;
  /// Per tensor operand: (operandIdx, currentLayout, producerOp).
  /// producerOp may be null for func args.
  struct OperandInfo {
    size_t operandIdx;
    TTNNLayoutAttr layout; // from IR encoding
    Operation *producerOp; // may be null
  };
  llvm::SmallVector<OperandInfo> operands;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTPROPAGATIONTYPES_H
