// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::memref {
class AllocOp;
} // namespace mlir::memref

namespace mlir::tt::d2m {

class GenericOp;

/// Compute the CBType and create a d2m.get_cb op for a given operand of a
/// generic op.  Results are cached to ensure each (generic, operand) pair gets
/// exactly one CB value.  Port numbers are assigned sequentially and do NOT
/// correspond to operand indices.
Value getCB(Operation *opUsingCB, Value cbGenericOperand,
            RewriterBase &rewriter);

/// Find the memref.alloc operation that produces a given value, potentially
/// through a chain of view-like operations. Returns the alloc op if found,
/// null otherwise.
memref::AllocOp findAllocOp(Value value);

} // namespace mlir::tt::d2m

#endif
