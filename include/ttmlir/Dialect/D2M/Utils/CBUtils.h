// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace memref {
class AllocOp;
} // namespace memref
} // namespace mlir

namespace mlir::tt::d2m {

class GenericOp;

/// Cache for CB values: maps (GenericOp pointer, operand index) -> CB value.
using CBCache = DenseMap<std::pair<Operation *, unsigned>, Value>;

/// Track the next available CB port number per generic op.
using PortCounter = DenseMap<Operation *, unsigned>;

/// Find the next available port number for a generic by scanning existing
/// d2m.get_cb ops in the region.
unsigned getNextAvailablePort(Region &region, PortCounter &portCounters,
                              Operation *genericOp);

/// Compute the CBType and create a d2m.get_cb op for a given operand of a
/// generic op.  Results are cached to ensure each (generic, operand) pair gets
/// exactly one CB value.  Port numbers are assigned sequentially and do NOT
/// correspond to operand indices.
Value getOrCreateCB(GenericOp generic, Region &region, unsigned operandIndex,
                    RewriterBase &rewriter, CBCache &cache,
                    PortCounter &portCounters);

/// Find the CB value that corresponds to a memref operand in a generic op.
/// Creates CB values on demand via d2m.get_cb.
Value findAssociatedCB(Operation *op, Value memrefOperand,
                       RewriterBase &rewriter, CBCache &cache,
                       PortCounter &portCounters);

/// Find the memref.alloc operation that produces a given value, potentially
/// through a chain of view-like operations. Returns the alloc op if found,
/// null otherwise.
memref::AllocOp findAllocOp(Value value);

} // namespace mlir::tt::d2m

#endif
