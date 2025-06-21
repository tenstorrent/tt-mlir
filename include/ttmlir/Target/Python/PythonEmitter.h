// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_PYTHON_PYTHONEMITTER_H
#define TTMLIR_TARGET_PYTHON_PYTHONEMITTER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;
namespace tt {
namespace emitpy {

/// Translates the given operation to Python code. The operation or operations
/// in the region of 'op' need almost all be in EmitPy dialect.
LogicalResult translateToPython(Operation *op, raw_ostream &os);
} // namespace emitpy
} // namespace tt
} // namespace mlir

#endif // TTMLIR_TARGET_PYTHON_PYTHONEMITTER_H
