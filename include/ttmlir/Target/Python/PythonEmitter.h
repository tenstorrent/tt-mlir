// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_PYTHON_PYTHONEMITTER_H
#define TTMLIR_TARGET_PYTHON_PYTHONEMITTER_H

#include "mlir/Support/LLVM.h"

#include <optional>
#include <string>

namespace mlir {
class Operation;
namespace tt {
namespace emitpy {

/// Translates the given operation to Python code. The operation or operations
/// in the region of 'op' all need to be in EmitPy dialect, except for those
/// that are explicitly labeled as (dynamically) legal ops when converting to
/// EmitPy dialect. When `fileId` is provided, only the ops of the
/// `emitpy.file` with the matching id are emitted. Otherwise, all ops are
/// emitted.
///
LogicalResult
translateToPython(Operation *op, raw_ostream &os,
                  std::optional<std::string> fileId = std::nullopt);

} // namespace emitpy
} // namespace tt
} // namespace mlir

#endif // TTMLIR_TARGET_PYTHON_PYTHONEMITTER_H
