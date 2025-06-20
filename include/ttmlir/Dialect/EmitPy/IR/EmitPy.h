// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_EMITPY_IR_EMITPY_H
#define TTMLIR_DIALECT_EMITPY_IR_EMITPY_H

#include "mlir/IR/Dialect.h"

#include <variant>

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOpsDialect.h.inc"

namespace mlir {
namespace tt {
namespace emitpy {

// Either a literal string, or an placeholder for the fmtArgs.
struct Placeholder {};
using ReplacementItem = std::variant<StringRef, Placeholder>;

} // namespace emitpy
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_EMITPY_IR_EMITPY_H
