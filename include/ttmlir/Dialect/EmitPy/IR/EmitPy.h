// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_EMITPY_IR_EMITPY_H
#define TTMLIR_DIALECT_EMITPY_IR_EMITPY_H

#include "mlir/IR/Dialect.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOpsDialect.h.inc"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include <variant>

namespace mlir {
namespace tt {
namespace emitpy {

// Either a literal string, or an placeholder for the fmtArgs.
struct Placeholder {};
using ReplacementItem = std::variant<StringRef, Placeholder>;

void buildTerminatedBody(OpBuilder &builder, Location loc);

} // namespace emitpy
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_EMITPY_IR_EMITPY_H
