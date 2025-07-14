// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_EMITPY_IR_EMITPYTYPES_H
#define TTMLIR_DIALECT_EMITPY_IR_EMITPYTYPES_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOpsTypes.h.inc"

#endif // TTMLIR_DIALECT_EMITPY_IR_EMITPYTYPES_H
