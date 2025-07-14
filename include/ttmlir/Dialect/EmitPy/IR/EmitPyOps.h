// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_EMITPY_IR_EMITPYOPS_H
#define TTMLIR_DIALECT_EMITPY_IR_EMITPYOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h.inc"

#endif
