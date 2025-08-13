// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_EMITPY_IR_EMITPYOPS_H
#define TTMLIR_DIALECT_EMITPY_IR_EMITPYOPS_H

#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyInterfaces.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"

#include "mlir/IR/BuiltinOps.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h.inc"

#endif // TTMLIR_DIALECT_EMITPY_IR_EMITPYOPS_H
