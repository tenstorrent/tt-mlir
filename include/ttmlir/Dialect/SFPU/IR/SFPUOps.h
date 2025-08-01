// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPU_IR_SFPUOPS_H
#define TTMLIR_DIALECT_SFPU_IR_SFPUOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ttmlir/Dialect/SFPU/IR/SFPUOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/SFPU/IR/SFPUOps.h.inc"

#endif