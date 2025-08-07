// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIOPS_H
#define TTMLIR_DIALECT_SFPI_IR_SFPIOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ttmlir/Dialect/SFPI/IR/SFPITraits.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsEnums.h.inc"
#pragma clang diagnostic pop

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsAttrs.h.inc"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/SFPI/IR/SFPIOps.h.inc"

#endif
