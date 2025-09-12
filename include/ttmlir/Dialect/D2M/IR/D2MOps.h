// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MOPS_H
#define TTMLIR_DIALECT_D2M_IR_D2MOPS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include "ttmlir/Dialect/D2M/IR/D2MOpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypeDefs.h.inc"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOps.h.inc"

#endif // TTMLIR_DIALECT_D2M_IR_D2MOPS_H
