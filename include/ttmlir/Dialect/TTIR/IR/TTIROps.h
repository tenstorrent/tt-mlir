// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIROPS_H
#define TTMLIR_DIALECT_TTIR_IR_TTIROPS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.h.inc"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h.inc"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIRTileOps.h.inc"

#endif // TTMLIR_DIALECT_TTIR_IR_TTIROPS_H
