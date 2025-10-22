// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIROPS_H
#define TTMLIR_DIALECT_TTIR_IR_TTIROPS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTIR/IR/Utils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreTraits.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::tt::ttir {

inline void getDpsEffects(
    DestinationStyleOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (!llvm::isa<MemRefType>(operand.get().getType())) {
      continue;
    }
    if (op.isDpsInput(&operand)) {
      effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    } else {
      effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    }
  }
}

} // namespace mlir::tt::ttir

#include "ttmlir/Dialect/TTIR/IR/TTIROpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.h.inc"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h.inc"

#endif // TTMLIR_DIALECT_TTIR_IR_TTIROPS_H
