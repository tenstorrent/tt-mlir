// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::d2m {

// TODO: move to common util (used in generic region to funcs too)
static std::optional<unsigned> getCapturedOperandIndex(GenericOp op,
                                                       Value operand) {
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (opOperand.get() == operand) {
      return opOperand.getOperandNumber();
    }
  }
  return std::nullopt;
}

unsigned getCBOperandIdx(GenericOp generic, Value cbGenericOperand) {
  assert(mlir::isa<ttcore::CBLayoutAttr>(
             mlir::cast<MemRefType>(cbGenericOperand.getType()).getLayout()) &&
         "expected cb layout");
  auto operandIndex = getCapturedOperandIndex(generic, cbGenericOperand);
  assert(operandIndex && "expected captured operand");
  return *operandIndex;
}

memref::AllocOp findAllocOp(Value value) {
  while (value) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) {
      return nullptr;
    }

    if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(definingOp)) {
      return allocOp;
    }

    // Only trace through view-like operations (e.g., memref.collapse_shape,
    // memref.subview, memref.cast).
    if (mlir::isa<mlir::ViewLikeOpInterface>(definingOp)) {
      value = definingOp->getOperand(0);
      continue;
    }

    return nullptr;
  }
  return nullptr;
}

} // namespace mlir::tt::d2m
