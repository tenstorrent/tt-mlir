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

Value getCB(Operation *opUsingCB, Value cbGenericOperand,
            RewriterBase &rewriter) {
  GenericOp generic = opUsingCB->getParentOfType<GenericOp>();
  auto genericRegion =
      ttmlir::utils::getRegionWithParentOfType<GenericOp>(opUsingCB);
  assert(mlir::isa<ttcore::CBLayoutAttr>(
             mlir::cast<MemRefType>(cbGenericOperand.getType()).getLayout()) &&
         "expected cb layout");
  auto operandIndex = getCapturedOperandIndex(generic, cbGenericOperand);
  assert(operandIndex && "expected captured operand");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&genericRegion->front());
  auto cbType = CBType::get(generic.getContext(),
                            mlir::cast<ShapedType>(cbGenericOperand.getType()));
  auto getCBOp =
      rewriter.create<GetCBOp>(generic.getLoc(), cbType, *operandIndex);
  // llvm::errs() << "getting cb for operand: " << *operandIndex
  //              << " in generic position: " << generic.getLoc()
  //              << " with result: " << getCBOp.getResult() << "\n";
  return getCBOp.getResult();
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
