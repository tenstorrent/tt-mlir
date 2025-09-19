// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/AssociatedDMAWaits.h"
namespace mlir::tt::d2m {

static SmallVector<d2m::DMAWaitOp> findAssociatedDMAWaits(DMAOpInterface op) {
  SmallVector<OpOperand *> uses(llvm::map_range(
      op.getResult().getUses(), [](OpOperand &use) { return &use; }));
  SmallVector<d2m::DMAWaitOp> dmaWaits;
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *user = use->getOwner();
    if (auto dmaWaitOp = mlir::dyn_cast<d2m::DMAWaitOp>(user)) {
      dmaWaits.push_back(dmaWaitOp);
    } else if (user->hasTrait<OpTrait::IsTerminator>()) {
      // If this op is a terminator, we need to look through its parent's
      // uses
      unsigned resultIdx = use->getOperandNumber();
      auto parentResult = user->getParentOp()->getOpResult(resultIdx);
      auto range = llvm::map_range(parentResult.getUses(),
                                   [](OpOperand &use) { return &use; });
      assert(!range.empty());
      uses.append(range.begin(), range.end());
    } else {
      llvm_unreachable("Unexpected user of DMAOp");
    }
  }
  assert(!dmaWaits.empty());
  return dmaWaits;
}

AssociatedDMAWaits::AssociatedDMAWaits(Operation *op) {
  op->walk([&](DMAOpInterface dmaOp) {
    dmaWaitsMap[dmaOp] = findAssociatedDMAWaits(dmaOp);
  });
}

} // namespace mlir::tt::d2m
