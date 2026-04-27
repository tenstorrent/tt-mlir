// SPDX-FileCopyrightText: (c) 2026wr Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MHOISTCBALLOCS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MHoistCBAllocs : public impl::D2MHoistCBAllocsBase<D2MHoistCBAllocs> {
public:
  using impl::D2MHoistCBAllocsBase<D2MHoistCBAllocs>::D2MHoistCBAllocsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk(
        [&](d2m::GenericOp genericOp) { hoistCBAllocs(rewriter, genericOp); });
  }

private:
  void hoistCBAllocs(IRRewriter &rewriter, d2m::GenericOp genericOp) {
    // Collect allocs to hoist AND their operand indices BEFORE modifying the
    // IR.  Erasing allocs shifts positional lookups.
    SmallVector<memref::AllocOp> allocsToHoist;
    for (Region &region : genericOp->getRegions()) {
      region.walk([&](memref::AllocOp allocOp) {
        auto memrefType = allocOp.getType();
        if (mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout()) &&
            allocOp->getAttrOfType<IntegerAttr>("address")) {
          allocsToHoist.push_back(allocOp);
        }
      });
    }

    for (auto allocOp : allocsToHoist) {
      auto allocType = allocOp.getType();

      // Move the CB alloc outside the generic as an additionalArg.

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      auto externalAlloc =
          rewriter.create<memref::AllocOp>(genericOp.getLoc(), allocType);

      // Transfer address and alignment.
      if (auto addressAttr = allocOp->getAttrOfType<IntegerAttr>("address")) {
        externalAlloc->setAttr("address", addressAttr);
      }
      if (auto alignAttr = allocOp.getAlignmentAttr()) {
        externalAlloc.setAlignmentAttr(alignAttr);
      }

      // Add the external alloc as an additionalArg to the generic op.
      genericOp.getAdditionalArgsMutable().append(externalAlloc.getResult());

      // Replace all uses of the in-generic alloc with the additionalArg
      // and erase the alloc.  Types match so this is safe.
      rewriter.replaceOp(allocOp, externalAlloc.getResult());

      // Insert a dealloc right after the generic op to bound the
      // hoisted alloc's live range.
      {
        OpBuilder::InsertionGuard deallocGuard(rewriter);
        rewriter.setInsertionPointAfter(genericOp);
        rewriter.create<memref::DeallocOp>(genericOp.getLoc(),
                                           externalAlloc.getResult());
      }
    }

    // hoist aliased CBs
    genericOp->walk([&](memref::AllocOp allocOp) {
      if (auto aliasForOperandAttr =
              allocOp->getAttrOfType<IntegerAttr>("d2m.alias_for_operand")) {
        // llvm::errs() << "hoisting aliased CB: " <<
        // genericOp.getOperand(operandIndex) << "\n";
        rewriter.setInsertionPoint(genericOp);
        auto aliasedOperand =
            genericOp.getOperand(aliasForOperandAttr.getInt());
        // TODO: add back later
        // auto operandMemRefType =
        //    mlir::cast<MemRefType>(aliasedOperand.getType());
        // unsigned numStreamBuffers =
        //    ttmlir::utils::volume(operandMemRefType.getShape()) /
        //    ttmlir::utils::volume(allocOp.getType().getShape());
        // auto cbLayout = ttcore::CBLayoutAttr::get(
        //    &getContext(), allocOp.getType().getShape(),
        //    ttcore::getElementSizeBytes(allocOp.getType().getElementType()),
        //    numStreamBuffers);
        // auto newMemRefType = MemRefType::get(
        //    allocOp.getType().getShape(), allocOp.getType().getElementType(),
        //    cbLayout, allocOp.getType().getMemorySpace());
        auto externalAlias = rewriter.create<d2m::OperandAliasOp>(
            genericOp.getLoc(), allocOp.getType(), aliasedOperand);
        genericOp.getAdditionalArgsMutable().append(externalAlias.getResult());
        rewriter.replaceOp(allocOp, externalAlias.getResult());
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::d2m
