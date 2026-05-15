// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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

    WalkResult result = moduleOp->walk([&](d2m::GenericOp genericOp) {
      if (failed(hoistCBAllocs(rewriter, genericOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

private:
  memref::AllocOp attachCBLayoutAttribute(IRRewriter &rewriter,
                                          memref::AllocOp allocOp,
                                          int32_t numBuffers) {
    auto buffer = allocOp->getResult(0);
    auto bufferType = mlir::cast<MemRefType>(buffer.getType());
    auto shardShape = bufferType.getShape();
    Type elementType = bufferType.getElementType();
    auto cbLayout = ttcore::CBLayoutAttr::get(
        bufferType.getContext(), shardShape,
        ttcore::getElementSizeBytes(elementType), numBuffers);

    auto newMemRefType = MemRefType::get(shardShape, elementType, cbLayout,
                                         bufferType.getMemorySpace());
    auto oldAttrs = llvm::to_vector(allocOp->getAttrs());
    rewriter.setInsertionPoint(allocOp);
    auto newAllocOp =
        rewriter.replaceOpWithNewOp<memref::AllocOp>(allocOp, newMemRefType);
    // transfer all attributes from the old alloc to the new alloc
    for (auto attr : oldAttrs) {
      newAllocOp->setAttr(attr.getName(), attr.getValue());
    }

    for (auto *user : newAllocOp->getResult(0).getUsers()) {
      auto destinationStyleOp =
          mlir::dyn_cast<DestinationStyleOpInterface>(user);
      if (destinationStyleOp && user->getNumResults() > 0) {
        for (auto &operand : destinationStyleOp->getOpOperands()) {
          if (destinationStyleOp.isDpsInit(&operand) &&
              operand.get() == newAllocOp->getResult(0)) {
            // update op result that matches dps init
            OpResult tiedResult = destinationStyleOp.getTiedOpResult(&operand);
            tiedResult.setType(newMemRefType);
          }
        }
      }
    }

    return newAllocOp;
  }

  LogicalResult hoistCBAllocs(IRRewriter &rewriter, d2m::GenericOp genericOp) {
    // Collect allocs to hoist AND their operand indices BEFORE modifying the
    // IR.  Erasing allocs shifts positional lookups.
    SmallVector<memref::AllocOp> allocsToHoist;

    for (Region &region : genericOp->getRegions()) {
      WalkResult walkResult = region.walk([&](memref::AllocOp allocOp) {
        if (allocOp->getAttr("d2m.scratch_buffer")) {
          if (!allocOp->getAttrOfType<IntegerAttr>("address")) {
            allocOp.emitError("scratch buffer must have address attribute");
            return WalkResult::interrupt();
          }
          auto newAllocOp = attachCBLayoutAttribute(rewriter, allocOp, 1);
          allocsToHoist.push_back(newAllocOp);
        } else if (allocOp->getAttr("d2m.synchronized_buffer")) {
          if (allocOp->getAttr("d2m.compute_intermediate")) {
            // Skip hoisting, these are fake buffers added by fusion that are
            // guaranteed to not be used
            if (allocOp->getAttrOfType<IntegerAttr>("address")) {
              allocOp.emitError("compute intermediate buffer must not have "
                                "address attribute");
              return WalkResult::interrupt();
            }
          } else {
            if (!allocOp->getAttrOfType<IntegerAttr>("address")) {
              allocOp.emitError(
                  "synchronized buffer must have address attribute");
              return WalkResult::interrupt();
            }
            auto newAllocOp = attachCBLayoutAttribute(
                rewriter, allocOp,
                allocOp->getAttrOfType<IntegerAttr>("d2m.synchronized_buffer")
                    .getInt());
            allocsToHoist.push_back(newAllocOp);
          }
        } else {
          allocOp.emitError("unexpected alloc op: expected d2m.scratch_buffer "
                            "or d2m.synchronized_buffer attribute");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (walkResult.wasInterrupted()) {
        return failure();
      }
    }

    for (auto allocOp : allocsToHoist) {
      auto allocType = allocOp.getType();

      // Move the CB alloc outside the generic as an additionalArg.

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      auto externalAlloc =
          rewriter.create<memref::AllocOp>(genericOp.getLoc(), allocType);
      // transfer all attributes from the old alloc to the new alloc
      for (auto attr : allocOp->getAttrs()) {
        externalAlloc->setAttr(attr.getName(), attr.getValue());
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

    // Hoist aliased CBs.
    genericOp->walk([&](d2m::OperandAliasOp operandAliasOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      auto externalAlias = rewriter.create<d2m::OperandAliasOp>(
          genericOp.getLoc(), operandAliasOp.getType(),
          operandAliasOp.getMemref());
      genericOp.getAdditionalArgsMutable().append(externalAlias.getResult());
      rewriter.replaceOp(operandAliasOp, externalAlias.getResult());
    });

    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
