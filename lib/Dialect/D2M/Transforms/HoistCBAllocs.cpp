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
    SmallVector<std::pair<memref::AllocOp, int64_t>> allocsToHoist;
    for (Region &region : genericOp->getRegions()) {
      region.walk([&](memref::AllocOp allocOp) {
        auto memrefType = allocOp.getType();
        if (mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout()) &&
            allocOp->getAttrOfType<IntegerAttr>("address")) {
          int64_t idx = findRegularOperandIndex(genericOp, allocOp);
          allocsToHoist.push_back({allocOp, idx});
        }
      });
    }

    if (allocsToHoist.empty()) {
      return;
    }

    for (auto [allocOp, operandIdx] : allocsToHoist) {
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

      // Tag which regular operand this CB backs so D2MGenericRewriter
      // can override the CB at the correct port.
      if (operandIdx >= 0) {
        externalAlloc->setAttr("d2m.cb_for_operand",
                               rewriter.getI64IntegerAttr(operandIdx));
        // Copy VGM attrs from the operand's defining alloc so the
        // serializer can compute physical grid shapes.
        copyVGMFromOperand(genericOp, operandIdx, externalAlloc);
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
  }

  // Helper function to check if an operand is remote (i.e., implies data
  // movement). This includes view ops (view_layout) and buffers with
  // CBLayoutAttr (streaming circular buffers hoisted from the generic).
  static bool isRemoteOperand(Value operand) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      return false;
    }
    if (mlir::isa<ViewOpInterface>(defOp)) {
      return true;
    }
    // A buffer with CBLayoutAttr is a streaming CB that requires real
    // data movement from an external shard.
    if (auto memrefType = mlir::dyn_cast<MemRefType>(operand.getType())) {
      auto memspace = ttcore::getMemorySpace(memrefType);
      if (memspace == ttcore::MemorySpace::DeviceDRAM ||
          mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout())) {
        return true;
      }
    }
    return false;
  }

  /// Find the regular operand index for a CB alloc by tracing through
  /// its remote_load/remote_store memref operand back to the generic's
  /// operands.
  /// In the case that a load and store both use the same alloc and both are
  /// true (non-aliased) remote load/stores, consider the allocation as
  /// belonging _to the input operand_.
  static int64_t findRegularOperandIndex(d2m::GenericOp genericOp,
                                         memref::AllocOp allocOp) {
    for (Operation *user : allocOp.getResult().getUsers()) {
      if (auto remoteLoad = mlir::dyn_cast<d2m::RemoteLoadOp>(user)) {
        Value memref = remoteLoad.getMemref();
        for (unsigned i = 0; i < genericOp->getNumOperands(); ++i) {
          if (genericOp->getOperand(i) == memref) {
            return static_cast<int64_t>(i);
          }
        }
      }
      if (auto remoteStore = mlir::dyn_cast<d2m::RemoteStoreOp>(user)) {
        Value memref = remoteStore.getMemref();
        for (unsigned i = 0; i < genericOp->getNumOperands(); ++i) {
          if (genericOp->getOperand(i) == memref) {
            return static_cast<int64_t>(i);
          }
        }
      }
    }
    return -1;
  }

  /// Copy VGM attrs from the operand's defining alloc onto |externalAlloc|.
  static void copyVGMFromOperand(d2m::GenericOp genericOp, int64_t operandIdx,
                                 memref::AllocOp externalAlloc) {
    Value externalOperand = genericOp.getInputsAndOutputs()[operandIdx];
    auto copyAttrs = [&](Operation *src) {
      if (auto vgm = src->getAttrOfType<AffineMapAttr>(
              d2m::utils::kVirtualGridInverseMappingAttr)) {
        externalAlloc->setAttr(d2m::utils::kVirtualGridInverseMappingAttr, vgm);
      }
      if (auto fwd = src->getAttrOfType<AffineMapAttr>(
              d2m::utils::kVirtualGridForwardMappingAttr)) {
        externalAlloc->setAttr(d2m::utils::kVirtualGridForwardMappingAttr, fwd);
      }
    };
    if (auto viewOp = externalOperand.getDefiningOp<d2m::ViewLayoutOp>()) {
      if (auto alloc = viewOp.getInput().getDefiningOp<memref::AllocOp>()) {
        copyAttrs(alloc);
      }
    } else if (auto alloc = externalOperand.getDefiningOp<memref::AllocOp>()) {
      copyAttrs(alloc);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
