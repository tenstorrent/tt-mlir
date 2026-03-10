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
        if (mlir::isa<ttcore::CBBufferLayoutAttr>(memrefType.getLayout()) &&
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

      // The CBBufferLayoutAttr already carries the per-operand grid shape
      // (from bufferType in insertStream).  No grid derivation needed
      // here — just move the alloc outside.

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
        // Copy VGM attrs from the stream's storage buffer (if the operand
        // has a pre-existing stream).  The storage buffer carries the
        // target grid mapping needed by the serializer.
        copyVGMFromStreamStorage(genericOp, operandIdx, externalAlloc);
      }

      // Add the external alloc as an additionalArg to the generic op.
      genericOp.getAdditionalArgsMutable().append(externalAlloc.getResult());

      // Replace all uses of the in-generic alloc with the additionalArg
      // and erase the alloc.  Types match so this is safe.
      rewriter.replaceOp(allocOp, externalAlloc.getResult());
    }
  }

  /// Find the regular operand index for a CB alloc by tracing through
  /// its remote_load/remote_store memref operand back to the generic's
  /// operands.
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

  /// Copy VGM attrs from the stream's *storage* buffer onto |externalAlloc|.
  /// The storage buffer (second operand of stream_layout) carries the target
  /// grid mapping the serializer needs.
  static void copyVGMFromStreamStorage(d2m::GenericOp genericOp,
                                       int64_t operandIdx,
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
    if (auto streamOp = externalOperand.getDefiningOp<d2m::StreamLayoutOp>()) {
      // Copy from the storage buffer (second operand), not the input.
      if (auto storageAlloc =
              streamOp.getStorage().getDefiningOp<memref::AllocOp>()) {
        copyAttrs(storageAlloc);
      }
    } else if (auto viewOp =
                   externalOperand.getDefiningOp<d2m::ViewLayoutOp>()) {
      // View operand — try the view's input.
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
