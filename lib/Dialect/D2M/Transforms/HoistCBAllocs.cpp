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
          int64_t idx = findRegularOperandIndex(genericOp, region, allocOp);
          allocsToHoist.push_back({allocOp, idx});
        }
      });
    }

    if (allocsToHoist.empty()) {
      return;
    }

    for (auto [allocOp, operandIdx] : allocsToHoist) {
      if (operandIdx >= 0) {
        if (memref::AllocOp storageAlloc =
                getStreamStorageAlloc(genericOp, operandIdx)) {
          moveAttrsToStreamStorage(allocOp, storageAlloc);
          continue;
        }
      }

      auto allocType = allocOp.getType();

      // The CBLayoutAttr already carries the per-operand grid shape
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

  /// Find the regular operand index whose associated CB/allocation in |region|
  /// is |allocOp|. GenericOp::getOperandAlloc uses d2m.get_cb as the source of
  /// truth when present, and falls back to positional alloc order otherwise.
  static int64_t findRegularOperandIndex(d2m::GenericOp genericOp,
                                         Region &region,
                                         memref::AllocOp allocOp) {
    unsigned ioSize = genericOp.getInputsAndOutputs().size();
    for (unsigned operandIdx = 0; operandIdx < ioSize; ++operandIdx) {
      Value assocAlloc = d2m::GenericOp::getOperandAlloc(region, operandIdx);
      if (assocAlloc == allocOp.getResult()) {
        return static_cast<int64_t>(operandIdx);
      }
    }
    return -1;
  }

  static memref::AllocOp getStreamStorageAlloc(d2m::GenericOp genericOp,
                                               int64_t operandIdx) {
    Value operand = genericOp.getInputsAndOutputs()[operandIdx];
    auto streamOp = operand.getDefiningOp<d2m::StreamLayoutOp>();
    if (!streamOp) {
      return nullptr;
    }

    return streamOp.getStorage().getDefiningOp<memref::AllocOp>();
  }

  static void moveAttrsToStreamStorage(memref::AllocOp allocOp,
                                       memref::AllocOp storageAlloc) {
    if (auto addressAttr = allocOp->getAttrOfType<IntegerAttr>("address")) {
      storageAlloc->setAttr("address", addressAttr);
      allocOp->removeAttr("address");
    }
    if (auto alignAttr = allocOp.getAlignmentAttr()) {
      storageAlloc.setAlignmentAttr(alignAttr);
      allocOp->removeAttr("alignment");
    }

    llvm::SmallVector<StringAttr> discardableAttrNames;
    for (NamedAttribute attr : allocOp->getDiscardableAttrDictionary()) {
      storageAlloc->setDiscardableAttr(attr.getName(), attr.getValue());
      discardableAttrNames.push_back(attr.getName());
    }
    for (StringAttr attrName : discardableAttrNames) {
      allocOp->removeDiscardableAttr(attrName);
    }
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
