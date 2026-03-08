// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
    // IR.  getOperandAlloc uses positional counting, so erasing allocs shifts
    // positions and breaks subsequent lookups.
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

    auto defaultGridShape = genericOp.getPhysicalGridShape();

    for (auto [allocOp, operandIdx] : allocsToHoist) {
      auto allocType = allocOp.getType();

      // Derive grid shape from the corresponding external operand's device
      // layout (not the generic's physical grid, which may be a virtual grid
      // with a volume that exceeds the physical device).
      SmallVector<int64_t> gridShape(defaultGridShape);
      if (operandIdx >= 0) {
        Value externalOperand = genericOp.getInputsAndOutputs()[operandIdx];
        auto operandType = mlir::cast<ShapedType>(externalOperand.getType());
        if (auto devLayout = ttcore::getDeviceLayout(operandType)) {
          gridShape = SmallVector<int64_t>(devLayout.getGridShape(operandType));
        }
      }

      // External alloc uses the SAME type as the in-generic alloc
      // (shard-only + CBBufferLayoutAttr).  This allows direct replacement.

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

      // Store grid shape so MemrefAllocRewriter can derive [grid..shard..].
      externalAlloc->setAttr("d2m.grid_shape",
                             rewriter.getDenseI64ArrayAttr(gridShape));

      // Store which regular operand this CB backs so D2MGenericRewriter
      // can override the CB at the correct port.
      if (operandIdx >= 0) {
        externalAlloc->setAttr("d2m.cb_for_operand",
                               rewriter.getI64IntegerAttr(operandIdx));
        copyVirtualGridMappings(genericOp, operandIdx, externalAlloc);
      }

      // Add the external alloc as an additionalArg to the generic op.
      genericOp.getAdditionalArgsMutable().append(externalAlloc.getResult());

      // Replace all uses of the in-generic alloc with the additionalArg
      // and erase the alloc.  Types match so this is safe.
      rewriter.replaceOp(allocOp, externalAlloc.getResult());
    }
  }

  /// Find the regular operand index whose in-generic alloc matches |allocOp|.
  static int64_t findRegularOperandIndex(d2m::GenericOp genericOp,
                                         memref::AllocOp allocOp) {
    for (unsigned i = 0; i < genericOp.getInputsAndOutputs().size(); ++i) {
      for (Region &region : genericOp->getRegions()) {
        if (d2m::GenericOp::getOperandAlloc(region, i) == allocOp.getResult()) {
          return static_cast<int64_t>(i);
        }
      }
    }
    return -1;
  }

  /// Copy virtual grid mapping attrs from the external operand at
  /// |operandIdx| onto |externalAlloc|.
  static void copyVirtualGridMappings(d2m::GenericOp genericOp,
                                      int64_t operandIdx,
                                      memref::AllocOp externalAlloc) {
    Value externalOperand = genericOp.getInputsAndOutputs()[operandIdx];
    auto copyAttrs = [&](memref::AllocOp src) {
      if (auto vgm = src->getAttrOfType<AffineMapAttr>(
              d2m::utils::kVirtualGridInverseMappingAttr)) {
        externalAlloc->setAttr(d2m::utils::kVirtualGridInverseMappingAttr, vgm);
      }
      if (auto fwd = src->getAttrOfType<AffineMapAttr>(
              d2m::utils::kVirtualGridForwardMappingAttr)) {
        externalAlloc->setAttr(d2m::utils::kVirtualGridForwardMappingAttr, fwd);
      }
    };
    if (auto alloc = externalOperand.getDefiningOp<memref::AllocOp>()) {
      copyAttrs(alloc);
    } else if (auto streamOp =
                   externalOperand.getDefiningOp<d2m::StreamLayoutOp>()) {
      if (auto alloc = streamOp.getInput().getDefiningOp<memref::AllocOp>()) {
        copyAttrs(alloc);
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
