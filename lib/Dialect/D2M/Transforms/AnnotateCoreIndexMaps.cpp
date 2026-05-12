// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MANNOTATECOREINDEXMAPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
static void annotateCoreIndexOpsWithPhysicalToVirtualMaps(Operation *root) {
  root->walk([](GenericOp genericOp) {
    AffineMap physicalToVirtualMap = genericOp.getGrid().getPhysicalToVirtMap();
    if (!physicalToVirtualMap || physicalToVirtualMap.isEmpty()) {
      return;
    }

    genericOp.walk([&](CoreIndexOp coreIndexOp) {
      if (coreIndexOp->getParentOfType<GenericOp>() != genericOp) {
        return;
      }
      coreIndexOp.setPhysToVirtMapAttr(
          AffineMapAttr::get(physicalToVirtualMap));
    });
  });
}

class D2MAnnotateCoreIndexMaps
    : public impl::D2MAnnotateCoreIndexMapsBase<D2MAnnotateCoreIndexMaps> {
public:
  using impl::D2MAnnotateCoreIndexMapsBase<
      D2MAnnotateCoreIndexMaps>::D2MAnnotateCoreIndexMapsBase;

  void runOnOperation() final {
    annotateCoreIndexOpsWithPhysicalToVirtualMaps(getOperation());
  }
};
} // namespace

} // namespace mlir::tt::d2m
