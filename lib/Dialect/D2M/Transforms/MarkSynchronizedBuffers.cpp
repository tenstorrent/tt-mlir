// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMARKSYNCHRONIZEDBUFFERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MMarkSynchronizedBuffers
    : public impl::D2MMarkSynchronizedBuffersBase<D2MMarkSynchronizedBuffers> {
public:
  using impl::D2MMarkSynchronizedBuffersBase<
      D2MMarkSynchronizedBuffers>::D2MMarkSynchronizedBuffersBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk([&](d2m::GenericOp genericOp) {
      auto cbUsageInfo = utils::getCBUsageInfo(genericOp.getRegion(0));
      for (auto &[cb, usageInfo] : cbUsageInfo) {
        if (auto allocOp =
                mlir::dyn_cast<memref::AllocOp>(cb.getDefiningOp())) {
          allocOp->setAttr("d2m.synchronized_buffer",
                           rewriter.getI32IntegerAttr(numStreamBuffers));

          if (!usageInfo.consumers.empty() && !usageInfo.producers.empty()) {
            TT_assertv((usageInfo.consumers.size() == 1 &&
                        usageInfo.producers.size() == 1),
                       "expected exactly one consumer and one producer");
            auto *consumer = usageInfo.consumers.front();
            auto *producer = usageInfo.producers.front();
            if (mlir::isa<linalg::GenericOp>(consumer) &&
                mlir::isa<linalg::GenericOp>(producer)) {
              allocOp->setAttr("d2m.compute_intermediate",
                               rewriter.getUnitAttr());
            }
          }
        }
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::d2m
