// SPDX-FileCopyrightText: (c) 2026wr Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMARKSYNCHRONIZEDOPBUFFERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MMarkSynchronizedOpBuffers
    : public impl::D2MMarkSynchronizedOpBuffersBase<
          D2MMarkSynchronizedOpBuffers> {
public:
  using impl::D2MMarkSynchronizedOpBuffersBase<
      D2MMarkSynchronizedOpBuffers>::D2MMarkSynchronizedOpBuffersBase;

  void runOnOperation() final {
    // ModuleOp moduleOp = getOperation();
    // IRRewriter rewriter(&getContext());

    // moduleOp->walk([&](d2m::GenericOp genericOp) {
    //   if (failed(wrapLinalgAndTileOpsInSynchronizedRegion(rewriter,
    //   genericOp))) {
    //     emitError(genericOp->getLoc(),
    //               "Failed to mark synchronized op buffers for genericop: ")
    //         << genericOp;
    //   }
    //   return WalkResult::advance();
    // });
    //
    // moduleOp->walk([&](d2m::GenericOp genericOp) {
    //  if (failed(markSynchronizedOpBuffers(rewriter, genericOp))) {
    //    emitError(genericOp->getLoc(),
    //              "Failed to mark synchronized op buffers for genericop: ")
    //        << genericOp;
    //  }
    //  return WalkResult::advance();
    //});
  }

private:
};

} // namespace
} // namespace mlir::tt::d2m
