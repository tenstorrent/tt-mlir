// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MAFFINELICM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
} // namespace mlir::tt::d2m

namespace {

struct D2MAffineLICM
    : public mlir::tt::d2m::impl::D2MAffineLICMBase<D2MAffineLICM> {
  using D2MAffineLICMBase::D2MAffineLICMBase;

  void runOnOperation() override {
    llvm::SmallVector<mlir::LoopLikeOpInterface> loops;
    getOperation().walk(
        [&](mlir::affine::AffineForOp forOp) { loops.push_back(forOp); });
    getOperation().walk([&](mlir::affine::AffineParallelOp parallelOp) {
      loops.push_back(parallelOp);
    });

    // Process inner loops first so newly hoisted operations can be hoisted
    // again from enclosing loops when they are invariant there too.
    for (mlir::LoopLikeOpInterface loop : llvm::reverse(loops)) {
      mlir::moveLoopInvariantCode(loop);
    }
  }
};

} // namespace
