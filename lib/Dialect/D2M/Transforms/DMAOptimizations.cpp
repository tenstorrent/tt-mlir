// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MOPTIMIZEDMA
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Coalesce barriers for consecutive same-kind DMA sequences within a block.
// Reorders ops so all NOC transactions are issued before any barriers fire.
static void coalesceBarriers(Block &block) {
  // TODO(vtang): Implement barrier coalescing (Optimization 1).
}

// Defer write barriers from the current loop iteration to the next iteration.
static void deferWriteBarriers(scf::ForOp forOp) {
  // TODO(vtang): Implement write barrier deferral (Optimization 2).
}

// Defer mcast write barriers from the current loop iteration to the next.
static void deferMcastWriteBarriers(scf::ForOp forOp) {
  // TODO(vtang): Implement mcast write barrier deferral (Optimization 3).
}

class D2MOptimizeDMA : public impl::D2MOptimizeDMABase<D2MOptimizeDMA> {
public:
  using impl::D2MOptimizeDMABase<D2MOptimizeDMA>::D2MOptimizeDMABase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([](GenericOp generic) {
      for (Region &region : generic.getRegions()) {
        if (region.empty()) {
          continue;
        }
        if (generic.getRegionThreadType(region.getRegionNumber()) !=
            ThreadType::Datamovement) {
          continue;
        }

        for (Block &block : region) {
          coalesceBarriers(block);
        }

        region.walk([](scf::ForOp forOp) {
          deferWriteBarriers(forOp);
          deferMcastWriteBarriers(forOp);
        });
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
