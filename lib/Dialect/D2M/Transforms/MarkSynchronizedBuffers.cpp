// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <functional>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMARKSYNCHRONIZEDBUFFERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

static bool containsAccumulatingCompute(Operation *op) {
  if (isa<d2m::TileMatmulOp, d2m::TileMatmulBlockOp, d2m::TileReduceSumOp,
          d2m::TileReduceMaxOp, d2m::TileReduceMeanOp, d2m::TileSFPUReduceSumOp,
          d2m::TileSFPUReduceMaxOp>(op)) {
    return true;
  }

  return op
      ->walk([](Operation *nestedOp) {
        if (isa<d2m::TileMatmulOp, d2m::TileMatmulBlockOp, d2m::TileReduceSumOp,
                d2m::TileReduceMaxOp, d2m::TileReduceMeanOp,
                d2m::TileSFPUReduceSumOp, d2m::TileSFPUReduceMaxOp>(nestedOp)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

class D2MMarkSynchronizedBuffers
    : public impl::D2MMarkSynchronizedBuffersBase<D2MMarkSynchronizedBuffers> {
public:
  using impl::D2MMarkSynchronizedBuffersBase<
      D2MMarkSynchronizedBuffers>::D2MMarkSynchronizedBuffersBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());
    llvm::DenseMap<Operation *, bool> containsAccumulatingComputeCache;

    auto cachedContainsAccumulatingCompute = [&](Operation *op) {
      auto it = containsAccumulatingComputeCache.find(op);
      if (it != containsAccumulatingComputeCache.end()) {
        return it->second;
      }
      bool result = containsAccumulatingCompute(op);
      containsAccumulatingComputeCache[op] = result;
      return result;
    };

    moduleOp->walk([&](d2m::GenericOp genericOp) {
      auto cbUsageInfo = utils::getCBUsageInfo(genericOp.getRegion(0));

      // A loop-carried accumulator is written by accumulating compute through
      // an scf.for iter_arg: the fill writes the backing alloc, but the
      // matmul/reduce writes the iter_arg block argument (a distinct Value).
      // To detect this, follow the buffer through the iter_args of any scf.for
      // it initializes and check the producers of each carried value.
      llvm::DenseSet<Value> visited;
      std::function<bool(Value)> accumulates = [&](Value v) -> bool {
        if (!visited.insert(v).second) {
          return false;
        }
        if (auto it = cbUsageInfo.find(v); it != cbUsageInfo.end()) {
          for (Operation *producer : it->second.producers) {
            if (cachedContainsAccumulatingCompute(producer)) {
              return true;
            }
          }
        }
        for (OpOperand &use : v.getUses()) {
          if (auto forOp = mlir::dyn_cast<scf::ForOp>(use.getOwner())) {
            for (auto [i, init] : llvm::enumerate(forOp.getInitArgs())) {
              if (init == v && accumulates(forOp.getRegionIterArgs()[i])) {
                return true;
              }
            }
          }
        }
        return false;
      };

      for (auto &[cb, usageInfo] : cbUsageInfo) {
        // `cb` may be a block argument (e.g. an scf.for iter_arg carrying an
        // accumulator), in which case it has no defining op. Use
        // dyn_cast_or_null so such CBs are simply skipped rather than
        // tripping dyn_cast's non-null assertion.
        if (auto allocOp =
                mlir::dyn_cast_or_null<memref::AllocOp>(cb.getDefiningOp())) {
          bool forceHoistedCB =
              utils::isReductionScalerBuffer(allocOp.getOperation());
          int32_t bufferCount = numStreamBuffers;
          if (forceHoistedCB) {
            bufferCount = 1;
          } else {
            for (Operation *producer : usageInfo.producers) {
              if (cachedContainsAccumulatingCompute(producer)) {
                bufferCount = 1;
                break;
              }
            }
          }
          allocOp->setAttr("d2m.synchronized_buffer",
                           rewriter.getI32IntegerAttr(bufferCount));

          if (!forceHoistedCB && usageInfo.consumers.size() == 1 &&
              usageInfo.producers.size() == 1) {
            auto *consumer = usageInfo.consumers.front();
            auto *producer = usageInfo.producers.front();
            // A `compute_intermediate` buffer is assumed dead in L1 (the
            // producer->consumer value stays DST-resident, so Allocate /
            // HoistCBAllocs skip it). That assumption is violated when the
            // producer is a fill/zeros init feeding an in-place accumulate
            // (e.g. `acc = zeros(); acc += remote_load(...)` before a loop):
            // InsertDstRegisterAccess routes that value through L1, so the
            // buffer is genuinely used and must get a real CB. Don't tag it.
            bool producerIsFill =
                producer
                    ->walk([](d2m::TileFillOp) {
                      return WalkResult::interrupt();
                    })
                    .wasInterrupted();
            if (mlir::isa<linalg::GenericOp>(consumer) &&
                mlir::isa<linalg::GenericOp>(producer) && !producerIsFill) {
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
