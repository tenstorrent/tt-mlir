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
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

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
      for (auto &[cb, usageInfo] : cbUsageInfo) {
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

          bool isComputeLocal =
              !usageInfo.producers.empty() && !usageInfo.consumers.empty() &&
              llvm::all_of(usageInfo.producers,
                           [](Operation *producer) {
                             return mlir::isa<linalg::GenericOp>(producer);
                           }) &&
              llvm::all_of(usageInfo.consumers, [](Operation *consumer) {
                return mlir::isa<linalg::GenericOp>(consumer);
              });
          if (!forceHoistedCB && isComputeLocal) {
            allocOp->setAttr("d2m.compute_intermediate",
                             rewriter.getUnitAttr());
          }
        }
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::d2m
