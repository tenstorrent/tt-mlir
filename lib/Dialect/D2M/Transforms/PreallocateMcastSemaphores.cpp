// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MPREALLOCATEMCASTSEMAPHORES
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Attribute name for storing pre-allocated semaphore indices on RemoteLoadOp.
constexpr StringRef kPreallocatedSemaphoresAttr = "preallocated_semaphores";

// Check if a RemoteLoadOp needs multicast semaphores.
// Returns true if the op has mcastStartIndex and mcastShape.
static bool needsMcastSemaphores(RemoteLoadOp remoteLoad) {
  return !remoteLoad.getMcastStartIndex().empty() &&
         !remoteLoad.getMcastShape().empty();
}

// Recursively collect all RemoteLoadOps that need multicast semaphores.
static void collectMcastRemoteLoads(Block *block,
                                    SmallVectorImpl<RemoteLoadOp> &mcastLoads) {
  for (Operation &op : block->getOperations()) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
      collectMcastRemoteLoads(forOp.getBody(), mcastLoads);
      continue;
    }

    if (auto remoteLoad = mlir::dyn_cast<RemoteLoadOp>(&op)) {
      if (needsMcastSemaphores(remoteLoad)) {
        mcastLoads.push_back(remoteLoad);
      }
    }
  }
}

// Recursively collect all ScatterOps (all scatters need semaphores).
static void collectScatterOps(Block *block,
                              SmallVectorImpl<ScatterOp> &scatterOps) {
  for (Operation &op : block->getOperations()) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
      collectScatterOps(forOp.getBody(), scatterOps);
      continue;
    }

    if (auto scatter = mlir::dyn_cast<ScatterOp>(&op)) {
      scatterOps.push_back(scatter);
    }
  }
}

class D2MPreallocateMcastSemaphoresRewriter
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Skip if already processed (any op has the attribute).
    for (Region &region : generic->getRegions()) {
      if (region.empty()) {
        continue;
      }
      Block &block = region.front();
      SmallVector<RemoteLoadOp> mcastLoads;
      collectMcastRemoteLoads(&block, mcastLoads);
      for (RemoteLoadOp load : mcastLoads) {
        if (load->hasAttr(kPreallocatedSemaphoresAttr)) {
          return failure(); // Already processed
        }
      }
      SmallVector<ScatterOp> scatterOps;
      collectScatterOps(&block, scatterOps);
      for (ScatterOp scatter : scatterOps) {
        if (scatter->hasAttr(kPreallocatedSemaphoresAttr)) {
          return failure(); // Already processed
        }
      }
    }

    // Find all RemoteLoadOps that need multicast semaphores.
    // They can be in any region (typically the datamovement region).
    SmallVector<RemoteLoadOp> allMcastLoads;
    SmallVector<ScatterOp> allScatterOps;
    for (Region &region : generic->getRegions()) {
      if (region.empty()) {
        continue;
      }
      Block &block = region.front();
      collectMcastRemoteLoads(&block, allMcastLoads);
      collectScatterOps(&block, allScatterOps);
    }

    // If nothing needs semaphores, nothing to do.
    if (allMcastLoads.empty() && allScatterOps.empty()) {
      return failure();
    }

    Location loc = generic.getLoc();
    SemaphoreType semType = rewriter.getType<SemaphoreType>();

    // Helper: add N semaphore block arguments to all regions, return indices.
    auto addSemaphores = [&](unsigned count) -> SmallVector<unsigned> {
      SmallVector<unsigned> semIndices;
      for (Region &region : generic->getRegions()) {
        if (region.empty()) {
          continue;
        }
        Block &block = region.front();

        if (semIndices.empty()) {
          unsigned baseIdx = block.getNumArguments();
          for (unsigned i = 0; i < count; ++i) {
            semIndices.push_back(baseIdx + i);
          }
        }

        for (unsigned i = 0; i < count; ++i) {
          block.addArgument(semType, loc);
        }
      }
      return semIndices;
    };

    // For each multicast RemoteLoadOp: 2 semaphores
    // (receiversReady + senderFinished).
    for (RemoteLoadOp load : allMcastLoads) {
      SmallVector<unsigned> semIndices = addSemaphores(2);
      SmallVector<int64_t> indices(semIndices.begin(), semIndices.end());
      load->setAttr(kPreallocatedSemaphoresAttr,
                    rewriter.getI64ArrayAttr(indices));
    }

    // For each ScatterOp: 1 semaphore (scatterDone).
    for (ScatterOp scatter : allScatterOps) {
      SmallVector<unsigned> semIndices = addSemaphores(1);
      SmallVector<int64_t> indices(semIndices.begin(), semIndices.end());
      scatter->setAttr(kPreallocatedSemaphoresAttr,
                       rewriter.getI64ArrayAttr(indices));
    }

    return success();
  }
};

class D2MPreallocateMcastSemaphores
    : public impl::D2MPreallocateMcastSemaphoresBase<
          D2MPreallocateMcastSemaphores> {
public:
  using impl::D2MPreallocateMcastSemaphoresBase<
      D2MPreallocateMcastSemaphores>::D2MPreallocateMcastSemaphoresBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MPreallocateMcastSemaphoresRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
