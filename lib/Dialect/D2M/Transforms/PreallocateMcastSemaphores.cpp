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

class D2MPreallocateMcastSemaphoresRewriter
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Skip if already processed (any RemoteLoadOp has the attribute).
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
    }

    // Find all RemoteLoadOps that need multicast semaphores.
    // They can be in any region (typically the datamovement region).
    SmallVector<RemoteLoadOp> allMcastLoads;
    for (Region &region : generic->getRegions()) {
      if (region.empty()) {
        continue;
      }
      Block &block = region.front();
      collectMcastRemoteLoads(&block, allMcastLoads);
    }

    // If no multicast loads, nothing to do.
    if (allMcastLoads.empty()) {
      return failure();
    }

    // For each multicast RemoteLoadOp, we need to add 2 semaphore block
    // arguments to ALL regions of the generic op:
    // - receiversReadySemaphore
    // - senderFinishedSemaphore

    Location loc = generic.getLoc();
    SemaphoreType semType = rewriter.getType<SemaphoreType>();

    // Track the starting index of semaphores for each RemoteLoadOp.
    // We'll store these indices as attributes on the ops.
    SmallVector<std::pair<RemoteLoadOp, SmallVector<unsigned>>>
        loadToSemIndices;

    for (RemoteLoadOp load : allMcastLoads) {
      SmallVector<unsigned> semIndices;

      // Add 2 semaphore arguments to each region.
      for (Region &region : generic->getRegions()) {
        if (region.empty()) {
          continue;
        }
        Block &block = region.front();

        // Record the index before adding (same index in all regions).
        if (semIndices.empty()) {
          unsigned baseIdx = block.getNumArguments();
          semIndices.push_back(baseIdx);     // receiversReady index
          semIndices.push_back(baseIdx + 1); // senderFinished index
        }

        // Add the semaphore arguments.
        block.addArgument(semType, loc);
        block.addArgument(semType, loc);
      }

      loadToSemIndices.push_back({load, semIndices});
    }

    // Now set the attribute on each RemoteLoadOp with its semaphore indices.
    for (auto &[load, semIndices] : loadToSemIndices) {
      SmallVector<int64_t> indices(semIndices.begin(), semIndices.end());
      load->setAttr(kPreallocatedSemaphoresAttr,
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
