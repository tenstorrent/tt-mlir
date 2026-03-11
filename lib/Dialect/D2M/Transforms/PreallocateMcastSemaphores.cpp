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
    LocalSemaphoreType semType = rewriter.getType<LocalSemaphoreType>();

    // Track the additionalArgs index of semaphores for each RemoteLoadOp.
    SmallVector<std::pair<RemoteLoadOp, SmallVector<unsigned>>>
        loadToSemIndices;

    // Insert create_local_semaphore ops before the generic op.
    rewriter.setInsertionPoint(generic);

    for (RemoteLoadOp load : allMcastLoads) {
      // Record the index in additionalArgs before appending.
      unsigned baseAdditionalIdx = generic.getAdditionalArgs().size();
      SmallVector<unsigned> semIndices = {baseAdditionalIdx,
                                          baseAdditionalIdx + 1};

      // Create the two local semaphores (receiversReady, senderFinished).
      auto recvReady = rewriter.create<CreateLocalSemaphoreOp>(
          loc, semType, rewriter.getUI32IntegerAttr(0));
      auto senderFinished = rewriter.create<CreateLocalSemaphoreOp>(
          loc, semType, rewriter.getUI32IntegerAttr(0));

      // Append them to the generic op's additionalArgs.
      // NormalizeThreadArgs will later add block args for these
      // and replace uses of the CreateLocalSemaphoreOp results in regions.
      generic.getAdditionalArgsMutable().append(
          ValueRange{recvReady.getResult(), senderFinished.getResult()});

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
