// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSCHEDULEDMA
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Represents the assignment of DMA operations to a hardware thread.
// Each thread is responsible for a set of circular buffers (CBs).
struct DMAThreadAssignment {
  // Set of CB indices (block argument indices) assigned to this thread.
  DenseSet<unsigned> assignedCBs;

  // Estimated workload for this thread (number of DMA ops).
  size_t workload = 0;

  // Assigned NoC index for this thread (0 = DRAM reader, 1 = DRAM writer).
  int32_t nocIndex = -1;
};

// Collect all remote_load and remote_store operations from a block,
// recursively walking into nested scf.for loops.
static void
collectDMAOps(Block *block,
              SmallVectorImpl<std::pair<Operation *, unsigned>> &dmaOps) {
  for (Operation &op : block->getOperations()) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
      collectDMAOps(forOp.getBody(), dmaOps);
      continue;
    }

    if (auto remoteLoad = mlir::dyn_cast<RemoteLoadOp>(&op)) {
      // Get the CB operand and find which block argument it corresponds to.
      Value cb = remoteLoad.getCb();
      if (auto blockArg = mlir::dyn_cast_or_null<BlockArgument>(cb)) {
        dmaOps.push_back({&op, blockArg.getArgNumber()});
      }
    } else if (auto remoteStore = mlir::dyn_cast<RemoteStoreOp>(&op)) {
      Value cb = remoteStore.getCb();
      if (auto blockArg = mlir::dyn_cast_or_null<BlockArgument>(cb)) {
        dmaOps.push_back({&op, blockArg.getArgNumber()});
      }
    }
  }
}

// Score for deciding which thread gets which NoC.
// Priority is given to the thread that has the larger mcast shards.
// Unicast reads take priority over unicast writes.
struct NocScore {
  int64_t mcastShardSize = 0;
  int64_t unicastBias = 0;

  bool operator>(const NocScore &other) const {
    if (mcastShardSize != other.mcastShardSize) {
      return mcastShardSize > other.mcastShardSize;
    }
    return unicastBias > other.unicastBias;
  }
};

// The thread with the higher NocScore gets NoC0.
// Default is thread0 gets NoC0, thread1 gets NoC1.
static void assignNocIndices(
    SmallVectorImpl<DMAThreadAssignment> &assignments,
    const SmallVectorImpl<std::pair<Operation *, unsigned>> &dmaOps) {
  TT_assertv(int(assignments.size()) == 2, "Expect exactly 2 DM threads");

  NocScore scores[2];
  for (const auto &[op, cbIdx] : dmaOps) {
    for (int t = 0; t < 2; ++t) {
      if (!assignments[t].assignedCBs.contains(cbIdx)) {
        continue;
      }

      if (auto load = mlir::dyn_cast<RemoteLoadOp>(op)) {
        if (ttcore::getMemorySpace(load.getMemref()) !=
            ttcore::MemorySpace::DeviceDRAM) {
          continue;
        }
        if (load.isMcast()) {
          auto layout = ttcore::getDeviceLayout(load.getMemref());
          if (layout) {
            auto shardShape = SmallVector<int64_t>(
                layout.getShardShape(load.getShapedType()));
            if (auto tileType = mlir::dyn_cast<ttcore::TileType>(
                    load.getShapedType().getElementType())) {
              shardShape = tileType.getScalarShape(shardShape);
            }
            int64_t size = 1;
            for (int64_t dim : shardShape) {
              size *= dim;
            }
            scores[t].mcastShardSize = std::max(scores[t].mcastShardSize, size);
          }
        } else {
          scores[t].unicastBias += 2;
        }
      } else if (auto store = mlir::dyn_cast<RemoteStoreOp>(op)) {
        if (ttcore::getMemorySpace(store.getMemref()) ==
            ttcore::MemorySpace::DeviceDRAM) {
          scores[t].unicastBias -= 1;
        }
      }
    }
  }
  bool swap = scores[1] > scores[0];
  assignments[0].nocIndex = swap ? 1 : 0;
  assignments[1].nocIndex = swap ? 0 : 1;
}

// Assign CBs to threads to balance workload.
// Returns a vector of DMAThreadAssignment, one per hardware thread.
static SmallVector<DMAThreadAssignment>
assignCBsToThreads(const DenseMap<unsigned, size_t> &cbWorkloads,
                   unsigned numThreads) {
  SmallVector<DMAThreadAssignment> assignments(numThreads);

  // Sort CBs by workload (descending) for greedy assignment.
  SmallVector<std::pair<unsigned, size_t>> sortedCBs;
  for (const auto &[cbIdx, workload] : cbWorkloads) {
    sortedCBs.push_back({cbIdx, workload});
  }
  llvm::sort(sortedCBs,
             [](const auto &a, const auto &b) { return a.second > b.second; });

  // Greedy assignment: assign each CB to the thread with smallest workload.
  for (const auto &[cbIdx, workload] : sortedCBs) {
    // Find thread with minimum workload.
    unsigned minThreadIdx = 0;
    size_t minWorkload = assignments[0].workload;
    for (unsigned i = 1; i < numThreads; ++i) {
      if (assignments[i].workload < minWorkload) {
        minWorkload = assignments[i].workload;
        minThreadIdx = i;
      }
    }

    assignments[minThreadIdx].assignedCBs.insert(cbIdx);
    assignments[minThreadIdx].workload += workload;
  }

  return assignments;
}

// Check if an operation should be kept in a thread based on CB assignments.
// Returns true if the operation uses a CB assigned to this thread.
static bool shouldKeepOpForThread(Operation *op,
                                  const DenseSet<unsigned> &assignedCBs) {
  if (auto remoteLoad = mlir::dyn_cast<RemoteLoadOp>(op)) {
    Value cb = remoteLoad.getCb();
    if (auto blockArg = mlir::dyn_cast_or_null<BlockArgument>(cb)) {
      return assignedCBs.contains(blockArg.getArgNumber());
    }
  } else if (auto remoteStore = mlir::dyn_cast<RemoteStoreOp>(op)) {
    Value cb = remoteStore.getCb();
    if (auto blockArg = mlir::dyn_cast_or_null<BlockArgument>(cb)) {
      return assignedCBs.contains(blockArg.getArgNumber());
    }
  }
  return false;
}

// Recursively erase DMA ops not assigned to this thread.
// Also removes ops that become dead as a result.
static void filterOpsForThread(PatternRewriter &rewriter, Block *block,
                               const DenseSet<unsigned> &assignedCBs) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;

    for (Operation &op : block->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      // Recurse into nested loops.
      if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
        filterOpsForThread(rewriter, forOp.getBody(), assignedCBs);
        continue;
      }

      // Check if this is a DMA op.
      if (mlir::isa<RemoteLoadOp, RemoteStoreOp>(&op)) {
        if (!shouldKeepOpForThread(&op, assignedCBs)) {
          if (op.use_empty()) {
            toErase.push_back(&op);
            changed = true;
          }
        }
      }
    }

    for (Operation *op : llvm::reverse(toErase)) {
      rewriter.eraseOp(op);
    }
  }
}

class D2MScheduleDMARewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  D2MScheduleDMARewriter(MLIRContext *context, unsigned numDatamovementThreads)
      : OpRewritePattern<GenericOp>(context),
        numDatamovementThreads(numDatamovementThreads) {}

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Check if this generic has exactly 2 regions: datamovement + compute.
    if (generic.getNumRegions() != 2) {
      return failure();
    }

    // Verify first region is datamovement and second is compute.
    if (generic.getRegionThreadType(0) != ThreadType::Datamovement ||
        generic.getRegionThreadType(1) != ThreadType::Compute) {
      return failure();
    }

    Region &dmRegion = generic.getRegion(0);
    if (dmRegion.empty()) {
      return failure();
    }
    Block *dmBlock = &dmRegion.front();

    // Check that there are no illegal semaphore ops in the datamovement region.
    // Replicating these across multiple threads would create a race condition
    // on the shared semaphore.
    if (failed(utils::checkForIllegalSemaphoreOps(dmBlock))) {
      return failure();
    }

    // Collect all DMA operations and their CB associations.
    SmallVector<std::pair<Operation *, unsigned>> dmaOps;
    collectDMAOps(dmBlock, dmaOps);

    // If no DMA ops, nothing to split.
    if (dmaOps.empty()) {
      return failure();
    }

    // Count workload per CB.
    DenseMap<unsigned, size_t> cbWorkloads;
    for (const auto &[op, cbIdx] : dmaOps) {
      cbWorkloads[cbIdx]++;
    }

    // Determine number of threads to use.
    unsigned numThreadsToUse = std::min(
        static_cast<unsigned>(cbWorkloads.size()), numDatamovementThreads);

    // Not enough CBs to warrant splitting but still need to assign nocIndex on
    // the existing single DM thread before returning failure.
    if (numThreadsToUse <= 1 || cbWorkloads.size() <= 1) {
      bool writesDRAM = llvm::any_of(dmaOps, [](const auto &entry) {
        auto store = mlir::dyn_cast<RemoteStoreOp>(entry.first);
        return store && ttcore::getMemorySpace(store.getMemref()) ==
                            ttcore::MemorySpace::DeviceDRAM;
      });
      int nocIndex = writesDRAM ? 1 : 0;
      generic.setThreadsAttr(rewriter.getArrayAttr(
          {rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement, nullptr,
                                        nocIndex),
           generic.getThreadsAttr().getValue()[1]}));
      return failure();
    }

    // Assign CBs to threads.
    SmallVector<DMAThreadAssignment> assignments =
        assignCBsToThreads(cbWorkloads, numThreadsToUse);

    assignNocIndices(assignments, dmaOps);

    // Create new thread attributes: N datamovement threads + 1 compute thread.
    SmallVector<Attribute> threads;
    for (unsigned i = 0; i < numThreadsToUse; ++i) {
      threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement,
                                                     /*kernelSymbol=*/nullptr,
                                                     assignments[i].nocIndex));
    }
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));

    // Create new generic op with N+1 regions.
    auto newGeneric = rewriter.create<GenericOp>(
        generic.getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(), rewriter.getArrayAttr(threads),
        generic.getScratchInputsAttr(),
        /*numRegions*/ numThreadsToUse + 1);

    // Get the original DM block's argument types.
    SmallVector<Type> argTypes(dmBlock->getArgumentTypes().begin(),
                               dmBlock->getArgumentTypes().end());
    SmallVector<Location> argLocs(argTypes.size(), generic.getLoc());

    // Clone the DM region into each new DM region.
    for (unsigned i = 0; i < numThreadsToUse; ++i) {
      Block *newDMBlock = &newGeneric.getRegion(i).emplaceBlock();
      newDMBlock->addArguments(argTypes, argLocs);

      IRMapping mapping;
      for (unsigned j = 0; j < dmBlock->getNumArguments(); ++j) {
        mapping.map(dmBlock->getArgument(j), newDMBlock->getArgument(j));
      }

      rewriter.setInsertionPointToStart(newDMBlock);
      for (Operation &op : dmBlock->getOperations()) {
        rewriter.clone(op, mapping);
      }

      // Filter to keep only DMA ops for this thread's assigned CBs.
      filterOpsForThread(rewriter, newDMBlock, assignments[i].assignedCBs);
    }

    // Clone the compute region to the new generic (not move, to preserve SSA).
    Region &computeRegion = generic.getRegion(1);
    if (!computeRegion.empty()) {
      Block *computeBlock = &computeRegion.front();
      Block *newComputeBlock = &newGeneric.getRegions().back().emplaceBlock();

      SmallVector<Type> computeArgTypes(
          computeBlock->getArgumentTypes().begin(),
          computeBlock->getArgumentTypes().end());
      SmallVector<Location> computeArgLocs(computeArgTypes.size(),
                                           generic.getLoc());
      newComputeBlock->addArguments(computeArgTypes, computeArgLocs);

      IRMapping computeMapping;
      for (unsigned j = 0; j < computeBlock->getNumArguments(); ++j) {
        computeMapping.map(computeBlock->getArgument(j),
                           newComputeBlock->getArgument(j));
      }

      rewriter.setInsertionPointToStart(newComputeBlock);
      for (Operation &op : computeBlock->getOperations()) {
        rewriter.clone(op, computeMapping);
      }
    }

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }

private:
  unsigned numDatamovementThreads;
};

class D2MScheduleDMA : public impl::D2MScheduleDMABase<D2MScheduleDMA> {
public:
  using impl::D2MScheduleDMABase<D2MScheduleDMA>::D2MScheduleDMABase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    auto systemDesc = moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
        mlir::tt::ttcore::SystemDescAttr::name);
    TT_assert(systemDesc);

    auto chipDesc = systemDesc.getChipDescs().front();
    unsigned numDatamovementThreads = chipDesc.getNumDatamovementThreads();

    // If only 1 DMA thread available, nothing to schedule.
    if (numDatamovementThreads <= 1) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<D2MScheduleDMARewriter>(&getContext(), numDatamovementThreads);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
