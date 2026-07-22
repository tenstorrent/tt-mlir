// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

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
  // Set of CB indices (generic operand indices) assigned to this thread.
  DenseSet<unsigned> assignedCBs;

  // Estimated workload for this thread (number of DMA ops).
  size_t workload = 0;

  // Assigned hardware DM core.
  // For WH/BH: 1 = DRAM reader, 0 = DRAM writer.
  int32_t dmCoreIndex = -1;
};

// Collect all DMA ops from a block, recursively walking into nested scf.for
// loops.
static void
collectDMAOps(Block *block,
              SmallVectorImpl<std::pair<Operation *, unsigned>> &dmaOps) {
  for (Operation &op : block->getOperations()) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
      collectDMAOps(forOp.getBody(), dmaOps);
      continue;
    }
    if (auto ifOp = mlir::dyn_cast<scf::IfOp>(&op)) {
      collectDMAOps(ifOp.thenBlock(), dmaOps);
      if (Block *elseBlock = ifOp.elseBlock()) {
        collectDMAOps(elseBlock, dmaOps);
      }
      continue;
    }

    if (auto dmaOp = mlir::dyn_cast<ShardDMAOpInterface>(&op)) {
      dmaOps.push_back({&op, dmaOp.getCBPort()});
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

// Wormhole/Blackhole have 2 DMs and 2 NoCs. After CBs have already been
// load-balanced across two threads, choose which thread should use NoC0 versus
// NoC1, then store the DM core index that maps to that NoC (see
// ttcore::getDmCoreDefaultNoc for the canonical convention).
static void assignNoCsToThreads(
    SmallVectorImpl<DMAThreadAssignment> &assignments,
    const SmallVectorImpl<std::pair<Operation *, unsigned>> &dmaOps) {
  TT_assertv(assignments.size() == 2u, "Expect exactly 2 DM threads");

  auto deviceAttr = ttcore::lookupDevice(dmaOps.front().first);

  NocScore scores[2];
  for (const auto &[op, cbIdx] : dmaOps) {
    for (int t = 0; t < 2; ++t) {
      if (!assignments[t].assignedCBs.contains(cbIdx)) {
        continue;
      }

      if (auto load = mlir::dyn_cast_or_null<RemoteLoadOp>(op)) {
        if (ttcore::getMemorySpace(load.getMemref()) !=
            ttcore::MemorySpace::DeviceDRAM) {
          continue;
        }
        if (load.isMcast()) {
          TT_assertv(scores[t].mcastShardSize == 0,
                     "There can only be one mcast load per thread.");
          auto layout = ttcore::getDeviceLayout(load.getMemref());
          if (layout) {
            auto memrefType =
                mlir::cast<MemRefType>(load.getMemref().getType());
            scores[t].mcastShardSize =
                deviceAttr.getShardSizeInBytes(memrefType, 1, false);
          }
        } else {
          scores[t].unicastBias += 2;
        }
      } else if (auto store = mlir::dyn_cast_or_null<RemoteStoreOp>(op)) {
        if (ttcore::getMemorySpace(store.getMemref()) ==
            ttcore::MemorySpace::DeviceDRAM) {
          scores[t].unicastBias -= 1;
        }
      }
    }
  }
  bool swapNocs = scores[1] > scores[0];
  ttcore::NocIndex thread0Noc =
      swapNocs ? ttcore::NocIndex::Noc1 : ttcore::NocIndex::Noc0;
  ttcore::NocIndex thread1Noc =
      swapNocs ? ttcore::NocIndex::Noc0 : ttcore::NocIndex::Noc1;
  // Inverse of ttcore::getDmCoreDefaultNoc for WH/BH.
  assignments[0].dmCoreIndex = thread0Noc == ttcore::NocIndex::Noc0 ? 1 : 0;
  assignments[1].dmCoreIndex = thread1Noc == ttcore::NocIndex::Noc0 ? 1 : 0;
}

// There is no NoC choice to make, use the thread index as the DM core index.
static void assignDmCoreIndicesForSingleNoC(
    SmallVectorImpl<DMAThreadAssignment> &assignments) {
  for (size_t index = 0; index < assignments.size(); ++index) {
    assignments[index].dmCoreIndex = static_cast<int32_t>(index);
  }
}

static void assignDmCoreIndices(
    SmallVectorImpl<DMAThreadAssignment> &assignments,
    const SmallVectorImpl<std::pair<Operation *, unsigned>> &dmaOps,
    unsigned numDatamovementThreads) {
  if (numDatamovementThreads == 2) {
    // WH/BH case. 2 DMs and 2 NoCs, so need to assign each NoC to a thread.
    assignNoCsToThreads(assignments, dmaOps);
    return;
  }

  TT_assertv(numDatamovementThreads == 6u, "Expect 6 DM cores");
  assignDmCoreIndicesForSingleNoC(assignments);
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
  if (auto dmaOp = mlir::dyn_cast<ShardDMAOpInterface>(op)) {
    return assignedCBs.contains(dmaOp.getCBPort());
  }
  return false;
}

// Recursively erase DMA ops not assigned to this thread.
// Also removes ops that become dead as a result.
//
// `keepBarrier` designates this thread as the owner of the CCL start barrier
// (DeviceSynchronizeOp). The barrier has no CB, so it would otherwise survive
// the clone-into-every-DM-thread and run once per thread -- each replica emits
// a fabric sem increment + wait, so a core with N DM threads sends N barrier
// increments to its peers, overshooting the expected count and deadlocking.
// The barrier must run exactly once per core, on the thread that owns the
// cross-device store it guards; erase it from every other DM thread.
static void filterOpsForThread(PatternRewriter &rewriter, Block *block,
                               const DenseSet<unsigned> &assignedCBs,
                               bool keepBarrier) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;

    for (Operation &op : block->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      // Recurse into nested loops / conditionals.
      if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
        filterOpsForThread(rewriter, forOp.getBody(), assignedCBs, keepBarrier);
        continue;
      }
      if (auto ifOp = mlir::dyn_cast<scf::IfOp>(&op)) {
        filterOpsForThread(rewriter, ifOp.thenBlock(), assignedCBs,
                           keepBarrier);
        if (Block *elseBlock = ifOp.elseBlock()) {
          filterOpsForThread(rewriter, elseBlock, assignedCBs, keepBarrier);
        }
        continue;
      }

      // The CCL start barrier belongs to a single DM thread (see above).
      if (mlir::isa<DeviceSynchronizeOp>(&op)) {
        if (!keepBarrier && op.use_empty()) {
          toErase.push_back(&op);
          changed = true;
        }
        continue;
      }

      // Check if this is a DMA op.
      if (mlir::isa<ShardDMAOpInterface>(&op)) {
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

    if (failed(utils::checkForIllegalSemaphoreOps(dmBlock))) {
      return failure();
    }

    // Option D (temporary): a kernel with explicit semaphore mutations
    // (semaphore_inc / semaphore_set) is kept on a SINGLE datamovement thread.
    // Splitting the DM region across NOC processors clones every op into each
    // thread, so a mutation would run once per thread -- a race / wrong count
    // on the shared semaphore. Keeping one DM thread makes the mutation run
    // exactly once. This sidesteps the general problem (pin each semaphore op
    // to one DM thread + local-barrier the observers) until that lands; see the
    // TODO in DMAUtils.cpp's checkForIllegalSemaphoreOps and
    // tools/d2m-jit/unified_semaphore_design.md.
    // core_read (a core->core NoC read) is also kept single-DM-thread for now:
    // it is not a ShardDMAOpInterface so it would not be assigned/filtered by a
    // NOC split, and the streaming sync against it is not yet proven.
    bool keepSingleDMThread = false;
    dmBlock->walk([&](Operation *op) {
      if (isa<SemaphoreIncOp, SemaphoreSetOp, CoreReadOp, CoreWriteOp>(op)) {
        keepSingleDMThread = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // Multiple cross-device fabric sends must stay on a single DM thread. Each
    // NOC thread that issues a fabric op creates and tears down its own
    // fabric_connection_manager (setup_fabric_connections /
    // close_fabric_connections); two managers racing the same link on one core
    // deadlock. A *single* fabric send is safe on whatever thread it lands on,
    // and pinning is actively harmful there -- a fused matmul+all_gather needs
    // the matmul to split its reader/writer across DM threads (only the
    // device_synchronize barrier is pinned, below), so collapsing it to one
    // thread breaks the matmul CB handshake. Only pin when >1 fabric send (a
    // cross-device remote_store, i.e. with a startDevice mcast range) would
    // otherwise be distributed across threads -- e.g. a per-tile streaming
    // all_gather.
    unsigned numFabricSends = 0;
    dmBlock->walk([&](RemoteStoreOp store) {
      if (!store.getStartDevice().empty()) {
        ++numFabricSends;
      }
    });
    if (numFabricSends > 1) {
      keepSingleDMThread = true;
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

    // Not enough CBs to warrant splitting but still need to assign a DM core on
    // the existing single DM thread before returning failure.
    if (numThreadsToUse <= 1 || cbWorkloads.size() <= 1) {
      bool writesDRAM = llvm::any_of(dmaOps, [](const auto &entry) {
        auto store = mlir::dyn_cast_or_null<RemoteStoreOp>(entry.first);
        return store && ttcore::getMemorySpace(store.getMemref()) ==
                            ttcore::MemorySpace::DeviceDRAM;
      });
      int32_t dmCoreIndex;
      if (numDatamovementThreads == 2) {
        dmCoreIndex = writesDRAM ? 0 : 1;
      } else {
        dmCoreIndex = 0;
      }
      generic.setThreadsAttr(rewriter.getArrayAttr({
          rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement, nullptr,
                                       dmCoreIndex),
          generic.getThreadsAttr().getValue()[1],
      }));
      // The single DM thread owns any fabric send; pin the fabric config to its
      // DM core.
      if (auto fabricConfig = generic.getFabricConnectionConfigAttr()) {
        generic.setFabricConnectionConfigAttr(fabricConfig.getWithThread(
            rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement, nullptr,
                                         dmCoreIndex)));
      }
      return failure();
    }

    // Assign CBs to threads.
    SmallVector<DMAThreadAssignment> assignments =
        assignCBsToThreads(cbWorkloads, numThreadsToUse);

    assignDmCoreIndices(assignments, dmaOps, numDatamovementThreads);

    // The CCL start barrier (device_synchronize) has no CB and must run exactly
    // once per core. Pin it to the thread that owns the cross-device store it
    // guards (the fabric send); fall back to thread 0 if there is no fabric
    // store. filterOpsForThread erases the barrier from every other DM thread.
    unsigned barrierOwnerIdx = 0;
    for (const auto &[op, cbIdx] : dmaOps) {
      auto store = mlir::dyn_cast<RemoteStoreOp>(op);
      if (store && !store.getStartDevice().empty()) {
        for (unsigned i = 0; i < numThreadsToUse; ++i) {
          if (assignments[i].assignedCBs.contains(cbIdx)) {
            barrierOwnerIdx = i;
            break;
          }
        }
        break;
      }
    }

    // Create new thread attributes: N datamovement threads + 1 compute thread.
    SmallVector<Attribute> threads;
    for (unsigned i = 0; i < numThreadsToUse; ++i) {
      threads.push_back(rewriter.getAttr<ThreadAttr>(
          ThreadType::Datamovement,
          /*kernelSymbol=*/nullptr, assignments[i].dmCoreIndex));
    }
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));

    // Pin the fabric config to the DM core of the thread that owns the fabric
    // send (barrierOwnerIdx); that is the thread that issues the fabric ops.
    auto fabricConfig = generic.getFabricConnectionConfigAttr();
    if (fabricConfig) {
      fabricConfig = fabricConfig.getWithThread(rewriter.getAttr<ThreadAttr>(
          ThreadType::Datamovement, /*kernelSymbol=*/nullptr,
          assignments[barrierOwnerIdx].dmCoreIndex));
    }

    // Create new generic op with N+1 regions.
    auto newGeneric = rewriter.create<GenericOp>(
        generic.getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(), rewriter.getArrayAttr(threads),
        fabricConfig,
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

      // Filter to keep only DMA ops for this thread's assigned CBs (and the
      // start barrier only on its designated owner thread).
      filterOpsForThread(rewriter, newDMBlock, assignments[i].assignedCBs,
                         /*keepBarrier=*/i == barrierOwnerIdx);
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
    const unsigned numDatamovementThreads =
        numDmCores != 0 ? numDmCores : chipDesc.getNumDatamovementThreads();

    // If only 1 DMA thread available, nothing to schedule.
    if (numDatamovementThreads == 1) {
      return;
    }
    if (numDatamovementThreads != 2 && numDatamovementThreads != 6) {
      moduleOp.emitError("d2m-schedule-dma only supports 2 or 6 DM cores");
      signalPassFailure();
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
