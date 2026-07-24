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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

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

struct RemoteMemrefAccess {
  DenseSet<unsigned> loadCBs;
  DenseSet<unsigned> storeCBs;
};

struct CBAffinityGroup {
  DenseSet<unsigned> cbs;
  size_t workload = 0;
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

static bool groupsOverlap(const DenseSet<unsigned> &lhs,
                          const DenseSet<unsigned> &rhs) {
  return llvm::any_of(lhs, [&](unsigned cb) { return rhs.contains(cb); });
}

static void collectRemoteMemrefRoots(Value memref,
                                     SmallVectorImpl<Value> &roots,
                                     llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(memref).second) {
    return;
  }

  Operation *definingOp = memref.getDefiningOp();
  if (auto view = mlir::dyn_cast_if_present<ViewOpInterface>(definingOp)) {
    for (Value input : view.getCompositeInputs()) {
      collectRemoteMemrefRoots(input, roots, visited);
    }
    return;
  }
  if (auto view =
          mlir::dyn_cast_if_present<mlir::ViewLikeOpInterface>(definingOp)) {
    collectRemoteMemrefRoots(view.getViewSource(), roots, visited);
    return;
  }
  if (auto cast = mlir::dyn_cast_if_present<memref::CastOp>(definingOp)) {
    collectRemoteMemrefRoots(cast.getSource(), roots, visited);
    return;
  }

  roots.push_back(memref);
}

static SmallVector<Value> collectRemoteMemrefRoots(Value memref) {
  SmallVector<Value> roots;
  llvm::SmallPtrSet<Value, 4> visited;
  collectRemoteMemrefRoots(memref, roots, visited);
  return roots;
}

// Assign CBs to threads to balance workload. CBs that load from and store to
// the same remote memref stay on one thread so potentially dependent accesses
// are ordered by that thread's NoC barriers.
// Returns a vector of DMAThreadAssignment, one per hardware thread.
static SmallVector<DMAThreadAssignment>
assignCBsToThreads(const DenseMap<unsigned, size_t> &cbWorkloads,
                   ArrayRef<std::pair<Operation *, unsigned>> dmaOps,
                   unsigned numThreads) {
  DenseMap<Value, RemoteMemrefAccess> accesses;
  for (const auto &[op, cbIdx] : dmaOps) {
    if (auto load = mlir::dyn_cast<RemoteLoadOp>(op)) {
      for (Value root : collectRemoteMemrefRoots(load.getMemref())) {
        accesses[root].loadCBs.insert(cbIdx);
      }
    } else if (auto store = mlir::dyn_cast<RemoteStoreOp>(op)) {
      for (Value root : collectRemoteMemrefRoots(store.getMemref())) {
        accesses[root].storeCBs.insert(cbIdx);
      }
    }
  }

  SmallVector<DenseSet<unsigned>> affinitySets;
  for (const auto &[memref, access] : accesses) {
    (void)memref;
    if (access.loadCBs.empty() || access.storeCBs.empty()) {
      continue;
    }

    DenseSet<unsigned> merged(access.loadCBs.begin(), access.loadCBs.end());
    merged.insert(access.storeCBs.begin(), access.storeCBs.end());
    for (size_t i = 0; i < affinitySets.size();) {
      if (!groupsOverlap(merged, affinitySets[i])) {
        ++i;
        continue;
      }
      merged.insert(affinitySets[i].begin(), affinitySets[i].end());
      affinitySets.erase(affinitySets.begin() + i);
    }
    affinitySets.push_back(std::move(merged));
  }

  DenseSet<unsigned> groupedCBs;
  SmallVector<CBAffinityGroup> groups;
  for (DenseSet<unsigned> &cbs : affinitySets) {
    CBAffinityGroup group;
    group.cbs = std::move(cbs);
    for (unsigned cb : group.cbs) {
      group.workload += cbWorkloads.lookup(cb);
      groupedCBs.insert(cb);
    }
    groups.push_back(std::move(group));
  }
  for (const auto &[cbIdx, workload] : cbWorkloads) {
    if (!groupedCBs.contains(cbIdx)) {
      groups.push_back({DenseSet<unsigned>{cbIdx}, workload});
    }
  }

  llvm::sort(groups,
             [](const CBAffinityGroup &lhs, const CBAffinityGroup &rhs) {
               return lhs.workload > rhs.workload;
             });

  SmallVector<DMAThreadAssignment> assignments(
      std::min(numThreads, static_cast<unsigned>(groups.size())));
  // Greedy assignment: assign each affinity group to the least-loaded thread.
  for (const CBAffinityGroup &group : groups) {
    // Find thread with minimum workload.
    unsigned minThreadIdx = 0;
    size_t minWorkload = assignments[0].workload;
    for (unsigned i = 1; i < assignments.size(); ++i) {
      if (assignments[i].workload < minWorkload) {
        minWorkload = assignments[i].workload;
        minThreadIdx = i;
      }
    }

    assignments[minThreadIdx].assignedCBs.insert(group.cbs.begin(),
                                                 group.cbs.end());
    assignments[minThreadIdx].workload += group.workload;
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

    // Assign CBs to threads. Read/write affinity groups may reduce the number
    // of independent work units below the number of physical DMA cores.
    SmallVector<DMAThreadAssignment> assignments =
        assignCBsToThreads(cbWorkloads, dmaOps, numDatamovementThreads);
    unsigned numThreadsToUse = assignments.size();

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
      return failure();
    }

    assignDmCoreIndices(assignments, dmaOps, numDatamovementThreads);

    // Create new thread attributes: N datamovement threads + 1 compute thread.
    SmallVector<Attribute> threads;
    for (unsigned i = 0; i < numThreadsToUse; ++i) {
      threads.push_back(rewriter.getAttr<ThreadAttr>(
          ThreadType::Datamovement,
          /*kernelSymbol=*/nullptr, assignments[i].dmCoreIndex));
    }
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));

    // Create new generic op with N+1 regions.
    auto newGeneric = rewriter.create<GenericOp>(
        generic.getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(), rewriter.getArrayAttr(threads),
        generic.getFabricConnectionConfigAttr(),
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
