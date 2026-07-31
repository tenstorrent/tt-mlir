// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/MapVector.h"

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

    if (auto dmaOp = mlir::dyn_cast<ShardDMAOpInterface>(&op)) {
      dmaOps.push_back({&op, dmaOp.getCBPort()});
    }
  }
}

// Collect all scalar L1 accesses from a block, recursively walking into nested
// scf.for loops.
static void collectScalarL1Accesses(Block *block,
                                    SmallVectorImpl<Operation *> &accesses) {
  for (Operation &op : block->getOperations()) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(&op)) {
      collectScalarL1Accesses(forOp.getBody(), accesses);
      continue;
    }
    if (utils::getScalarL1AccessMemref(&op)) {
      accesses.push_back(&op);
    }
  }
}

// Forward-propagate from `value` to every DMA op that transitively consumes it,
// following op results and loop-carried values.
static void collectDependentDMAOps(Value value, DenseSet<Value> &seen,
                                   DenseSet<Operation *> &dmaOps) {
  if (!seen.insert(value).second) {
    return;
  }
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (mlir::isa<ShardDMAOpInterface>(user)) {
      dmaOps.insert(user);
      continue;
    }
    if (auto loop = mlir::dyn_cast<LoopLikeOpInterface>(user)) {
      // An init operand flows to the matching region iter arg and loop result.
      if (BlockArgument iterArg = loop.getTiedLoopRegionIterArg(&use)) {
        collectDependentDMAOps(iterArg, seen, dmaOps);
      }
      if (OpResult result = loop.getTiedLoopResult(&use)) {
        collectDependentDMAOps(result, seen, dmaOps);
      }
      continue;
    }
    // A yielded value flows out to the loop result and round to the matching
    // iter arg. Without this a value read inside a loop and accumulated across
    // iterations -- the shape a dependent load in a loop takes -- looks like it
    // reaches no transfer at all.
    if (auto yield = mlir::dyn_cast<scf::YieldOp>(user)) {
      if (auto forOp = mlir::dyn_cast<scf::ForOp>(yield->getParentOp())) {
        unsigned idx = use.getOperandNumber();
        collectDependentDMAOps(forOp.getResult(idx), seen, dmaOps);
        collectDependentDMAOps(forOp.getRegionIterArg(idx), seen, dmaOps);
      }
      continue;
    }
    for (Value result : user->getResults()) {
      collectDependentDMAOps(result, seen, dmaOps);
    }
  }
}

// Groups of CBs that must be scheduled onto the same data movement thread.
//
// A scalar L1 read of a CB is a dependent load: the value it produces feeds the
// address of a later transfer. Nothing couples the two -- the reader takes no
// part in the CB wait/pop protocol -- so the only thing that orders the read
// after the transfer that fills the CB is being on the same thread, where
// program order plus that transfer's own barrier applies. So the CB holding the
// data and the CBs of the transfers consuming the loaded value form one
// indivisible scheduling unit.
class CBAffinityGroups {
public:
  void unite(unsigned a, unsigned b) {
    unsigned rootA = find(a);
    unsigned rootB = find(b);
    if (rootA != rootB) {
      parent[rootB] = rootA;
    }
  }

  unsigned find(unsigned port) {
    auto [it, inserted] = parent.try_emplace(port, port);
    if (inserted || it->second == port) {
      return port;
    }
    unsigned root = find(it->second);
    // Path compression. Re-look-up: the recursion may have rehashed the map.
    parent[port] = root;
    return root;
  }

private:
  DenseMap<unsigned, unsigned> parent;
};

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

// A set of CBs that must land on the same thread, with their combined workload.
struct CBGroup {
  SmallVector<unsigned> cbs;
  size_t workload = 0;
};

// Partition the CBs into co-scheduling groups. CBs with no affinity constraint
// each form a singleton group, which is the common case.
//
// Iterates cbWorkloads directly and keys by union-find root in a MapVector, so
// with no affinity constraints the groups come out as singletons in exactly the
// order the CBs were previously enumerated -- keeping the greedy assignment,
// and therefore the NoC choice, unchanged for every existing schedule.
static SmallVector<CBGroup>
buildCBGroups(const DenseMap<unsigned, size_t> &cbWorkloads,
              CBAffinityGroups &affinity) {
  llvm::MapVector<unsigned, CBGroup> groupsByRoot;
  for (const auto &[cbIdx, workload] : cbWorkloads) {
    CBGroup &group = groupsByRoot[affinity.find(cbIdx)];
    group.cbs.push_back(cbIdx);
    group.workload += workload;
  }

  SmallVector<CBGroup> groups;
  for (auto &[root, group] : groupsByRoot) {
    groups.push_back(std::move(group));
  }
  return groups;
}

// Assign CB groups to threads to balance workload.
// Returns a vector of DMAThreadAssignment, one per hardware thread.
static SmallVector<DMAThreadAssignment>
assignCBsToThreads(ArrayRef<CBGroup> groups, unsigned numThreads) {
  SmallVector<DMAThreadAssignment> assignments(numThreads);

  // Sort groups by workload (descending) for greedy assignment.
  SmallVector<const CBGroup *> sortedGroups;
  for (const CBGroup &group : groups) {
    sortedGroups.push_back(&group);
  }
  llvm::sort(sortedGroups, [](const CBGroup *a, const CBGroup *b) {
    return a->workload > b->workload;
  });

  // Greedy assignment: assign each group to the thread with smallest workload.
  for (const CBGroup *group : sortedGroups) {
    // Find thread with minimum workload.
    unsigned minThreadIdx = 0;
    size_t minWorkload = assignments[0].workload;
    for (unsigned i = 1; i < numThreads; ++i) {
      if (assignments[i].workload < minWorkload) {
        minWorkload = assignments[i].workload;
        minThreadIdx = i;
      }
    }

    assignments[minThreadIdx].assignedCBs.insert(group->cbs.begin(),
                                                 group->cbs.end());
    assignments[minThreadIdx].workload += group->workload;
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
  if (std::optional<unsigned> port = getScalarL1AccessPort(op)) {
    return assignedCBs.contains(*port);
  }
  return false;
}

// Recursively erase DMA ops and scalar L1 accesses not assigned to this thread.
// Also removes ops that become dead as a result.
//
// Scalar accesses are filtered alongside DMA ops rather than left to a later
// DCE pass: a scalar *store* has write side effects, so it is not trivially
// dead and would otherwise be replicated into -- and executed by -- every DM
// thread the region was cloned into.
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

      // Check if this is a DMA op or a scalar L1 access.
      if (mlir::isa<ShardDMAOpInterface>(&op) ||
          utils::getScalarL1AccessMemref(&op)) {
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

    // Constrain the schedule so a dependent load and the transfer feeding it
    // share a thread. An access we cannot attribute to a CB port -- a
    // region-local scratch allocation, say -- cannot be filtered per thread and
    // would be replicated into all of them, so refuse to split at all.
    SmallVector<Operation *> scalarAccesses;
    collectScalarL1Accesses(dmBlock, scalarAccesses);

    CBAffinityGroups affinity;
    bool hasUnattributableAccess = false;
    for (Operation *access : scalarAccesses) {
      std::optional<unsigned> port = getScalarL1AccessPort(access);
      if (!port) {
        hasUnattributableAccess = true;
        break;
      }
      if (access->getNumResults() == 0) {
        continue;
      }
      DenseSet<Value> seen;
      DenseSet<Operation *> dependentDMAOps;
      collectDependentDMAOps(access->getResult(0), seen, dependentDMAOps);
      for (Operation *dmaOp : dependentDMAOps) {
        affinity.unite(*port,
                       mlir::cast<ShardDMAOpInterface>(dmaOp).getCBPort());
      }
    }

    SmallVector<CBGroup> cbGroups = buildCBGroups(cbWorkloads, affinity);

    // Determine number of threads to use.
    unsigned numThreadsToUse =
        hasUnattributableAccess
            ? 1
            : std::min(static_cast<unsigned>(cbGroups.size()),
                       numDatamovementThreads);

    // Not enough independent CB groups to warrant splitting but still need to
    // assign a DM core on the existing single DM thread before returning
    // failure.
    if (numThreadsToUse <= 1 || cbGroups.size() <= 1) {
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

    // Assign CBs to threads.
    SmallVector<DMAThreadAssignment> assignments =
        assignCBsToThreads(cbGroups, numThreadsToUse);

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

      // Filter to keep only DMA ops and scalar accesses for this thread's
      // assigned CBs. Port attribution works on the clones directly: the new
      // generic carries the same operand values, and a scalar access names a
      // buffer defined outside the region, which cloning leaves untouched.
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
