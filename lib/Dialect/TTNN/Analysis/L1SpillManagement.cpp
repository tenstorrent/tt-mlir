// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cstdint>
#include <queue>

namespace mlir::tt::ttnn {

// Safety margin for the CB fragmentation check, expressed as a fraction of
// the L1 budget per core.  The compile-time address simulator cannot model
// transient internal allocations that ops (e.g. matmul, SDPA) make and
// release during program execution.  These "ghost" allocations fragment the
// runtime free list, pushing subsequent tensor allocations to lower
// addresses than the simulator predicts.  The cushion accounts for this
// gap so that the CB region (growing bottom-up) never overlaps with tensor
// buffers at runtime.
static constexpr double kCBFragCushionFraction = 0.10;

//===----------------------------------------------------------------------===//
// SumL1MemoryTracker
//===----------------------------------------------------------------------===//

op_constraint_validation::ValidationResult
SumL1MemoryTracker::validate(Operation *op,
                             llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                             const OpConfig &config) const {
  // Subtract L1 sizes of input tensors that are already tracked in the live
  // set. OpModel's overallPeakL1Usage already accounts for input buffers, so
  // including them in additionalL1Usage would double-count.
  uint64_t inputOverlap = 0;
  llvm::DenseSet<Value> seen;
  for (auto operand : op->getOperands()) {
    if (!seen.insert(operand).second) {
      continue;
    }
    auto it = tensorSizes.find(operand);
    if (it != tensorSizes.end()) {
      inputOverlap += it->second;
    }
  }
  uint64_t additionalL1 =
      currentOccupied > inputOverlap ? currentOccupied - inputOverlap : 0;
  return validateBackendDirect(op, inputLayouts, config, additionalL1);
}

op_constraint_validation::ValidationResult
SumL1MemoryTracker::validateBackendDirect(
    Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
    const OpConfig &config, uint64_t additionalL1Usage) const {
  if (backendValidator) {
    return backendValidator(op, inputLayouts, config, additionalL1Usage);
  }
  return op_constraint_validation::validateOperation(op, inputLayouts, config,
                                                     additionalL1Usage);
}

uint64_t SumL1MemoryTracker::getOccupiedL1() const { return currentOccupied; }

void SumL1MemoryTracker::init(uint64_t l1BudgetPerCore) {
  l1Budget = l1BudgetPerCore;
  currentOccupied = 0;
  tensorSizes.clear();
  freeList.clear();
  freeList.push_back({0, l1Budget});
  tensorAddresses.clear();
  aliasGroups.clear();
}

SumL1MemoryTracker::Snapshot SumL1MemoryTracker::takeSnapshot() const {
  return {freeList, tensorAddresses, aliasGroups, currentOccupied};
}

void SumL1MemoryTracker::restoreSnapshot(const Snapshot &snapshot) {
  freeList = snapshot.freeList;
  tensorAddresses = snapshot.tensorAddresses;
  aliasGroups = snapshot.aliasGroups;
  currentOccupied = snapshot.currentOccupied;
}

void SumL1MemoryTracker::allocateAddress(Value result, uint64_t l1SizePerCore) {
  if (l1SizePerCore == 0 || l1Budget == 0) {
    return;
  }
  uint64_t alignedSize =
      (l1SizePerCore + kL1Alignment - 1) & ~(kL1Alignment - 1);

  // Walk free list from highest address (reverse) for top-down first-fit.
  for (int i = static_cast<int>(freeList.size()) - 1; i >= 0; --i) {
    if (freeList[i].size() >= alignedSize) {
      // Allocate from the top of this block.
      uint64_t allocStart = freeList[i].end - alignedSize;
      tensorAddresses[result] = {allocStart, alignedSize};
      // First allocation in this slot: open the alias group at refcount 1
      // and account the buffer once. Aliases joining via allocateAddressAt
      // do not re-bump currentOccupied.
      aliasGroups[allocStart] = {/*count=*/1, /*rawSize=*/l1SizePerCore};
      currentOccupied += l1SizePerCore;

      // Shrink or remove the free block.
      if (freeList[i].size() == alignedSize) {
        freeList.erase(freeList.begin() + i);
      } else {
        freeList[i].end = allocStart;
      }
      return;
    }
  }
  // No fit — log warning. Sum tracker still works; frag check will catch it.
  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "Address simulator: no fit for {} bytes (aligned {})",
               l1SizePerCore, alignedSize);
}

void SumL1MemoryTracker::addTensor(Value result, uint64_t l1SizePerCore) {
  tensorSizes[result] = l1SizePerCore;
  allocateAddress(result, l1SizePerCore);
}

void SumL1MemoryTracker::allocateAddressAt(Value result, Value srcAtSameAddr) {
  auto srcIt = tensorAddresses.find(srcAtSameAddr);
  assert(srcIt != tensorAddresses.end() &&
         "allocateAddressAt: src must already be in tensorAddresses");
  // Snapshot the slot before inserting `result`, since the insert may
  // rehash and invalidate `srcIt`.
  auto slot = srcIt->second;
  tensorAddresses[result] = slot;
  auto groupIt = aliasGroups.find(slot.first);
  assert(groupIt != aliasGroups.end() &&
         "allocateAddressAt: aliasGroups entry missing for src's slot");
  ++groupIt->second.count;
}

void SumL1MemoryTracker::addTensorAtAddress(Value result,
                                            uint64_t l1SizePerCore,
                                            Value srcAtSameAddr) {
  allocateAddressAt(result, srcAtSameAddr);
  tensorSizes[result] = l1SizePerCore;
}

void SumL1MemoryTracker::freeAddress(Value result) {
  auto addrIt = tensorAddresses.find(result);
  if (addrIt == tensorAddresses.end()) {
    return;
  }
  auto [freedStart, freedSize] = addrIt->second;
  uint64_t freedEnd = freedStart + freedSize;
  tensorAddresses.erase(addrIt);

  // Last alias of this slot reclaims the buffer; earlier ones just
  // detach from the alias group.
  auto groupIt = aliasGroups.find(freedStart);
  assert(groupIt != aliasGroups.end() &&
         "freeAddress: aliasGroups entry missing for live slot");
  if (--groupIt->second.count > 0) {
    return;
  }
  currentOccupied -= groupIt->second.rawSize;
  aliasGroups.erase(groupIt);

  // Find insertion point in sorted freeList (sorted by start address).
  size_t insertIdx = 0;
  while (insertIdx < freeList.size() &&
         freeList[insertIdx].start < freedStart) {
    ++insertIdx;
  }

  // Check merge with previous and next free blocks.
  bool mergePrev = insertIdx > 0 && freeList[insertIdx - 1].end == freedStart;
  bool mergeNext =
      insertIdx < freeList.size() && freeList[insertIdx].start == freedEnd;

  if (mergePrev && mergeNext) {
    freeList[insertIdx - 1].end = freeList[insertIdx].end;
    freeList.erase(freeList.begin() + insertIdx);
  } else if (mergePrev) {
    freeList[insertIdx - 1].end = freedEnd;
  } else if (mergeNext) {
    freeList[insertIdx].start = freedStart;
  } else {
    freeList.insert(freeList.begin() + insertIdx, {freedStart, freedEnd});
  }
}

void SumL1MemoryTracker::logState() const {
  for ([[maybe_unused]] const auto &entry : tensorAddresses) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "Tensor {}: address {} - {} (size {})", entry.first,
                 entry.second.first, entry.second.first + entry.second.second,
                 entry.second.second);
  }

  for ([[maybe_unused]] const auto &entry : freeList) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "Free block: address {} - {} (size {})", entry.start,
                 entry.end, entry.size());
  }
}

void SumL1MemoryTracker::removeTensorFromSizes(Value result) {
  // Size-table-only — used by eviction, which then `restoreSnapshot +
  // replay` rebuilds currentOccupied. Address-side path is `freeAddress`.
  tensorSizes.erase(result);
}

void SumL1MemoryTracker::removeTensor(Value result) {
  removeTensorFromSizes(result);
  freeAddress(result);
}

bool SumL1MemoryTracker::hasTensor(Value result) const {
  return tensorSizes.count(result);
}

bool SumL1MemoryTracker::hasTensorAddress(Value result) const {
  return tensorAddresses.count(result);
}

uint64_t SumL1MemoryTracker::getTensorSize(Value result) const {
  auto it = tensorSizes.find(result);
  return it != tensorSizes.end() ? it->second : 0;
}

uint64_t SumL1MemoryTracker::getLowestOccupiedAddress() const {
  if (tensorAddresses.empty()) {
    return l1Budget;
  }
  uint64_t lowest = l1Budget;
  for (const auto &entry : tensorAddresses) {
    lowest = std::min(lowest, entry.second.first);
  }
  return lowest;
}

std::optional<uint64_t>
SumL1MemoryTracker::wouldAllocateAt(uint64_t l1SizePerCore) const {
  if (l1SizePerCore == 0 || l1Budget == 0) {
    return l1Budget; // Trivially fits at the top.
  }
  uint64_t alignedSize =
      (l1SizePerCore + kL1Alignment - 1) & ~(kL1Alignment - 1);
  // Walk free list from highest address (reverse) for top-down first-fit.
  for (int i = static_cast<int>(freeList.size()) - 1; i >= 0; --i) {
    if (freeList[i].size() >= alignedSize) {
      return freeList[i].end - alignedSize;
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// L1SpillManagement
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
L1SpillManagement<MemoryTracker>::L1SpillManagement(
    func::FuncOp func, ttcore::GridAttr deviceGrid, uint64_t l1BudgetPerCore,
    std::unique_ptr<L1SpillObserver> observer)
    : func(func), deviceGrid(deviceGrid), l1BudgetPerCore(l1BudgetPerCore),
      cbFragCushion(
          static_cast<uint64_t>(kCBFragCushionFraction * l1BudgetPerCore)) {
  if (observer) {
    observer_ = std::move(observer);
  } else {
    observer_ = std::make_unique<L1SpillObserver>();
  }
  memoryTracker.init(l1BudgetPerCore);
}

//===----------------------------------------------------------------------===//
// extractOpConfigFromIR
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
OpConfig
L1SpillManagement<MemoryTracker>::extractOpConfigFromIR(Operation *op) {
  if (op->getNumResults() == 0) {
    return OpConfig{};
  }
  auto tensorType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  auto layout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
  OpConfig config(layout);

  llvm::TypeSwitch<Operation *>(op)
      .Case<Conv2dOp>([&](auto convOp) {
        Conv2dAttrs attrs;
        attrs.conv2dConfig = convOp.getConv2dConfig();
        attrs.deviceComputeKernelConfig = convOp.getComputeConfig();
        config.opSpecificAttrs = std::move(attrs);
      })
      .template Case<ConvTranspose2dOp>([&](auto convOp) {
        Conv2dAttrs attrs;
        attrs.conv2dConfig = convOp.getConv2dConfig();
        config.opSpecificAttrs = std::move(attrs);
      })
      .template Case<MatmulOp, LinearOp>([&](auto matmulOp) {
        MatmulAttrs attrs;
        attrs.matmulProgramConfig = matmulOp.getMatmulProgramConfig();
        attrs.computeKernelConfig = matmulOp.getComputeConfig();
        config.opSpecificAttrs = std::move(attrs);
      })
      .Default([&](Operation *defaultOp) {
        // Drop the output layout hint for multi-output ops. This is needed
        // because those ops don't have 1:1 relationship between memory config
        // and outputs, which can create weird bugs in op validation. It is
        // safest to ignore the hint in this case.
        if (llvm::count_if(defaultOp->getResults(), [](Value r) {
              return mlir::isa<RankedTensorType>(r.getType());
            }) > 1) {
          config.outputLayout = TTNNLayoutAttr{};
        }
      });

  return config;
}

//===----------------------------------------------------------------------===//
// evictFarthestUse
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
Value L1SpillManagement<MemoryTracker>::evictFarthestUse() {
  while (!liveSet.empty()) {
    auto [lastUse, candidateVal] = liveSet.top();
    liveSet.pop();

    // Skip already-evicted entries (lazy deletion).
    if (!liveValues.count(candidateVal)) {
      continue;
    }

    // Skip reshards we inserted for future consumers — evicting them defeats
    // their purpose.
    if (insertedReshardValues.count(candidateVal)) {
      continue;
    }

    liveValues.erase(candidateVal);
    return candidateVal;
  }
  return Value();
}

//===----------------------------------------------------------------------===//
// applyOutputConfig
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::applyOutputConfig(
    Operation *op, const op_constraint_validation::ValidationResult &result) {
  TTNNLayoutAttr chosenLayout = result.getFirstActualOutputLayout();
  if (!chosenLayout) {
    return;
  }

  // Update all tensor results with their respective layouts.
  for (auto opResult : op->getResults()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(opResult.getType());
    if (!tensorType) {
      continue;
    }

    size_t ri = opResult.getResultNumber();
    assert(ri < result.actualOutputLayouts.size() &&
           "validation result has fewer output layouts than tensor results");
    TTNNLayoutAttr resultLayout = result.actualOutputLayouts[ri];
    assert(resultLayout && "result layout is null for tensor result");

    llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

    // Preserve quantized element types.
    Type originalElementType = tensorType.getElementType();
    Type newElementType = originalElementType;
    if (!mlir::isa<mlir::quant::QuantizedType>(originalElementType)) {
      newElementType = resultLayout.getScalarElementType();
    }

    RankedTensorType newTensorType =
        RankedTensorType::get(tensorShape, newElementType, resultLayout);
    opResult.setType(newTensorType);
  }
}

//===----------------------------------------------------------------------===//
// collectDownstreamConsumers
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
llvm::SmallVector<Operation *>
L1SpillManagement<MemoryTracker>::collectDownstreamConsumers(
    Operation *changed) {
  // After spillToDram, a result may have a ToMemoryConfigOp user (spill op).
  // Follow through spill ops to find the actual downstream consumers.
  llvm::SmallVector<Operation *> consumers;
  for (auto changedResult : changed->getResults()) {
    for (Operation *user : changedResult.getUsers()) {
      if (isa<ToMemoryConfigOp>(user)) {
        for (Operation *consumer : user->getResult(0).getUsers()) {
          consumers.push_back(consumer);
        }
      } else {
        consumers.push_back(user);
      }
    }
  }
  return consumers;
}

//===----------------------------------------------------------------------===//
// revalidateConsumers
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::revalidateConsumers(
    Operation *changedOp, int64_t currentPos,
    const llvm::DenseMap<Operation *, int64_t> &positionMap) {
  // Worklist of ops whose output changed — seed with the victim/changed op.
  llvm::SmallVector<Operation *> worklist;
  worklist.push_back(changedOp);
  llvm::DenseSet<Operation *> visited;

  while (!worklist.empty()) {
    Operation *changed = worklist.pop_back_val();
    if (!visited.insert(changed).second) {
      continue;
    }

    llvm::SmallVector<Operation *> consumers =
        collectDownstreamConsumers(changed);

    for (Operation *consumer : consumers) {
      // Only revalidate already-processed ops (before current position).
      auto posIt = positionMap.find(consumer);
      if (posIt == positionMap.end() || posIt->second >= currentPos) {
        continue; // Main loop will handle this op.
      }

      // Skip non-validated ops.
      if (!mlir::dyn_cast<OpModel>(consumer)) {
        continue;
      }
      if (isa<ToLayoutOp>(consumer)) {
        continue;
      }

      auto inputLayouts = utils::extractInputLayouts(consumer);
      auto config = extractOpConfigFromIR(consumer);
      auto result = memoryTracker.validate(consumer, inputLayouts, config);

      bool outputChanged =
          result.isSuccess() &&
          result.getFirstActualOutputLayout() != config.outputLayout;
      observer_->onRevalidationCascade(changed, consumer, outputChanged);

      if (outputChanged) {
        // Backend returned a different output layout for this consumer.
        // Update consumer's IR to match.
        applyOutputConfig(consumer, result);
        // Update memory tracker for any live results of this consumer.
        // Even split approximation — see addResultsToLiveSet in run().
        size_t numTensorResults = 0;
        for (auto r : consumer->getResults()) {
          if (mlir::isa<RankedTensorType>(r.getType())) {
            ++numTensorResults;
          }
        }
        uint64_t perResultL1 =
            numTensorResults > 0 ? result.outputL1Usage / numTensorResults : 0;
        for (auto r : consumer->getResults()) {
          if (memoryTracker.hasTensor(r)) {
            memoryTracker.removeTensor(r);
            memoryTracker.addTensor(r, perResultL1);
          }
        }

        TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                     "  REVALIDATE: consumer {0} output layout changed "
                     "-- cascading to its consumers",
                     ttmlir::opToString(consumer));

        // Consumer's output changed -- cascade to its consumers.
        worklist.push_back(consumer);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// buildScheduleData
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
typename L1SpillManagement<MemoryTracker>::ScheduleData
L1SpillManagement<MemoryTracker>::buildScheduleData() {
  ScheduleData data;

  // Build schedule (ops in IR order = topological order). Include sink ops
  // (paged_fill_cache / paged_update_cache / fill_cache) even though they
  // have no tensor result — their operand uses must be visible to
  // computeLastUsePositions so that values consumed only by a cache write
  // (e.g. per-layer KV typecasts) are kept alive in L1 until the cache
  // write actually executes, not freed at the layer-local to_memory_config.
  func->walk([&](Operation *op) {
    if (!optimizer_utils::isBeamSearchTarget(op)) {
      return;
    }
    data.schedule.push_back(op);
  });

  // Compute per-result last-use positions.
  data.lastUsePositions = computeLastUsePositions(data.schedule);

  // Build death schedule: position -> results whose last use is at that
  // position.
  for (auto &[val, lastUse] : data.lastUsePositions) {
    data.deathSchedule[lastUse].push_back(val);
  }

  // Build position map for revalidateConsumers.
  for (int64_t i = 0; i < static_cast<int64_t>(data.schedule.size()); ++i) {
    data.positionMap[data.schedule[i]] = i;
  }

  return data;
}

//===----------------------------------------------------------------------===//
// processDeadTensors
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::processDeadTensors(
    int64_t pos, const ScheduleData &data) {
  auto it = data.deathSchedule.find(pos - 1);
  if (it == data.deathSchedule.end()) {
    return;
  }
  for (Value deadVal : it->second) {
    if (liveValues.erase(deadVal)) {
      l1EventLog.push_back({L1Event::kDealloc, deadVal, 0, /*skipped=*/false});
      memoryTracker.removeTensor(deadVal);
      Operation *deadOp = deadVal.getDefiningOp();
      observer_->onDeadRemoval(deadOp, pos, memoryTracker.getOccupiedL1());
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "  [pos={0}] DEAD: {1}, L1 now {2}/{3}", pos,
                   ttmlir::opToString(deadOp), memoryTracker.getOccupiedL1(),
                   l1BudgetPerCore);
    }
  }
}

//===----------------------------------------------------------------------===//
// ensureFitsL1
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::willAliasSourceInL1(
    Operation *op) const {
  // canReshapeBeView already guarantees op is a ReshapeOp with operand(0).
  // Use hasTensorAddress (not hasTensor): aliasing calls allocateAddressAt,
  // which requires the source to occupy a simulated address slot. A tensor
  // can be size-tracked but not address-tracked (e.g. zero-size or no-fit),
  // in which case it cannot be aliased. This matches the replay path.
  return canReshapeBeView(op) &&
         memoryTracker.hasTensorAddress(op->getOperand(0));
}

template <typename MemoryTracker>
uint64_t L1SpillManagement<MemoryTracker>::ensureFitsL1(Operation *op,
                                                        int64_t pos,
                                                        ScheduleData &data,
                                                        uint64_t cbPeakUsage,
                                                        uint64_t l1Size) {
  // A view-eligible reshape aliases its source's existing L1 slot
  // (addResultsToLiveSet uses addTensorAtAddress), so it consumes no fresh
  // L1. Skip the fit / CB-overlap checks that assume a new allocation —
  // otherwise wouldAllocateAt(l1Size) can falsely report no-fit and evict
  // the very source the reshape is about to alias.
  if (willAliasSourceInL1(op)) {
    return l1Size;
  }

  auto speculativeAddr = memoryTracker.wouldAllocateAt(l1Size);
  if (!speculativeAddr) {
    l1Size = handleNoFit(op, pos, data, l1Size);
    speculativeAddr = memoryTracker.wouldAllocateAt(l1Size);
  }
  // Always run the overlap check when the op declares a non-zero CB peak;
  // pass UINT64_MAX as the "no L1 output" sentinel so wouldCBsOverlapTensors
  // compares cushionedCBUsage strictly against the lowest-existing tensor
  // address.
  uint64_t addrForCheck =
      speculativeAddr.value_or(std::numeric_limits<uint64_t>::max());
  if (cbPeakUsage > 0 &&
      wouldCBsOverlapTensors(op, pos, cbPeakUsage, addrForCheck)) {
    l1Size = handleFragmentation(op, pos, data, cbPeakUsage, l1Size);
  }
  return l1Size;
}

//===----------------------------------------------------------------------===//
// handleOOM
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::handleOOM(
    Operation *op, int64_t pos, llvm::ArrayRef<OpResult> tensorResults,
    ScheduleData &data, std::function<void(uint64_t)> addResultsToLiveSet) {
  observer_->onOOM(op, pos, memoryTracker.getOccupiedL1());
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    OOM: validation failed, trying demotion/eviction");

  // Evict from live set (farthest last-use first) until validation succeeds.
  // Uses evictUntil which inserts reshards for already-processed consumers
  // instead of cascade revalidation, preserving downstream sharded layouts.
  // handleOOM is only called when validation already failed — skip the initial
  // redundant validate and go straight to eviction.
  auto inputLayouts = utils::extractInputLayouts(op);
  auto config = extractOpConfigFromIR(op);
  auto result =
      op_constraint_validation::ValidationResult::outOfMemoryError("");
  evictUntil(pos, data, [&]() {
    inputLayouts = utils::extractInputLayouts(op);
    result = memoryTracker.validate(op, inputLayouts, config);
    return result.isSuccess();
  });

  if (result.isSuccess()) {
    uint64_t l1Size = result.outputL1Usage;
    if (l1Size > 0) {
      l1Size = ensureFitsL1(op, pos, data, result.cbPeakUsage, l1Size);
    } else {
      // DRAM-output op: validate's byte-budget check accounts for CB usage in
      // total but not for CB-vs-tensor address overlap. Evict low-address
      // tensors that would collide with the op's CB region.
      evictForDramCBGrowth(op, pos, data);
    }
    // Eviction may have inserted a reshard for `op`, shifting it past `pos`;
    // bail so run()'s sweep reprocesses the reshard and then `op`.
    if (data.schedule[pos] != op) {
      return;
    }
    if (l1Size > 0) {
      addResultsToLiveSet(l1Size);
      observer_->onLiveAdded(op, pos, l1Size, pos,
                             memoryTracker.getOccupiedL1());

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    ADDED (after eviction): L1 now {0}/{1} ({2} tensors)",
                   memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                   liveValues.size());
    }
  } else {
    // Stage 3: Op exceeds L1 budget even with no other live tensors — the op
    // itself is too large for the configured cap. Demote its output to DRAM.
    // No evictForDramCBGrowth needed: evictUntil already drained the live set.
    observer_->onSelfSpill(op, pos);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    DEMOTE SELF: op exceeds budget alone");
    demoteToDram(op);
  }
}

//===----------------------------------------------------------------------===//
// insertEventIntoLog
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::insertEventIntoLog(size_t pos,
                                                          L1Event event) {
  // Insert the event and shift every index >= pos in both index structures
  // so the invariant "addressSnapshots[i] = state before event i" and
  // "allocEventIndex[v] = i <=> l1EventLog[i] is v's kAlloc" are preserved.
  l1EventLog.insert(l1EventLog.begin() + pos, event);
  for (auto &[val, idx] : allocEventIndex) {
    if (idx >= pos) {
      ++idx;
    }
  }
  decltype(addressSnapshots) shifted;
  shifted.reserve(addressSnapshots.size());
  for (auto &[key, snap] : addressSnapshots) {
    shifted[key >= pos ? key + 1 : key] = std::move(snap);
  }
  addressSnapshots = std::move(shifted);
}

//===----------------------------------------------------------------------===//
// markEvictedAndRebuild
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::markEvictedAndRebuild(Value victim) {
  // O(1) lookup of victim's alloc event.
  auto idxIt = allocEventIndex.find(victim);
  assert(idxIt != allocEventIndex.end() &&
         "victim not found in allocEventIndex");
  size_t allocIdx = idxIt->second;

  // Mark alloc event as skipped.
  l1EventLog[allocIdx].skipped = true;

  // Search forward from allocIdx for the dealloc event (at most one).
  for (size_t i = allocIdx + 1; i < l1EventLog.size(); ++i) {
    if (l1EventLog[i].tensor == victim &&
        l1EventLog[i].kind == L1Event::kDealloc) {
      l1EventLog[i].skipped = true;
      break;
    }
  }

  // Restore snapshot taken before the victim's allocation.
  auto snapIt = addressSnapshots.find(allocIdx);
  assert(snapIt != addressSnapshots.end() &&
         "snapshot not found for evicted tensor");
  memoryTracker.restoreSnapshot(snapIt->second);

  // Replay events from the alloc point forward, skipping evicted tensors.
  // Update snapshots during replay so future evictions see accurate state.
  for (size_t i = allocIdx; i < l1EventLog.size(); ++i) {
    if (l1EventLog[i].skipped) {
      continue;
    }
    if (l1EventLog[i].kind == L1Event::kAlloc) {
      addressSnapshots[i] = memoryTracker.takeSnapshot();
      // View-eligible reshape: alias if src is still address-tracked in
      // this replay, otherwise fresh-allocate (the counterfactual when
      // src was evicted to DRAM).
      Operation *defOp = l1EventLog[i].tensor.getDefiningOp();
      if (defOp && canReshapeBeView(defOp) &&
          memoryTracker.hasTensorAddress(defOp->getOperand(0))) {
        memoryTracker.allocateAddressAt(l1EventLog[i].tensor,
                                        defOp->getOperand(0));
      } else {
        memoryTracker.allocateAddress(l1EventLog[i].tensor,
                                      l1EventLog[i].sizePerCore);
      }
    } else {
      memoryTracker.freeAddress(l1EventLog[i].tensor);
    }
  }
}

//===----------------------------------------------------------------------===//
// evictUntil
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::evictValue(
    Value victim, int64_t pos, ScheduleData &data,
    Operation *skipReshardConsumer) {
  liveValues.erase(victim);
  Operation *victimOp = victim.getDefiningOp();
  uint64_t freedBytes = memoryTracker.getTensorSize(victim);

  // Save original L1 layout before spilling.
  auto tensorType = mlir::cast<RankedTensorType>(victim.getType());
  TTNNLayoutAttr originalL1Layout =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    EVICT: {0} (L1: {1} bytes)", ttmlir::opToString(victimOp),
               freedBytes);
  observer_->onEviction(victimOp, pos, freedBytes);

  spillToDram(victim);
  memoryTracker.removeTensorFromSizes(victim);

  // Restore the original L1-sharded layout for consumers that need it:
  // - Past consumers (pos < currentPos): always restore. They were already
  //   validated assuming L1-sharded input; without reshard the IR is broken.
  //   Log a kAlloc for the reshard just before the consumer's first alloc event
  //   and a kDealloc just after its last alloc event. Because the victim is
  //   always scheduled before its consumers, allocIdx < consumerFirstAllocIdx,
  //   so the markEvictedAndRebuild replay below always passes through both
  //   events and updates the consumer's alloc snapshots to include the
  //   reshard's slot. Future replays then handle the reshard naturally: it is
  //   allocated before the consumer (correct addresses) and freed after (no
  //   stuck slot). Neither event is added to allocEventIndex, so the reshard
  //   can never be selected as an eviction victim.
  // - Future consumers (pos >= currentPos): probe the consumer's layout
  //   constraints with the new (DRAM-after-spill) inputs in isolation
  //   (additionalL1Usage=0 — we don't know what L1 will be occupied at the
  //   consumer's position yet). If a hard input-layout constraint rejects
  //   it (e.g. paged_update_cache requires sharded input), insert reshard into
  //   IR and schedule so the forward sweep processes it naturally, logging it
  //   like any other live tensor.
  //
  //   Otherwise leave the consumer to read DRAM — that's the actual L1
  //   saving the spill bought us.

  // Collect all insertions first; apply largest-position-first so that
  // inserting later events does not shift the positions of earlier ones.
  struct TransientInsertion {
    size_t pos;
    L1Event event;
  };
  llvm::SmallVector<TransientInsertion> transientInsertions;

  if (originalL1Layout.hasL1BufferType()) {
    for (Operation *consumer : collectDownstreamConsumers(victimOp)) {
      // Sibling-spill passes the op it is making all-DRAM here: don't reshard
      // its operand back to L1. Other (past/future) consumers of the operand
      // are still restored below — they need the L1 layout.
      if (consumer == skipReshardConsumer) {
        continue;
      }
      auto posIt = data.positionMap.find(consumer);
      if (posIt == data.positionMap.end()) {
        continue;
      }
      if (!mlir::dyn_cast<OpModel>(consumer)) {
        continue;
      }
      if (isa<ToLayoutOp>(consumer)) {
        continue;
      }

      llvm::SmallVector<unsigned, 2> spilledOperandIdx;
      for (unsigned i = 0; i < consumer->getNumOperands(); ++i) {
        Operation *defOp = consumer->getOperand(i).getDefiningOp();
        if (isa_and_nonnull<ToMemoryConfigOp>(defOp) &&
            defOp->getOperand(0) == victim) {
          spilledOperandIdx.push_back(i);
        }
      }
      if (spilledOperandIdx.empty()) {
        continue;
      }

      // Capture the position now: insertReshardIntoSchedule below adds a key to
      // positionMap, which can rehash and invalidate posIt.
      int64_t consumerPos = posIt->second;
      bool isPastConsumer = consumerPos < pos;
      bool needsReshard = isPastConsumer;
      if (!needsReshard) {
        auto consumerInputs = utils::extractInputLayouts(consumer);
        auto consumerConfig = extractOpConfigFromIR(consumer);
        auto consumerResult = memoryTracker.validateBackendDirect(
            consumer, consumerInputs, consumerConfig,
            /*additionalL1Usage=*/0);
        needsReshard = consumerResult.isMetalBackendError();
      }

      if (!needsReshard) {
        continue;
      }

      for (unsigned i : spilledOperandIdx) {
        TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                     "    RESHARD: inserting L1 reshard before {0} "
                     "(operand {1} changed L1→DRAM after eviction of {2})",
                     consumer->getName(), i, ttmlir::opToString(victimOp));
        insertReshardForConsumer(consumer, i, originalL1Layout);
        Value reshardResult = consumer->getOperand(i);
        Operation *reshardOp = reshardResult.getDefiningOp();

        // Aligned per-core L1 size from validating the inserted reshard;
        // fall back to the layout estimate if validation can't model it.
        auto reshardValidation = memoryTracker.validateBackendDirect(
            reshardOp, utils::extractInputLayouts(reshardOp),
            extractOpConfigFromIR(reshardOp), /*additionalL1Usage=*/0);
        uint64_t reshardSizePerCore =
            reshardValidation.isSuccess()
                ? reshardValidation.outputL1Usage
                : utils::getPerCoreL1Usage(
                      originalL1Layout,
                      ttmlir::utils::volume(originalL1Layout.getGridShape()));

        if (isPastConsumer) {
          // Find the consumer's first and last output alloc event indices so
          // we can bracket the reshard's transient occupation in the log.
          size_t firstAllocIdx = std::numeric_limits<size_t>::max();
          size_t lastAllocIdx = 0;
          for (auto result : consumer->getResults()) {
            auto logIt = allocEventIndex.find(result);
            if (logIt != allocEventIndex.end()) {
              firstAllocIdx = std::min(firstAllocIdx, logIt->second);
              lastAllocIdx = std::max(lastAllocIdx, logIt->second);
            }
          }
          // If the consumer has no L1 results (all DRAM outputs), it has no
          // allocEventIndex entries and we cannot bracket the reshard in the
          // log. The reshard is still inserted in the IR for correctness, but
          // its transient slot goes untracked in the address simulation. This
          // is a known gap: the CB overlap check for this consumer was done
          // when it was originally processed (with the victim at its then-
          // current address), but subsequent evictions may have rearranged
          // the free list so the reshard lands at a lower address at runtime.
          if (firstAllocIdx != std::numeric_limits<size_t>::max()) {
            // kAlloc goes at firstAllocIdx (just before consumer's first
            // result alloc) and kDealloc goes at lastAllocIdx+1 (just after
            // consumer's last result alloc). Inserting largest-first keeps
            // firstAllocIdx valid when the kDealloc insertion is processed.
            // Note: reshardResult is NOT added to allocEventIndex — it should
            // anyway never appear as an eviction victim.
            transientInsertions.push_back(
                {lastAllocIdx + 1,
                 {L1Event::kDealloc, reshardResult, 0, false}});
            transientInsertions.push_back(
                {firstAllocIdx,
                 {L1Event::kAlloc, reshardResult, reshardSizePerCore, false}});
          }
        } else {
          // Insert the reshard into the schedule so the forward sweep processes
          // it naturally, adding it to liveValues and the tracker. Each
          // insertion shifts the consumer right by one, so advance consumerPos.
          insertReshardIntoSchedule(reshardOp, reshardResult,
                                    reshardSizePerCore, consumerPos, data);
          ++consumerPos;
        }
      }
    }
  }

  // Apply transient event insertions largest-position-first so earlier
  // positions are not invalidated by later insertions.
  llvm::sort(transientInsertions,
             [](const TransientInsertion &a, const TransientInsertion &b) {
               return a.pos > b.pos;
             });
  for (const auto &ins : transientInsertions) {
    insertEventIntoLog(ins.pos, ins.event);
  }

  markEvictedAndRebuild(victim);
}

//===----------------------------------------------------------------------===//
// insertReshardIntoSchedule
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::insertReshardIntoSchedule(
    Operation *reshardOp, Value reshardResult, uint64_t reshardSizePerCore,
    int64_t consumerPos, ScheduleData &data) {
  // Register so evictFarthestUse and sibling eviction guards know not to
  // evict this value — it exists specifically to feed its consumer.
  insertedReshardValues.insert(reshardResult);

  // Insert the reshard before the consumer in the schedule vector, shifting
  // the consumer and all subsequent ops by +1.
  data.schedule.insert(data.schedule.begin() + consumerPos, reshardOp);

  // Shift positionMap for all ops at positions >= consumerPos.
  for (auto &[op, opPos] : data.positionMap) {
    if (opPos >= consumerPos) {
      ++opPos;
    }
  }
  data.positionMap[reshardOp] = consumerPos;

  // Existing lastUsePositions are not shifted. An insertion adds 1 to every
  // position at or after consumerPos, which preserves their relative order,
  // and relative order is all liveSet's farthest-last-use priority needs. The
  // reshard's own priority is arbitrary (reshards are never evicted; see
  // evictFarthestUse), so consumerPos is just convenient. deathSchedule
  // (shifted) owns freeing.
  int64_t reshardLastUse = consumerPos + 1;
  data.lastUsePositions[reshardResult] = consumerPos;

  // Rebuild deathSchedule with shifted keys (simpler than in-place shift on a
  // DenseMap where keys can't be updated).
  llvm::DenseMap<int64_t, llvm::SmallVector<Value>> newDeathSchedule;
  for (auto &[deathPos, vals] : data.deathSchedule) {
    int64_t shiftedPos = (deathPos >= consumerPos) ? deathPos + 1 : deathPos;
    newDeathSchedule[shiftedPos] = std::move(vals);
  }
  newDeathSchedule[reshardLastUse].push_back(reshardResult);
  data.deathSchedule = std::move(newDeathSchedule);
}

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::evictUntil(
    int64_t pos, ScheduleData &data, std::function<bool()> shouldStop) {
  // Helper: true if every remaining live value is a non-evictable inserted
  // reshard. Avoids exhausting the liveSet heap through stale entries when
  // no evictable candidates exist.
  auto onlyReshardsRemain = [&]() {
    return llvm::all_of(liveValues, [&](Value v) {
      return insertedReshardValues.count(v) > 0;
    });
  };

  while (!shouldStop() && !liveValues.empty() && !onlyReshardsRemain()) {
    Value victim = evictFarthestUse();
    if (!victim) {
      break;
    }
    evictValue(victim, pos, data);
  }
  return shouldStop();
}

//===----------------------------------------------------------------------===//
// handleNoFit
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
uint64_t L1SpillManagement<MemoryTracker>::handleNoFit(Operation *op,
                                                       int64_t pos,
                                                       ScheduleData &data,
                                                       uint64_t outputL1Size) {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    NO_FIT: output {0} bytes can't fit contiguously, evicting",
               outputL1Size);

  bool resolved = evictUntil(pos, data, [&]() {
    return memoryTracker.wouldAllocateAt(outputL1Size).has_value();
  });
  if (!resolved) {
    demoteToDram(op);
    evictForDramCBGrowth(op, pos, data);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    NO_FIT: eviction exhausted, demoting to DRAM");
    return 0;
  }

  // Re-validate after eviction freed space.
  auto freshInputLayouts = utils::extractInputLayouts(op);
  auto freshConfig = extractOpConfigFromIR(op);
  auto freshResult = memoryTracker.validate(op, freshInputLayouts, freshConfig);
  if (freshResult.isSuccess()) {
    applyOutputConfig(op, freshResult);
    uint64_t freshL1 = freshResult.outputL1Usage;
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    NO_FIT resolved: L1 now {0}/{1}",
                 memoryTracker.getOccupiedL1(), l1BudgetPerCore);
    return freshL1;
  }

  demoteToDram(op);
  evictForDramCBGrowth(op, pos, data);
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    NO_FIT: validation failed after eviction, demoting to "
               "DRAM");
  return 0;
}

//===----------------------------------------------------------------------===//
// handleFragmentation
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
uint64_t L1SpillManagement<MemoryTracker>::handleFragmentation(
    Operation *op, int64_t pos, ScheduleData &data, uint64_t cbPeakUsage,
    uint64_t outputL1Size) {
  // Add the same safety cushion as wouldCBsOverlapTensors to account for
  // unmodeled runtime fragmentation from transient internal op allocations.
  uint64_t cushionedCBUsage = cbPeakUsage + cbFragCushion;

  // Evict tensors (farthest last-use first) until CB overlap resolves.
  evictUntil(pos, data, [&]() {
    auto specAddr = memoryTracker.wouldAllocateAt(outputL1Size);
    if (!specAddr) {
      return false;
    }
    uint64_t effLowest =
        std::min(*specAddr, memoryTracker.getLowestOccupiedAddress());
    return cushionedCBUsage <= effLowest;
  });

  // After eviction, re-check both conditions with the updated free list.
  auto freshOutputAddr = memoryTracker.wouldAllocateAt(outputL1Size);
  if (!freshOutputAddr) {
    demoteToDram(op);
    evictForDramCBGrowth(op, pos, data);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    FRAG_DEMOTE (still no-fit after eviction): output to "
                 "DRAM");
    return 0;
  }

  uint64_t freshEffectiveLowest =
      std::min(*freshOutputAddr, memoryTracker.getLowestOccupiedAddress());
  if (cushionedCBUsage > freshEffectiveLowest) {
    demoteToDram(op);
    evictForDramCBGrowth(op, pos, data);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    FRAG_DEMOTE (CB overlap persists): cb={0}+cushion={1}"
                 "={2} > effectiveLowest={3}",
                 cbPeakUsage, cbFragCushion, cushionedCBUsage,
                 freshEffectiveLowest);
    return 0;
  }

  // Re-extract layouts after eviction and re-validate.
  auto inputLayouts = utils::extractInputLayouts(op);
  auto config = extractOpConfigFromIR(op);
  auto freshResult = memoryTracker.validate(op, inputLayouts, config);
  if (freshResult.isSuccess()) {
    uint64_t freshL1 = freshResult.outputL1Usage;
    applyOutputConfig(op, freshResult);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    FRAG_RESOLVED: L1 now {0}/{1}",
                 memoryTracker.getOccupiedL1(), l1BudgetPerCore);
    memoryTracker.logState();
    return freshL1;
  }

  // If validation failed with a backend constraint error (not OOM) after
  // partial eviction, some ops (e.g. concat) require homogeneous input
  // layouts and now see a mix of DRAM (spilled) and L1 (still live) inputs.
  // Spill all remaining L1 operands of the op to DRAM so the op sees
  // homogeneous DRAM inputs, then fall through to demoteToDram to also
  // put the output in DRAM. We intentionally do NOT validate with the
  // original (possibly sharded) output config after sibling spill:
  // tt-metal concat does not reject interleaved inputs paired with a
  // sharded output memory config at validation time, but runtime will
  // crash with a JIT kernel build failure
  // (https://github.com/tenstorrent/tt-metal/issues/41469). Keeping the
  // op all-DRAM avoids that runtime bug.
  if (freshResult.status !=
      op_constraint_validation::ValidationStatus::OutOfMemoryError) {
    // Collect L1 operands that need spilling (can't mutate liveValues
    // while iterating op operands via liveSet).
    llvm::SmallVector<Value> toEvict;
    for (Value operand : op->getOperands()) {
      if (liveValues.count(operand)) {
        toEvict.push_back(operand);
      }
    }
    for (Value victim : toEvict) {
      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    SPILL_SIBLING_OPERAND: evicting {0} to DRAM to "
                   "resolve backend constraint failure for {1}",
                   ttmlir::opToString(victim.getDefiningOp()), op->getName());
      // skipReshardConsumer=op: don't reshard `op`'s operand back to L1.
      // Validating after spilling one operand (others still L1) would report a
      // mixed-input failure and reshard it straight back, defeating the
      // homogeneous spill. Past/future consumers of the operand are still
      // restored, so their IR stays valid.
      evictValue(victim, pos, data, /*skipReshardConsumer=*/op);
    }
  }

  demoteToDram(op);
  evictForDramCBGrowth(op, pos, data);
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    FRAG_DEMOTE: output to DRAM");
  return 0;
}

//===----------------------------------------------------------------------===//
// wouldCBsOverlapTensors
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::wouldCBsOverlapTensors(
    Operation *op, int64_t pos, uint64_t cbPeakUsage,
    uint64_t speculativeOutputAddr) {
  // Check if the op's CB region (growing bottom-up from base) would overlap
  // with any live tensor or the speculative output tensor, OR if the
  // fragmentation cushion alone exceeds the lowest tensor address (catches
  // tight-fit scenarios where cbPeakUsage=0 but runtime allocator
  // fragmentation from preceding ops' internal allocations can prevent
  // contiguous allocation).
  // See: https://github.com/tenstorrent/tt-mlir/issues/7396
  uint64_t lowestExistingAddr = memoryTracker.getLowestOccupiedAddress();
  uint64_t effectiveLowest =
      std::min(speculativeOutputAddr, lowestExistingAddr);

  // Add a safety cushion to account for unmodeled runtime fragmentation
  // from transient internal op allocations (ghost holes).
  uint64_t cushionedCBUsage = cbPeakUsage + cbFragCushion;

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    CB FRAG CHECK for {0}: cbPeakUsage={1}, cushion={2}, "
               "speculativeOutputAddr={3}, lowestExistingAddr={4}, "
               "effectiveLowest={5}",
               op->getName(), cbPeakUsage, cbFragCushion, speculativeOutputAddr,
               lowestExistingAddr, effectiveLowest);

  if (cushionedCBUsage > effectiveLowest) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    FRAG_RISK: cb={0}+cushion={1}={2} > "
                 "effectiveLowest={3} (occupied={4})",
                 cbPeakUsage, cbFragCushion, cushionedCBUsage, effectiveLowest,
                 memoryTracker.getOccupiedL1());
    memoryTracker.logState();
    observer_->onFragmentationDemote(op, pos, cbPeakUsage, cbPeakUsage,
                                     /*inputL1Size=*/0, /*holeL1Size=*/0,
                                     effectiveLowest,
                                     memoryTracker.getOccupiedL1());
    return true;
  }

  return false;
}

// NOTE: spillCount tracking in run() is approximate after extracting
// handleOOM -- eviction spills inside handleOOM are not counted. The
// observer receives individual spill events regardless.

//===----------------------------------------------------------------------===//
// run
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::run() {
  ScheduleData data = buildScheduleData();

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "=== L1 Memory View (compile-time, validation-based) ===\n"
               "  Budget per core: {0} bytes\n"
               "  Schedule size: {1} ops",
               l1BudgetPerCore, data.schedule.size());

  observer_->onSpillStart(func.getName(), l1BudgetPerCore,
                          data.schedule.size());

  [[maybe_unused]] int64_t spillCount = 0;

  // Farthest-last-use eviction sweep with validation-based enforcement.
  for (int64_t pos = 0; pos < static_cast<int64_t>(data.schedule.size());
       ++pos) {
    Operation *op = data.schedule[pos];

    // Eviction can insert a reshard for `op`, shifting `op` past `pos`. Rewind
    // to the first inserted reshard (always at `pos`) so the sweep processes it
    // (and any others) before reprocessing `op`. Returns true when the caller
    // should `continue` the sweep.
    auto rewindIfScheduleShifted = [&]() {
      if (data.schedule[pos] == op) {
        return false;
      }
      --pos;
      return true;
    };

    processDeadTensors(pos, data);

    // Sink ops (paged_fill_cache / paged_update_cache / fill_cache) are in
    // the schedule only so computeLastUsePositions sees their operand uses
    // (which keeps their L1-resident input tensors alive in the tracker
    // until the cache write actually executes). They have no tensor result,
    // so addResultsToLiveSet / extractOpConfigFromIR / validate do not apply.
    if (optimizer_utils::isSinkOp(op)) {
      continue;
    }

    // ToLayoutOp with L1 output: workaround-inserted and always immediately
    // consumed by the target op. MemoryLayoutPropagation skips these, and
    // pre-decomposition OpModel is inaccurate.
    // Use farthest-last-use eviction for live tensors if needed to create room,
    // but do not add the output to liveValues — it is not a long-lived L1
    // tenant and will be gone (or dead) before any subsequent eviction decision
    // matters. Being absent from liveValues also means evictAllFromL1 (e.g.
    // triggered by DistributedRMSNormOp's isNotImplemented) cannot spill it.
    if (isa<ToLayoutOp>(op)) {
      auto resultType =
          mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
      auto lo =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(resultType.getEncoding());
      assert(lo && "ToLayoutOp result must have TTNNLayoutAttr encoding");
      if (lo.hasL1BufferType()) {
        uint64_t derivedL1 = utils::getPerCoreL1Usage(
            lo, ttmlir::utils::volume(lo.getGridShape()));
        TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                     "  [pos={0}] L1_TOLAYOUT: {1}, derivedL1={2} bytes, "
                     "occupied={3}/{4}",
                     pos, ttmlir::opToString(op), derivedL1,
                     memoryTracker.getOccupiedL1(), l1BudgetPerCore);
        // CBPeakUsage fixed to 0 as we have no validation result for ToLayoutOp
        // itself which is yet to be decomposed.
        ensureFitsL1(op, pos, data, /*cbPeakUsage=*/0, derivedL1);
        continue;
      }
      // DRAM ToLayoutOp: fall through to standard processing.
    }

    // Determine if the op outputs to L1 by inspecting its result types
    // directly. DRAM-output ops still need CB overlap checking against live L1
    // tensors -- skip only if there are no live L1 tensors that could clash.
    // Ops that can't be checked return NotImplemented from validation, which
    // triggers a full spill regardless of live set size.
    bool hasL1Output = llvm::any_of(op->getResults(), [](OpResult r) {
      auto tt = mlir::dyn_cast<RankedTensorType>(r.getType());
      auto lo = tt ? mlir::dyn_cast_or_null<TTNNLayoutAttr>(tt.getEncoding())
                   : nullptr;
      return lo && lo.hasL1BufferType();
    });

    if (!hasL1Output) {
      if (liveValues.empty()) {
        continue;
      }
    }

    // Count tensor results and compute per-result L1 budget.
    llvm::SmallVector<OpResult> tensorResults;
    for (auto r : op->getResults()) {
      if (mlir::isa<RankedTensorType>(r.getType())) {
        tensorResults.push_back(r);
      }
    }
    size_t numTensorResults = tensorResults.size();

    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "  [pos={0}] PROCESS: {1}\n"
                 "    hasL1Output: {2}, tensor results: {3}\n"
                 "    occupied L1 before: {4}/{5} ({6} tensors)",
                 pos, ttmlir::opToString(op), hasL1Output, numTensorResults,
                 memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                 liveValues.size());

    // Helper: add all tensor results to the live set with per-result L1 sizes.
    // TODO(rpavlovic): Even split is an approximation. Multi-output ops may
    // have unbalanced per-result L1 sizes. An allocator-backed tracker would
    // compute exact per-result sizes.
    // See: https://github.com/tenstorrent/tt-mlir/issues/7295
    auto addResultsToLiveSet = [&](uint64_t totalL1) {
      uint64_t perResultL1 =
          numTensorResults > 0 ? totalL1 / numTensorResults : 0;
      for (auto r : tensorResults) {
        Value val = r;
        auto luIt = data.lastUsePositions.find(val);
        assert(luIt != data.lastUsePositions.end() &&
               "scheduled value missing its lastUsePositions entry");
        int64_t resultLastUse = luIt->second;
        // Snapshot before allocation and record event for replay.
        allocEventIndex[val] = l1EventLog.size();
        addressSnapshots[l1EventLog.size()] = memoryTracker.takeSnapshot();
        l1EventLog.push_back(
            {L1Event::kAlloc, val, perResultL1, /*skipped=*/false});

        // View-eligible reshape: alias src's buffer instead of carving a
        // fresh slot. See `addTensorAtAddress`. Must stay in sync with the
        // willAliasSourceInL1 short-circuit in ensureFitsL1.
        Operation *defOp = val.getDefiningOp();
        if (defOp && willAliasSourceInL1(defOp)) {
          memoryTracker.addTensorAtAddress(val, perResultL1,
                                           defOp->getOperand(0));
        } else {
          memoryTracker.addTensor(val, perResultL1);
        }

        liveValues.insert(val);
        liveSet.push({resultLastUse, val});
      }
    };

    // Extract current config and input layouts for validation.
    auto inputLayouts = utils::extractInputLayouts(op);
    auto config = extractOpConfigFromIR(op);

    // Validate op with current occupied L1.
    auto result = memoryTracker.validate(op, inputLayouts, config);

    if (result.isNotImplemented()) {
      // Op not validated by backend — evict all live L1 tensors. Without
      // OpModel constraints we cannot know how much L1 the op needs, so the
      // only safe choice is a full flush. Insert spills right before the
      // trigger op so earlier consumers can still read from L1.
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "    NOT_IMPLEMENTED: evicting all live L1 tensors for {0}",
                   ttmlir::opToString(op));
      evictAllFromL1(pos, data, op);
      ++spillCount;
      continue;
    }

    if (result.isSuccess()) {
      uint64_t l1Size = result.outputL1Usage;

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    VALIDATION SUCCESS: op {0}, "
                   "cbPeakUsage={1}, outputL1={2} bytes",
                   ttmlir::opToString(op), result.cbPeakUsage, l1Size);

      l1Size = ensureFitsL1(op, pos, data, result.cbPeakUsage, l1Size);
      if (rewindIfScheduleShifted()) {
        continue;
      }
      if (l1Size == 0) {
        continue;
      }
      addResultsToLiveSet(l1Size);
      observer_->onLiveAdded(op, pos, l1Size, pos,
                             memoryTracker.getOccupiedL1());

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    ADDED: L1 now {0}/{1} ({2} tensors)",
                   memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                   liveValues.size());
      continue;
    }

    // Non-OOM backend constraint error: the op rejects its current
    // input/output layout combination. Eviction can't help (the failure
    // isn't from L1 pressure), so skip the OOM path and demote the output
    // to DRAM as a graceful fallback rather than crash on a hard
    // constraint.
    if (result.isMetalBackendError()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    BACKEND_ERROR at pos {0} for {1}: {2}. "
                   "Demoting to DRAM.",
                   pos, ttmlir::opToString(op), result.errorMessage);
      demoteToDram(op);
      ++spillCount;
      continue;
    }

    handleOOM(op, pos, tensorResults, data, addResultsToLiveSet);
    if (rewindIfScheduleShifted()) {
      continue;
    }
  }

  // Print final memory view summary.
  observer_->onSpillEnd(spillCount, memoryTracker.getOccupiedL1(),
                        liveValues.size());
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "=== L1 Spill Summary ===\n"
               "  Total spills: {0}\n"
               "  Final live L1: {1}/{2} ({3} tensors)",
               spillCount, memoryTracker.getOccupiedL1(), l1BudgetPerCore,
               liveValues.size());
}

//===----------------------------------------------------------------------===//
// computeLastUsePositions
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
llvm::DenseMap<Value, int64_t>
L1SpillManagement<MemoryTracker>::computeLastUsePositions(
    const llvm::SmallVector<Operation *> &schedule) {
  // Build position map.
  llvm::DenseMap<Operation *, int64_t> positionMap;
  for (int64_t i = 0; i < static_cast<int64_t>(schedule.size()); ++i) {
    positionMap[schedule[i]] = i;
  }

  // For each tensor result of each op, find the maximum position among its
  // users. Per-result granularity enables precise L1 reclamation for
  // multi-output ops.
  llvm::DenseMap<Value, int64_t> resultLastUse;
  for (int64_t i = 0; i < static_cast<int64_t>(schedule.size()); ++i) {
    Operation *op = schedule[i];

    for (auto result : op->getResults()) {
      if (!mlir::isa<RankedTensorType>(result.getType())) {
        continue;
      }

      int64_t lastUse = i; // Default: last use is the op itself.
      for (auto &use : result.getUses()) {
        Operation *user = use.getOwner();
        auto posIt = positionMap.find(user);
        if (posIt != positionMap.end()) {
          lastUse = std::max(lastUse, posIt->second);
        }
      }

      resultLastUse[result] = lastUse;
    }
  }

  return resultLastUse;
}

//===----------------------------------------------------------------------===//
// demoteToDram
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::demoteToDram(Operation *op) {
  for (auto opResult : op->getResults()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(opResult.getType());
    if (!tensorType) {
      continue;
    }
    auto layoutAttr =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    if (!layoutAttr || !layoutAttr.hasL1BufferType()) {
      continue;
    }
    TTNNLayoutAttr dramLayout =
        TTNNLayoutAttr::Builder(layoutAttr, tensorType.getShape())
            .setBufferType(BufferType::DRAM)
            .setMemoryLayout(TensorMemoryLayout::Interleaved)
            .build();
    RankedTensorType newType =
        utils::RankedTensorTypeFactory::create(tensorType, dramLayout);
    opResult.setType(newType);
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer, "Demoted to DRAM: {0}",
               ttmlir::opToString(op));
}

//===----------------------------------------------------------------------===//
// evictAllFromL1
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::evictAllFromL1(int64_t pos,
                                                      ScheduleData &data,
                                                      Operation *triggerOp) {
  llvm::SmallVector<Operation *> evictedOps;
  for (Value victim : liveValues) {
    uint64_t freedBytes = memoryTracker.getTensorSize(victim);
    Operation *victimOp = victim.getDefiningOp();
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    EVICT_ALL: {0} (L1: {1} bytes)",
                 ttmlir::opToString(victimOp), freedBytes);
    observer_->onEviction(victimOp, pos, freedBytes);
    spillToDramBeforeTrigger(victim, triggerOp);
    memoryTracker.removeTensor(victim);
    evictedOps.push_back(victimOp);
  }
  liveValues.clear();
  // Mark all events as skipped and reset tracker to empty state.
  for (auto &event : l1EventLog) {
    event.skipped = true;
  }
  memoryTracker.init(l1BudgetPerCore);
  // INVARIANT: spillToDramBeforeTrigger requires a full tracker reset.
  // Anything else here means the address simulator is out of sync with
  // the IR's L1 occupancy.
  assert(memoryTracker.getOccupiedL1() == 0 &&
         "evictAllFromL1: tracker not fully reset after flush");

  // Revalidate consumers after all evictions to avoid revalidating against
  // transient intermediate IR states.
  for (Operation *victimOp : evictedOps) {
    revalidateConsumers(victimOp, pos, data.positionMap);
  }
}

//===----------------------------------------------------------------------===//
// evictForCBOverlap
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::evictForCBOverlap(
    uint64_t cushionedCBUsage, int64_t pos, ScheduleData &data) {
  evictUntil(pos, data, [&]() {
    return cushionedCBUsage <= memoryTracker.getLowestOccupiedAddress();
  });
}

//===----------------------------------------------------------------------===//
// evictForDramCBGrowth
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::evictForDramCBGrowth(
    Operation *op, int64_t pos, ScheduleData &data) {

  auto inputLayouts = utils::extractInputLayouts(op);
  auto config = extractOpConfigFromIR(op);
  auto result = memoryTracker.validateBackendDirect(op, inputLayouts, config,
                                                    /*additionalL1Usage=*/0);
  if (!result.isSuccess()) {
    op->emitError("L1SpillManagement: DRAM output config failed validation "
                  "after demotion (")
        << result.errorMessage << "); this indicates a compiler bug";
    compilationFailed = true;
    return;
  }
  if (result.cbPeakUsage == 0) {
    return;
  }

  uint64_t dramCBCushioned = result.cbPeakUsage + cbFragCushion;

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    DRAM_CB_CHECK: dramCBPeak={0}, cushion={1}, "
               "cushionedDramCB={2}, lowestExisting={3}",
               result.cbPeakUsage, cbFragCushion, dramCBCushioned,
               memoryTracker.getLowestOccupiedAddress());

  evictForCBOverlap(dramCBCushioned, pos, data);

  // If the CB region still overlaps a live tensor after eviction, that tensor
  // is an inserted reshard we cannot evict (it restores a required L1 input).
  // Demotion freed the output but not this input, so the op cannot be placed:
  // its CBs would clobber an L1 input it requires. Fail with a clear error
  // rather than emit IR that overflows L1 at runtime.
  if (!liveValues.empty() &&
      dramCBCushioned > memoryTracker.getLowestOccupiedAddress()) {
    op->emitError("L1SpillManagement: ")
        << op->getName()
        << " requires an L1 input restored by an inserted reshard, but its "
           "circular-buffer region overlaps that reshard's L1 slot and the "
           "reshard cannot be evicted or relocated; the op cannot be placed "
           "within the L1 budget";
    compilationFailed = true;
  }
}

//===----------------------------------------------------------------------===//
// spillToDram / spillToDramBeforeTrigger / spillToDramImpl
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::spillToDram(Value result) {
  spillToDramImpl(result, /*insertBefore=*/nullptr);
}

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::spillToDramBeforeTrigger(
    Value result, Operation *triggerOp) {
  assert(triggerOp && "spillToDramBeforeTrigger requires a non-null trigger");
  spillToDramImpl(result, triggerOp);
}

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::spillToDramImpl(
    Value result, Operation *insertBefore) {
  Operation *defOp = result.getDefiningOp();
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(result.getType());
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  if (!layoutAttr.hasL1BufferType()) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    WARNING: spillToDram called on already-DRAM tensor: {0}",
                 ttmlir::opToString(defOp));
    return;
  }

  // Create DRAM interleaved layout.
  TTNNLayoutAttr dramLayout =
      TTNNLayoutAttr::Builder(layoutAttr, tensorType.getShape())
          .setBufferType(BufferType::DRAM)
          .setMemoryLayout(TensorMemoryLayout::Interleaved)
          .build();
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    SPILL_TO_DRAM: {0}, new layout: {1}",
               ttmlir::opToString(defOp), dramLayout);

  RankedTensorType newTensorType =
      utils::RankedTensorTypeFactory::create(tensorType, dramLayout);

  OpBuilder builder(defOp->getContext());
  if (insertBefore) {
    builder.setInsertionPoint(insertBefore);
  } else {
    builder.setInsertionPointAfter(defOp);
  }
  Location loc = ttmlir::utils::appendLocationSuffix(defOp->getLoc(), "_spill");

  // Save uses that should be reconnected to the DRAM spill. When insertBefore
  // is set, only reconnect uses at or after the insertion point — earlier uses
  // can still read the original L1 value directly.
  llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
  for (auto &use : result.getUses()) {
    if (insertBefore && use.getOwner()->isBeforeInBlock(insertBefore)) {
      continue; // This use is before the spill point — keep reading from L1.
    }
    uses.emplace_back(use.getOwner(), use.getOperandNumber());
  }

  Operation *spillOp =
      builder.create<ToMemoryConfigOp>(loc, newTensorType, result);

  for (auto &[useOp, operandIdx] : uses) {
    useOp->setOperand(operandIdx, spillOp->getResult(0));
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "Inserted spill-to-DRAM op: {0}", ttmlir::opToString(spillOp));
}

//===----------------------------------------------------------------------===//
// insertReshardForConsumer
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::insertReshardForConsumer(
    Operation *consumer, unsigned operandIdx, TTNNLayoutAttr originalL1Layout) {
  Value spillOutput = consumer->getOperand(operandIdx);
  auto spillTensorType = mlir::cast<RankedTensorType>(spillOutput.getType());
  auto spillLayout = mlir::cast<TTNNLayoutAttr>(spillTensorType.getEncoding());

  // Skip if operand is already in L1 (e.g., reshard already inserted).
  if (spillLayout.hasL1BufferType()) {
    return;
  }

  // Build target tensor type with original L1 layout.
  RankedTensorType reshardType =
      utils::RankedTensorTypeFactory::create(spillTensorType, originalL1Layout);

  // Insert ToMemoryConfigOp before consumer.
  OpBuilder builder(consumer);
  Location loc =
      ttmlir::utils::appendLocationSuffix(consumer->getLoc(), "_reshard");

  auto reshardOp =
      builder.create<ToMemoryConfigOp>(loc, reshardType, spillOutput);
  consumer->setOperand(operandIdx, reshardOp->getResult(0));

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "Inserted reshard op: {0} before {1}",
               ttmlir::opToString(reshardOp.getOperation()),
               ttmlir::opToString(consumer));
}

//===----------------------------------------------------------------------===//
// Explicit template instantiation
//===----------------------------------------------------------------------===//

template class L1SpillManagement<SumL1MemoryTracker>;

} // namespace mlir::tt::ttnn
