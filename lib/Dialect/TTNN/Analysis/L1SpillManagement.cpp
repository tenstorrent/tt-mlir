// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
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

// Kill switches (TTMLIR_OPT_DISABLE=<name>[,<name>...]):
//   demote_producer       copy-based eviction for convolutions evicted right
//                         after they ran (instead of a DRAM-output program)
//   transient_model       do not model op-internal buffers / freed inputs in
//                         the address simulation
//   cb_zone_compact       no ttnn.reallocate compaction before eviction when
//                         a live tensor sits in the CB region
//   large_tensor_shortlived  keep the 40 % large-tensor guard for tensors
//                         consumed by the very next op as well
static bool optDisabled(const char *name) {
  const char *env = std::getenv("TTMLIR_OPT_DISABLE");
  return env && llvm::StringRef(env).contains(name);
}
static bool producerDemoteDisabled() {
  static const bool disabled = optDisabled("demote_producer");
  return disabled;
}
static bool transientModelDisabled() {
  static const bool disabled = optDisabled("transient_model");
  return disabled;
}
static bool cbZoneCompactDisabled() {
  static const bool disabled = optDisabled("cb_zone_compact");
  return disabled;
}
static bool shortLivedExemptionDisabled() {
  static const bool disabled = optDisabled("large_tensor_shortlived");
  return disabled;
}

// Release-build probe for attributing a spill/eviction to the branch that
// caused it. TTMLIR_DEBUG compiles to a no-op in Release (TTMLIR_ENABLE_DEBUG_LOGS
// is undefined), so this writes to stderr directly, gated on TTMLIR_SPILL_PROBE.
static bool spillProbeEnabled() {
  static const bool enabled = (std::getenv("TTMLIR_SPILL_PROBE") != nullptr);
  return enabled;
}
static std::string probeOpName(mlir::Operation *op) {
  if (!op) {
    return "<null>";
  }
  std::string label = op->getName().getStringRef().str();
  if (auto nameLoc = mlir::dyn_cast<NameLoc>(op->getLoc())) {
    label += "@";
    label += nameLoc.getName().str();
  }
  return label;
}
#define SPILL_PROBE(...)                                                       \
  do {                                                                         \
    if (spillProbeEnabled()) {                                                 \
      llvm::errs() << "[SPILL_PROBE] " << llvm::formatv(__VA_ARGS__) << "\n";  \
    }                                                                          \
  } while (0)

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

void SumL1MemoryTracker::allocateAddress(Value result, uint64_t l1SizePerCore,
                                         uint64_t transientPerCore,
                                         Value freedBefore,
                                         bool transientAfterFree) {
  if (l1SizePerCore == 0 || l1Budget == 0) {
    return;
  }
  uint64_t alignedSize =
      (l1SizePerCore + kL1Alignment - 1) & ~(kL1Alignment - 1);

  // Model the op's internal buffers: carve them out first (top-down), place
  // the output, then give the carved block back. The hole they leave above
  // the output is what the hardware allocator sees after the op.
  std::optional<uint64_t> trStart;
  uint64_t alignedTransient = 0;
  if (transientPerCore > 0) {
    alignedTransient =
        (transientPerCore + kL1Alignment - 1) & ~(kL1Alignment - 1);
  }
  if (transientPerCore > 0 && !transientAfterFree) {
    trStart = findTopDown(freeList, alignedTransient);
    if (trStart) {
      carve(freeList, *trStart, alignedTransient);
    }
  }
  // An input the op frees before its output is released now: the output
  // can take its place (a 3x3 conv's output lands where its input was, the
  // halo hole is below it, not above).
  if (freedBefore && tensorAddresses.count(freedBefore)) {
    freeAddress(freedBefore);
  }
  // conv2d re-places its halo after that free (reallocate_halo_output):
  // it lands at the top of the merged space and the output goes below it.
  if (transientPerCore > 0 && transientAfterFree) {
    trStart = findTopDown(freeList, alignedTransient);
    if (trStart) {
      carve(freeList, *trStart, alignedTransient);
    }
  }
  std::optional<uint64_t> allocStart = findTopDown(freeList, alignedSize);
  if (!allocStart) {
    llvm_unreachable("allocateAddress: no contiguous free block; caller must "
                     "pre-check wouldAllocateAt");
  }
  carve(freeList, *allocStart, alignedSize);
  tensorAddresses[result] = {*allocStart, alignedSize};
  // First allocation in this slot: open the alias group at refcount 1
  // and account the buffer once. Aliases joining via allocateAddressAt
  // do not re-bump currentOccupied.
  aliasGroups[*allocStart] = {/*count=*/1, /*rawSize=*/l1SizePerCore};
  currentOccupied += l1SizePerCore;
  if (trStart) {
    insertFreeBlock(*trStart, *trStart + alignedTransient);
  }
}

void SumL1MemoryTracker::addTensor(Value result, uint64_t l1SizePerCore,
                                   uint64_t transientPerCore,
                                   Value freedBefore,
                                   bool transientAfterFree) {
  tensorSizes[result] = l1SizePerCore;
  allocateAddress(result, l1SizePerCore, transientPerCore, freedBefore,
                  transientAfterFree);
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
  insertFreeBlock(freedStart, freedEnd);
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
SumL1MemoryTracker::getTensorAddress(Value result) const {
  auto it = tensorAddresses.find(result);
  if (it == tensorAddresses.end()) {
    return std::nullopt;
  }
  return it->second.first;
}

llvm::SmallVector<Value>
SumL1MemoryTracker::getValuesAboveVirtualThreshold(uint64_t threshold) const {
  llvm::SmallVector<Value> result;
  for (const auto &[val, addrSize] : tensorAddresses) {
    if (addrSize.first > threshold) {
      result.push_back(val);
    }
  }
  return result;
}

std::optional<uint64_t>
SumL1MemoryTracker::wouldAllocateAt(uint64_t l1SizePerCore,
                                    uint64_t transientPerCore,
                                    Value freedBefore,
                                    bool transientAfterFree) const {
  if (l1SizePerCore == 0 || l1Budget == 0) {
    return l1Budget; // Trivially fits at the top.
  }
  uint64_t alignedSize =
      (l1SizePerCore + kL1Alignment - 1) & ~(kL1Alignment - 1);
  auto freedIt = freedBefore ? tensorAddresses.find(freedBefore)
                             : tensorAddresses.end();
  // Freeing an alias of a slot that another live value still shares does
  // not release the slot (see freeAddress).
  bool hasFreed = freedIt != tensorAddresses.end();
  if (hasFreed) {
    auto groupIt = aliasGroups.find(freedIt->second.first);
    if (groupIt != aliasGroups.end() && groupIt->second.count > 1) {
      hasFreed = false;
    }
  }
  if (transientPerCore == 0 && !hasFreed) {
    return findTopDown(freeList, alignedSize);
  }
  // The op's internal buffers are allocated first (with the input still
  // live), then the input it frees is released, then the output is placed.
  llvm::SmallVector<FreeBlock> blocks = freeList;
  uint64_t alignedTransient =
      (transientPerCore + kL1Alignment - 1) & ~(kL1Alignment - 1);
  // The transient buffers are allocated while the input is still live; when
  // the op re-places them after the free they end up at the top of the
  // merged free space, otherwise they stay below the freed slot.
  if (transientPerCore > 0 && !findTopDown(blocks, alignedTransient)) {
    return std::nullopt;
  }
  if (transientPerCore > 0 && !transientAfterFree) {
    carve(blocks, *findTopDown(blocks, alignedTransient), alignedTransient);
  }
  if (hasFreed) {
    release(blocks, freedIt->second.first,
            freedIt->second.first + freedIt->second.second);
  }
  if (transientPerCore > 0 && transientAfterFree) {
    std::optional<uint64_t> trStart = findTopDown(blocks, alignedTransient);
    if (!trStart) {
      return std::nullopt;
    }
    carve(blocks, *trStart, alignedTransient);
  }
  return findTopDown(blocks, alignedSize);
}

std::optional<uint64_t>
SumL1MemoryTracker::findTopDown(const llvm::SmallVector<FreeBlock> &blocks,
                                uint64_t alignedSize) {
  // Walk free list from highest address (reverse) for top-down first-fit.
  for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
    if (blocks[i].size() >= alignedSize) {
      return blocks[i].end - alignedSize;
    }
  }
  return std::nullopt;
}

void SumL1MemoryTracker::carve(llvm::SmallVector<FreeBlock> &blocks,
                               uint64_t start, uint64_t size) {
  uint64_t end = start + size;
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i].start <= start && end <= blocks[i].end) {
      if (blocks[i].start == start && blocks[i].end == end) {
        blocks.erase(blocks.begin() + i);
      } else if (blocks[i].start == start) {
        blocks[i].start = end;
      } else if (blocks[i].end == end) {
        blocks[i].end = start;
      } else {
        FreeBlock upper{end, blocks[i].end};
        blocks[i].end = start;
        blocks.insert(blocks.begin() + i + 1, upper);
      }
      return;
    }
  }
  llvm_unreachable("carve: range is not inside one free block");
}

void SumL1MemoryTracker::release(llvm::SmallVector<FreeBlock> &blocks,
                                 uint64_t freedStart, uint64_t freedEnd) {
  // Find insertion point in sorted list (sorted by start address).
  size_t insertIdx = 0;
  while (insertIdx < blocks.size() && blocks[insertIdx].start < freedStart) {
    ++insertIdx;
  }
  // Check merge with previous and next free blocks.
  bool mergePrev = insertIdx > 0 && blocks[insertIdx - 1].end == freedStart;
  bool mergeNext =
      insertIdx < blocks.size() && blocks[insertIdx].start == freedEnd;
  if (mergePrev && mergeNext) {
    blocks[insertIdx - 1].end = blocks[insertIdx].end;
    blocks.erase(blocks.begin() + insertIdx);
  } else if (mergePrev) {
    blocks[insertIdx - 1].end = freedEnd;
  } else if (mergeNext) {
    blocks[insertIdx].start = freedStart;
  } else {
    blocks.insert(blocks.begin() + insertIdx, {freedStart, freedEnd});
  }
}

void SumL1MemoryTracker::insertFreeBlock(uint64_t freedStart,
                                         uint64_t freedEnd) {
  release(freeList, freedStart, freedEnd);
}

//===----------------------------------------------------------------------===//
// L1SpillManagement
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
L1SpillManagement<MemoryTracker>::L1SpillManagement(
    func::FuncOp func, ttcore::GridAttr deviceGrid, uint64_t l1BudgetPerCore,
    uint64_t usableL1Size, std::unique_ptr<L1SpillObserver> observer)
    : func(func), deviceGrid(deviceGrid), l1BudgetPerCore(l1BudgetPerCore),
      cbFragCushion(
          static_cast<uint64_t>(kCBFragCushionFraction * l1BudgetPerCore)),
      l1DeadZone(usableL1Size > l1BudgetPerCore
                     ? usableL1Size - l1BudgetPerCore
                     : 0) {
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
      if (isa<ToTensorSpecOp>(consumer)) {
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
      SPILL_PROBE("  DEAD pos={0} val={1} occupied={2} nTracked={3}", pos,
                  probeOpName(deadOp), memoryTracker.getOccupiedL1(),
                  memoryTracker.getTrackedCount());
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
  return canReshapeBeView(op) &&
         memoryTracker.hasTensorAddress(op->getOperand(0));
}

template <typename MemoryTracker>
Value L1SpillManagement<MemoryTracker>::opFreedInput(Operation *op) {
  auto convFrees = [](auto convOp) -> Value {
    auto cfg = convOp.getConv2dConfigAttr();
    if (cfg && cfg.getDeallocateActivation() &&
        cfg.getDeallocateActivation().getValue()) {
      return convOp.getInput();
    }
    return Value();
  };
  if (auto conv = mlir::dyn_cast<Conv2dOp>(op)) {
    return convFrees(conv);
  }
  if (auto convT = mlir::dyn_cast<ConvTranspose2dOp>(op)) {
    return convFrees(convT);
  }
  if (auto pool = mlir::dyn_cast<MaxPool2dOp>(op)) {
    return pool.getDeallocateInput() ? pool.getInput() : Value();
  }
  if (auto pool = mlir::dyn_cast<AvgPool2dOp>(op)) {
    return pool.getDeallocateInput() ? pool.getInput() : Value();
  }
  return Value();
}

template <typename MemoryTracker>
uint64_t L1SpillManagement<MemoryTracker>::ensureFitsL1(
    Operation *op, int64_t pos, ScheduleData &data, uint64_t opL1Usage,
    uint64_t cbPeakUsage, uint64_t l1Size, uint64_t l1BuffersPeakUsage) {
  SPILL_PROBE("ENTER pos={0} op={1} opL1Usage={2} cbPeakUsage={3} "
              "transientPeak={4} l1Size={5} budget={6} deadZone={7} "
              "cushion={8} occupied={9} trackedLowest={10} nTracked={11}",
              pos, probeOpName(op), opL1Usage, cbPeakUsage, l1BuffersPeakUsage,
              l1Size, l1BudgetPerCore, l1DeadZone, cbFragCushion,
              memoryTracker.getOccupiedL1(),
              memoryTracker.getLowestOccupiedAddress(),
              memoryTracker.getTrackedCount());
  pendingTransientBytes = 0;
  pendingFreedBefore = Value();
  pendingTransientAfterFree = false;
  {
    Value freed = opFreedInput(op);
    if (freed && !transientModelDisabled() &&
        memoryTracker.hasTensorAddress(freed) &&
        data.lastUsePositions.lookup(freed) == pos) {
      pendingFreedBefore = freed;
      // conv2d / conv_transpose2d move the halo after freeing the input
      // (tt-metal reallocate_halo_output defaults to true); pools do so
      // only when asked.
      auto convHaloMoved = [](auto convOp) {
        auto cfg = convOp.getConv2dConfigAttr();
        if (cfg && cfg.getReallocateHaloOutput()) {
          return cfg.getReallocateHaloOutput().getValue();
        }
        return true; // tt-metal default
      };
      if (auto pool = mlir::dyn_cast<MaxPool2dOp>(op)) {
        pendingTransientAfterFree = pool.getReallocateHaloOutput();
      } else if (auto pool = mlir::dyn_cast<AvgPool2dOp>(op)) {
        pendingTransientAfterFree = pool.getReallocateHaloOutput();
      } else if (auto conv = mlir::dyn_cast<Conv2dOp>(op)) {
        pendingTransientAfterFree = convHaloMoved(conv);
      } else if (auto convT = mlir::dyn_cast<ConvTranspose2dOp>(op)) {
        pendingTransientAfterFree = convHaloMoved(convT);
      } else {
        pendingTransientAfterFree = true;
      }
    }
  }

  // A view-eligible reshape aliases its source's existing L1 slot, so it
  // consumes no fresh L1. Skip fit/CB-overlap checks that assume new allocation.
  if (willAliasSourceInL1(op)) {
    SPILL_PROBE("  EXIT aliasesSource pos={0}", pos);
    return l1Size;
  }

  // For DRAM-output ops (l1Size==0) the simulation allocates no L1 tensor, but
  // the op's kernel still allocates circular buffers growing upward from
  // l1_unreserved_base.  With large compute configs (e.g. HiFi2 + fp32_dest_acc)
  // the CB peak can exceed 600 KB and overlap with live L1 tensors that were
  // placed by earlier ops.  Without this check the overlap is only detected at
  // runtime by validate_circular_buffer_region → TT_THROW.
  if (l1Size == 0 && cbPeakUsage > 0) {
    uint64_t cushionedCB = cbPeakUsage + cbFragCushion;
    uint64_t lowestExisting = memoryTracker.getLowestOccupiedAddress();
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    DRAM_OP_CB_CHECK: cbPeak={0}, cushion={1}, "
                 "cushioned={2}, lowestExisting={3}",
                 cbPeakUsage, cbFragCushion, cushionedCB, lowestExisting);
    SPILL_PROBE("  PATH=DRAM_OP_CB pos={0} cbPeak={1} cushioned={2} lowestExisting={3}",
                pos, cbPeakUsage, cushionedCB, lowestExisting);
    if (cushionedCB > lowestExisting) {
      SPILL_PROBE("  ACTION=DRAM_OP_CB_EVICT pos={0} cushioned={1} > lowestExisting={2}",
                  pos, cushionedCB, lowestExisting);
      evictForCBOverlap(cushionedCB, pos, data);
    }
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !isa<ToLayoutOp>(defOp)) {
        continue;
      }
      auto resultType = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!resultType) {
        continue;
      }
      auto lo = mlir::dyn_cast<TTNNLayoutAttr>(resultType.getEncoding());
      if (lo && lo.hasL1BufferType()) {
        spillToDram(operand);
      }
    }
    return 0;
  }

  // The first fit check already honours an input the op frees before its
  // output (a 3x3 conv whose output takes the place of its input), otherwise
  // a 553 KB output next to a 369 KB input it is about to free looks like a
  // no-fit and the input is evicted to DRAM for nothing.
  auto speculativeAddr = memoryTracker.wouldAllocateAt(
      l1Size, /*transientPerCore=*/0, pendingFreedBefore, /*afterFree=*/false);
  if (!speculativeAddr) {
    SPILL_PROBE("  ACTION=NO_FIT pos={0} l1Size={1} occupied={2}", pos, l1Size,
                memoryTracker.getOccupiedL1());
    l1Size = handleNoFit(op, pos, data, opL1Usage, l1Size);
    speculativeAddr = memoryTracker.wouldAllocateAt(
        l1Size, /*transientPerCore=*/0, pendingFreedBefore, /*afterFree=*/false);
  }
  if (l1Size > 0 && speculativeAddr) {
    // CB-zone eviction: make room for this op's full runtime L1 footprint.
    //
    // The footprint is cbPeakUsage (static circular buffers) PLUS the op's
    // transient tensor buffers (l1BuffersPeakUsage, tt-metal graph-capture
    // l1_buffers_peak_per_core). The transient part is what the compile-time
    // address simulator cannot see -- these "ghost" buffers are allocated and
    // released inside the program and fragment the runtime free list -- and it
    // is exactly what the fixed 10% cushion was approximating.
    //
    // The gate is a pure capacity sum over SIZES and never consults the
    // simulated address model:
    //
    //     cbPeak + transientPeak + occupiedL1 + cushion  >  budget
    //
    // Two earlier formulations were tried and reverted, both address-based
    // (cushioned CB vs getLowestOccupiedAddress()); both failed the full BEV
    // model with `validate_circular_buffer_region` clashes because the
    // simulated addresses are hundreds of KB off the hardware's. Sizes are
    // reliable; addresses are not. The previous size-free gate
    // (cbPeak > l1DeadZone, deadZone = the 5% cap margin) fired on any op
    // whose CBs exceeded ~68 KB and, on block A single camera 0, evicted 39
    // tensors that were in no danger; it also MISSED ops whose CBs are tiny
    // but whose transient buffers are over 1 MB (e.g. conv2d with
    // l1_buffers_peak 1,160,448 on the full BEV model).
    //
    // When the gate fires, the reserve handed to the victim selector is the
    // full footprint, so enough is evicted to actually fit it.
    // The op's own L1 operands are already inside l1BuffersPeakUsage: the
    // op-model graph capture creates every input tensor inside the trace, so
    // l1_buffers_peak_per_core counts them (SumL1MemoryTracker::validate
    // subtracts them from additionalL1Usage for the same reason). Only the
    // OTHER live tensors add to the footprint. Counting operands twice made
    // the gate fire on a block-sharded conv2d input and evict it; the conv
    // then ran a different program (DRAM input, internal sharding, 1.3 MB of
    // CBs) that clashed with the remaining L1 tensors.
    llvm::SmallDenseSet<Value, 8> opOperands;
    uint64_t operandBytesInOccupied = 0;
    for (Value operand : op->getOperands()) {
      if (!opOperands.insert(operand).second) {
        continue;
      }
      if (memoryTracker.hasTensor(operand)) {
        operandBytesInOccupied += memoryTracker.getTensorSize(operand);
      }
    }
    uint64_t occupied = memoryTracker.getOccupiedL1();
    uint64_t othersOccupied =
        occupied > operandBytesInOccupied ? occupied - operandBytesInOccupied : 0;

    uint64_t footprint = cbPeakUsage + l1BuffersPeakUsage;
    uint64_t needed = footprint + othersOccupied + cbFragCushion;
    bool cbZoneFires = footprint > 0 && footprint <= l1BudgetPerCore &&
                       needed > l1BudgetPerCore;
    SPILL_PROBE("  CB_ZONE_TEST pos={0} cbPeak={1} transientPeak={2} footprint={3} "
                "occupied={4} operandBytes={5} needed={6} budget={7} "
                "nTracked={8} fires={9}",
                pos, cbPeakUsage, l1BuffersPeakUsage, footprint, occupied,
                operandBytesInOccupied, needed, l1BudgetPerCore,
                memoryTracker.getTrackedCount(), cbZoneFires);
    if (cbZoneFires) {
      uint64_t cbVT = l1BudgetPerCore > footprint ? l1BudgetPerCore - footprint : 0;
      bool anyEvicted = false;
      for (Value victim : memoryTracker.getValuesAboveVirtualThreshold(cbVT)) {
        if (!liveValues.count(victim)) {
          continue;
        }
        // Never evict an already-inserted reshard: it exists specifically to
        // supply an L1-sharded input to its consumer. Evicting it would trigger
        // inserting another reshard for the same consumer, creating an
        // unbounded chain of reshards (infinite loop / O(n^2) positionMap
        // iteration). evictUntil / evictFarthestUse already honour this guard;
        // apply the same rule here in the direct CB-zone eviction path.
        if (insertedReshardValues.count(victim)) {
          continue;
        }
        // Never evict this op's own operands: they are part of the validated
        // footprint, and moving one to DRAM changes the program the op runs
        // (and its CB / transient sizes) without re-validation.
        if (opOperands.count(victim)) {
          continue;
        }
        SPILL_PROBE("  ACTION=CB_ZONE_EVICT pos={0} footprint={1} cbVT={2} victim={3}",
                    pos, footprint, cbVT, probeOpName(victim.getDefiningOp()));
        size_t cbZoneCampaignMin = SIZE_MAX;
        evictValue(victim, pos, data, cbZoneCampaignMin);
        anyEvicted = true;
      }
      if (anyEvicted) {
        speculativeAddr = memoryTracker.wouldAllocateAt(l1Size);
        if (!speculativeAddr) {
          l1Size = handleNoFit(op, pos, data, opL1Usage, l1Size);
          speculativeAddr = memoryTracker.wouldAllocateAt(l1Size);
        }
      }
    }
    // Large-tensor fragmentation guard: if a single tensor exceeds ~40% of
    // the L1 budget, runtime fragmentation from untracked L1 allocations
    // (e.g. ToLayoutOp outputs that are excluded from liveValues by design)
    // can leave insufficient contiguous space even when the simulation's free
    // list shows a fit. For example, in Block A the conv2d outputs
    // 589,824 B/core but is immediately followed by to_memory_config → DRAM;
    // at runtime three prior ToLayoutOp outputs of 303,104 B each have been
    // allocated, the middle one freed, leaving only 420,576 B contiguous
    // (OOM). The simulation sees none of this because ToLayoutOp results are
    // deliberately not added to liveValues. Unconditionally sending any tensor
    // above this threshold to DRAM is conservative but correct: such tensors
    // are always close to half the budget and provide no net benefit when the
    // risk of fragmentation-induced OOM outweighs L1 savings.
    // Threshold = 40% × budget (530,261 B on WH N150) rejects the 589,824 B
    // Block A conv2d output while leaving smaller tensors (≤303,104 B) in L1.
    static constexpr double kMaxSingleTensorFraction = 0.40;
    // A tensor consumed by the very next op only (last use at pos + 1) is
    // not exposed to the fragmentation this guard protects against: no
    // untracked allocation can happen between its producer and its consumer.
    int64_t resultLastUse = -1;
    for (Value r : op->getResults()) {
      auto it = data.lastUsePositions.find(r);
      if (it != data.lastUsePositions.end()) {
        resultLastUse = std::max(resultLastUse, it->second);
      }
    }
    bool shortLived =
        !shortLivedExemptionDisabled() && resultLastUse == pos + 1;
    SPILL_PROBE("  LARGE_TENSOR_TEST pos={0} l1Size={1} threshold={2} shortLived={3} fires={4}",
                pos, l1Size,
                static_cast<uint64_t>(kMaxSingleTensorFraction *
                                      static_cast<double>(l1BudgetPerCore)),
                shortLived,
                !shortLived &&
                    l1Size > static_cast<uint64_t>(
                                 kMaxSingleTensorFraction *
                                 static_cast<double>(l1BudgetPerCore)));
    if (!shortLived &&
        l1Size > static_cast<uint64_t>(kMaxSingleTensorFraction *
                                       static_cast<double>(l1BudgetPerCore))) {
      SPILL_PROBE("  ACTION=LARGE_TENSOR_FRAG pos={0} l1Size={1} budget={2}",
                  pos, l1Size, l1BudgetPerCore);
      // Only the oversized OUTPUT goes to DRAM. The op's L1 operands stay
      // where they are: they are part of the footprint the op was validated
      // with, and evicting them turned a single DRAM write into a DRAM round
      // trip of the input as well (spill after the producer + the consumer
      // reading it back from DRAM), e.g. a 3x3 conv whose 368 KB input was
      // spilled and re-read only because its 553 KB output was too large.
      // evictForDramCBGrowth below re-validates the DRAM-output program and
      // evicts live tensors if its circular buffers grew.
      demoteToDram(op);
      evictForDramCBGrowth(op, pos, data, cbPeakUsage);
      return 0;
    }
    // The op's transient buffers (everything in l1BuffersPeakUsage beyond
    // its own operands and output, e.g. a conv2d halo) need a contiguous
    // region as well. The op-model allocates them in an empty L1 and cannot
    // see fragmentation; the hardware allocator can refuse ("largest free
    // block" smaller than the request) although the bytes are there.
    // Peak = max(operands + transient, transient + output) when the op frees
    // its input before allocating its output (conv2d deallocate_activation),
    // so transient = peak - max(operands, output) is the safe estimate.
    // An op that keeps its inputs holds inputs, transient and output at the
    // peak, so its transient is peak - inputs - output.
    uint64_t resident = opFreedInput(op)
                            ? std::max(operandBytesInOccupied, l1Size)
                            : operandBytesInOccupied + l1Size;
    if (l1BuffersPeakUsage > resident) {
      uint64_t transientBytes = l1BuffersPeakUsage - resident;
      bool fitsNow = compactForTransientBuffers(op, pos, data, l1Size, transientBytes);
      SPILL_PROBE("  TRANSIENT_TEST pos={0} transient={1} output={2} fits={3}", pos,
                  transientBytes, l1Size, fitsNow);
      // The output is placed while the transient buffers exist; remember
      // them so the live-set allocation leaves the same hole above it.
      pendingTransientBytes = transientModelDisabled() ? 0 : transientBytes;
      speculativeAddr = memoryTracker.wouldAllocateAt(
          l1Size, pendingTransientBytes, pendingFreedBefore,
          pendingTransientAfterFree);
      if (!speculativeAddr) {
        l1Size = handleNoFit(op, pos, data, opL1Usage, l1Size);
        speculativeAddr = memoryTracker.wouldAllocateAt(
            l1Size, pendingTransientBytes, pendingFreedBefore,
            pendingTransientAfterFree);
        if (!speculativeAddr || l1Size == 0) {
          return l1Size;
        }
      }
    }
    bool cbOverlap = wouldCBsOverlapTensors(op, pos, cbPeakUsage, *speculativeAddr);
    SPILL_PROBE("  CB_OVERLAP_TEST pos={0} cbPeak={1} addr={2} fires={3}", pos,
                cbPeakUsage, *speculativeAddr, cbOverlap);
    if (cbOverlap) {
      SPILL_PROBE("  ACTION=CB_OVERLAP_FRAGMENTATION pos={0}", pos);
      l1Size =
          handleFragmentation(op, pos, data, opL1Usage, cbPeakUsage, l1Size);
    }
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
  bool fitsAfterEviction = evictUntil(pos, data, [&]() {
    inputLayouts = utils::extractInputLayouts(op);
    result = memoryTracker.validate(op, inputLayouts, config);
    return result.isSuccess();
  });
  // evictUntil already failed the pass on unresolvable fragmentation; bail
  // before touching the truncated tracker.
  if (compilationFailed) {
    return;
  }

  // fitsAfterEviction ⟹ validate() succeeded against a fully-placed live set.
  if (fitsAfterEviction) {
    uint64_t l1Size = result.outputL1Usage;
    if (l1Size > 0) {
      l1Size = ensureFitsL1(op, pos, data, l1Size, result.cbPeakUsage, l1Size,
                            result.l1BuffersPeakUsage);
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
bool L1SpillManagement<MemoryTracker>::markEvictedAndRebuild(
    Value victim, size_t &campaignMin) {
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

  // Fold this victim into the campaign minimum and rebuild the whole suffix
  // from there. Restoring from the per-victim allocIdx would be unsound:
  // farthest-last-use victims are unordered w.r.t. alloc order, so a later
  // victim could restore a base predating an earlier eviction and resurrect an
  // already-spilled tensor. Replaying from the minimum with all accumulated
  // skips re-applies every eviction.
  campaignMin = std::min(campaignMin, allocIdx);
  return replayFrom(campaignMin);
}

//===----------------------------------------------------------------------===//
// replayFrom
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::replayFrom(size_t startIdx) {
  auto snapIt = addressSnapshots.find(startIdx);
  assert(snapIt != addressSnapshots.end() &&
         "snapshot not found for replay start index");
  memoryTracker.restoreSnapshot(snapIt->second);

  // Replay events from startIdx forward, skipping evicted tensors. Update
  // snapshots during replay so future restores see the current schedule.
  for (size_t i = startIdx; i < l1EventLog.size(); ++i) {
    if (l1EventLog[i].skipped) {
      continue;
    }
    if (l1EventLog[i].kind == L1Event::kDealloc) {
      memoryTracker.freeAddress(l1EventLog[i].tensor);
      continue;
    }
    addressSnapshots[i] = memoryTracker.takeSnapshot();
    // View-eligible reshape: alias if src is still address-tracked in this
    // replay (consumes no fresh L1), otherwise fresh-allocate (the
    // counterfactual when src was evicted to DRAM).
    Operation *defOp = l1EventLog[i].tensor.getDefiningOp();
    if (defOp && canReshapeBeView(defOp) &&
        memoryTracker.hasTensorAddress(defOp->getOperand(0))) {
      memoryTracker.allocateAddressAt(l1EventLog[i].tensor,
                                      defOp->getOperand(0));
      continue;
    }
    // Pre-check allocateAddress's hard contract and stop at the first no-fit:
    // a still-live tensor has no contiguous slot here (fragmentation). The
    // partial tracker is discarded by the next eviction's restore-from-min
    // (or, at terminal exhaustion, by failing the pass) — it is never trusted.
    if (!memoryTracker.wouldAllocateAt(l1EventLog[i].sizePerCore,
                                       l1EventLog[i].transientPerCore,
                                       l1EventLog[i].freedBefore,
                                       l1EventLog[i].transientAfterFree)) {
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "Replay: no contiguous fit for {0} bytes",
                   l1EventLog[i].sizePerCore);
      return false;
    }
    memoryTracker.allocateAddress(l1EventLog[i].tensor,
                                  l1EventLog[i].sizePerCore,
                                  l1EventLog[i].transientPerCore,
                                  l1EventLog[i].freedBefore,
                                  l1EventLog[i].transientAfterFree);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// evictUntil
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::evictValue(
    Value victim, int64_t pos, ScheduleData &data, size_t &campaignMin,
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

  // Place the DRAM copy right after the victim's last already-scheduled
  // consumer (position < pos) instead of right after the producer. Those
  // consumers were validated with the L1 value and keep reading it; only the
  // consumers at or after `pos` -- the ones the eviction is for -- read the
  // DRAM copy. Spilling at the producer forced every earlier consumer onto a
  // DRAM round trip (spill + reshard back to L1), e.g. a pool that follows
  // its conv immediately while a far-away skip connection is the real reason
  // the value is spilled. The L1 copy is still freed before `pos`, which is
  // what the eviction needs.
  Operation *insertSpillBefore = nullptr;
  int64_t lastPastPos = -1;
  for (Operation *user : victim.getUsers()) {
    auto userPosIt = data.positionMap.find(user);
    if (userPosIt == data.positionMap.end() || userPosIt->second >= pos ||
        userPosIt->second <= lastPastPos) {
      continue;
    }
    if (user->getBlock() != victimOp->getBlock() || !user->getNextNode()) {
      continue;
    }
    lastPastPos = userPosIt->second;
    insertSpillBefore = user->getNextNode();
  }
  SPILL_PROBE("  EVICT pos={0} victim={1} lastPastConsumerPos={2} spillAfterLastPast={3}",
              pos, probeOpName(victimOp), lastPastPos, insertSpillBefore != nullptr);
  // A convolution whose output is evicted right after it ran can write DRAM
  // itself: that saves the reshard tt-metal does inside the op to reach the
  // requested L1 grid before the separate copy. Only when the DRAM-output
  // program needs no more circular buffers than the validated L1 one.
  // A pointwise (1x1) conv2d runs as a sharded matmul; with an interleaved
  // output tt-metal switches program and its circular buffers no longer match
  // the op model's estimate (observed: 1.8 MB of CBs at runtime for a 611 KB
  // estimate), so only convolutions with a spatial kernel are demoted.
  auto isPointwiseConv = [](Operation *op) {
    auto conv = mlir::dyn_cast<Conv2dOp>(op);
    if (!conv) {
      return false;
    }
    auto ks = conv.getKernelSize();
    return ks.size() == 2 && ks[0] == 1 && ks[1] == 1;
  };
  bool demotedProducer = false;
  if (!insertSpillBefore && victimOp && !producerDemoteDisabled() &&
      mlir::isa<Conv2dOp, ConvTranspose2dOp>(victimOp) &&
      !isPointwiseConv(victimOp) && victimOp->getNumResults() == 1 &&
      data.positionMap.lookup(victimOp) == pos - 1) {
    auto inputLayouts = utils::extractInputLayouts(victimOp);
    auto l1Config = extractOpConfigFromIR(victimOp);
    auto l1Result = op_constraint_validation::validateOperation(
        victimOp, inputLayouts, l1Config, /*additionalL1Usage=*/0);
    demoteToDram(victimOp);
    auto dramConfig = extractOpConfigFromIR(victimOp);
    auto dramResult = op_constraint_validation::validateOperation(
        victimOp, inputLayouts, dramConfig, /*additionalL1Usage=*/0);
    if (dramResult.isSuccess() &&
        (!l1Result.isSuccess() ||
         dramResult.cbPeakUsage <= l1Result.cbPeakUsage)) {
      demotedProducer = true;
      SPILL_PROBE("  ACTION=DEMOTE_PRODUCER pos={0} victim={1} cbL1={2} cbDram={3}",
                  pos, probeOpName(victimOp), l1Result.cbPeakUsage,
                  dramResult.cbPeakUsage);
    } else {
      victim.setType(tensorType);
    }
  }
  if (demotedProducer) {
    // Consumers read the DRAM result directly; nothing to insert.
  } else if (insertSpillBefore) {
    spillToDramImpl(victim, insertSpillBefore);
  } else {
    spillToDram(victim);
  }
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
      if (isa<ToTensorSpecOp>(consumer)) {
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

  return markEvictedAndRebuild(victim, campaignMin);
}

//===----------------------------------------------------------------------===//
// compactForTransientBuffers
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::compactForTransientBuffers(
    Operation *op, int64_t pos, ScheduleData &data, uint64_t outputL1Size,
    uint64_t transientBytes) {
  // Output and transient buffers both allocate top-down first-fit. They fit
  // when each finds a block, and, if both land in the same free block (same
  // block end), when that block holds both.
  auto fits = [&]() {
    if (pendingFreedBefore) {
      // The transient buffers are allocated with the input still live; the
      // output after the input is freed (and the transient re-placed when
      // the op does that).
      return memoryTracker.wouldAllocateAt(transientBytes).has_value() &&
             memoryTracker
                 .wouldAllocateAt(outputL1Size, transientBytes,
                                  pendingFreedBefore,
                                  pendingTransientAfterFree)
                 .has_value();
    }
    auto outAddr = memoryTracker.wouldAllocateAt(outputL1Size);
    auto trAddr = memoryTracker.wouldAllocateAt(transientBytes);
    if (!outAddr || !trAddr) {
      return false;
    }
    uint64_t outEnd = *outAddr + outputL1Size, trEnd = *trAddr + transientBytes;
    uint64_t gap = outEnd > trEnd ? outEnd - trEnd : trEnd - outEnd;
    if (gap < 64) {
      return memoryTracker.wouldAllocateAt(outputL1Size + transientBytes)
          .has_value();
    }
    return true;
  };
  uint64_t occupied = memoryTracker.getOccupiedL1();
  if (pendingFreedBefore && memoryTracker.hasTensor(pendingFreedBefore)) {
    // The freed input's bytes are available again for the output.
    uint64_t freedSize = memoryTracker.getTensorSize(pendingFreedBefore);
    occupied = occupied > freedSize ? occupied - freedSize : 0;
  }
  uint64_t freeBytes = l1BudgetPerCore > occupied ? l1BudgetPerCore - occupied : 0;
  // The simulated addresses are only approximate (the hardware allocator
  // sees the ops' internal buffers, this model does not), so a contiguous
  // block in the model is no guarantee. Worst case, one live tensor sitting
  // in the middle of the free space halves the largest block: when the
  // transient buffers need more than half of the free bytes, compact.
  bool riskySplit =
      !liveValues.empty() && transientBytes * 2 + outputL1Size > freeBytes;
  if (fits() && !riskySplit) {
    return true;
  }
  SPILL_PROBE("  TRANSIENT_FRAG pos={0} op={1} transient={2} output={3} free={4} "
              "trackedLowest={5} simFits={6} riskySplit={7}",
              pos, probeOpName(op), transientBytes, outputL1Size, freeBytes,
              memoryTracker.getLowestOccupiedAddress(), fits(), riskySplit);
  if (freeBytes < outputL1Size + transientBytes) {
    return false; // capacity, not fragmentation: the eviction paths handle it
  }

  // Candidates: the op's own L1 operands first, then every other live L1
  // tensor (a reallocated tensor lands at the top of L1, so moving all of
  // them leaves one contiguous free region below).
  llvm::SmallVector<Value> candidates;
  for (Value operand : op->getOperands()) {
    if (liveValues.count(operand) && !llvm::is_contained(candidates, operand)) {
      candidates.push_back(operand);
    }
  }
  for (Value live : liveValues) {
    if (!llvm::is_contained(candidates, live) &&
        !insertedReshardValues.count(live)) {
      candidates.push_back(live);
    }
  }
  bool changed = false;
  for (Value operand : candidates) {
    uint64_t size = memoryTracker.getTensorSize(operand);
    if (!reallocateLiveValue(op, pos, data, operand)) {
      continue;
    }
    changed = true;
    SPILL_PROBE("  ACTION=REALLOCATE pos={0} size={1} newLowest={2} fits={3}",
                pos, size, memoryTracker.getLowestOccupiedAddress(), fits());
    if (fits() && !riskySplit) {
      break;
    }
  }
  return changed && fits();
}

//===----------------------------------------------------------------------===//
// reallocateLiveValue
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::reallocateLiveValue(Operation *op,
                                                           int64_t pos,
                                                           ScheduleData &data,
                                                           Value operand) {
  if (!liveValues.count(operand) || !memoryTracker.hasTensorAddress(operand)) {
    return false;
  }
  auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType) {
    return false;
  }
  Operation *defOp = operand.getDefiningOp();
  if (!defOp || defOp->getBlock() != op->getBlock()) {
    return false;
  }
  OpBuilder builder(op);
  auto reallocOp = builder.create<ReallocateOp>(
      ttmlir::utils::appendLocationSuffix(op->getLoc(), "_reallocate"),
      tensorType, operand);
  Value newVal = reallocOp.getResult();
  // The op and every later user read the new buffer (ttnn::reallocate
  // frees the old one); earlier users are untouched.
  for (OpOperand &use : llvm::make_early_inc_range(operand.getUses())) {
    Operation *user = use.getOwner();
    if (user == reallocOp.getOperation()) {
      continue;
    }
    if (user->getBlock() == op->getBlock() &&
        !user->isBeforeInBlock(reallocOp.getOperation())) {
      use.set(newVal);
    }
  }
  // Simulation: free the old slot, place the new one top-down first-fit.
  uint64_t size = memoryTracker.getTensorSize(operand);
  l1EventLog.push_back({L1Event::kDealloc, operand, 0, /*skipped=*/false});
  memoryTracker.removeTensor(operand);
  liveValues.erase(operand);
  allocEventIndex[newVal] = l1EventLog.size();
  addressSnapshots[l1EventLog.size()] = memoryTracker.takeSnapshot();
  l1EventLog.push_back({L1Event::kAlloc, newVal, size, /*skipped=*/false});
  memoryTracker.addTensor(newVal, size);
  liveValues.insert(newVal);
  // The moved value dies where its last user runs. lastUsePositions keeps the
  // numbering of the original schedule (only the relative order matters to
  // liveSet), while deathSchedule is re-keyed after every schedule insertion
  // and owns the freeing, so its key must come from the current positionMap:
  // the stale number would file the value in a bucket that has already been
  // processed and leave it live for the rest of the function.
  int64_t lastUse = pos;
  for (Operation *user : newVal.getUsers()) {
    auto posIt = data.positionMap.find(user);
    if (posIt != data.positionMap.end()) {
      lastUse = std::max(lastUse, posIt->second);
    }
  }
  data.lastUsePositions[newVal] = lastUse;
  SPILL_PROBE("  REALLOC_LIVE pos={0} operand={1} lastUse={2}", pos,
              probeOpName(defOp), lastUse);
  // The old value's death entry (if any) is harmless: it is no longer live.
  data.deathSchedule[lastUse].push_back(newVal);
  liveSet.push({lastUse, newVal});
  observer_->onEviction(defOp, pos, 0); // recorded as a placement change
  if (pendingFreedBefore == operand) {
    pendingFreedBefore = newVal; // the op now frees the moved tensor
  }
  // The moved tensor keeps its size entry under the new value; the old
  // value's size entry is gone with removeTensor. Callers that log the
  // size must read it before calling.
  return true;
}

//===----------------------------------------------------------------------===//
// compactForCBZone
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
bool L1SpillManagement<MemoryTracker>::compactForCBZone(
    Operation *op, int64_t pos, ScheduleData &data, uint64_t outputL1Size,
    uint64_t cushionedCBUsage) {
  auto clear = [&]() {
    auto specAddr = memoryTracker.wouldAllocateAt(
        outputL1Size, pendingTransientBytes, pendingFreedBefore,
        pendingTransientAfterFree);
    if (!specAddr) {
      return false;
    }
    uint64_t effLowest =
        std::min(*specAddr, memoryTracker.getLowestOccupiedAddress());
    return cushionedCBUsage <= effLowest;
  };
  uint64_t occupied = memoryTracker.getOccupiedL1();
  uint64_t freeBytes = l1BudgetPerCore > occupied ? l1BudgetPerCore - occupied : 0;
  if (freeBytes < outputL1Size + pendingTransientBytes) {
    return false; // capacity, not placement: the eviction path handles it
  }
  // Tensors below the CB top, the op's own operands first (they are read by
  // the op whose program defines the CB region).
  llvm::SmallVector<Value> candidates;
  for (Value operand : op->getOperands()) {
    if (liveValues.count(operand) && !llvm::is_contained(candidates, operand)) {
      candidates.push_back(operand);
    }
  }
  llvm::SmallVector<std::pair<uint64_t, Value>> inZone;
  for (Value live : liveValues) {
    auto addr = memoryTracker.getTensorAddress(live);
    if (!addr || *addr >= cushionedCBUsage || llvm::is_contained(candidates, live) ||
        insertedReshardValues.count(live)) {
      continue;
    }
    inZone.emplace_back(*addr, live);
  }
  llvm::sort(inZone, [](const auto &a, const auto &b) { return a.first < b.first; });
  for (auto &[addr, live] : inZone) {
    candidates.push_back(live);
  }
  bool changed = false;
  for (Value v : candidates) {
    if (clear()) {
      break;
    }
    uint64_t size =
        memoryTracker.hasTensor(v) ? memoryTracker.getTensorSize(v) : 0;
    if (!reallocateLiveValue(op, pos, data, v)) {
      continue;
    }
    changed = true;
    SPILL_PROBE("  ACTION=CB_ZONE_REALLOCATE pos={0} size={1} newLowest={2} clear={3}",
                pos, size, memoryTracker.getLowestOccupiedAddress(), clear());
  }
  return changed && clear();
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

  // A fresh campaign: every rebuild restores from the smallest alloc index
  // evicted in this campaign and replays the whole suffix with all accumulated
  // skips (see evictValue / markEvictedAndRebuild).
  size_t campaignMin = SIZE_MAX;

  // shouldStop() inspects tracker state, so it is only trustworthy after a
  // rebuild that placed every live tensor. The entry state qualifies — the
  // forward sweep placed everything via the infallible allocateAddress.
  bool rebuildPlacedAll = true;
  while (!rebuildPlacedAll || !shouldStop()) {
    if (liveValues.empty() || onlyReshardsRemain()) {
      break;
    }
    Value victim = evictFarthestUse();
    if (!victim) {
      break;
    }
    rebuildPlacedAll = evictValue(victim, pos, data, campaignMin);
  }

  // Demoting the current op cannot help — the unplaceable tensor is in the
  // already-scheduled suffix (or a non-evictable inserted reshard), independent
  // of the op whose output is not yet added. Fail loudly instead of shipping a
  // layout that clashes at runtime.
  if (!rebuildPlacedAll) {
    Operation *blockedOp = data.schedule[pos];
    blockedOp->emitError(
        "L1SpillManagement: a live tensor has no contiguous L1 placement even "
        "after evicting every spillable tensor (fragmentation: a non-evictable "
        "inserted reshard or irreducible working set cannot be relocated)");
    compilationFailed = true;
    return false;
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
                                                       uint64_t opL1Usage,
                                                       uint64_t outputL1Size) {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    NO_FIT: output {0} bytes can't fit contiguously, evicting",
               outputL1Size);

  bool fitsAfterEviction = evictUntil(pos, data, [&]() {
    return memoryTracker.wouldAllocateAt(outputL1Size).has_value();
  });
  // evictUntil already failed the pass on unresolvable fragmentation; bail
  // before touching the truncated tracker.
  if (compilationFailed) {
    return 0;
  }
  if (!fitsAfterEviction) {
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
  uint64_t freshL1 = freshResult.outputL1Usage;
  // Honor allocateAddress's contract: the size handed back must still fit
  // (re-validate could in principle return a larger output than was checked).
  if (freshResult.isSuccess() &&
      (freshL1 == 0 || memoryTracker.wouldAllocateAt(freshL1))) {
    applyOutputConfig(op, freshResult);
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
    Operation *op, int64_t pos, ScheduleData &data, uint64_t opL1Usage,
    uint64_t cbPeakUsage, uint64_t outputL1Size) {
  // Add the same safety cushion as wouldCBsOverlapTensors to account for
  // unmodeled runtime fragmentation from transient internal op allocations.
  uint64_t cushionedCBUsage = cbPeakUsage + cbFragCushion;

  // First choice: move the tensors that sit in the CB region to the top of
  // L1 (ttnn.reallocate, ~10-20 us) instead of sending them to DRAM. This is
  // the usual case after an op whose internal buffers left a hole above its
  // output: the next output was pushed down into the CB region although L1
  // is mostly empty.
  if (!cbZoneCompactDisabled() &&
      compactForCBZone(op, pos, data, outputL1Size, cushionedCBUsage)) {
    SPILL_PROBE("  ACTION=CB_ZONE_COMPACTED pos={0} lowest={1}", pos,
                memoryTracker.getLowestOccupiedAddress());
    auto inputLayouts = utils::extractInputLayouts(op);
    auto config = extractOpConfigFromIR(op);
    auto freshResult = memoryTracker.validate(op, inputLayouts, config);
    uint64_t freshL1 = freshResult.outputL1Usage;
    if (freshResult.isSuccess() &&
        (freshL1 == 0 ||
         memoryTracker.wouldAllocateAt(freshL1, pendingTransientBytes,
                                       pendingFreedBefore,
                                       pendingTransientAfterFree))) {
      applyOutputConfig(op, freshResult);
      return freshL1;
    }
  }

  // Evict (farthest last-use first) until the output has a contiguous fit AND
  // the CB region no longer overlaps the lowest live tensor.
  bool fitsAfterEviction = evictUntil(pos, data, [&]() {
    auto specAddr = memoryTracker.wouldAllocateAt(outputL1Size);
    if (!specAddr) {
      return false;
    }
    uint64_t effLowest =
        std::min(*specAddr, memoryTracker.getLowestOccupiedAddress());
    return cushionedCBUsage <= effLowest;
  });
  // evictUntil already failed the pass on unresolvable fragmentation; bail
  // before touching the truncated tracker.
  if (compilationFailed) {
    return 0;
  }

  // Eviction was exhausted (op's own output won't fit / CB still overlaps).
  // Demote this op's output to DRAM rather than ship a clashing layout.
  if (!fitsAfterEviction) {
    demoteToDram(op);
    evictForDramCBGrowth(op, pos, data);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    FRAG_DEMOTE: eviction exhausted, output to DRAM");
    return 0;
  }

  // Re-extract layouts after eviction and re-validate.
  auto inputLayouts = utils::extractInputLayouts(op);
  auto config = extractOpConfigFromIR(op);
  auto freshResult = memoryTracker.validate(op, inputLayouts, config);
  uint64_t freshL1 = freshResult.outputL1Usage;
  // Honor allocateAddress's contract: the size handed back must still fit
  // largest free block.
  if (freshResult.isSuccess() &&
      (freshL1 == 0 || memoryTracker.wouldAllocateAt(freshL1))) {
    applyOutputConfig(op, freshResult);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    FRAG_RESOLVED: L1 now {0}/{1}",
                 memoryTracker.getOccupiedL1(), l1BudgetPerCore);
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
    // Its own mini-campaign: campaignMin accumulates across the sibling spills.
    size_t campaignMin = SIZE_MAX;
    bool fitsAfterSiblingSpill = true;
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
      //
      // Those reshards can fragment L1; keep the last result, since the replay
      // reflects the fully-spilled state and early fragmentation clears as the
      // rest spill.
      fitsAfterSiblingSpill = evictValue(victim, pos, data, campaignMin,
                                         /*skipReshardConsumer=*/op);
    }
    if (!fitsAfterSiblingSpill) {
      op->emitError(
          "L1SpillManagement: sibling-operand spill left a live tensor with no "
          "contiguous L1 placement (fragmentation cannot be resolved by "
          "demoting this op)");
      compilationFailed = true;
      return 0;
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
    // A handler hit unresolvable fragmentation and failed the pass; stop the
    // sweep rather than process further ops against a discarded result.
    if (compilationFailed) {
      break;
    }
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

    // ToTensorSpecOp requires special handling due to the fact it is a complex
    // op which decomposes into few real TTNN ops. MemoryLayoutPropagation skips
    // these, and pre-decomposition OpModel is impossible.
    if (isa<ToTensorSpecOp>(op)) {
      auto resultType =
          mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
      auto lo =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(resultType.getEncoding());
      assert(lo && "ToLayoutOp result must have TTNNLayoutAttr encoding");
      if (lo.hasL1BufferType()) {
        // Case with L1 output: workaround-inserted and always immediately
        // consumed by the target op.
        // Use farthest-last-use eviction for live tensors if needed to create
        // room, but do not add the output to liveValues — it is not a
        // long-lived L1 tenant and will be gone (or dead) before any subsequent
        // eviction decision matters. Being absent from liveValues also means
        // evictAllFromL1 (e.g. triggered by DistributedRMSNormOp's
        // isNotImplemented) cannot spill it.
        uint64_t derivedL1 = utils::getPerCoreL1Usage(
            lo, ttmlir::utils::volume(lo.getGridShape()));
        TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                     "  [pos={0}] L1_TOLAYOUT: {1}, derivedL1={2} bytes, "
                     "occupied={3}/{4}",
                     pos, ttmlir::opToString(op), derivedL1,
                     memoryTracker.getOccupiedL1(), l1BudgetPerCore);
        // CBPeakUsage fixed to 0 as we have no validation result for ToLayoutOp
        // itself which is yet to be decomposed.
        ensureFitsL1(op, pos, data, derivedL1, /*cbPeakUsage=*/0, derivedL1,
                     /*l1BuffersPeakUsage=*/0);
      }

      // Regardless of whether the ToLayoutOp's output is L1 or DRAM, we have no
      // way of validating this op since it is yet to be decomposed. Usually
      // this is workaround inserted op, so we can skip it.
      continue;
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
        uint64_t transient = pendingTransientBytes;
        Value freedBefore = pendingFreedBefore;
        bool afterFree = pendingTransientAfterFree;
        pendingTransientBytes = 0; // only the first result carries them
        pendingFreedBefore = Value();
        pendingTransientAfterFree = false;
        l1EventLog.push_back({L1Event::kAlloc, val, perResultL1,
                              /*skipped=*/false, transient, freedBefore,
                              afterFree});

        // View-eligible reshape: alias src's buffer instead of carving a
        // fresh slot. See `addTensorAtAddress`. Must stay in sync with the
        // willAliasSourceInL1 short-circuit in ensureFitsL1.
        Operation *defOp = val.getDefiningOp();
        if (defOp && willAliasSourceInL1(defOp)) {
          memoryTracker.addTensorAtAddress(val, perResultL1,
                                           defOp->getOperand(0));
        } else {
          // The fit was checked with these parameters in ensureFitsL1, but
          // evictions and compaction since then may have changed the free
          // list. Degrade the model rather than abort: without the
          // transient, then without the early free.
          if (!memoryTracker.wouldAllocateAt(perResultL1, transient,
                                             freedBefore, afterFree)) {
            SPILL_PROBE("  ALLOC_FALLBACK pos={0} size={1} transient={2}",
                        pos, perResultL1, transient);
            transient = 0;
            l1EventLog.back().transientPerCore = 0;
            if (!memoryTracker.wouldAllocateAt(perResultL1, 0, freedBefore,
                                               false)) {
              freedBefore = Value();
              l1EventLog.back().freedBefore = Value();
            }
            afterFree = false;
            l1EventLog.back().transientAfterFree = false;
          }
          memoryTracker.addTensor(val, perResultL1, transient, freedBefore,
                                  afterFree);
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

      // The backend can report outputL1Usage == 0 for an op whose IR result
      // type is L1 (seen on the full BEV model: ttnn.reshape
      // 1x3x1280x2304 -> 1x1x96x92160 with an L1-interleaved result). The
      // runtime honours the IR type, so the buffer WILL exist (276,480 B/core
      // in that case) while the tracker would carry 0 for it; the consumer's
      // CB/capacity checks then under-count L1 by that much and the program
      // clashes with its circular buffers at runtime. Fall back to the
      // layout-derived size, as is already done for ToTensorSpecOp.
      if (l1Size == 0 && hasL1Output) {
        uint64_t derived = 0;
        for (auto r : tensorResults) {
          auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
              mlir::cast<RankedTensorType>(r.getType()).getEncoding());
          if (lo && lo.hasL1BufferType()) {
            derived += utils::getPerCoreL1Usage(
                lo, ttmlir::utils::volume(lo.getGridShape()));
          }
        }
        SPILL_PROBE("  FIXUP=L1_RESULT_ZERO_USAGE pos={0} op={1} derived={2}",
                    pos, probeOpName(op), derived);
        l1Size = derived;
      }

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    VALIDATION SUCCESS: op {0}, "
                   "cbPeakUsage={1}, outputL1={2} bytes",
                   ttmlir::opToString(op), result.cbPeakUsage, l1Size);

      l1Size = ensureFitsL1(op, pos, data, l1Size, result.cbPeakUsage, l1Size,
                            result.l1BuffersPeakUsage);
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

//===----------------------------------------------------------------------===//
// spillLiveUntrackedToLayoutOutputs
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::spillLiveUntrackedToLayoutOutputs(
    int64_t pos, ScheduleData &data) {
  llvm::SmallVector<Value> toSpill;
  for (auto &[val, lastUse] : data.lastUsePositions) {
    if (lastUse < pos) {
      continue;
    }
    Operation *defOp = val.getDefiningOp();
    if (!defOp || !isa<ToLayoutOp>(defOp)) {
      continue;
    }
    auto posIt = data.positionMap.find(defOp);
    if (posIt == data.positionMap.end() || posIt->second >= pos) {
      continue;
    }
    auto tensorType = mlir::dyn_cast<RankedTensorType>(val.getType());
    if (!tensorType) {
      continue;
    }
    auto lo = mlir::dyn_cast<TTNNLayoutAttr>(tensorType.getEncoding());
    if (!lo || !lo.hasL1BufferType()) {
      continue;
    }
    toSpill.push_back(val);
  }
  for (Value val : toSpill) {
    SPILL_PROBE("  ACTION=UNTRACKED_TOLAYOUT_SPILL pos={0}", pos);
    spillToDram(val);
  }
}

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
    Operation *op, int64_t pos, ScheduleData &data,
    uint64_t knownCBPeak) {

  auto inputLayouts = utils::extractInputLayouts(op);
  auto config = extractOpConfigFromIR(op);
  auto result =
      op_constraint_validation::validateOperation(op, inputLayouts, config,
                                                  /*additionalL1Usage=*/0);


  if (!result.isSuccess()) {
    op->emitError("L1SpillManagement: DRAM output config failed validation "
                  "after demotion (")
        << result.errorMessage << "); this indicates a compiler bug";
    compilationFailed = true;
    return;
  }

  // Use the larger of the fresh query result and the known CB peak from before
  // demotion. The fresh query may return cbPeakUsage=0 if the DRAM-output
  // path doesn't model internal sub-operation CBs (e.g., conv2d_L1 uses a
  // block_sharded L1 reshard internally even when the output is DRAM).
  uint64_t effectiveCBPeak = std::max(result.cbPeakUsage, knownCBPeak);
  if (effectiveCBPeak == 0) {
    return;
  }

  uint64_t dramCBCushioned = effectiveCBPeak + cbFragCushion;

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    DRAM_CB_CHECK: effectiveCBPeak={0} (fresh={1}, known={2}), "
               "cushion={3}, cushionedDramCB={4}, lowestExisting={5}",
               effectiveCBPeak, result.cbPeakUsage, knownCBPeak, cbFragCushion,
               dramCBCushioned, memoryTracker.getLowestOccupiedAddress());

  evictForCBOverlap(dramCBCushioned, pos, data);

  // ToLayoutOp outputs are excluded from liveValues and tensorAddresses, so
  // evictForCBOverlap cannot reach them. They occupy real hardware L1 and can
  // land within the op's CB region, causing validate_circular_buffer_region to
  // throw, so they have to be spilled explicitly.
  spillLiveUntrackedToLayoutOutputs(pos, data);
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
