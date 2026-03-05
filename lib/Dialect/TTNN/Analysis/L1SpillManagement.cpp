// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <queue>

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// SumL1MemoryTracker
//===----------------------------------------------------------------------===//

op_constraint_validation::ValidationResult
SumL1MemoryTracker::validate(Operation *op,
                             llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                             const OpConfig &config) const {
  return op_constraint_validation::validateOperation(op, inputLayouts, config,
                                                     currentOccupied);
}

uint64_t SumL1MemoryTracker::getOccupiedL1() const { return currentOccupied; }

void SumL1MemoryTracker::addTensor(Value result, uint64_t l1SizePerCore) {
  tensorSizes[result] = l1SizePerCore;
  currentOccupied += l1SizePerCore;
}

void SumL1MemoryTracker::removeTensor(Value result) {
  auto it = tensorSizes.find(result);
  if (it != tensorSizes.end()) {
    currentOccupied -= it->second;
    tensorSizes.erase(it);
  }
}

bool SumL1MemoryTracker::hasTensor(Value result) const {
  return tensorSizes.count(result);
}

uint64_t SumL1MemoryTracker::getTensorSize(Value result) const {
  auto it = tensorSizes.find(result);
  return it != tensorSizes.end() ? it->second : 0;
}

//===----------------------------------------------------------------------===//
// L1SpillManagement
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
L1SpillManagement<MemoryTracker>::L1SpillManagement(
    func::FuncOp func, ttcore::GridAttr deviceGrid, uint64_t l1BudgetPerCore,
    std::unique_ptr<L1SpillObserver> observer)
    : func(func), deviceGrid(deviceGrid), l1BudgetPerCore(l1BudgetPerCore) {
  if (observer) {
    observer_ = std::move(observer);
  } else {
    observer_ = std::make_unique<L1SpillObserver>();
  }
}

//===----------------------------------------------------------------------===//
// extractOpConfigFromIR
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
OpConfig
L1SpillManagement<MemoryTracker>::extractOpConfigFromIR(Operation *op) {
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
      .Default([](Operation *) {});

  return config;
}

//===----------------------------------------------------------------------===//
// makeL1InterleavedConfig
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
OpConfig
L1SpillManagement<MemoryTracker>::makeL1InterleavedConfig(Operation *op) {
  OpConfig config = extractOpConfigFromIR(op);
  config.outputLayout = config.outputLayout.withBufferType(BufferType::L1)
                            .withMemoryLayout(TensorMemoryLayout::Interleaved);
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

    liveValues.erase(candidateVal);
    return candidateVal;
  }
  return Value();
}

//===----------------------------------------------------------------------===//
// applyDemotedConfig
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::applyDemotedConfig(
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
    TTNNLayoutAttr resultLayout = (ri < result.actualOutputLayouts.size())
                                      ? result.actualOutputLayouts[ri]
                                      : chosenLayout;
    if (!resultLayout) {
      continue;
    }

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

  // Update layout attribute for ops that have layout interface (op-level).
  if (auto opWithLayoutIF = mlir::dyn_cast<TTNNLayoutOpInterface>(op)) {
    opWithLayoutIF.setLayoutAttr(
        LayoutAttr::get(op->getContext(), chosenLayout.getLayout()));
  }

  // Update output data type attribute (op-level, uses result 0's layout).
  if (auto dtypeOp = mlir::dyn_cast<TTNNDtypeOpInterface>(op)) {
    ttcore::DataTypeAttr newDataTypeAttr =
        ttcore::DataTypeAttr::get(op->getContext(), chosenLayout.getDataType());
    dtypeOp.setDtypeAttr(newDataTypeAttr);
  }

  // Update L1 usage attribute.
  if (chosenLayout.hasL1BufferType() && result.outputL1Usage > 0) {
    OpBuilder builder(op->getContext());
    op->setAttr("ttnn.output_l1_usage",
                builder.getI64IntegerAttr(result.outputL1Usage));
  } else {
    op->removeAttr("ttnn.output_l1_usage");
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
      if (isa<ToLayoutOp, ToMemoryConfigOp>(consumer)) {
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
        applyDemotedConfig(consumer, result);
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

  // Build schedule (ops in IR order = topological order).
  func->walk([&](Operation *op) {
    if (op->getNumResults() == 0) {
      return;
    }
    if (isa<EmptyOp>(op)) {
      return;
    }
    bool hasTensorResult = llvm::any_of(op->getResults(), [](OpResult r) {
      return mlir::isa<RankedTensorType>(r.getType());
    });
    if (!hasTensorResult) {
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
// handleOOM
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::handleOOM(
    Operation *op, int64_t pos, llvm::ArrayRef<OpResult> tensorResults,
    const ScheduleData &data, uint64_t opL1Usage,
    std::function<void(uint64_t)> addResultsToLiveSet) {
  observer_->onOOM(op, pos, memoryTracker.getOccupiedL1());
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "    OOM: validation failed, trying demotion/eviction");

  auto inputLayouts = utils::extractInputLayouts(op);
  auto config = extractOpConfigFromIR(op);

  // Stage 1: Demote current op to L1 interleaved.
  // Matmul: skip L1-interleaved demotion (see getOutputHints).
  if (!isa<MatmulOp, LinearOp>(op)) {
    OpConfig l1InterleavedConfig = makeL1InterleavedConfig(op);
    auto demoteResult =
        memoryTracker.validate(op, inputLayouts, l1InterleavedConfig);

    if (demoteResult.isSuccess()) {
      observer_->onDemotion(op, pos, /*success=*/true,
                            demoteResult.outputL1Usage);
      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    DEMOTED to L1 interleaved: outputL1Usage={0}",
                   demoteResult.outputL1Usage);
      applyDemotedConfig(op, demoteResult);
      addResultsToLiveSet(demoteResult.outputL1Usage);
      return;
    }
    observer_->onDemotion(op, pos, /*success=*/false, 0);
  }

  // Stage 2: Evict from live set (Belady: farthest last-use first).
  auto result = memoryTracker.validate(op, inputLayouts, config);
  while (!result.isSuccess() && !liveValues.empty()) {
    Value victim = evictFarthestUse();
    if (!victim) {
      break;
    }

    Operation *victimOp = victim.getDefiningOp();
    uint64_t freedBytes = memoryTracker.getTensorSize(victim);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    EVICT: {0} (L1: {1} bytes)", ttmlir::opToString(victimOp),
                 freedBytes);
    observer_->onEviction(victimOp, pos, freedBytes);

    spillToDram(victim);
    memoryTracker.removeTensor(victim);

    revalidateConsumers(victimOp, pos, data.positionMap);

    // Re-extract input layouts (victim may have been input to current op).
    inputLayouts = utils::extractInputLayouts(op);
    result = memoryTracker.validate(op, inputLayouts, config);
  }

  if (result.isSuccess()) {
    uint64_t l1Size =
        result.outputL1Usage > 0 ? result.outputL1Usage : opL1Usage;
    addResultsToLiveSet(l1Size);
    observer_->onLiveAdded(op, pos, l1Size, pos, memoryTracker.getOccupiedL1());

    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    ADDED (after eviction): L1 now {0}/{1} ({2} tensors)",
                 memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                 liveValues.size());
  } else {
    // Stage 3: Op exceeds budget alone -- spill all results to DRAM.
    observer_->onSelfSpill(op, pos);
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    SPILL SELF: op exceeds budget alone");
    for (auto r : tensorResults) {
      spillToDram(r);
    }
  }
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

  // Belady's algorithm sweep with validation-based eviction.
  for (int64_t pos = 0; pos < static_cast<int64_t>(data.schedule.size());
       ++pos) {
    Operation *op = data.schedule[pos];

    processDeadTensors(pos, data);

    // Skip ops without L1 output annotation.
    auto l1Attr = op->getAttrOfType<IntegerAttr>("ttnn.output_l1_usage");
    if (!l1Attr) {
      continue;
    }

    uint64_t opL1Usage = l1Attr.getValue().getZExtValue();

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
                 "    output L1: {2} bytes, tensor results: {3}\n"
                 "    occupied L1 before: {4}/{5} ({6} tensors)",
                 pos, ttmlir::opToString(op), opL1Usage, numTensorResults,
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
        int64_t resultLastUse =
            (luIt != data.lastUsePositions.end()) ? luIt->second : pos;
        memoryTracker.addTensor(val, perResultL1);
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
      // Op not validated by backend — skip.
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "    NOT_IMPLEMENTED: skipping validation for {0}",
                   ttmlir::opToString(op));
      continue;
    }

    if (result.isSuccess()) {
      // Validation passed -- add all tensor results to live set.
      uint64_t l1Size =
          result.outputL1Usage > 0 ? result.outputL1Usage : opL1Usage;
      addResultsToLiveSet(l1Size);
      observer_->onLiveAdded(op, pos, l1Size, pos,
                             memoryTracker.getOccupiedL1());

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    ADDED: L1 now {0}/{1} ({2} tensors)",
                   memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                   liveValues.size());
      continue;
    }

    // Backend constraint error: demote directly to DRAM.
    if (result.isMetalBackendError()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    WARNING: Backend constraint error at pos {0} for {1}: "
                   "{2}. Demoting to DRAM (may indicate a layout propagation "
                   "bug).",
                   pos, ttmlir::opToString(op), result.errorMessage);
      for (auto r : tensorResults) {
        spillToDram(r);
      }
      ++spillCount;
      continue;
    }

    handleOOM(op, pos, tensorResults, data, opL1Usage, addResultsToLiveSet);
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

  // Step 4: Cleanup L1 usage attributes.
  cleanupL1UsageAttrs();
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
// spillToDram
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::spillToDram(Value result) {
  Operation *defOp = result.getDefiningOp();
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(result.getType());
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  // Create DRAM interleaved layout.
  TTNNLayoutAttr dramLayout =
      layoutAttr.withBufferType(BufferType::DRAM)
          .withMemoryLayout(TensorMemoryLayout::Interleaved);
  RankedTensorType newTensorType =
      utils::RankedTensorTypeFactory::create(tensorType, dramLayout);

  MemoryConfigAttr memConfigAttr = MemoryConfigAttr::get(
      defOp->getContext(), dramLayout.getMemLayout(),
      BufferTypeAttr::get(defOp->getContext(), BufferType::DRAM),
      utils::createShardSpecIfNeeded(dramLayout, deviceGrid));

  OpBuilder builder(defOp->getContext());
  builder.setInsertionPointAfter(defOp);
  Location loc = ttmlir::utils::appendLocationSuffix(defOp->getLoc(), "_spill");

  // Save all uses, insert ToMemoryConfigOp, reconnect uses.
  llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
  for (auto &use : result.getUses()) {
    uses.emplace_back(use.getOwner(), use.getOperandNumber());
  }

  Operation *spillOp = builder.create<ToMemoryConfigOp>(loc, newTensorType,
                                                        result, memConfigAttr);

  for (auto &[useOp, operandIdx] : uses) {
    useOp->setOperand(operandIdx, spillOp->getResult(0));
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "Inserted spill-to-DRAM op: {0}", spillOp);
}

//===----------------------------------------------------------------------===//
// cleanupL1UsageAttrs
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::cleanupL1UsageAttrs() {
  func->walk([](Operation *op) {
    if (op->hasAttr("ttnn.output_l1_usage")) {
      op->removeAttr("ttnn.output_l1_usage");
    }
  });
}

//===----------------------------------------------------------------------===//
// Explicit template instantiation
//===----------------------------------------------------------------------===//

template class L1SpillManagement<SumL1MemoryTracker>;

} // namespace mlir::tt::ttnn
