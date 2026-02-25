// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

void SumL1MemoryTracker::addTensor(Operation *op, uint64_t l1SizePerCore) {
  tensorSizes[op] = l1SizePerCore;
  currentOccupied += l1SizePerCore;
}

void SumL1MemoryTracker::removeTensor(Operation *op) {
  auto it = tensorSizes.find(op);
  if (it != tensorSizes.end()) {
    currentOccupied -= it->second;
    tensorSizes.erase(it);
  }
}

bool SumL1MemoryTracker::hasTensor(Operation *op) const {
  return tensorSizes.count(op);
}

uint64_t SumL1MemoryTracker::getTensorSize(Operation *op) const {
  auto it = tensorSizes.find(op);
  return it != tensorSizes.end() ? it->second : 0;
}

//===----------------------------------------------------------------------===//
// L1SpillManagement
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
L1SpillManagement<MemoryTracker>::L1SpillManagement(func::FuncOp func,
                                                    ttcore::GridAttr deviceGrid,
                                                    uint64_t l1BudgetPerCore)
    : func(func), deviceGrid(deviceGrid), l1BudgetPerCore(l1BudgetPerCore) {}

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
Operation *L1SpillManagement<MemoryTracker>::evictFarthestUse() {
  while (!liveSet.empty()) {
    auto [lastUse, candidateOp] = liveSet.top();
    liveSet.pop();

    // Skip already-evicted entries (lazy deletion).
    if (!liveOps.count(candidateOp)) {
      continue;
    }

    liveOps.erase(candidateOp);
    return candidateOp;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// applyDemotedConfig
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::applyDemotedConfig(
    Operation *op, const op_constraint_validation::ValidationResult &result) {
  TTNNLayoutAttr chosenLayout = result.actualOutputLayout;
  if (!chosenLayout) {
    return;
  }

  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  // Preserve quantized element types.
  Type originalElementType = tensorType.getElementType();
  Type newElementType = originalElementType;
  if (!mlir::isa<mlir::quant::QuantizedType>(originalElementType)) {
    newElementType = chosenLayout.getScalarElementType();
  }

  RankedTensorType newTensorType =
      RankedTensorType::get(tensorShape, newElementType, chosenLayout);

  // Update layout attribute for ops that have layout interface.
  if (auto opWithLayoutIF = mlir::dyn_cast<TTNNLayoutOpInterface>(op)) {
    opWithLayoutIF.setLayoutAttr(
        LayoutAttr::get(op->getContext(), chosenLayout.getLayout()));
  }

  // Update result type.
  op->getResult(0).setType(newTensorType);

  // Update output data type attribute.
  if (auto dtypeOp = mlir::dyn_cast<TTNNDtypeOpInterface>(op)) {
    ttcore::DataTypeAttr newDataTypeAttr =
        ttcore::DataTypeAttr::get(op->getContext(), chosenLayout.getDataType());
    dtypeOp.setDtypeAttr(newDataTypeAttr);
  }

  // Update DPS operand (EmptyOp).
  if (isa<mlir::DestinationStyleOpInterface>(op)) {
    BufferType bufferType = chosenLayout.getBufferType();
    TensorMemoryLayoutAttr tensorMemoryLayoutAttr = chosenLayout.getMemLayout();

    op->getOperands().back().setType(newTensorType);
    EmptyOp emptyOp =
        mlir::cast<EmptyOp>(op->getOperands().back().getDefiningOp());

    emptyOp.setDtype(chosenLayout.getDataType());
    if (chosenLayout.isTiled()) {
      emptyOp.setLayout(ttnn::Layout::Tile);
    } else {
      emptyOp.setLayout(ttnn::Layout::RowMajor);
    }

    emptyOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
        op->getContext(), tensorMemoryLayoutAttr,
        BufferTypeAttr::get(op->getContext(), bufferType),
        utils::createShardSpecIfNeeded(chosenLayout, deviceGrid)));
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

    // Find all downstream consumers of `changed`.
    // After spillToDram, changed->getResult(0) may have a ToMemoryConfigOp
    // user (spill op). Consumers use the spill op's result.
    // For non-spilled ops (demoted in a previous cascade step), consumers
    // use changed->getResult(0) directly.
    llvm::SmallVector<Operation *> consumers;
    for (Operation *user : changed->getResult(0).getUsers()) {
      if (isa<ToMemoryConfigOp>(user)) {
        // Spill op — collect its users (the original consumers).
        for (Operation *consumer : user->getResult(0).getUsers()) {
          consumers.push_back(consumer);
        }
      } else {
        consumers.push_back(user);
      }
    }

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

      if (result.isSuccess() &&
          result.actualOutputLayout != config.outputLayout) {
        // Backend returned a different output layout for this consumer.
        // Update consumer's IR to match.
        applyDemotedConfig(consumer, result);
        // Update memory tracker if consumer is live.
        if (memoryTracker.hasTensor(consumer)) {
          memoryTracker.removeTensor(consumer);
          memoryTracker.addTensor(consumer, result.outputL1Usage);
        }

        TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                     "  REVALIDATE: consumer {0} output layout changed "
                     "-- cascading to its consumers",
                     ttmlir::opToString(consumer));

        // Consumer's output changed — cascade to its consumers.
        worklist.push_back(consumer);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// run
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::run() {
  // Step 1: Build schedule (ops in IR order = topological order).
  llvm::SmallVector<Operation *> schedule;
  func->walk([&](Operation *op) {
    if (op->getNumResults() == 0) {
      return;
    }
    if (isa<EmptyOp>(op)) {
      return;
    }
    if (!isa<RankedTensorType>(op->getResult(0).getType())) {
      return;
    }
    schedule.push_back(op);
  });

  // Step 2: Compute last-use positions and build position map.
  llvm::DenseMap<Operation *, int64_t> lastUsePositions =
      computeLastUsePositions(schedule);

  // Build position map for revalidateConsumers.
  llvm::DenseMap<Operation *, int64_t> positionMap;
  for (int64_t i = 0; i < static_cast<int64_t>(schedule.size()); ++i) {
    positionMap[schedule[i]] = i;
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "=== L1 Memory View (compile-time, validation-based) ===\n"
               "  Budget per core: {0} bytes\n"
               "  Schedule size: {1} ops",
               l1BudgetPerCore, schedule.size());

  [[maybe_unused]] int64_t spillCount = 0;

  // Step 3: Belady's algorithm sweep with validation-based eviction.
  for (int64_t pos = 0; pos < static_cast<int64_t>(schedule.size()); ++pos) {
    Operation *op = schedule[pos];

    // Remove dead tensors (lastUse < pos) from the live set.
    // We use lazy deletion: check top of heap and pop if dead.
    while (!liveSet.empty()) {
      auto [lastUse, liveOp] = liveSet.top();
      if (!liveOps.count(liveOp) || lastUse < pos) {
        liveSet.pop();
        if (liveOps.count(liveOp) && lastUse < pos) {
          memoryTracker.removeTensor(liveOp);
          liveOps.erase(liveOp);
          TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                       "  [pos={0}] DEAD: {1}, L1 now {2}/{3}", pos,
                       ttmlir::opToString(liveOp),
                       memoryTracker.getOccupiedL1(), l1BudgetPerCore);
        }
        continue;
      }
      break;
    }

    // Skip ops without L1 output annotation.
    auto l1Attr = op->getAttrOfType<IntegerAttr>("ttnn.output_l1_usage");
    if (!l1Attr) {
      continue;
    }

    uint64_t opL1Usage = l1Attr.getValue().getZExtValue();
    int64_t opLastUse = lastUsePositions.count(op) ? lastUsePositions[op] : pos;

    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "  [pos={0}] PROCESS: {1}\n"
                 "    output L1: {2} bytes, last use: pos {3}\n"
                 "    occupied L1 before: {4}/{5} ({6} tensors)",
                 pos, ttmlir::opToString(op), opL1Usage, opLastUse,
                 memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                 liveOps.size());

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
      // Validation passed — add to live set.
      uint64_t l1Size =
          result.outputL1Usage > 0 ? result.outputL1Usage : opL1Usage;
      memoryTracker.addTensor(op, l1Size);
      liveOps.insert(op);
      liveSet.push({opLastUse, op});

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    ADDED: L1 now {0}/{1} ({2} tensors)",
                   memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                   liveOps.size());
      continue;
    }

    // OOM — try demoting current op, then evict if needed.
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "    OOM: validation failed, trying demotion/eviction");

    // Stage 1: Demote current op to L1 interleaved (no eviction needed).
    // Skip for MatmulOp/LinearOp — L1-interleaved output is strictly worse
    // than DRAM-interleaved for matmul: no program config is generated for
    // non-sharded output (generateMatmulProgramConfig returns nullopt),
    // causing runtime to fall back to MatmulMultiCoreProgramConfig with
    // hardcoded HiFi4. Fall through to Stage 2 (eviction) or Stage 3
    // (spill to DRAM) instead.
    if (!isa<MatmulOp, LinearOp>(op)) {
      OpConfig l1InterleavedConfig = makeL1InterleavedConfig(op);
      auto demoteResult =
          memoryTracker.validate(op, inputLayouts, l1InterleavedConfig);

      if (demoteResult.isSuccess()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                     "    DEMOTED to L1 interleaved: outputL1Usage={0}",
                     demoteResult.outputL1Usage);
        applyDemotedConfig(op, demoteResult);
        uint64_t l1Size = demoteResult.outputL1Usage;
        memoryTracker.addTensor(op, l1Size);
        liveOps.insert(op);
        liveSet.push({opLastUse, op});
        continue;
      }
    }

    // Stage 2: Evict from live set (Belady: farthest last-use first).
    // Re-validate with original sharded config after each eviction.
    result = memoryTracker.validate(op, inputLayouts, config);
    while (!result.isSuccess() && !liveOps.empty()) {
      Operation *victim = evictFarthestUse();
      if (!victim) {
        break;
      }

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    EVICT: {0} (L1: {1} bytes)", ttmlir::opToString(victim),
                   memoryTracker.getTensorSize(victim));

      spillToDram(victim);
      memoryTracker.removeTensor(victim);
      ++spillCount;

      // Re-validate victim's consumers that were already processed.
      revalidateConsumers(victim, pos, positionMap);

      // Re-extract input layouts (victim may have been input to current op).
      inputLayouts = utils::extractInputLayouts(op);
      result = memoryTracker.validate(op, inputLayouts, config);
    }

    if (result.isSuccess()) {
      uint64_t l1Size =
          result.outputL1Usage > 0 ? result.outputL1Usage : opL1Usage;
      memoryTracker.addTensor(op, l1Size);
      liveOps.insert(op);
      liveSet.push({opLastUse, op});

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    ADDED (after eviction): L1 now {0}/{1} ({2} tensors)",
                   memoryTracker.getOccupiedL1(), l1BudgetPerCore,
                   liveOps.size());
    } else {
      // Op exceeds budget alone — spill self to DRAM.
      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "    SPILL SELF: op exceeds budget alone");
      spillToDram(op);
      ++spillCount;
    }
  }

  // Print final memory view summary.
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "=== L1 Spill Summary ===\n"
               "  Total spills: {0}\n"
               "  Final live L1: {1}/{2} ({3} tensors)",
               spillCount, memoryTracker.getOccupiedL1(), l1BudgetPerCore,
               liveOps.size());

  // Step 4: Cleanup L1 usage attributes.
  cleanupL1UsageAttrs();
}

//===----------------------------------------------------------------------===//
// computeLastUsePositions
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
llvm::DenseMap<Operation *, int64_t>
L1SpillManagement<MemoryTracker>::computeLastUsePositions(
    const llvm::SmallVector<Operation *> &schedule) {
  // Build position map.
  llvm::DenseMap<Operation *, int64_t> positionMap;
  for (int64_t i = 0; i < static_cast<int64_t>(schedule.size()); ++i) {
    positionMap[schedule[i]] = i;
  }

  // For each op, find the maximum position among its users.
  llvm::DenseMap<Operation *, int64_t> lastUsePositions;
  for (int64_t i = 0; i < static_cast<int64_t>(schedule.size()); ++i) {
    Operation *op = schedule[i];
    int64_t lastUse = i; // Default: last use is the op itself.

    for (auto &use : op->getResult(0).getUses()) {
      Operation *user = use.getOwner();
      auto posIt = positionMap.find(user);
      if (posIt != positionMap.end()) {
        lastUse = std::max(lastUse, posIt->second);
      }
    }

    lastUsePositions[op] = lastUse;
  }

  return lastUsePositions;
}

//===----------------------------------------------------------------------===//
// spillToDram
//===----------------------------------------------------------------------===//

template <typename MemoryTracker>
void L1SpillManagement<MemoryTracker>::spillToDram(Operation *op) {
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  TTNNLayoutAttr layoutAttr =
      mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

  // Create DRAM interleaved layout.
  TTNNLayoutAttr dramLayout =
      layoutAttr.withBufferType(BufferType::DRAM)
          .withMemoryLayout(TensorMemoryLayout::Interleaved);
  RankedTensorType newTensorType =
      utils::RankedTensorTypeFactory::create(tensorType, dramLayout);

  MemoryConfigAttr memConfigAttr = MemoryConfigAttr::get(
      op->getContext(), dramLayout.getMemLayout(),
      BufferTypeAttr::get(op->getContext(), BufferType::DRAM),
      utils::createShardSpecIfNeeded(dramLayout, deviceGrid));

  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  Location loc = ttmlir::utils::appendLocationSuffix(op->getLoc(), "_spill");

  // Save all uses, insert ToMemoryConfigOp, reconnect uses.
  llvm::SmallVector<std::pair<Operation *, unsigned>> uses;
  for (auto &use : op->getResult(0).getUses()) {
    uses.emplace_back(use.getOwner(), use.getOperandNumber());
  }

  Operation *spillOp = builder.create<ToMemoryConfigOp>(
      loc, newTensorType, op->getResult(0), memConfigAttr);

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
