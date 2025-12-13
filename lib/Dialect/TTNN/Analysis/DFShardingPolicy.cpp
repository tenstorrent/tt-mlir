// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisProgressTracker.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Scheduler/Scheduler.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Chain Merge Helper Functions
//===----------------------------------------------------------------------===//

// Build a map from Operation* to schedule position for O(1) lookup.
static llvm::DenseMap<Operation *, int64_t>
buildSchedulePositionMap(const llvm::SmallVector<Operation *> &schedule) {
  llvm::DenseMap<Operation *, int64_t> posMap;
  for (size_t i = 0; i < schedule.size(); ++i) {
    posMap[schedule[i]] = static_cast<int64_t>(i);
  }
  return posMap;
}

// Build a map from Operation* to the chain index containing it.
static llvm::DenseMap<Operation *, size_t>
buildOpToChainMap(const std::vector<L1ChainConfig> &l1ChainConfigs) {
  llvm::DenseMap<Operation *, size_t> opToChain;
  for (size_t chainIdx = 0; chainIdx < l1ChainConfigs.size(); ++chainIdx) {
    for (const auto &spec : l1ChainConfigs[chainIdx].getOpL1MemSpecs()) {
      opToChain[spec.op] = chainIdx;
    }
  }
  return opToChain;
}

// Get the output tensor size in bytes for the last op in a chain.
static uint64_t getChainOutputSizeBytes(const L1ChainConfig &chain) {
  const auto &specs = chain.getOpL1MemSpecs();
  if (specs.empty()) {
    return 0;
  }

  const OpL1MemSpec &lastSpec = specs.back();
  if (!lastSpec.config.outputLayout) {
    return 0;
  }

  return lastSpec.config.outputLayout.getShardSizeInBytes();
}

// Build a map from Operation* to its resolved output layout within a chain.
static llvm::DenseMap<Operation *, TTNNLayoutAttr>
buildResolvedLayoutMap(const L1ChainConfig &chain) {
  llvm::DenseMap<Operation *, TTNNLayoutAttr> layoutMap;
  for (const auto &spec : chain.getOpL1MemSpecs()) {
    if (spec.config.outputLayout) {
      layoutMap[spec.op] = spec.config.outputLayout;
    }
  }
  return layoutMap;
}

// Build input layouts for an op using resolved configs from the chain.
// Priority order for each operand:
// 1. If there's a reshard entry for this edge, use reshard output layout
// 2. If producer is in chain, use its resolved output layout
// 3. Otherwise extract from IR (external operand)
// Returns std::nullopt if any layout cannot be determined.
static std::optional<std::vector<TTNNLayoutAttr>>
buildInputLayoutsFromResolvedConfigs(
    Operation *op,
    const llvm::DenseMap<Operation *, TTNNLayoutAttr> &resolvedLayoutMap,
    const llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap) {
  std::vector<TTNNLayoutAttr> inputLayouts;
  inputLayouts.reserve(op->getNumOperands());

  TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
               "Building input layouts for op {} with {} operands",
               op->getName(), op->getNumOperands());

  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    Operation *producerOp = operand.getDefiningOp();

    // First check if this edge has a reshard entry - use reshard output layout.
    // Resharding can occur on any edge, including between chain ops.
    Edge edge(producerOp, op, i);
    auto reshardIt = memReconfigEntryMap.find(edge);
    if (reshardIt != memReconfigEntryMap.end()) {
      const MemReconfigEntry &entry = reshardIt->second;
      if (entry.hasSelectedReshardOutputConfigBitIndex()) {
        size_t bitIdx = entry.getSelectedReshardOutputConfigBitIndex();
        auto configIt = entry.reshardOutputConfigMap.find(bitIdx);
        if (configIt != entry.reshardOutputConfigMap.end() &&
            !configIt->second.empty()) {
          TTNNLayoutAttr reshardLayout = configIt->second[0].outputLayout;
          if (reshardLayout) {
            TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                         "  Operand {}: using reshard output layout", i);
            inputLayouts.push_back(reshardLayout);
            continue;
          }
        }
      }
    }

    // No reshard - check if producer has a resolved layout in the chain.
    if (producerOp) {
      auto it = resolvedLayoutMap.find(producerOp);
      if (it != resolvedLayoutMap.end()) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "  Operand {}: using resolved layout from chain op {}", i,
                     producerOp->getName());
        inputLayouts.push_back(it->second);
        continue;
      }
    }

    // Producer not in chain and no reshard - extract layout from IR.
    auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType) {
      if (auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
              tensorType.getEncoding())) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "  Operand {}: using layout from IR (external: {})", i,
                     producerOp ? producerOp->getName().getStringRef()
                                : "block arg");
        inputLayouts.push_back(layout);
        continue;
      }
    }

    // Cannot determine layout - bail out.
    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "  Operand {}: cannot determine layout, bailing out", i);
    return std::nullopt;
  }

  TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
               "Successfully built {} input layouts for op {}",
               inputLayouts.size(), op->getName());
  return inputLayouts;
}

// Validate that Chain B can execute with Chain A's output in L1.
// This validates each op in Chain B (up to and including joinOp) with
// Chain A's output as additional L1 usage. At the join op, we also
// validate with Chain A's output layout as the RHS input.
static bool validateChainBWithMergedInput(
    const L1ChainConfig &chainB, Operation *joinOp, TTNNLayoutAttr rhsLayout,
    size_t rhsOperandIndex, uint64_t chainAOutputSize,
    const llvm::DenseMap<Operation *, int64_t> &schedulePositionMap) {

  auto joinOpPosIt = schedulePositionMap.find(joinOp);
  if (joinOpPosIt == schedulePositionMap.end()) {
    return false;
  }
  int64_t joinOpPos = joinOpPosIt->second;

  // Build a map of resolved output layouts for ops in Chain B.
  llvm::DenseMap<Operation *, TTNNLayoutAttr> resolvedLayoutMap =
      buildResolvedLayoutMap(chainB);

  const auto &memReconfigEntryMap = chainB.getMemReconfigEntryMap();

  // Validate each op in Chain B up to and including the join op.
  for (const auto &spec : chainB.getOpL1MemSpecs()) {
    Operation *op = spec.op;

    auto opPosIt = schedulePositionMap.find(op);
    if (opPosIt == schedulePositionMap.end()) {
      return false;
    }
    int64_t opPos = opPosIt->second;

    // Build input layouts using resolved configs from the chain.
    auto inputLayoutsOpt = buildInputLayoutsFromResolvedConfigs(
        op, resolvedLayoutMap, memReconfigEntryMap);
    if (!inputLayoutsOpt) {
      return false;
    }
    std::vector<TTNNLayoutAttr> inputLayouts = std::move(*inputLayoutsOpt);

    // If this is the join op, replace the RHS operand layout with Chain A's
    // output.
    if (op == joinOp && rhsOperandIndex < inputLayouts.size()) {
      inputLayouts[rhsOperandIndex] = rhsLayout;
    }

    // Validate the operation with Chain A's output as additional L1 usage.
    op_constraint_validation::ValidationResult result =
        op_constraint_validation::validateOperation(
            op, inputLayouts, spec.config, chainAOutputSize);

    if (!result.isSuccess()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Chain merge validation failed at op {}: {}",
                   ttmlir::opToString(op), result.errorMessage);
      return false;
    }

    // Stop after validating the join op - Chain A output is consumed there.
    if (opPos >= joinOpPos) {
      break;
    }
  }

  return true;
}

// Validate that Chain C can execute with Chain A's output in L1.
// Chain A (operand 0 producer) executes first, then Chain C executes while
// Chain A's output stays in L1. This is needed for 3-way merges where both
// Chain A and Chain C feed into Chain B.
static bool validateChainWithPredecessorInL1(const L1ChainConfig &chainC,
                                             uint64_t predecessorOutputSize) {

  // Build a map of resolved output layouts for ops in Chain C.
  llvm::DenseMap<Operation *, TTNNLayoutAttr> resolvedLayoutMap =
      buildResolvedLayoutMap(chainC);

  const auto &memReconfigEntryMap = chainC.getMemReconfigEntryMap();

  // Validate each op in Chain C with predecessor's output as additional L1.
  for (const auto &spec : chainC.getOpL1MemSpecs()) {
    Operation *op = spec.op;

    // Build input layouts using resolved configs from the chain.
    auto inputLayoutsOpt = buildInputLayoutsFromResolvedConfigs(
        op, resolvedLayoutMap, memReconfigEntryMap);
    if (!inputLayoutsOpt) {
      return false;
    }
    std::vector<TTNNLayoutAttr> inputLayouts = std::move(*inputLayoutsOpt);

    // Validate the operation with predecessor's output as additional L1 usage.
    op_constraint_validation::ValidationResult result =
        op_constraint_validation::validateOperation(
            op, inputLayouts, spec.config, predecessorOutputSize);

    if (!result.isSuccess()) {
      TTMLIR_DEBUG(
          ttmlir::LogComponent::DFShardingPolicy,
          "Chain C validation failed at op {} (with predecessor L1={}): {}",
          ttmlir::opToString(op), predecessorOutputSize, result.errorMessage);
      return false;
    }
  }

  return true;
}

// Represents a potential merge candidate.
struct MergeCandidate {
  size_t chainAIdx;
  Operation *joinOp;
  size_t operandIdx;
  uint64_t chainAOutputSize;
  TTNNLayoutAttr chainAOutputLayout;

  // Returns true if this is an operand 0 merge (main path continuation).
  bool isOperand0Merge() const { return operandIdx == 0; }
};

// Find all chains (Chain A candidates) whose output can be merged into Chain B.
// This searches for ops in Chain B that have operands coming from the last op
// of another completed chain. For operand 0, we only check the first op in
// Chain B (since that's where the chain would have continued if scheduling
// allowed). For operands > 0, we check all ops in Chain B.
static llvm::SmallVector<MergeCandidate> findAllMergeCandidates(
    size_t chainBIndex, const std::vector<L1ChainConfig> &l1ChainConfigs,
    const llvm::DenseMap<Operation *, size_t> &opToChainMap) {

  llvm::SmallVector<MergeCandidate> candidates;
  const L1ChainConfig &chainB = l1ChainConfigs[chainBIndex];

  TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
               "findAllMergeCandidates: chainBIndex={}, chainB has {} ops, "
               "first op: {}",
               chainBIndex, chainB.getOpL1MemSpecs().size(),
               ttmlir::opToString(chainB.getOpL1MemSpecs()[0].op));

  // Iterate through ops in Chain B to find potential join points.
  bool isFirstOpInChain = true;
  for (const auto &spec : chainB.getOpL1MemSpecs()) {
    Operation *op = spec.op;

    TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                 "  Checking op {} with {} operands (isFirstOp={})",
                 ttmlir::opToString(op), op->getNumOperands(),
                 isFirstOpInChain);

    // For the first op in chain, also check operand 0 (the main input path).
    // This handles cases where the chain should have continued but didn't due
    // to scheduling order (parallel branches completing at different times).
    size_t startOperandIdx = isFirstOpInChain ? 0 : 1;

    for (size_t operandIdx = startOperandIdx; operandIdx < op->getNumOperands();
         ++operandIdx) {
      Operation *producerOp = op->getOperand(operandIdx).getDefiningOp();
      if (!producerOp) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "    Operand {}: no defining op (block arg?)", operandIdx);
        continue;
      }

      TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                   "    Operand {}: producer op = {}", operandIdx,
                   producerOp->getName());

      // Find which chain produces this operand using the map.
      auto chainIt = opToChainMap.find(producerOp);
      if (chainIt == opToChainMap.end()) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "    Operand {}: producer {} NOT in opToChainMap",
                     operandIdx, producerOp->getName());
        continue;
      }

      size_t chainAIdx = chainIt->second;
      TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                   "    Operand {}: producer in chain {}", operandIdx,
                   chainAIdx);

      if (chainAIdx == chainBIndex) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "    Operand {}: same chain, skipping", operandIdx);
        continue;
      }

      const L1ChainConfig &chainA = l1ChainConfigs[chainAIdx];

      // Chain A must be completed.
      if (chainA.getState() != L1ChainState::Completed) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "    Chain {} not Completed (state={}), skipping",
                     chainAIdx, chainA.getStateString());
        continue;
      }

      // Check if producerOp is the last op of Chain A.
      if (chainA.getLastOp() != producerOp) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "    Producer {} is not last op of chain {}, skipping",
                     producerOp->getName(), chainAIdx);
        continue;
      }

      // Note: No need to verify schedule ordering - the scheduler guarantees
      // producers are scheduled before consumers (topological order).

      // Get Chain A's output layout and size.
      const auto &chainASpecs = chainA.getOpL1MemSpecs();
      if (chainASpecs.empty()) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "    Chain {} has empty specs, skipping", chainAIdx);
        continue;
      }
      TTNNLayoutAttr chainAOutputLayout =
          chainASpecs.back().config.outputLayout;
      if (!chainAOutputLayout) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "    Chain {} has null output layout, skipping",
                     chainAIdx);
        continue;
      }
      uint64_t chainAOutputSize = getChainOutputSizeBytes(chainA);

      TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                   "    FOUND candidate: chain {} -> chain {} (size={} bytes)",
                   chainAIdx, chainBIndex, chainAOutputSize);

      // Found a candidate!
      candidates.push_back(
          {chainAIdx, op, operandIdx, chainAOutputSize, chainAOutputLayout});
    }

    isFirstOpInChain = false;
  }

  TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
               "findAllMergeCandidates: found {} candidates for chain {}",
               candidates.size(), chainBIndex);

  return candidates;
}

//===----------------------------------------------------------------------===//
// Concat Input Chain Constraints
//===----------------------------------------------------------------------===//

// Determine the required input memory layout for a concat op based on its dim.
// Returns std::nullopt if concat doesn't have a specific sharding constraint.
//
// Backend constraints:
// - Block sharded: NOT supported for concat
// - Width concat (dim = last) requires HEIGHT_SHARDED inputs
// - Height concat (dim = second-to-last) requires WIDTH_SHARDED inputs
//
static std::optional<TensorMemoryLayout>
getConcatRequiredInputMemLayout(Operation *concatOp) {
  auto concat = llvm::dyn_cast<ttnn::ConcatOp>(concatOp);
  if (!concat) {
    return std::nullopt;
  }

  // Get the concat dimension
  int32_t dim = concat.getDim();

  // Get the rank of the input tensor
  auto inputType = mlir::cast<RankedTensorType>(concat.getOperand(0).getType());
  int64_t rank = inputType.getRank();

  // Normalize negative dim
  if (dim < 0) {
    dim += rank;
  }

  // Backend constraints for sharded concat:
  // - Width concat (dim = last) requires HEIGHT_SHARDED
  // - Height concat (dim = second-to-last) requires WIDTH_SHARDED
  if (dim == rank - 1) {
    // Width concat -> needs height sharded inputs
    return TensorMemoryLayout::HeightSharded;
  }
  if (dim == rank - 2) {
    // Height concat -> needs width sharded inputs
    return TensorMemoryLayout::WidthSharded;
  }

  // Other dims - return nullopt to not constrain (will likely fail validation)
  return std::nullopt;
}

// Pre-pass to set preferred output memory layout for chains that feed into
// or consume from concat. This allows pickOpShardConfigs to prefer compatible
// sharding types, enabling seamless chain merging.
//
// Must be called after chain building but before resolution.
//
static void
setConcatChainPreferences(std::vector<L1ChainConfig> &l1ChainConfigs) {

  // Build temporary op-to-chain map
  llvm::DenseMap<Operation *, size_t> opToChainMap;
  for (size_t i = 0; i < l1ChainConfigs.size(); ++i) {
    for (const auto &spec : l1ChainConfigs[i].getOpL1MemSpecs()) {
      opToChainMap[spec.op] = i;
    }
  }

  // Find concat chains and set preferences for their input and output chains
  for (size_t concatChainIdx = 0; concatChainIdx < l1ChainConfigs.size();
       ++concatChainIdx) {
    L1ChainConfig &concatChain = l1ChainConfigs[concatChainIdx];
    if (!concatChain.isConcatChain) {
      continue;
    }

    Operation *concatOp = concatChain.getOpL1MemSpecs()[0].op;
    auto requiredMemLayout = getConcatRequiredInputMemLayout(concatOp);

    if (!requiredMemLayout) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Concat {}: no specific sharding preference for this dim",
                   ttmlir::opToString(concatOp));
      continue;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Concat {} prefers {} inputs", ttmlir::opToString(concatOp),
                 *requiredMemLayout == TensorMemoryLayout::HeightSharded
                     ? "height_sharded"
                     : "width_sharded");

    // Set preference for input chains (producers of concat's operands)
    for (size_t i = 0; i < concatOp->getNumOperands(); ++i) {
      Operation *producerOp = concatOp->getOperand(i).getDefiningOp();
      if (!producerOp) {
        continue;
      }

      auto chainIt = opToChainMap.find(producerOp);
      if (chainIt == opToChainMap.end()) {
        continue;
      }

      size_t inputChainIdx = chainIt->second;
      L1ChainConfig &inputChain = l1ChainConfigs[inputChainIdx];

      inputChain.preferredOutputMemLayout = *requiredMemLayout;

      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "  Input chain {}: set preferredOutputMemLayout to {}",
                   inputChainIdx,
                   *requiredMemLayout == TensorMemoryLayout::HeightSharded
                       ? "height_sharded"
                       : "width_sharded");
    }

    // Set preference for output chain (consumer of concat's result)
    // This enables seamless merging between concat and its consumer.
    // Concat's output will have the same memory layout as its inputs,
    // so the consumer should prefer the same layout.
    if (concatOp->hasOneUse()) {
      Operation *consumerOp = *concatOp->user_begin();
      auto consumerChainIt = opToChainMap.find(consumerOp);
      if (consumerChainIt != opToChainMap.end()) {
        size_t consumerChainIdx = consumerChainIt->second;
        L1ChainConfig &consumerChain = l1ChainConfigs[consumerChainIdx];

        // Only set if consumer's first op is the concat user
        if (!consumerChain.getOpL1MemSpecs().empty() &&
            consumerChain.getOpL1MemSpecs()[0].op == consumerOp) {
          consumerChain.preferredOutputMemLayout = *requiredMemLayout;

          TTMLIR_DEBUG(
              ttmlir::LogComponent::DFShardingPolicy,
              "  Consumer chain {}: set preferredOutputMemLayout to {}",
              consumerChainIdx,
              *requiredMemLayout == TensorMemoryLayout::HeightSharded
                  ? "height_sharded"
                  : "width_sharded");
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Concat Chain Resolution
//===----------------------------------------------------------------------===//

// Resolve concat chains by validating that all incoming L1-sharded inputs
// can be consumed directly. This is called after regular chains are resolved
// but before chain merging.
//
// For each concat chain:
// 1. Find all input chains (producers of concat's operands)
// 2. Check if all input chains are Completed (successfully L1-sharded)
// 3. Validate concat can consume all L1 inputs by querying the backend
// 4. Validate N-way merge: each subsequent input chain can execute while
//    previous chains' outputs stay in L1
// 5. If successful, mark input chains with spillEndToDRAM = false and
//    complete the concat chain with the backend-determined output layout
//
static void resolveConcatChains(
    std::vector<L1ChainConfig> &l1ChainConfigs,
    const llvm::DenseMap<Operation *, size_t> &opToChainMap,
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs) {

  for (size_t concatChainIdx = 0; concatChainIdx < l1ChainConfigs.size();
       ++concatChainIdx) {
    L1ChainConfig &concatChain = l1ChainConfigs[concatChainIdx];

    if (!concatChain.isConcatChain) {
      continue;
    }

    Operation *concatOp = concatChain.getOpL1MemSpecs()[0].op;

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Resolving concat chain {} for op {}", concatChainIdx,
                 ttmlir::opToString(concatOp));

    // Find all input chains (producers of concat's operands)
    llvm::SmallVector<size_t> inputChainIndices;
    std::vector<TTNNLayoutAttr> inputLayouts;
    bool allInputChainsCompleted = true;

    for (size_t i = 0; i < concatOp->getNumOperands(); ++i) {
      Operation *producerOp = concatOp->getOperand(i).getDefiningOp();
      if (!producerOp) {
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "  Operand {}: no defining op (block arg), cannot do L1 "
                     "optimization",
                     i);
        allInputChainsCompleted = false;
        break;
      }

      auto chainIt = opToChainMap.find(producerOp);
      if (chainIt == opToChainMap.end()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "  Operand {}: producer {} not in any chain", i,
                     producerOp->getName());
        allInputChainsCompleted = false;
        break;
      }

      const L1ChainConfig &inputChain = l1ChainConfigs[chainIt->second];
      if (inputChain.getState() != L1ChainState::Completed) {
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "  Operand {}: input chain {} not Completed (state={})", i,
                     chainIt->second, inputChain.getStateString());
        allInputChainsCompleted = false;
        break;
      }

      if (inputChain.getLastOp() != producerOp) {
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "  Operand {}: producer {} is not last op of chain {}", i,
                     producerOp->getName(), chainIt->second);
        allInputChainsCompleted = false;
        break;
      }

      inputChainIndices.push_back(chainIt->second);
      TTNNLayoutAttr outputLayout =
          inputChain.getOpL1MemSpecs().back().config.outputLayout;
      inputLayouts.push_back(outputLayout);

      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "  Operand {}: from chain {}, layout: {}", i,
                   chainIt->second, outputLayout);
    }

    if (!allInputChainsCompleted) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Concat chain {}: not all inputs are L1-sharded, marking "
                   "as Failed",
                   concatChainIdx);
      concatChain.fail();
      continue;
    }

    // Validate concat can consume all L1 inputs
    // Try each legal config for concat to find one that works with L1 inputs.
    // The config must have an L1 sharded output that matches the inputs'
    // memory layout type.
    const std::vector<OpConfig> &concatLegalConfigs =
        legalConfigs.lookup(concatOp);

    if (concatLegalConfigs.empty()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Concat chain {}: no legal configs available",
                   concatChainIdx);
      concatChain.fail();
      continue;
    }

    // For sharded concat, the backend requires output grid to match input grid.
    // Construct an output layout based on the first input's layout, keeping
    // the same grid and memory layout type.
    TTNNLayoutAttr firstInputLayout = inputLayouts[0];
    RankedTensorType concatOutputType =
        mlir::cast<RankedTensorType>(concatOp->getResult(0).getType());
    llvm::ArrayRef<int64_t> outputShape = concatOutputType.getShape();

    // Build output layout with same grid/memory config as first input
    TTNNLayoutAttr outputLayout = firstInputLayout.withTensorShape(outputShape);

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Concat chain {}: constructed output layout from input: {}",
                 concatChainIdx, outputLayout);

    OpConfig selectedConfig;
    selectedConfig.outputLayout = outputLayout;

    op_constraint_validation::ValidationResult result =
        op_constraint_validation::validateOperation(
            concatOp, inputLayouts, selectedConfig, /*additionalL1Usage=*/0);

    if (!result.isSuccess()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Concat chain {}: validation failed: {}", concatChainIdx,
                   result.errorMessage);
      concatChain.fail();
      continue;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Concat chain {}: validation succeeded with output layout: {}",
                 concatChainIdx, outputLayout);

    // Validate N-way merge: each subsequent input chain can execute while
    // previous chains' outputs stay in L1
    bool nWayMergeValid = true;
    uint64_t accumulatedL1 = 0;

    for (size_t i = 0; i < inputChainIndices.size(); ++i) {
      if (i > 0) {
        // Validate chain[i] can execute while previous chains' outputs stay
        // in L1
        const L1ChainConfig &chainI = l1ChainConfigs[inputChainIndices[i]];
        if (!validateChainWithPredecessorInL1(chainI, accumulatedL1)) {
          TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                       "Concat chain {}: N-way merge failed - chain {} cannot "
                       "execute with {} bytes of predecessor output in L1",
                       concatChainIdx, inputChainIndices[i], accumulatedL1);
          nWayMergeValid = false;
          break;
        }
      }
      accumulatedL1 +=
          getChainOutputSizeBytes(l1ChainConfigs[inputChainIndices[i]]);
    }

    if (!nWayMergeValid) {
      concatChain.fail();
      continue;
    }

    // Success! Mark input chains as spillEndToDRAM = false
    for (size_t idx : inputChainIndices) {
      l1ChainConfigs[idx].spillEndToDRAM = false;
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Concat chain {}: marking input chain {} spillEndToDRAM = "
                   "false",
                   concatChainIdx, idx);
    }

    // Complete concat chain with the validated config
    // Use the actual output layout from backend validation if available,
    // otherwise use the selected config's output layout
    if (result.actualOutputLayout) {
      selectedConfig.outputLayout = result.actualOutputLayout;
    }

    llvm::DenseMap<Operation *, OpConfig> selectedConfigs;
    selectedConfigs[concatOp] = selectedConfig;
    llvm::DenseMap<Edge, MemReconfigEntry> emptyReconfigMap;

    // Transition state from Built -> Resolved before completing
    concatChain.resolve();
    concatChain.complete(selectedConfigs, emptyReconfigMap);

    // Set spillEndToDRAM based on output layout
    if (!selectedConfig.outputLayout.hasDRAMBufferType()) {
      concatChain.spillEndToDRAM = true;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Concat chain {} resolved successfully with output layout: {}",
                 concatChainIdx, selectedConfig.outputLayout);
  }
}

//===----------------------------------------------------------------------===//
// Chain Merge Post-Processing
//===----------------------------------------------------------------------===//

// Post-processing pass to merge chains where possible.
// This allows predecessor chains' outputs to stay in L1 and be consumed
// by Chain B, avoiding unnecessary DRAM spills.
//
// Two separate scenarios are handled:
//
// Scenario 1 (Classic RHS merge):
//   Chain C (operand > 0) -> Chain B
//   Validate: Chain B can execute with Chain C's output in L1
//
// Scenario 2 (3-way merge - special case):
//   Detected when first op of Chain B is a 2-operand op with candidates
//   on BOTH operand 0 and operand 1.
//   Chain A (operand 0) executes first -> output stays in L1
//   Chain C (operand 1) executes second -> output stays in L1
//   Chain B's first op consumes both (no validation needed for Chain B
//   since Chain A flows directly into it on operand 0)
//   Validate: Chain C can execute with Chain A's output in L1
//
static void applyChainMerges(std::vector<L1ChainConfig> &l1ChainConfigs,
                             const llvm::SmallVector<Operation *> &schedule) {
  // Build lookup maps for O(1) access.
  llvm::DenseMap<Operation *, int64_t> schedulePositionMap =
      buildSchedulePositionMap(schedule);
  llvm::DenseMap<Operation *, size_t> opToChainMap =
      buildOpToChainMap(l1ChainConfigs);

  // Track chains that have received a merge (one-level limit per chain).
  llvm::DenseSet<size_t> mergedIntoChains;

  for (size_t chainBIdx = 0; chainBIdx < l1ChainConfigs.size(); ++chainBIdx) {
    L1ChainConfig &chainB = l1ChainConfigs[chainBIdx];

    // Skip non-completed chains.
    if (chainB.getState() != L1ChainState::Completed) {
      continue;
    }

    // One-level limit: skip if Chain B already received a merge.
    if (mergedIntoChains.contains(chainBIdx)) {
      continue;
    }

    // Find all merge candidates for Chain B.
    llvm::SmallVector<MergeCandidate> candidates =
        findAllMergeCandidates(chainBIdx, l1ChainConfigs, opToChainMap);

    if (candidates.empty()) {
      continue;
    }

    // Check for 3-way merge scenario (Scenario 2):
    // - First op of Chain B has exactly 2 operands
    // - We have candidates on both operand 0 and operand 1
    // - Both candidates target the same op (first op of Chain B)
    Operation *firstOpInChainB = chainB.getOpL1MemSpecs().front().op;
    const MergeCandidate *operand0Candidate = nullptr;
    const MergeCandidate *operand1Candidate = nullptr;

    if (firstOpInChainB->getNumOperands() == 2) {
      for (const auto &candidate : candidates) {
        if (candidate.joinOp == firstOpInChainB) {
          if (candidate.operandIdx == 0) {
            operand0Candidate = &candidate;
          } else if (candidate.operandIdx == 1) {
            operand1Candidate = &candidate;
          }
        }
      }
    }

    // Scenario 2: 3-way merge
    if (operand0Candidate && operand1Candidate) {
      // Chain A (operand 0) executes first, then Chain C (operand 1).
      // Validate that Chain C can execute with Chain A's output in L1.
      const L1ChainConfig &chainC =
          l1ChainConfigs[operand1Candidate->chainAIdx];

      if (validateChainWithPredecessorInL1(
              chainC, operand0Candidate->chainAOutputSize)) {
        // 3-way merge is valid! Apply both merges.
        L1ChainConfig &chainA = l1ChainConfigs[operand0Candidate->chainAIdx];
        L1ChainConfig &chainCMut = l1ChainConfigs[operand1Candidate->chainAIdx];

        chainA.spillEndToDRAM = false;
        chainCMut.spillEndToDRAM = false;
        mergedIntoChains.insert(chainBIdx);

        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "3-way merge applied: [chain {} (op0) + chain {} (op1) -> "
                     "chain {}] (join op: {}, tensor sizes: {} + {} bytes)",
                     operand0Candidate->chainAIdx, operand1Candidate->chainAIdx,
                     chainBIdx, firstOpInChainB->getName(),
                     operand0Candidate->chainAOutputSize,
                     operand1Candidate->chainAOutputSize);
        continue; // Move to next chain
      }

      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "3-way merge rejected: Chain C (chain {}) cannot execute "
                   "with Chain A's (chain {}) output in L1",
                   operand1Candidate->chainAIdx, operand0Candidate->chainAIdx);
      // Fall through to try classic RHS merge
    }

    // Scenario 1: Classic RHS merge (operand > 0 only, or operand 0 for
    // single-operand ops). For multi-operand ops, operand 0 merges are skipped
    // since they are handled by chain continuation during building. But for
    // single-operand ops (like relu after concat), we allow operand 0 merge.
    // Exception: concat chains are allowed to merge on operand 0 even for
    // multi-operand consumers, since concat chains can't be continued during
    // chain building.
    const MergeCandidate *bestCandidate = nullptr;
    uint64_t bestSize = 0;

    for (const auto &candidate : candidates) {
      // For operand 0 candidates, only allow if:
      // 1. The join op has exactly 1 operand (single-operand ops), OR
      // 2. The source chain is a concat chain (can't be continued during build)
      if (candidate.isOperand0Merge() &&
          candidate.joinOp->getNumOperands() != 1 &&
          !l1ChainConfigs[candidate.chainAIdx].isConcatChain) {
        continue;
      }

      // Validate Chain B can execute with the candidate's output in L1.
      if (!validateChainBWithMergedInput(
              chainB, candidate.joinOp, candidate.chainAOutputLayout,
              candidate.operandIdx, candidate.chainAOutputSize,
              schedulePositionMap)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "Chain merge candidate rejected: validation failed for "
                     "merging chain {} into chain {} at op {}",
                     candidate.chainAIdx, chainBIdx,
                     candidate.joinOp->getName());
        continue;
      }

      // Select this candidate if it has the largest output size so far.
      if (candidate.chainAOutputSize > bestSize) {
        bestSize = candidate.chainAOutputSize;
        bestCandidate = &candidate;
      }
    }

    if (!bestCandidate) {
      continue;
    }

    // Apply the merge.
    L1ChainConfig &chainToMerge = l1ChainConfigs[bestCandidate->chainAIdx];
    chainToMerge.spillEndToDRAM = false;
    mergedIntoChains.insert(chainBIdx);

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "{} merge applied: [chain {} -> chain {}] "
                 "(join op: {}, operand: {}, tensor size: {} bytes)",
                 bestCandidate->isOperand0Merge() ? "Operand0" : "RHS",
                 bestCandidate->chainAIdx, chainBIdx,
                 bestCandidate->joinOp->getName(), bestCandidate->operandIdx,
                 bestCandidate->chainAOutputSize);
  }
}

void DFShardingPolicy::run() {
  func::FuncOp funcToProcess = nullptr;

  rootOp->walk([&](func::FuncOp func) {
    if (ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }

    funcToProcess = func;
    deviceAttr = ttcore::lookupDevice(func);
    mlir::tt::scheduler::Scheduler scheduler(&func);
    l1ChainConfigs->push_back(L1ChainConfig());
    llvm::SmallVector<mlir::Operation *> scheduleableOps;
    Operation *currentOp = nullptr;

    // Produce shard chain configs.
    // 1. Schedule ops in DFS order.
    // 2. Check if currentOp has a valid successor. (no forking for now)
    // 3. Check if currentOp/nextOp pair is valid for sharding.
    // 4. Op is considered sharded if its output is sharded to L1.
    //
    while (scheduler.hasUnscheduledOps()) {
      scheduleableOps = scheduler.getSchedulableOps();

      // Before starting a sharding chain, schedule layout/memory management ops
      // first until they are exhausted from schedulable ops.
      // TODO(nobradovic) :
      // We need to examine type of memory op and determine if for
      // example we have a space in DRAM to perform this?(system->dram, double
      // check this)
      //
      if (l1ChainConfigs->back().isEmpty()) {
        for (auto *op : scheduleableOps) {
          if (isa<ToLayoutOp>(op)) {
            currentOp = op;
            break;
          }
        }
      }

      if (currentOp == nullptr) {
        currentOp = scheduleableOps[0];
      }

      // Schedule currentOp.
      //
      scheduler.scheduleOp(currentOp);

      // Skip starting sharding chain if currentOp is a memory management op.
      //
      if (l1ChainConfigs->back().isEmpty() && isa<ToLayoutOp>(currentOp)) {
        currentOp = nullptr;
        continue;
      }

      // Consider sharding only if we found at least single legal config for
      // the current op.
      bool validForSharding =
          llvm::isa<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp, ttnn::AddOp,
                    ttnn::MultiplyOp, ttnn::ReluOp, ttnn::Relu6Op,
                    ttnn::TypecastOp, ttnn::SiluOp, ttnn::MatmulOp,
                    ttnn::LinearOp, ttnn::MinimumOp, ttnn::RMSNormOp,
                    ttnn::GeluOp, ttnn::NegOp, ttnn::RsqrtOp, ttnn::ConcatOp,
                    ttnn::PowScalarOp, ttnn::MeanOp, ttnn::SliceStaticOp,
                    ttnn::RotaryEmbeddingOp>(currentOp) &&
          legalConfigs.lookup(currentOp).size() > 0;

      // Special handling for ConcatOp: isolate it into its own single-op
      // chain. This allows us to handle concat specially by:
      // 1. Breaking any incoming chain at concat
      // 2. Starting a new chain after concat's user
      // 3. Resolving concat without ShardSolver by validating if all
      //    incoming L1-sharded inputs can be consumed directly
      if (llvm::isa<ttnn::ConcatOp>(currentOp) && validForSharding) {
        // First, finalize any current chain that was being built
        if (!l1ChainConfigs->back().isEmpty()) {
          l1ChainConfigs->back().build();
          l1ChainConfigs->push_back(L1ChainConfig());
        }

        // Create a single-op chain for concat
        OpL1MemSpec concatSpec;
        concatSpec.op = currentOp;
        concatSpec.tensorSplitFactor = 1;
        l1ChainConfigs->back().addOpL1MemSpec(std::move(concatSpec));
        l1ChainConfigs->back().isConcatChain = true;
        l1ChainConfigs->back().build();

        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "Created isolated concat chain for op {}",
                     ttmlir::opToString(currentOp));

        // Start a new chain for subsequent ops
        l1ChainConfigs->push_back(L1ChainConfig());
        currentOp = nullptr;
        continue;
      }

      // Check for next op only if there are still unscheduled ops
      Operation *nextOp = nullptr;
      if (scheduler.hasUnscheduledOps()) {
        scheduleableOps = scheduler.getSchedulableOps();

        // Check if currentOp has a valid successor.
        //
        for (auto *op : scheduleableOps) {
          for (auto operand : op->getOperands()) {
            if (operand.getDefiningOp() == currentOp) {
              nextOp = op;
              break;
            }
          }
        }
      }

      if (validForSharding) {
        OpL1MemSpec shardSpec;
        shardSpec.op = currentOp;

        // Hardcoded tensor split factor for now, until pipeline OP
        // support is added.
        //
        shardSpec.tensorSplitFactor = 1;
        l1ChainConfigs->back().addOpL1MemSpec(std::move(shardSpec));

        if (nextOp && currentOp->hasOneUse()) {
          // Only if nextOp is valid and currentOp is not a fork keep
          // growing the chain.
          if (nextOp->getOperand(0).getDefiningOp() != currentOp) {
            // Only continue chain if nextOp uses currentOp as first operand.
            // Here we break the chain if not.
            TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                         "Breaking L1 chain at op {} as it is not first "
                         "operand of next op {}",
                         ttmlir::opToString(currentOp),
                         ttmlir::opToString(nextOp));
            currentOp = nullptr;
          } else {
            // Don't continue chain into ConcatOp - it needs its own chain
            if (llvm::isa<ttnn::ConcatOp>(nextOp)) {
              TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                           "Breaking L1 chain at op {} because next op {} is "
                           "ConcatOp",
                           ttmlir::opToString(currentOp),
                           ttmlir::opToString(nextOp));
              currentOp = nullptr;
            } else {
              currentOp = nextOp;
              continue;
            }
          }
        }
      }

      currentOp = nullptr;

      if (!l1ChainConfigs->back().isEmpty()) {
        l1ChainConfigs->back().build();
        l1ChainConfigs->push_back(L1ChainConfig());
      }
    }

    (*schedule)[func] = scheduler.getSchedule();
  });

  // Remove trailing empty chain config if present (from last push_back in loop)
  if (!l1ChainConfigs->empty() && l1ChainConfigs->back().isEmpty()) {
    l1ChainConfigs->pop_back();
  }

  for ([[maybe_unused]] L1ChainConfig &l1ChainConfig : *l1ChainConfigs) {
    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy, "L1 chain config {}",
                 l1ChainConfig);
  }

  // Set preferred output memory layout for chains feeding into concat.
  // This must be done before resolution so pickOpShardConfigs can use it.
  setConcatChainPreferences(*l1ChainConfigs);

  // Resolve shard chain configs.
  //
  mlir::tt::ttnn::MemoryLayoutAnalysisProgressTracker progressTracker;
  progressTracker.startAnalysis(funcToProcess, l1ChainConfigs->size(),
                                "DFShardingPolicy");

  for (size_t chainIndex = 0; chainIndex < l1ChainConfigs->size();
       ++chainIndex) {
    L1ChainConfig &l1ChainConfig = (*l1ChainConfigs)[chainIndex];

    // Skip concat chains - they are resolved separately after all regular
    // chains are processed, so we can check if their inputs are L1-sharded.
    if (l1ChainConfig.isConcatChain) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Skipping concat chain {} - will resolve later", chainIndex);
      continue;
    }

    // Count operations in this chain
    size_t numOpsInChain = l1ChainConfig.getOpL1MemSpecs().size();
    Operation *firstOp = l1ChainConfig.getOpL1MemSpecs()[0].op;
    progressTracker.startL1Chain(firstOp, chainIndex, numOpsInChain);
    ShardSolver shardSolver = l1ChainConfig.resolveWithSolver(
        tensorTypePossibleLayouts, legalConfigs, usableL1CacheSize,
        overrideReshardEdges, overrideOutputLayout);

    if (l1ChainConfig.getState() == L1ChainState::Failed) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Failed to resolve L1 chain config {}", l1ChainConfig);
      progressTracker.finishL1Chain(firstOp, chainIndex, false);
      continue;
    }

    pickOpShardConfigs(shardSolver, l1ChainConfig);

    ShardSolverSolution resolvedShardSolution = shardSolver.finish();
    l1ChainConfig.complete(resolvedShardSolution.selectedOpConfig,
                           resolvedShardSolution.memReconfigEntryMap);

    // TODO(odjuricic): Add constraint check if op can write to dram.
    if (!resolvedShardSolution.selectedOpConfig[l1ChainConfig.getLastOp()]
             .outputLayout.hasDRAMBufferType()) {
      l1ChainConfig.spillEndToDRAM = true;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Resolved L1 chain config {}", l1ChainConfig);

    progressTracker.finishL1Chain(firstOp, chainIndex, true);
  }

  // Build op-to-chain map for concat resolution and chain merging.
  llvm::DenseMap<Operation *, size_t> opToChainMap =
      buildOpToChainMap(*l1ChainConfigs);

  // Resolve concat chains now that all regular chains are processed.
  // This allows us to check if concat's input chains are L1-sharded.
  resolveConcatChains(*l1ChainConfigs, opToChainMap, legalConfigs);

  // Post-processing: attempt to merge chains where Chain A's output can stay
  // in L1 and be consumed as RHS by a join op in Chain B.
  if (funcToProcess) {
    applyChainMerges(*l1ChainConfigs, (*schedule)[funcToProcess]);
  }

  progressTracker.finishAnalysis(funcToProcess);
}

void DFShardingPolicy::pickOpShardConfigs(ShardSolver &shardSolver,
                                          const L1ChainConfig &l1ChainConfig) {

  assert(l1ChainConfig.getState() == L1ChainState::Resolved);
  llvm::DenseMap<Operation *, SmallVector<float, 64>> accMaxCoreUsage =
      shardSolver.produceMaxCoreUsage();

  // Get last op and preferred layout for concat compatibility
  Operation *lastOp = l1ChainConfig.getLastOp();
  std::optional<TensorMemoryLayout> preferredMemLayout =
      l1ChainConfig.preferredOutputMemLayout;

  // If there's a preferred layout for the last op, try to process it FIRST.
  // This ensures the constraint propagates backward through ShardSolver,
  // forcing earlier ops to pick configs compatible with the preferred layout.
  bool lastOpProcessed = false;
  if (preferredMemLayout) {
    ShardSolver::RemainingConfigAttrs validConfigs = shardSolver.at(lastOp);
    const OpConfig *preferredConfig = nullptr;
    float preferredMaxCoreUsage = 0;

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Last op {}: looking for preferred {} layout among {} configs",
                 lastOp->getName(),
                 *preferredMemLayout == TensorMemoryLayout::HeightSharded
                     ? "height_sharded"
                     : "width_sharded",
                 validConfigs.size());

    for (auto configIterator = validConfigs.begin();
         configIterator != validConfigs.end(); ++configIterator) {
      assert(configIterator->outputLayout.getMemLayout() &&
             "TensorMemoryLayout is not set");

      TensorMemoryLayout currentMemLayout =
          configIterator->outputLayout.getMemLayout().getValue();
      float coreUsage = accMaxCoreUsage[lastOp][configIterator.index()];

      TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                   "  Config {}: {} (coreUsage={})", configIterator.index(),
                   currentMemLayout == TensorMemoryLayout::HeightSharded
                       ? "height_sharded"
                   : currentMemLayout == TensorMemoryLayout::WidthSharded
                       ? "width_sharded"
                       : "block_sharded",
                   coreUsage);

      if (currentMemLayout == *preferredMemLayout) {
        // Config matches preferred layout - pick highest core usage among these
        if (!preferredConfig || coreUsage > preferredMaxCoreUsage) {
          preferredConfig = configIterator.get();
          preferredMaxCoreUsage = coreUsage;
        }
      }
    }

    // If we found a preferred config, set it first to propagate constraints
    if (preferredConfig) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Last op {}: selected preferred {} layout",
                   lastOp->getName(),
                   *preferredMemLayout == TensorMemoryLayout::HeightSharded
                       ? "height_sharded"
                       : "width_sharded");
      shardSolver.set(lastOp, *preferredConfig);
      lastOpProcessed = true;
    } else {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Last op {}: preferred {} layout NOT found in valid configs",
                   lastOp->getName(),
                   *preferredMemLayout == TensorMemoryLayout::HeightSharded
                       ? "height_sharded"
                       : "width_sharded");
    }
  }

  // Process remaining ops in order
  for (const auto &shardSpec : l1ChainConfig.getOpL1MemSpecs()) {
    Operation *op = shardSpec.op;

    // Skip last op if already processed above
    if (lastOpProcessed && op == lastOp) {
      continue;
    }

    ShardSolver::RemainingConfigAttrs validConfigs = shardSolver.at(op);
    const OpConfig *selectedConfig = validConfigs.begin().get();
    float maxCoreUsage = 0;

    for (auto configIterator = validConfigs.begin();
         configIterator != validConfigs.end(); ++configIterator) {
      if (accMaxCoreUsage[op][configIterator.index()] > maxCoreUsage) {
        maxCoreUsage = accMaxCoreUsage[op][configIterator.index()];
        selectedConfig = configIterator.get();
      } else if (accMaxCoreUsage[op][configIterator.index()] == maxCoreUsage) {
        assert(configIterator->outputLayout.getMemLayout() &&
               "TensorMemoryLayout is not set");

        TensorMemoryLayout currentMemLayout =
            configIterator->outputLayout.getMemLayout().getValue();
        TensorMemoryLayout selectedMemLayout =
            selectedConfig->outputLayout.getMemLayout().getValue();

        // Default tie-breaker: prefer layout that is not BlockSharded.
        if (currentMemLayout != ttnn::TensorMemoryLayout::BlockSharded &&
            selectedMemLayout == ttnn::TensorMemoryLayout::BlockSharded) {
          selectedConfig = configIterator.get();
        }
      }
    }

    shardSolver.set(op, *selectedConfig);
  }
}

} // namespace mlir::tt::ttnn
