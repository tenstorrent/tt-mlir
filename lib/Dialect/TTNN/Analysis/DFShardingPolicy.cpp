// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisProgressTracker.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Scheduler/Scheduler.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Diagnostics.h"

#include <memory>

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

// Forward declaration for L1Reservation (defined later in the file).
struct L1Reservation;
static uint64_t
getActiveL1Reservations(int64_t schedulePos,
                        const std::vector<L1Reservation> &reservations);

// Static empty containers for validation calls that don't need reservations.
static const llvm::DenseMap<Operation *, int64_t> kEmptySchedulePositionMap;
static const std::vector<L1Reservation> kEmptyL1Reservations;

// Validate that Chain B can execute with Chain A's output in L1.
// This validates each op in Chain B (up to and including joinOp) with
// Chain A's output as additional L1 usage. At the join op, we also
// validate with Chain A's output layout as the RHS input.
// Also accounts for any active L1 reservations at each op's position.
static bool validateChainBWithMergedInput(
    const L1ChainConfig &chainB, Operation *joinOp, TTNNLayoutAttr rhsLayout,
    size_t rhsOperandIndex, uint64_t chainAOutputSize,
    const llvm::DenseMap<Operation *, int64_t> &schedulePositionMap,
    const std::vector<L1Reservation> &l1Reservations) {

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

    // Calculate total additional L1: chain merge + active reservations
    uint64_t reservedL1 = getActiveL1Reservations(opPos, l1Reservations);
    uint64_t totalAdditionalL1 = chainAOutputSize + reservedL1;

    // Validate the operation with total additional L1 usage.
    op_constraint_validation::ValidationResult result =
        op_constraint_validation::validateOperation(
            op, inputLayouts, spec.config, totalAdditionalL1);

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
// Also accounts for any active L1 reservations at each op's position.
static bool validateChainWithPredecessorInL1(
    const L1ChainConfig &chainC, uint64_t predecessorOutputSize,
    const llvm::DenseMap<Operation *, int64_t> &schedulePositionMap,
    const std::vector<L1Reservation> &l1Reservations) {

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

    // Calculate total additional L1: predecessor + active reservations
    uint64_t reservedL1 = 0;
    auto posIt = schedulePositionMap.find(op);
    if (posIt != schedulePositionMap.end()) {
      reservedL1 = getActiveL1Reservations(posIt->second, l1Reservations);
    }
    uint64_t totalAdditionalL1 = predecessorOutputSize + reservedL1;

    // Validate the operation with total additional L1 usage.
    op_constraint_validation::ValidationResult result =
        op_constraint_validation::validateOperation(
            op, inputLayouts, spec.config, totalAdditionalL1);

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

  // Skip chains with no ops (e.g., chains created only for memReconfigEntryMap)
  if (chainB.getOpL1MemSpecs().empty()) {
    return candidates;
  }

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
                 stringifyTensorMemoryLayout(*requiredMemLayout));

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
                   stringifyTensorMemoryLayout(*requiredMemLayout));
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
              stringifyTensorMemoryLayout(*requiredMemLayout));
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
// 5. If successful, mark input chains with spillLocation = None and
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
        if (!validateChainWithPredecessorInL1(chainI, accumulatedL1,
                                              kEmptySchedulePositionMap,
                                              kEmptyL1Reservations)) {
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

    // Success! Mark input chains as spillLocation = None (no spill needed)
    for (size_t idx : inputChainIndices) {
      l1ChainConfigs[idx].spillLocation = SpillLocation::None;
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Concat chain {}: marking input chain {} spillLocation = "
                   "None",
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

    // Set spillLocation based on output layout
    if (!selectedConfig.outputLayout.hasDRAMBufferType()) {
      concatChain.spillLocation = SpillLocation::DRAM;
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
                             const llvm::SmallVector<Operation *> &schedule,
                             const std::vector<L1Reservation> &l1Reservations) {
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

    // Skip chains with no ops (e.g., chains created only for
    // memReconfigEntryMap)
    if (chainB.getOpL1MemSpecs().empty()) {
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
              chainC, operand0Candidate->chainAOutputSize, schedulePositionMap,
              l1Reservations)) {
        // 3-way merge is valid! Apply both merges.
        L1ChainConfig &chainA = l1ChainConfigs[operand0Candidate->chainAIdx];
        L1ChainConfig &chainCMut = l1ChainConfigs[operand1Candidate->chainAIdx];

        chainA.spillLocation = SpillLocation::None;
        chainCMut.spillLocation = SpillLocation::None;
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
              schedulePositionMap, l1Reservations)) {
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
    chainToMerge.spillLocation = SpillLocation::None;
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

//===----------------------------------------------------------------------===//
// L1 Reservation Timeline
//===----------------------------------------------------------------------===//

// Represents a reservation of L1 memory for an op's output that stays in L1
// across multiple consumer chains. Used for forked ops like reshape that
// benefit from staying in L1 interleaved instead of spilling to DRAM.
struct L1Reservation {
  Operation *sourceOp; // The op whose output is reserved in L1
  int64_t startPos;    // Schedule position where reservation starts
  int64_t endPos;      // Schedule position where reservation ends (last user)
  uint64_t sizeBytes;  // L1 size reserved in bytes
};

// Compute total L1 reserved at a given schedule position.
// This is O(n) for now; can be optimized with Fenwick tree later.
static uint64_t
getActiveL1Reservations(int64_t schedulePos,
                        const std::vector<L1Reservation> &reservations) {
  uint64_t total = 0;
  for (const auto &res : reservations) {
    if (schedulePos >= res.startPos && schedulePos <= res.endPos) {
      total += res.sizeBytes;
    }
  }
  return total;
}

// Validate an op with L1 interleaved config.
// Returns the L1 size in bytes if valid, 0 otherwise.
static uint64_t tryGetL1InterleavedSize(Operation *op) {
  if (op->getNumResults() == 0) {
    return 0;
  }

  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!outputType) {
    return 0;
  }

  auto currentLayout =
      mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());
  if (!currentLayout || currentLayout.hasL1BufferType()) {
    return 0;
  }

  TTNNLayoutAttr l1Layout =
      currentLayout.withBufferType(BufferType::L1)
          .withMemoryLayout(TensorMemoryLayout::Interleaved);

  // Convert input layouts to L1 interleaved for validation
  std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(op);
  for (auto &inputLayout : inputLayouts) {
    inputLayout = inputLayout.withBufferType(BufferType::L1)
                      .withMemoryLayout(TensorMemoryLayout::Interleaved);
  }

  OpConfig l1Config;
  l1Config.outputLayout = l1Layout;

  op_constraint_validation::ValidationResult result =
      op_constraint_validation::validateOperation(op, inputLayouts, l1Config,
                                                  0);

  return result.isSuccess() ? result.outputL1Usage : 0;
}

// Validate that all chains in the active range [startPos, endPos] can execute
// with the given additional L1 reservation.
static bool validateChainsWithReservation(
    const std::vector<L1ChainConfig> &l1ChainConfigs,
    const llvm::DenseMap<Operation *, int64_t> &schedulePositionMap,
    const std::vector<L1Reservation> &existingReservations, int64_t startPos,
    int64_t endPos, uint64_t additionalL1) {

  for (const auto &chain : l1ChainConfigs) {
    if (chain.getState() != L1ChainState::Completed) {
      continue;
    }

    // Check if this chain overlaps with the reservation range
    bool overlaps = false;
    for (const auto &spec : chain.getOpL1MemSpecs()) {
      auto posIt = schedulePositionMap.find(spec.op);
      if (posIt == schedulePositionMap.end()) {
        continue;
      }
      int64_t opPos = posIt->second;
      if (opPos >= startPos && opPos <= endPos) {
        overlaps = true;
        break;
      }
    }

    if (!overlaps) {
      continue;
    }

    // Validate each op in the overlapping chain
    llvm::DenseMap<Operation *, TTNNLayoutAttr> resolvedLayoutMap =
        buildResolvedLayoutMap(chain);
    const auto &memReconfigEntryMap = chain.getMemReconfigEntryMap();

    for (const auto &spec : chain.getOpL1MemSpecs()) {
      auto posIt = schedulePositionMap.find(spec.op);
      if (posIt == schedulePositionMap.end()) {
        continue;
      }
      int64_t opPos = posIt->second;

      // Only validate ops within the reservation range
      if (opPos < startPos || opPos > endPos) {
        continue;
      }

      // Get existing reservations at this position
      uint64_t existingL1 =
          getActiveL1Reservations(opPos, existingReservations);

      // Build input layouts
      auto inputLayoutsOpt = buildInputLayoutsFromResolvedConfigs(
          spec.op, resolvedLayoutMap, memReconfigEntryMap);
      if (!inputLayoutsOpt) {
        return false;
      }

      // Validate with both existing and new reservation
      op_constraint_validation::ValidationResult result =
          op_constraint_validation::validateOperation(
              spec.op, *inputLayoutsOpt, spec.config,
              existingL1 + additionalL1);

      if (!result.isSuccess()) {
        TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                     "L1 reservation validation failed at op {}: {}",
                     spec.op->getName(), result.errorMessage);
        return false;
      }
    }
  }

  return true;
}

// Check if an operation is the last op in a chain that spills to DRAM.
// Returns the chain index if found, -1 otherwise.
static int64_t
findSpilledChainForOp(Operation *op,
                      const std::vector<L1ChainConfig> &l1ChainConfigs) {
  for (size_t chainIdx = 0; chainIdx < l1ChainConfigs.size(); ++chainIdx) {
    const auto &chain = l1ChainConfigs[chainIdx];
    if (chain.getState() != L1ChainState::Completed) {
      continue;
    }
    if (chain.spillLocation != SpillLocation::DRAM) {
      continue;
    }
    const auto &specs = chain.getOpL1MemSpecs();
    if (!specs.empty() && specs.back().op == op) {
      return static_cast<int64_t>(chainIdx);
    }
  }
  return -1;
}

static void applyL1ReservationsForReshapes(
    std::vector<L1ChainConfig> &l1ChainConfigs,
    const llvm::SmallVector<Operation *> &schedule,
    const llvm::DenseMap<Operation *, int64_t> &schedulePositionMap,
    std::vector<L1Reservation> &l1Reservations) {

  for (Operation *op : schedule) {
    if (!isa<ttnn::ReshapeOp>(op)) {
      continue;
    }

    // Get schedule position
    auto posIt = schedulePositionMap.find(op);
    if (posIt == schedulePositionMap.end()) {
      continue;
    }
    int64_t reshapePos = posIt->second;

    // Check if reshape's input comes from a chain that spills to DRAM.
    // If so and reshape is the only user, avoid the DRAM spill.
    Value reshapeInput = op->getOperand(0);
    Operation *inputDefOp = reshapeInput.getDefiningOp();

    if (inputDefOp && reshapeInput.hasOneUse()) {
      int64_t spilledChainIdx =
          findSpilledChainForOp(inputDefOp, l1ChainConfigs);

      if (spilledChainIdx >= 0) {
        auto inputPosIt = schedulePositionMap.find(inputDefOp);
        if (inputPosIt != schedulePositionMap.end()) {
          int64_t inputPos = inputPosIt->second;

          // Reshape must immediately follow the chain's last op
          if (reshapePos == inputPos + 1) {
            L1ChainConfig &chain =
                l1ChainConfigs[static_cast<size_t>(spilledChainIdx)];
            chain.spillLocation = SpillLocation::L1Interleaved;

            TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                         "Avoiding DRAM spill for chain {} feeding reshape {}",
                         spilledChainIdx, ttmlir::opToString(op));
          }
        }
      }
    }

    // Try to get L1 interleaved size via backend validation
    uint64_t l1Size = tryGetL1InterleavedSize(op);
    if (l1Size == 0) {
      continue;
    }

    // Find last user position
    int64_t lastUserPos = reshapePos;
    for (Operation *user : op->getUsers()) {
      auto userPosIt = schedulePositionMap.find(user);
      if (userPosIt != schedulePositionMap.end()) {
        lastUserPos = std::max(lastUserPos, userPosIt->second);
      }
    }

    // Validate all chains in the range can execute with this reservation
    if (!validateChainsWithReservation(l1ChainConfigs, schedulePositionMap,
                                       l1Reservations, reshapePos, lastUserPos,
                                       l1Size)) {
      continue;
    }

    l1Reservations.push_back({op, reshapePos, lastUserPos, l1Size});

    // Create L1ChainConfig for reshape so optimizer updates its layout
    auto outputType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
    auto currentLayout = mlir::cast<TTNNLayoutAttr>(outputType.getEncoding());
    TTNNLayoutAttr l1Layout = currentLayout.withBufferType(BufferType::L1);

    L1ChainConfig reshapeChain;
    OpL1MemSpec reshapeSpec;
    reshapeSpec.op = op;
    reshapeSpec.tensorSplitFactor = 1;
    reshapeSpec.config.outputLayout = l1Layout;
    reshapeChain.addOpL1MemSpec(std::move(reshapeSpec));
    reshapeChain.build();
    reshapeChain.resolve();

    llvm::DenseMap<Operation *, OpConfig> selectedConfigs;
    selectedConfigs[op] = OpConfig{l1Layout};
    llvm::DenseMap<Edge, MemReconfigEntry> emptyReconfigMap;
    reshapeChain.complete(selectedConfigs, emptyReconfigMap);
    reshapeChain.spillLocation = SpillLocation::None;

    l1ChainConfigs.push_back(std::move(reshapeChain));

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "L1 reservation: reshape {} in L1 (pos [{}, {}], {} bytes)",
                 ttmlir::opToString(op), reshapePos, lastUserPos, l1Size);
  }
}

// Find the schedule position of the last user of a value.
static int64_t findLastUserPosition(
    Value output,
    const llvm::DenseMap<Operation *, int64_t> &schedulePositionMap) {
  int64_t lastPos = -1;
  for (Operation *user : output.getUsers()) {
    auto posIt = schedulePositionMap.find(user);
    if (posIt != schedulePositionMap.end()) {
      lastPos = std::max(lastPos, posIt->second);
    }
  }
  return lastPos;
}

// Find which operand index of a consumer corresponds to a given value.
static std::optional<size_t> findOperandIndex(Operation *consumer,
                                              Value forkOutput) {
  for (size_t i = 0; i < consumer->getNumOperands(); ++i) {
    if (consumer->getOperand(i) == forkOutput) {
      return i;
    }
  }
  return std::nullopt;
}

// Check if a sharded layout would cause inefficient in0_block_w for matmul.
// For WIDTH_SHARDED and BLOCK_SHARDED, metal computes:
//   in0_block_w = gcd(shard_shape[1] / TILE_SIZE, K)
// When shard_width <= TILE_WIDTH (32), in0_block_w will be forced to 1,
// resulting in poor DRAM bandwidth utilization.
// HEIGHT_SHARDED is optimal as it uses in0_block_w = K.
// Returns true if this layout would be inefficient for the given matmul
// consumer.
static bool wouldCauseInefficientMatmulInput(Operation *consumer,
                                             size_t operandIndex,
                                             TTNNLayoutAttr layout) {
  // Only check for matmul/linear ops with input A (operand 0)
  if (!mlir::isa<ttnn::MatmulOp, ttnn::LinearOp>(consumer)) {
    return false;
  }
  if (operandIndex != 0) {
    return false; // Only input A affects in0_block_w
  }

  auto memLayoutOpt = layout.getMemLayoutOpt();
  if (!memLayoutOpt) {
    return false;
  }

  // WIDTH_SHARDED and BLOCK_SHARDED both use gcd(shard_width/TILE, K) formula
  // HEIGHT_SHARDED uses in0_block_w = K (optimal), so skip it
  if (*memLayoutOpt != TensorMemoryLayout::WidthSharded &&
      *memLayoutOpt != TensorMemoryLayout::BlockSharded) {
    return false;
  }

  // Get scalar shard shape - [height, width] in elements (not tiles)
  llvm::SmallVector<int64_t> shardShape = layout.getScalarShardShape();
  if (shardShape.size() < 2) {
    return false;
  }

  // If shard width <= TILE_WIDTH (32 elements), in0_block_w will be 1
  // (inefficient)
  int64_t shardWidth = shardShape.back();
  if (shardWidth <= static_cast<int64_t>(TILE_WIDTH)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "  Matmul {} would have in0_block_w=1 with sharded "
                 "input (shard_width={} <= {})",
                 ttmlir::opToString(consumer), shardWidth, TILE_WIDTH);
    return true;
  }

  return false;
}

// Helper to compute ceiling division
static inline int64_t divUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Forward declaration - defined below
static void setMatmulProgramConfigForShardedOutput(Operation *op,
                                                   const OpConfig &config);

// RAII guard to temporarily set matmul program config during validation.
// Saves the original config on construction, sets the new config, and restores
// the original on destruction unless commit() is called.
class MatmulProgramConfigGuard {
public:
  MatmulProgramConfigGuard(Operation *op, const OpConfig &config)
      : matmulOp(mlir::dyn_cast<ttnn::MatmulOp>(op)) {
    if (matmulOp) {
      savedConfig = matmulOp.getMatmulProgramConfigAttr();
      // Set the program config for validation
      setMatmulProgramConfigForShardedOutput(op, config);
    }
  }

  ~MatmulProgramConfigGuard() {
    if (matmulOp && !committed) {
      matmulOp.setMatmulProgramConfigAttr(savedConfig);
    }
  }

  // Call this when validation succeeds and we want to keep the config
  void commit() { committed = true; }

  // Check if this is a matmul op
  bool isMatmul() const { return matmulOp != nullptr; }

private:
  ttnn::MatmulOp matmulOp;
  mlir::Attribute savedConfig;
  bool committed = false;
};

// Compute the bounding box grid dimensions from a layout's shard grid.
// Returns (gridX, gridY) representing the physical compute grid.
static std::pair<int64_t, int64_t>
getPhysicalGridDimensions(TTNNLayoutAttr layout) {
  ttcore::GridAttr shardGrid = layout.getGrid();
  AffineMap mapping = shardGrid.getMapping();

  // Use toCoreRangeSet to get physical core coordinates
  auto coreRanges =
      ttcore::utils::toCoreRangeSet(shardGrid.getShape(), mapping);

  // Compute bounding box of all core ranges
  int64_t maxX = 0;
  int64_t maxY = 0;
  for (const auto &[loc, size] : coreRanges) {
    maxX = std::max(maxX, static_cast<int64_t>(loc[0] + size[0]));
    maxY = std::max(maxY, static_cast<int64_t>(loc[1] + size[1]));
  }

  return {maxX, maxY};
}

// Set MatmulMultiCoreReuseMultiCast1DProgramConfig on a matmul op.
// Used when input is interleaved, width sharded, or height sharded.
// For width sharded output: mcast_in0=true (multicast input A along width)
// For height sharded output: mcast_in0=false (multicast input B along height)
static void setMatmul1DProgramConfig(ttnn::MatmulOp matmulOp, int64_t Mt,
                                     int64_t Nt, int64_t Kt,
                                     TTNNLayoutAttr outputLayout,
                                     TensorMemoryLayout outputMemLayout) {
  MLIRContext *ctx = matmulOp.getContext();

  // Get physical grid dimensions from the shard spec
  auto [gridX, gridY] = getPhysicalGridDimensions(outputLayout);
  int64_t numCores = gridX * gridY;

  // For width sharded output: mcast_in0=true
  // per_core_M = Mt (all M tiles per core)
  // per_core_N = ceil(Nt / numCores)
  bool mcastIn0 = (outputMemLayout == TensorMemoryLayout::WidthSharded);
  int64_t perCoreM, perCoreN;

  if (mcastIn0) {
    perCoreM = Mt;
    perCoreN = divUp(Nt, numCores);
  } else {
    // Height sharded: mcast_in0=false
    perCoreM = divUp(Mt, numCores);
    perCoreN = Nt;
  }

  // in0_block_w: determines how many K tiles are loaded per iteration.
  // Larger values improve DRAM bandwidth utilization but increase L1 pressure.
  // Constraint: Kt % in0_block_w == 0
  //
  // For height sharded (mcast_in0=false): use full Kt for optimal bandwidth
  // (matches tt-metal's behavior in create_simple_matmul_program_config)
  //
  // For width sharded (mcast_in0=true): balance DRAM bandwidth vs L1 pressure
  // - Small Nt (<=128 tiles / 4096 elements): use in0_block_w=8 for better DRAM
  // BW
  // - Large Nt (>128 tiles / 4096 elements): use in0_block_w=2 to reduce L1
  // pressure Analysis on decoder model showed:
  //   Q/K/V proj (N=512-2048): in0_block_w=8 -> 2x better DRAM BW
  //   FFN up/gate (N=8192): in0_block_w=2 -> avoids L1 pressure, better perf
  constexpr int64_t kLargeNtThreshold = 128; // 128 tiles = 4096 elements

  int64_t in0BlockW;
  if (!mcastIn0) {
    // Height sharded: use full Kt (optimal for DRAM bandwidth)
    in0BlockW = Kt;
  } else {
    // Width sharded: choose based on output N dimension
    if (Nt > kLargeNtThreshold) {
      // Large N: use smaller in0_block_w to reduce L1 pressure
      if (Kt % 2 == 0) {
        in0BlockW = 2;
      } else {
        in0BlockW = 1;
      }
    } else {
      // Small N: use larger in0_block_w for better DRAM bandwidth
      if (Kt % 8 == 0) {
        in0BlockW = 8;
      } else if (Kt % 4 == 0) {
        in0BlockW = 4;
      } else if (Kt % 2 == 0) {
        in0BlockW = 2;
      } else {
        in0BlockW = 1;
      }
    }
  }

  // out_block_h/w: use per_core values
  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;

  // out_subblock_h/w: conservative values that should fit in L1
  // Constraint: out_subblock_h * out_subblock_w <= 8 for safe dest register
  // usage
  int64_t outSubblockH = 1;
  int64_t outSubblockW = std::min(outBlockW, static_cast<int64_t>(8));

  // Create attributes
  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);
  auto hopCoresAttr = CoreRangeSetAttr::get(ctx, {});

  auto programConfig = MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      ctx,
      gridAttr,                            // compute_with_storage_grid_size
      static_cast<uint64_t>(in0BlockW),    // in0_block_w
      static_cast<uint64_t>(outSubblockH), // out_subblock_h
      static_cast<uint64_t>(outSubblockW), // out_subblock_w
      static_cast<uint64_t>(outBlockH),    // out_block_h
      static_cast<uint64_t>(outBlockW),    // out_block_w
      static_cast<uint64_t>(perCoreM),     // per_core_m
      static_cast<uint64_t>(perCoreN),     // per_core_n
      false,                               // fuse_batch
      nullptr,                             // fused_activation
      mcastIn0,                            // mcast_in0
      false,                               // gather_in0
      hopCoresAttr,                        // hop_cores
      0,                                   // num_global_cb_receivers
      false);                              // untilize_out

  matmulOp.setMatmulProgramConfigAttr(programConfig);

  TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
               "  Set MatmulMultiCoreReuseMultiCast1DProgramConfig: "
               "grid={}x{}, per_core_m={}, per_core_n={}, in0_block_w={}, "
               "mcast_in0={}",
               gridX, gridY, perCoreM, perCoreN, in0BlockW, mcastIn0);
}

// Set MatmulMultiCoreReuseMultiCastProgramConfig on a matmul op.
// Used when input is block sharded (requires 2D multicast).
static void setMatmul2DProgramConfig(ttnn::MatmulOp matmulOp, int64_t Mt,
                                     int64_t Nt, int64_t Kt,
                                     TTNNLayoutAttr outputLayout) {
  MLIRContext *ctx = matmulOp.getContext();

  // Get physical grid dimensions from the shard spec
  auto [gridX, gridY] = getPhysicalGridDimensions(outputLayout);

  // For block sharded, distribute M and N across the 2D grid
  int64_t perCoreM = divUp(Mt, gridY);
  int64_t perCoreN = divUp(Nt, gridX);

  // in0_block_w: determines how many K tiles are loaded per iteration.
  // Constraint: Kt % in0_block_w == 0
  // For block sharded, try larger divisors for better DRAM bandwidth.
  int64_t in0BlockW;
  if (Kt % 8 == 0) {
    in0BlockW = 8;
  } else if (Kt % 4 == 0) {
    in0BlockW = 4;
  } else if (Kt % 2 == 0) {
    in0BlockW = 2;
  } else {
    in0BlockW = 1;
  }

  // out_block_h/w: use per_core values
  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;

  // out_subblock_h/w: conservative values
  int64_t outSubblockH = 4;
  int64_t outSubblockW = 2;
  if (outSubblockW > perCoreN) {
    outSubblockH = 1;
    outSubblockW = std::min(perCoreN, static_cast<int64_t>(8));
  }

  auto programConfig = MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
      ctx,
      CoreCoordAttr::get(ctx, gridX, gridY), // compute_with_storage_grid_size
      static_cast<uint64_t>(in0BlockW),      // in0_block_w
      static_cast<uint64_t>(outSubblockH),   // out_subblock_h
      static_cast<uint64_t>(outSubblockW),   // out_subblock_w
      static_cast<uint64_t>(outBlockH),      // out_block_h
      static_cast<uint64_t>(outBlockW),      // out_block_w
      static_cast<uint64_t>(perCoreM),       // per_core_m
      static_cast<uint64_t>(perCoreN),       // per_core_n
      false,                                 // transpose_mcast
      nullptr,                               // fused_activation
      false);                                // fuse_batch

  matmulOp.setMatmulProgramConfigAttr(programConfig);

  TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
               "  Set MatmulMultiCoreReuseMultiCastProgramConfig: "
               "grid={}x{}, per_core_m={}, per_core_n={}, in0_block_w={}",
               gridX, gridY, perCoreM, perCoreN, in0BlockW);
}

// For matmul with sharded output, set an explicit program config to ensure
// consistent behavior between compile-time validation and runtime execution.
//
// Background: tt-metal's create_simple_matmul_program_config() selects
// program config based on available L1 memory. At compile-time, mock tensors
// don't occupy L1. At runtime, input tensors occupy L1. This mismatch can cause
// different program config selection, leading to runtime errors.
//
// By setting an explicit program config, we bypass tt-metal's dynamic
// selection. Config type depends on input sharding:
// - Interleaved/Width sharded/Height sharded input -> 1D mcast config
// - Block sharded input -> 2D mcast config
//
// TODO(#xxxx): Math fidelity defaults to LoFi when program_config is set.
// In tt-metal's matmul_op.cpp, when has_program_config=true, the default
// math_fidelity is LoFi (see create_matmul_struct). To override this, we need
// to add device_compute_kernel_config support to TTNN_MatmulOp:
// 1. Add compute_config attr to TTNN_MatmulOp in TTNNOps.td
// 2. Add serialization in TTNNToFlatbuffer.cpp and Target.fbs
// 3. Pass compute_kernel_config in
// runtime/lib/ttnn/operations/matmul/matmul.cpp Until then, matmuls with
// program_config will use LoFi math fidelity.
static void setMatmulProgramConfigForShardedOutput(Operation *op,
                                                   const OpConfig &config) {
  auto matmulOp = mlir::dyn_cast<ttnn::MatmulOp>(op);
  if (!matmulOp) {
    return;
  }

  // If program config is already set, don't override
  if (matmulOp.getMatmulProgramConfigAttr()) {
    return;
  }

  // Check if output is sharded
  if (!config.outputLayout) {
    return;
  }

  TensorMemoryLayout outputMemLayout =
      config.outputLayout.getMemLayout().getValue();
  if (outputMemLayout != TensorMemoryLayout::WidthSharded &&
      outputMemLayout != TensorMemoryLayout::HeightSharded &&
      outputMemLayout != TensorMemoryLayout::BlockSharded) {
    return;
  }

  // Get output shape to calculate Mt, Nt
  if (op->getNumResults() == 0) {
    return;
  }
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return;
  }

  llvm::ArrayRef<int64_t> outShape = resultType.getShape();
  if (outShape.size() < 2) {
    return;
  }

  // Get input A shape and layout
  auto inputA = matmulOp.getA();
  auto inputAType = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  if (!inputAType) {
    return;
  }
  llvm::ArrayRef<int64_t> aShape = inputAType.getShape();
  if (aShape.size() < 2) {
    return;
  }

  TTNNLayoutAttr inputALayout = utils::getLayoutAttrFromTensor(inputAType);

  // Calculate Mt, Nt, Kt (dimensions in tiles)
  int64_t M = outShape[outShape.size() - 2];
  int64_t N = outShape[outShape.size() - 1];
  int64_t K = aShape[aShape.size() - 1];
  int64_t Mt = divUp(M, TILE_HEIGHT);
  int64_t Nt = divUp(N, TILE_WIDTH);
  int64_t Kt = divUp(K, TILE_WIDTH);

  // Determine input A memory layout
  TensorMemoryLayout inputMemLayout = TensorMemoryLayout::Interleaved;
  if (inputALayout && inputALayout.hasShardedL1TensorMemoryLayout()) {
    inputMemLayout = inputALayout.getMemLayout().getValue();
  }

  TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
               "  Setting matmul program config: Mt={}, Nt={}, Kt={}, "
               "inputMemLayout={}, outputMemLayout={}",
               Mt, Nt, Kt, static_cast<int>(inputMemLayout),
               static_cast<int>(outputMemLayout));

  // Select config type based on input sharding
  TTNNLayoutAttr outputLayout = config.outputLayout;
  if (inputMemLayout == TensorMemoryLayout::BlockSharded) {
    // Block sharded input requires 2D mcast config
    setMatmul2DProgramConfig(matmulOp, Mt, Nt, Kt, outputLayout);
  } else {
    // Interleaved, width sharded, or height sharded input uses 1D mcast config
    setMatmul1DProgramConfig(matmulOp, Mt, Nt, Kt, outputLayout,
                             outputMemLayout);
  }
}

// Validate that an op can accept a specific input layout and produce its
// expected output layout. For matmul/linear ops, uses withIgnorePhysicalLayout
// during validation. Returns true only if validation succeeds AND the actual
// output layout matches the config's expected output layout.
//
// IMPORTANT: For matmul ops with sharded output, the caller MUST use
// MatmulProgramConfigGuard before calling this function to ensure the program
// config is set during validation and properly restored after.
static bool validateOpWithInputLayout(Operation *op, size_t inputOperandIndex,
                                      TTNNLayoutAttr inputLayout,
                                      const OpConfig &config) {
  // Build input layouts - use provided layout for the specified operand,
  // extract current layouts for other operands
  std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(op);
  if (inputOperandIndex >= inputLayouts.size()) {
    return false;
  }
  inputLayouts[inputOperandIndex] = inputLayout;

  // For matmul/linear ops without fused activation, use
  // withIgnorePhysicalLayout to avoid strict grid matching during validation
  // (similar to preprocessFirstOp). When activation is present, we need full
  // layout because the internal unary op for activation cannot handle partial
  // memory configs (crashes in validate_shard_spec).
  // TODO(tt-metal#34500): Remove activation check once tt-metal handles partial
  // memory configs in fused activations.
  bool useIgnorePhysicalLayout = false;
  if (auto matmulOp = mlir::dyn_cast<ttnn::MatmulOp>(op)) {
    useIgnorePhysicalLayout = !matmulOp.getActivation().has_value();
  } else if (auto linearOp = mlir::dyn_cast<ttnn::LinearOp>(op)) {
    useIgnorePhysicalLayout = !linearOp.getActivation().has_value();
  }

  OpConfig testConfig = config;
  TTNNLayoutAttr expectedLayout = config.outputLayout;
  if (useIgnorePhysicalLayout && testConfig.outputLayout) {
    testConfig.outputLayout =
        testConfig.outputLayout.withIgnorePhysicalLayout(true);
  }

  op_constraint_validation::ValidationResult result =
      op_constraint_validation::validateOperation(op, inputLayouts, testConfig,
                                                  0);
  if (!result.isSuccess()) {
    return false;
  }

  // Verify the actual output layout matches the expected config layout.
  // This ensures the op will produce the layout we planned for.
  if (result.actualOutputLayout != expectedLayout) {
    TTMLIR_TRACE(ttmlir::LogComponent::DFShardingPolicy,
                 "  Op actual output {} != expected {}",
                 result.actualOutputLayout, expectedLayout);
    return false;
  }

  return true;
}

// Get the selected OpConfig for an operation.
// If the op is in a chain, returns the config from opL1MemSpecs.
// Otherwise, creates a config from the IR's output layout.
static std::optional<OpConfig>
getSelectedConfig(Operation *op,
                  const llvm::DenseMap<Operation *, size_t> &opToChainMap,
                  const std::vector<L1ChainConfig> &l1ChainConfigs) {
  auto chainIt = opToChainMap.find(op);
  if (chainIt != opToChainMap.end()) {
    // Op is in a chain - find its config in opL1MemSpecs
    const auto &specs = l1ChainConfigs[chainIt->second].getOpL1MemSpecs();
    for (const auto &spec : specs) {
      if (spec.op == op) {
        return spec.config;
      }
    }
  }

  // Op not in chain - get layout from IR
  if (op->getNumResults() == 0) {
    return std::nullopt;
  }
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.getEncoding()) {
    return std::nullopt;
  }
  TTNNLayoutAttr layout = utils::getLayoutAttrFromTensor(resultType);
  return OpConfig(layout, OpConfig::OpSpecificAttrs{});
}

// Apply L1 reservations for fork ops (ops with multiple users).
// This tries to keep forked tensors in L1 instead of spilling to DRAM,
// avoiding redundant DRAM reads by multiple consumers.
//
// Algorithm:
// 1. For each chain that spills to DRAM and has a forked output (multiple
// users)
// 2. Try passing the chain's sharded output layout to all consumers
// 3. If that fails, try L1 interleaved as fallback
// 4. Validate memory pressure across all chains in the fork span
// 5. If valid, update spill location and create L1 reservation
static void applyL1ReservationsForForkOps(
    std::vector<L1ChainConfig> &l1ChainConfigs,
    const llvm::SmallVector<Operation *> &schedule,
    const llvm::DenseMap<Operation *, int64_t> &schedulePositionMap,
    std::vector<L1Reservation> &l1Reservations) {

  // Build op to chain map for looking up selected configs
  llvm::DenseMap<Operation *, size_t> opToChainMap =
      buildOpToChainMap(l1ChainConfigs);

  for (auto &chain : l1ChainConfigs) {
    if (chain.getState() != L1ChainState::Completed) {
      continue;
    }
    if (chain.spillLocation != SpillLocation::DRAM) {
      continue;
    }

    Operation *lastOp = chain.getLastOp();
    if (lastOp->getNumResults() == 0) {
      continue;
    }

    Value forkOutput = lastOp->getResult(0);

    // Step 1: Check if this is a fork (multiple users)
    if (forkOutput.hasOneUse()) {
      continue;
    }

    // Get fork op's resolved sharded layout
    const auto &specs = chain.getOpL1MemSpecs();
    if (specs.empty()) {
      continue;
    }
    TTNNLayoutAttr shardedLayout = specs.back().config.outputLayout;
    if (!shardedLayout) {
      continue;
    }

    // Get schedule position
    auto forkPosIt = schedulePositionMap.find(lastOp);
    if (forkPosIt == schedulePositionMap.end()) {
      continue;
    }
    int64_t forkPos = forkPosIt->second;

    // Find last user position
    int64_t lastUserPos = findLastUserPosition(forkOutput, schedulePositionMap);
    if (lastUserPos < 0) {
      continue;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Fork op {} has {} users, span [{}, {}]",
                 ttmlir::opToString(lastOp),
                 std::distance(forkOutput.getUsers().begin(),
                               forkOutput.getUsers().end()),
                 forkPos, lastUserPos);

    // Step 2: Try passing sharded layout to all consumers
    // Use guards to track program configs that need to be committed on success
    std::vector<std::unique_ptr<MatmulProgramConfigGuard>> shardedGuards;
    bool allConsumersValidWithSharded = true;
    for (Operation *user : forkOutput.getUsers()) {
      auto operandIdx = findOperandIndex(user, forkOutput);
      if (!operandIdx) {
        allConsumersValidWithSharded = false;
        break;
      }
      // Check if this would cause inefficient in0_block_w for matmul
      if (wouldCauseInefficientMatmulInput(user, *operandIdx, shardedLayout)) {
        allConsumersValidWithSharded = false;
        break;
      }
      // Get consumer's selected config and validate with fork input
      auto selectedConfig =
          getSelectedConfig(user, opToChainMap, l1ChainConfigs);
      if (!selectedConfig) {
        allConsumersValidWithSharded = false;
        break;
      }
      // Create guard to set program config before validation
      auto guard =
          std::make_unique<MatmulProgramConfigGuard>(user, *selectedConfig);
      if (!validateOpWithInputLayout(user, *operandIdx, shardedLayout,
                                     *selectedConfig)) {
        allConsumersValidWithSharded = false;
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "  Consumer {} cannot accept sharded input",
                     ttmlir::opToString(user));
        break;
      }
      shardedGuards.push_back(std::move(guard));
    }

    if (allConsumersValidWithSharded) {
      // Validate memory pressure with sharded layout
      uint64_t l1Size = shardedLayout.getShardSizeInBytes();
      if (validateChainsWithReservation(l1ChainConfigs, schedulePositionMap,
                                        l1Reservations, forkPos, lastUserPos,
                                        l1Size)) {
        // SUCCESS: Keep sharded, no spill needed - commit all guards
        for (auto &guard : shardedGuards) {
          guard->commit();
        }
        chain.spillLocation = SpillLocation::None;
        l1Reservations.push_back({lastOp, forkPos, lastUserPos, l1Size});
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "Fork op {}: keeping sharded in L1 ({} bytes)",
                     ttmlir::opToString(lastOp), l1Size);
        continue;
      }
    }
    // Guards will auto-restore on scope exit if not committed

    // Step 3: Try L1 interleaved fallback
    // Use tryGetL1InterleavedSize to validate and get L1 size
    uint64_t l1InterleavedSize = tryGetL1InterleavedSize(lastOp);
    if (l1InterleavedSize == 0) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Fork op {}: L1 interleaved validation failed for op itself",
                   ttmlir::opToString(lastOp));
      continue;
    }

    TTNNLayoutAttr l1InterleavedLayout =
        shardedLayout.withBufferType(BufferType::L1)
            .withMemoryLayout(TensorMemoryLayout::Interleaved);

    // Use guards to track program configs for L1 interleaved path
    std::vector<std::unique_ptr<MatmulProgramConfigGuard>> l1Guards;
    bool allConsumersValidWithL1Interleaved = true;
    for (Operation *user : forkOutput.getUsers()) {
      auto operandIdx = findOperandIndex(user, forkOutput);
      if (!operandIdx) {
        allConsumersValidWithL1Interleaved = false;
        break;
      }
      // Get consumer's selected config and validate with L1 interleaved input
      auto selectedConfig =
          getSelectedConfig(user, opToChainMap, l1ChainConfigs);
      if (!selectedConfig) {
        allConsumersValidWithL1Interleaved = false;
        break;
      }
      TTMLIR_TRACE(
          ttmlir::LogComponent::DFShardingPolicy,
          "  Validating consumer {} with L1 interleaved input and config {}",
          ttmlir::opToString(user), selectedConfig->outputLayout);

      // Create guard to set program config before validation
      auto guard =
          std::make_unique<MatmulProgramConfigGuard>(user, *selectedConfig);
      if (!validateOpWithInputLayout(user, *operandIdx, l1InterleavedLayout,
                                     *selectedConfig)) {
        allConsumersValidWithL1Interleaved = false;
        TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                     "  Consumer {} cannot accept L1 interleaved input",
                     ttmlir::opToString(user));
        break;
      }
      l1Guards.push_back(std::move(guard));
    }

    if (!allConsumersValidWithL1Interleaved) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Fork op {}: consumers cannot accept L1 interleaved, "
                   "keeping DRAM spill",
                   ttmlir::opToString(lastOp));
      continue;
    }

    // Step 4: Validate memory pressure with L1 interleaved reservation
    if (!validateChainsWithReservation(l1ChainConfigs, schedulePositionMap,
                                       l1Reservations, forkPos, lastUserPos,
                                       l1InterleavedSize)) {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Fork op {}: L1 interleaved fails memory validation, "
                   "keeping DRAM spill",
                   ttmlir::opToString(lastOp));
      continue;
    }

    // SUCCESS: Use L1 interleaved - commit all guards
    for (auto &guard : l1Guards) {
      guard->commit();
    }
    chain.spillLocation = SpillLocation::L1Interleaved;
    l1Reservations.push_back({lastOp, forkPos, lastUserPos, l1InterleavedSize});
    TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                 "Fork op {}: using L1 interleaved ({} bytes)",
                 ttmlir::opToString(lastOp), l1InterleavedSize);
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
                    ttnn::PowScalarOp, ttnn::SliceStaticOp,
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
      l1ChainConfig.spillLocation = SpillLocation::DRAM;
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

  // Post-processing: apply L1 reservations for forked ops (like reshape)
  // BEFORE chain merges so merges account for reserved L1 memory.
  std::vector<L1Reservation> l1Reservations;
  if (funcToProcess) {
    llvm::DenseMap<Operation *, int64_t> schedulePositionMap =
        buildSchedulePositionMap((*schedule)[funcToProcess]);
    applyL1ReservationsForReshapes(*l1ChainConfigs, (*schedule)[funcToProcess],
                                   schedulePositionMap, l1Reservations);

    // Apply L1 reservations for fork ops (ops with multiple users)
    applyL1ReservationsForForkOps(*l1ChainConfigs, (*schedule)[funcToProcess],
                                  schedulePositionMap, l1Reservations);

    // Attempt to merge chains where Chain A's output can stay in L1 and be
    // consumed as RHS by a join op in Chain B.
    applyChainMerges(*l1ChainConfigs, (*schedule)[funcToProcess],
                     l1Reservations);
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
                 stringifyTensorMemoryLayout(*preferredMemLayout),
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
                   stringifyTensorMemoryLayout(currentMemLayout), coreUsage);

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
                   stringifyTensorMemoryLayout(*preferredMemLayout));
      shardSolver.set(lastOp, *preferredConfig);
      lastOpProcessed = true;
    } else {
      TTMLIR_DEBUG(ttmlir::LogComponent::DFShardingPolicy,
                   "Last op {}: preferred {} layout NOT found in valid configs",
                   lastOp->getName(),
                   stringifyTensorMemoryLayout(*preferredMemLayout));
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
