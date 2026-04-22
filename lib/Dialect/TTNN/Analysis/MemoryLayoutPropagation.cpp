// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutPropagation.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/D2MOptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>

namespace mlir::tt::ttnn {

MemoryLayoutPropagation::MemoryLayoutPropagation(
    func::FuncOp func, ttcore::GridAttr deviceGrid,
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
    const TensorTypeLayoutsMap *tensorTypePossibleLayouts, size_t beamWidth,
    size_t maxInputCandidatesPerOperand, size_t maxReshardCandidatesPerType,
    std::unique_ptr<LayoutPropagationObserver> observer)
    : func(func), deviceGrid(deviceGrid), legalConfigs(legalConfigs),
      tensorTypePossibleLayouts(tensorTypePossibleLayouts),
      beamWidth(beamWidth),
      maxInputCandidatesPerOperand(maxInputCandidatesPerOperand),
      maxReshardCandidatesPerType(maxReshardCandidatesPerType) {
  if (observer) {
    this->observer = std::move(observer);
  } else {
    this->observer = std::make_unique<LayoutPropagationObserver>();
  }
}

MemoryLayoutPropagation::~MemoryLayoutPropagation() = default;

const BeamCandidate *
MemoryLayoutPropagation::getChosenCandidate(Operation *op) const {
  auto it = beamState.find(op);
  if (it == beamState.end() || it->second.empty()) {
    return nullptr;
  }
  size_t chosenIdx = 0;
  auto choiceIt = finalChoice.find(op);
  if (choiceIt != finalChoice.end()) {
    chosenIdx = choiceIt->second;
  }
  if (chosenIdx >= it->second.size()) {
    chosenIdx = 0;
  }
  return &it->second[chosenIdx];
}

/// Get the output layout for a specific result index from a beam candidate.
/// Falls back to configHint.outputLayout for single-output ops or when
/// outputLayouts is not populated.
static TTNNLayoutAttr getOutputLayoutForResult(const BeamCandidate &c,
                                               size_t ri) {
  if (!c.outputLayouts.empty() && ri < c.outputLayouts.size()) {
    return c.outputLayouts[ri];
  }
  return c.configHint.outputLayout;
}

/// Returns an optional filter predicate that rejects invalid input layouts
/// for a given op.  When the filter returns false, the candidate is removed.
/// Returns nullptr when no filtering is needed (all layouts accepted).
/// Delegates to per-op rule files via OpRuleBook.
static LayoutFilterFn getInputLayoutFilter(Operation *op, unsigned operandIdx) {
  return getRuleBook(op).getInputLayoutFilter(operandIdx);
}

/// Check if a layout is sharded.
static bool isShardedLayout(TTNNLayoutAttr layout) {
  if (!layout) {
    return false;
  }
  auto memLayout = layout.getMemLayout();
  return memLayout && isShardedMemoryLayout(memLayout.getValue());
}

/// Compute total bytes transferred from DRAM across all inputs.
static uint64_t
computeInputDramBytes(const std::vector<TTNNLayoutAttr> &inputLayouts) {
  uint64_t dramBytes = 0;
  for (const auto &layout : inputLayouts) {
    if (layout && !layout.hasL1BufferType()) {
      uint64_t tensorBytes = layout.getShardSizeInBytes();
      if (auto grid = layout.getGrid()) {
        for (auto dim : grid.getShape()) {
          tensorBytes *= dim;
        }
      }
      dramBytes += tensorBytes;
    }
  }
  return dramBytes;
}

/// Format input layouts as a comma-separated "bufType/memLayout" string.
[[maybe_unused]] static std::string
formatInputLayouts(const std::vector<TTNNLayoutAttr> &layouts) {
  std::string desc;
  llvm::raw_string_ostream os(desc);
  for (size_t i = 0; i < layouts.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << layouts[i].getBufferType() << "/" << layouts[i].getMemLayout();
  }
  return desc;
}

/// Apply op-specific configurations from the chosen candidate.
/// Delegates to per-op rule files via OpRuleBook.
static void applyOpSpecificAttrs(Operation *op,
                                 const BeamCandidate &candidate) {
  getRuleBook(op).applyOpSpecificAttrs(op, candidate);
}

std::optional<BeamCandidate> MemoryLayoutPropagation::evaluateHint(
    Operation *op, const OpConfig &hint, size_t hintIdx,
    const std::vector<TTNNLayoutAttr> &inputLayouts, bool anyReshard,
    const llvm::SmallVector<size_t> &producerCandidateIndices,
    const llvm::DenseMap<size_t, TTNNLayoutAttr> &reshardLayouts) {
  auto result =
      op_constraint_validation::validateOperation(op, inputLayouts, hint);
  if (result.isSuccess()) {
    BeamCandidate candidate;
    candidate.configHint =
        OpConfig(result.getFirstActualOutputLayout(), hint.opSpecificAttrs);
    candidate.score = scoreCandidate(op, hint, result, anyReshard);
    candidate.score.inputDramBytes = computeInputDramBytes(inputLayouts);
    candidate.validationResult = result;
    candidate.inputLayouts = inputLayouts;
    candidate.producerCandidateIndices = producerCandidateIndices;
    candidate.reshardLayouts = reshardLayouts;
    candidate.outputLayouts = result.actualOutputLayouts;

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "    VALID candidate for {0}: hint[{1}]={2} "
                 "score(L1={3},sharded={4},dramIn={5},"
                 "reshard={6},cores={7},l1use={8})",
                 op->getName(), hintIdx, candidate.configHint.toCompactString(),
                 candidate.score.isL1, candidate.score.isSharded,
                 candidate.score.inputDramBytes,
                 candidate.score.requiresReshard, candidate.score.coreCount,
                 candidate.score.outputL1Usage);

    observer->onEvaluation(op, hint, hintIdx, inputLayouts, /*valid=*/true,
                           &candidate, /*failureReason=*/"");

    return candidate;
  }

  // Log validation failures.
  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "    FAILED validation for {0}: hint[{1}]={2} error={3} "
               "inputs=[{4}] reshard={5}",
               op->getName(), hintIdx, hint.toCompactString(),
               result.errorMessage, formatInputLayouts(inputLayouts),
               anyReshard);

  observer->onEvaluation(op, hint, hintIdx, inputLayouts, /*valid=*/false,
                         /*candidate=*/nullptr, result.errorMessage);

  return std::nullopt;
}

// Observer-only recording helpers. These feed the decision-trace observer
// and do not participate in the layout propagation algorithm.
namespace observer_recording {

using BeamState =
    llvm::DenseMap<Operation *, llvm::SmallVector<BeamCandidate, 0>>;

static const BeamCandidate *
getChosenCandidate(Operation *op, const BeamState &beamState,
                   const llvm::DenseMap<Operation *, size_t> &finalChoice) {
  auto it = beamState.find(op);
  if (it == beamState.end() || it->second.empty()) {
    return nullptr;
  }
  size_t chosenIdx = 0;
  auto choiceIt = finalChoice.find(op);
  if (choiceIt != finalChoice.end()) {
    chosenIdx = choiceIt->second;
  }
  if (chosenIdx >= it->second.size()) {
    chosenIdx = 0;
  }
  return &it->second[chosenIdx];
}

void recordInplaceOps(func::FuncOp func, LayoutPropagationObserver *observer,
                      const BeamState &beamState, size_t &opIndex) {
  func->walk([&](Operation *op) {
    if (op->getNumResults() != 0) {
      return;
    }
    InplaceOpInfo info;
    info.op = op;
    bool hasTrackedProducer = false;
    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      if (!mlir::isa<RankedTensorType>(operand.getType())) {
        continue;
      }
      auto tensorType = mlir::cast<RankedTensorType>(operand.getType());
      auto layout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
      Operation *producer = operand.getDefiningOp();
      info.operands.push_back({idx, layout, producer});
      if (producer && beamState.count(producer)) {
        hasTrackedProducer = true;
      }
    }
    if (!hasTrackedProducer) {
      return;
    }
    observer->onInplaceOp(info);
    ++opIndex;
  });
}

void recordFinalChoices(
    func::FuncOp func, LayoutPropagationObserver *observer,
    const BeamState &beamState,
    const llvm::DenseMap<Operation *, size_t> &finalChoice) {
  size_t finalIdx = 0;
  func->walk([&](Operation *op) {
    const BeamCandidate *chosen =
        getChosenCandidate(op, beamState, finalChoice);
    if (!chosen) {
      return;
    }
    observer->onFinalChoice(op, finalIdx, *chosen);
    ++finalIdx;
  });
}

void recordEdges(func::FuncOp func, LayoutPropagationObserver *observer,
                 const BeamState &beamState,
                 const llvm::DenseMap<Operation *, size_t> &finalChoice) {
  func->walk([&](Operation *consumerOp) {
    const BeamCandidate *chosen =
        getChosenCandidate(consumerOp, beamState, finalChoice);
    if (!chosen) {
      return;
    }

    size_t tensorOperandIdx = 0;
    for (auto operand : consumerOp->getOperands()) {
      if (!mlir::isa<RankedTensorType>(operand.getType())) {
        continue;
      }
      Operation *producerOp = operand.getDefiningOp();
      if (producerOp && beamState.count(producerOp)) {
        bool hasReshard = chosen->reshardLayouts.count(tensorOperandIdx) > 0;
        TTNNLayoutAttr reshardLayout;
        if (hasReshard) {
          reshardLayout = chosen->reshardLayouts.lookup(tensorOperandIdx);
        }
        size_t producerResultIdx = 0;
        if (auto opResult = mlir::dyn_cast<OpResult>(operand)) {
          producerResultIdx = opResult.getResultNumber();
        }
        observer->onEdge(producerOp, consumerOp, tensorOperandIdx,
                         producerResultIdx, hasReshard, reshardLayout);
      }
      ++tensorOperandIdx;
    }
  });
}

} // namespace observer_recording

void MemoryLayoutPropagation::run() {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "MemoryLayoutPropagation::run() starting for func {0}",
               func.getName());

  observer->onStart(func.getName(), beamWidth);

  size_t opIndex = 0;
  // Forward pass: propagate layouts in scheduled (IR) order.
  func->walk([&](Operation *op) {
    if (!LegalOpLayoutAnalysis::isValidAnalysisTarget(op)) {
      return;
    }
    // Skip ops that don't implement the OpModel interface (e.g.,
    // ttcore.load_cached). These ops cannot be validated by the backend.
    if (!mlir::dyn_cast<OpModel>(op)) {
      return;
    }
    // Skip ToLayoutOp -- these are inserted by earlier passes and their
    // layouts should be preserved, not re-decided by layout propagation.
    if (isa<ToLayoutOp>(op)) {
      return;
    }
    if (!legalConfigs.count(op)) {
      return;
    }
    // Skip ops with no tensor operands (e.g., ttnn.full, ttnn.constant).
    // These are constant-like ops that should be const-evaled, not optimized.
    if (llvm::none_of(op->getOperands(), [](Value v) {
          return mlir::isa<RankedTensorType>(v.getType());
        })) {
      return;
    }
    // Skip ops whose operands all derive from constant/parameter arguments.
    // These ops (e.g., BFP8 typecast on weights) will be re-hoisted into
    // const_eval functions. Promoting their output to L1 would cause the
    // const_eval to return L1 tensors that starve other ops of L1 budget.
    bool allFromConstEval = op->getNumOperands() > 0 &&
                            llvm::all_of(op->getOperands(), [](Value operand) {
                              return ttcore::valueTracesToConstantArgs(operand);
                            });
    if (allFromConstEval) {
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "[op {0}] Skipping {1} @{2}: all operands from const_eval",
                   opIndex, op->getName(), op->getLoc());
      return;
    }
    // Ops feeding func.return are processed normally so their input
    // layouts get optimized. A to_memory_config to DRAM is inserted
    // before func.return in applyToIR.
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "[op {0}] Processing {1} @{2}, legalConfigs={3}", opIndex,
                 op->getName(), op->getLoc(),
                 legalConfigs.find(op)->second.size());

    beamState[op] = processOp(op);

    if (!beamState[op].empty()) {
      TTMLIR_DEBUG(
          ttmlir::LogComponent::GreedyOptimizer,
          "[op {0}] -> chosen: bufType={1}, memLayout={2}, "
          "coreCount={3}, isSharded={4}, isL1={5}, reshard={6} "
          "outputLayout={7}",
          opIndex, beamState[op][0].configHint.outputLayout.getBufferType(),
          beamState[op][0].configHint.outputLayout.getMemLayout(),
          beamState[op][0].score.coreCount, beamState[op][0].score.isSharded,
          beamState[op][0].score.isL1, beamState[op][0].score.requiresReshard,
          beamState[op][0].configHint.outputLayout);
    }
    ++opIndex;
  });

  observer_recording::recordInplaceOps(func, observer.get(), beamState,
                                       opIndex);

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "MemoryLayoutPropagation: processed {0} ops with beamWidth={1}",
               opIndex, beamWidth);

  // Backward pass: consolidate beam at fork points (only for K > 1).
  if (beamWidth > 1) {
    consolidateBeam();
  }

  observer_recording::recordFinalChoices(func, observer.get(), beamState,
                                         finalChoice);
  observer_recording::recordEdges(func, observer.get(), beamState, finalChoice);

  observer->onEnd(opIndex);

  // Apply resolved configs to IR.
  applyToIR();
}

llvm::SmallVector<BeamCandidate, 0>
MemoryLayoutPropagation::processOp(Operation *op) {
  // Step 1: Build input candidate sets (one set per operand).
  std::vector<std::vector<InputCandidate>> inputCandidateSets =
      getInputCandidateSets(op);

  // Step 2: Get output hints.
  auto it = legalConfigs.find(op);
  assert(it != legalConfigs.end());
  const std::vector<OpConfig> &configs = it->second;
  OutputHints outputHints = getOutputHints(op, configs);

  // Compute cross-product size for observer.
  size_t crossProduct =
      outputHints.hints.size() + outputHints.fallbackHints.size();
  for (const auto &set : inputCandidateSets) {
    crossProduct *= set.size();
  }

  observer->onOpSetup(op, inputCandidateSets, outputHints, crossProduct);

  // Log search space dimensions.
  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  processOp {0}: inputSets={1}, outputHints={2}, "
               "fallbackHints={3}",
               op->getName(), inputCandidateSets.size(),
               outputHints.hints.size(), outputHints.fallbackHints.size());

  // Log output hints detail.
  for (size_t hi = 0; hi < outputHints.hints.size(); ++hi) {
    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer, "    hint[{0}]: {1}",
                 hi, outputHints.hints[hi].toCompactString());
  }

  // Step 3: Cross-product evaluation.
  llvm::SmallVector<BeamCandidate, 0> candidates;

  // Helper: evaluate a hint and collect the candidate if valid.
  // Returns true if the result is sharded.
  auto tryHint =
      [&](const OpConfig &hint, size_t hintIdx,
          const std::vector<TTNNLayoutAttr> &inputLayouts, bool anyReshard,
          const llvm::SmallVector<size_t> &producerCandidateIndices,
          const llvm::DenseMap<size_t, TTNNLayoutAttr> &reshardLayouts)
      -> bool {
    auto result = evaluateHint(op, hint, hintIdx, inputLayouts, anyReshard,
                               producerCandidateIndices, reshardLayouts);
    if (result) {
      bool sharded = isShardedLayout(result->configHint.outputLayout);
      candidates.push_back(std::move(*result));
      return sharded;
    }
    return false;
  };

  size_t numOperandSets = inputCandidateSets.size();
  assert(numOperandSets > 0 &&
         "Ops with no tensor operands should be skipped before processOp");

  // Iterate cross-product of input candidates using index vector.
  llvm::SmallVector<size_t> indices(numOperandSets, 0);
  bool done = false;

  const auto &ruleBook = getRuleBook(op);

  // Advance the index vector (odometer-style).
  auto advanceIndices = [&]() {
    for (int i = static_cast<int>(numOperandSets) - 1; i >= 0; --i) {
      ++indices[i];
      if (indices[i] < inputCandidateSets[i].size()) {
        return;
      }
      indices[i] = 0;
      if (i == 0) {
        done = true;
      }
    }
  };

  while (!done) {
    // Build the current input combination.
    std::vector<TTNNLayoutAttr> inputLayouts;
    llvm::SmallVector<size_t> producerCandidateIndices;
    llvm::DenseMap<size_t, TTNNLayoutAttr> reshardLayouts;
    bool anyReshard = false;

    inputLayouts.reserve(numOperandSets);
    producerCandidateIndices.reserve(numOperandSets);

    for (size_t i = 0; i < numOperandSets; ++i) {
      const InputCandidate &ic = inputCandidateSets[i][indices[i]];
      inputLayouts.push_back(ic.layout);
      producerCandidateIndices.push_back(ic.producerCandidateIndex);
      if (ic.isReshard) {
        anyReshard = true;
        reshardLayouts[i] = ic.layout;
      }
    }

    // Op-specific pruning: skip invalid input combinations early.
    if (!ruleBook.isValidInputCombination(inputLayouts)) {
      advanceIndices();
      continue;
    }

    // Try primary output hints with this input combination.
    bool gotSharded = false;
    for (size_t hi = 0; hi < outputHints.hints.size(); ++hi) {
      if (!ruleBook.isValidOutputHintForInputs(outputHints.hints[hi],
                                               inputLayouts)) {
        continue;
      }
      if (tryHint(outputHints.hints[hi], hi, inputLayouts, anyReshard,
                  producerCandidateIndices, reshardLayouts)) {
        gotSharded = true;
      }
    }

    // Try fallback hints only if primary didn't produce a sharded result.
    if (!gotSharded && !outputHints.fallbackHints.empty()) {
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "    Primary hints non-sharded for {0}, trying {1} fallback "
                   "hints",
                   op->getName(), outputHints.fallbackHints.size());
      for (size_t fi = 0; fi < outputHints.fallbackHints.size(); ++fi) {
        if (!ruleBook.isValidOutputHintForInputs(outputHints.fallbackHints[fi],
                                                 inputLayouts)) {
          continue;
        }
        tryHint(outputHints.fallbackHints[fi], outputHints.hints.size() + fi,
                inputLayouts, anyReshard, producerCandidateIndices,
                reshardLayouts);
      }
    }

    advanceIndices();
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  processOp {0}: {1} valid candidates from cross-product",
               op->getName(), candidates.size());

  // Step 4: Sort by score descending, keep top-K.
  std::sort(candidates.begin(), candidates.end(),
            [op](const BeamCandidate &a, const BeamCandidate &b) {
              if (a.score != b.score) {
                return a.score > b.score;
              }
              return preferCandidate(op, a, b);
            });

  if (candidates.size() > beamWidth) {
    candidates.resize(beamWidth);
  }

  // Log all kept beam candidates for this op.
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    [[maybe_unused]] const auto &c = candidates[ci];
    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "  BEAM[{0}] {1}: outBuf={2} outMem={3} "
                 "score(L1={4},sharded={5},dramIn={6},reshard={7},"
                 "cores={8},l1use={9}) inputs=[{10}]",
                 ci, op->getName(), c.configHint.outputLayout.getBufferType(),
                 c.configHint.outputLayout.getMemLayout(), c.score.isL1,
                 c.score.isSharded, c.score.inputDramBytes,
                 c.score.requiresReshard, c.score.coreCount,
                 c.score.outputL1Usage, formatInputLayouts(c.inputLayouts));
  }

  bool usedDramFallback = false;

  // Fallback: if no valid candidate found, use DRAM interleaved.
  if (candidates.empty()) {
    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "No valid candidate for op {0} @{1}, falling back to DRAM "
                 "interleaved.",
                 op->getName(), op->getLoc());

    TTNNLayoutAttr dramLayout = getDRAMInterleavedFallback(op);
    if (dramLayout) {
      BeamCandidate fallback;
      fallback.configHint = OpConfig(dramLayout);
      fallback.score = LayoutScore(); // Lowest possible score.
      fallback.outputLayouts.assign(op->getNumResults(), dramLayout);
      candidates.push_back(std::move(fallback));
    }

    usedDramFallback = true;
  }

  observer->onBeamResult(op, candidates, usedDramFallback);

  return candidates;
}

bool MemoryLayoutPropagation::validateReshard(
    Operation *consumerOp, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr producerOutputLayout, TTNNLayoutAttr reshardLayout) {
  if (producerOutputLayout.isTiled() != reshardLayout.isTiled()) {
    // Reject reshards that change tiling.
    return false;
  }

  MemoryConfigAttr memConfig = MemoryConfigAttr::get(reshardLayout, deviceGrid);

  auto result = op_constraint_validation::validateOperation<ToMemoryConfigOp>(
      consumerOp, /*additionalL1Usage=*/0, deviceGrid, inputShape,
      producerOutputLayout, memConfig, reshardLayout);

  bool valid = result.isSuccess();

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  validateReshard: {0} -> {1}: {2}", producerOutputLayout,
               reshardLayout, valid ? "OK" : "FAILED");
  return valid;
}

void MemoryLayoutPropagation::addL1InterleavedFallbacks(
    std::vector<InputCandidate> &candidates, Operation *op,
    const llvm::SmallVector<BeamCandidate, 0> *producerBeam,
    Operation *producerOp, TTNNLayoutAttr currentLayout, size_t resultIdx,
    size_t maxCandidates) {
  bool hasL1Sharded = false;
  bool hasL1Interleaved = false;
  for (const auto &candidate : candidates) {
    if (candidate.layout.hasL1BufferType()) {
      auto ml = candidate.layout.getMemLayout();
      if (ml && isShardedMemoryLayout(ml.getValue())) {
        hasL1Sharded = true;
      } else {
        hasL1Interleaved = true;
      }
    }
  }

  if (!hasL1Sharded || hasL1Interleaved) {
    return;
  }

  TTNNLayoutAttr l1Interleaved =
      currentLayout.withBufferType(BufferType::L1)
          .withMemoryLayout(TensorMemoryLayout::Interleaved);
  // Add one L1-interleaved candidate per L1-sharded producer beam index.
  for (size_t pIdx = 0; pIdx < producerBeam->size(); ++pIdx) {
    if (candidates.size() >= maxCandidates) {
      break;
    }
    TTNNLayoutAttr prodOut =
        getOutputLayoutForResult((*producerBeam)[pIdx], resultIdx);
    if (!prodOut || !prodOut.hasL1BufferType()) {
      continue;
    }
    auto ml = prodOut.getMemLayout();
    if (!ml || !isShardedMemoryLayout(ml.getValue())) {
      continue;
    }
    auto inputShape =
        mlir::cast<RankedTensorType>(producerOp->getResult(resultIdx).getType())
            .getShape();
    if (!validateReshard(op, inputShape, prodOut, l1Interleaved)) {
      continue;
    }
    InputCandidate ic;
    ic.layout = l1Interleaved;
    ic.producerCandidateIndex = pIdx;
    ic.isReshard = true;
    candidates.push_back(ic);
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  operand: added L1-interleaved reshard candidates for {0}",
               op->getName());
}

void MemoryLayoutPropagation::applyInputLayoutFilter(
    std::vector<InputCandidate> &candidates, Operation *op, unsigned operandIdx,
    TTNNLayoutAttr currentLayout) {
  // Per-op, per-operand input layout filtering: remove candidates that the op
  // cannot consume efficiently (e.g. any sharded RHS for matmul, all sharded
  // for concatenate_heads).
  if (auto inputFilter = getInputLayoutFilter(op, operandIdx)) {
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](const InputCandidate &ic) {
                                      return !inputFilter(ic.layout);
                                    }),
                     candidates.end());
    // Guarantee at least one interleaved candidate remains.
    if (candidates.empty()) {
      auto tensorType =
          mlir::cast<RankedTensorType>(op->getOperand(operandIdx).getType());
      InputCandidate ic;
      ic.layout = currentLayout.withBufferType(BufferType::DRAM)
                      .withMemoryLayout(TensorMemoryLayout::Interleaved)
                      .withTensorShape(tensorType.getShape());
      ic.producerCandidateIndex = 0;
      ic.isReshard = true;
      candidates.push_back(ic);
    }
  }
}

void MemoryLayoutPropagation::addReshardCandidates(
    std::vector<InputCandidate> &candidates, Operation *op, unsigned operandIdx,
    Value operand, TTNNLayoutAttr currentLayout, RankedTensorType tensorType,
    const llvm::SmallVector<BeamCandidate, 0> *producerBeam,
    Operation *producerOp, size_t resultIdx, size_t maxCandidates) {
  // Skip reshards for operands derived from constant/parameter arguments.
  // These will be re-hoisted into const_eval.
  if (ttcore::valueTracesToConstantArgs(operand)) {
    return;
  }
  if (!shouldExploreReshards(op)) {
    return;
  }

  // Generate reshard candidates from each unique producer candidate layout.
  llvm::SmallVector<TTNNLayoutAttr> layoutsToExplore;
  layoutsToExplore.push_back(currentLayout);
  for (const auto &ic : candidates) {
    bool alreadyInList = false;
    for (const auto &existing : layoutsToExplore) {
      if (existing == ic.layout) {
        alreadyInList = true;
        break;
      }
    }
    if (!alreadyInList) {
      layoutsToExplore.push_back(ic.layout);
    }
  }

  // Enable interleaved-to-sharded reshards when no existing candidate that
  // survives the op's input filter offers a sharded layout. Without the
  // filter check, candidates that will be rejected later (e.g., width_sharded
  // inputs for matmul) would suppress interleaved-to-sharded exploration,
  // preventing valid reshard paths like interleaved → height_sharded.
  auto inputFilter = getInputLayoutFilter(op, operandIdx);
  bool hasAnyShardedCandidate = false;
  for (const auto &ic : candidates) {
    auto ml = ic.layout.getMemLayout();
    if (ml && isShardedMemoryLayout(ml.getValue()) &&
        (!inputFilter || inputFilter(ic.layout))) {
      hasAnyShardedCandidate = true;
      break;
    }
  }
  bool exploreInterleavedToSharded = !hasAnyShardedCandidate;

  // Compute max grid volume from output tensor tile count. For tiled outputs,
  // reshard grids larger than the output tile count are wasteful — the op can't
  // produce a sharded output on more cores than it has tiles.
  int64_t maxGridVolume = std::numeric_limits<int64_t>::max();
  auto outputType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  auto outputLayout =
      mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());
  if (outputLayout &&
      mlir::isa<ttcore::TileType>(outputLayout.getElementType())) {
    auto shape = outputType.getShape();
    int64_t cols = (shape.back() + TILE_WIDTH - 1) / TILE_WIDTH;
    int64_t rows =
        (outputType.getNumElements() / shape.back() + TILE_HEIGHT - 1) /
        TILE_HEIGHT;
    maxGridVolume = rows * cols;
  }

  // Collect unique reshard layouts across all base layouts.
  llvm::SmallVector<TTNNLayoutAttr> uniqueReshardLayouts;
  for (const auto &baseLayout : layoutsToExplore) {
    std::vector<TTNNLayoutAttr> reshardCandidates = generateReshardCandidates(
        tensorType, baseLayout, exploreInterleavedToSharded, maxGridVolume);
    for (const auto &reshardLayout : reshardCandidates) {
      // Dedup: skip if already in uniqueReshardLayouts or existing candidates.
      bool alreadyPresent = false;
      for (const auto &existing : uniqueReshardLayouts) {
        if (existing == reshardLayout) {
          alreadyPresent = true;
          break;
        }
      }
      if (!alreadyPresent) {
        for (const auto &existing : candidates) {
          if (existing.layout == reshardLayout) {
            alreadyPresent = true;
            break;
          }
        }
      }
      if (!alreadyPresent) {
        uniqueReshardLayouts.push_back(reshardLayout);
      }
    }
  }

  // Pre-filter reshard layouts that the op cannot consume, before the
  // fan-out. This prevents doomed layouts (e.g., width_sharded for matmul)
  // from consuming maxCandidates budget slots that height_sharded needs.
  if (inputFilter) {
    uniqueReshardLayouts.erase(std::remove_if(uniqueReshardLayouts.begin(),
                                              uniqueReshardLayouts.end(),
                                              [&](TTNNLayoutAttr layout) {
                                                return !inputFilter(layout);
                                              }),
                               uniqueReshardLayouts.end());
  }

  // Fan out: for each unique reshard layout, create one candidate per
  // producer beam index.
  size_t producerBeamSize = producerBeam ? producerBeam->size() : 1;
  for (const auto &reshardLayout : uniqueReshardLayouts) {
    if (candidates.size() >= maxCandidates) {
      break;
    }
    for (size_t pIdx = 0; pIdx < producerBeamSize; ++pIdx) {
      if (candidates.size() >= maxCandidates) {
        break;
      }
      // Validate the reshard is feasible for this producer candidate.
      TTNNLayoutAttr producerOutput;
      if (producerBeam) {
        producerOutput =
            getOutputLayoutForResult((*producerBeam)[pIdx], resultIdx);
      } else {
        // Func args / unresolved producers: use the current IR layout.
        producerOutput = currentLayout;
      }
      if (!producerOutput) {
        continue;
      }
      if (!validateReshard(op, tensorType.getShape(), producerOutput,
                           reshardLayout)) {
        continue;
      }
      InputCandidate ic;
      ic.layout = reshardLayout;
      ic.producerCandidateIndex = pIdx;
      ic.isReshard = true;
      candidates.push_back(ic);
    }
  }
}

std::vector<std::vector<InputCandidate>>
MemoryLayoutPropagation::getInputCandidateSets(Operation *op) {
  std::vector<std::vector<InputCandidate>> result;

  for (auto [operandIdx, operand] : llvm::enumerate(op->getOperands())) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType) {
      continue;
    }
    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    assert(currentLayout &&
           "Ranked tensor operand missing TTNNLayoutAttr encoding");

    std::vector<InputCandidate> candidatesForOperand;

    // Get the producer op's resolved layout from beam state.
    // Cache the lookup -- reused many times below.
    Operation *producerOp = operand.getDefiningOp();
    const llvm::SmallVector<BeamCandidate, 0> *producerBeam = nullptr;
    if (producerOp) {
      auto it = beamState.find(producerOp);
      if (it != beamState.end()) {
        producerBeam = &it->second;
      }
    }

    // For multi-output producers, determine which result this operand uses.
    size_t resultIdx = 0;
    if (auto opResult = mlir::dyn_cast<OpResult>(operand)) {
      resultIdx = opResult.getResultNumber();
    }

    if (producerBeam) {
      for (size_t k = 0; k < producerBeam->size() && k < beamWidth; ++k) {
        InputCandidate ic;
        ic.layout = getOutputLayoutForResult((*producerBeam)[k], resultIdx);
        ic.producerCandidateIndex = k;
        ic.isReshard = false;
        if (ic.layout) {
          candidatesForOperand.push_back(ic);
        }
      }
    }

    // If no producer in beam (func arg or unresolved), use current layout.
    if (candidatesForOperand.empty()) {
      InputCandidate ic;
      ic.layout = currentLayout;
      ic.producerCandidateIndex = 0;
      ic.isReshard = false;
      candidatesForOperand.push_back(ic);
    }

    if (producerBeam) {
      addL1InterleavedFallbacks(candidatesForOperand, op, producerBeam,
                                producerOp, currentLayout, resultIdx,
                                maxInputCandidatesPerOperand);
    }

    addReshardCandidates(candidatesForOperand, op, operandIdx, operand,
                         currentLayout, tensorType, producerBeam, producerOp,
                         resultIdx, maxInputCandidatesPerOperand);

    // Filter after all candidates (producer beam + reshards) are collected.
    // This rejects layouts the op cannot consume (e.g., any sharded RHS for
    // matmul) while preserving producer beam entries as reshard sources
    // during addReshardCandidates — a sharded producer can still serve as
    // the source for a valid reshard to an accepted layout.
    applyInputLayoutFilter(candidatesForOperand, op, operandIdx, currentLayout);

    // Cap per-operand candidate count to prevent cross-product explosion.
    // Non-reshard candidates (from producer beam) come first and are preserved;
    // reshard candidates at the tail get trimmed.
    // Worst case: 2 operands x 64 = 4096 cross-product combos, scored and
    // trimmed to beam K=8.
    if (candidatesForOperand.size() > maxInputCandidatesPerOperand) {
      candidatesForOperand.resize(maxInputCandidatesPerOperand);
    }

    TTMLIR_TRACE(
        ttmlir::LogComponent::GreedyOptimizer,
        "  operand {0}: {1} candidates (fromProducer={2}, "
        "reshards={3})",
        result.size(), candidatesForOperand.size(),
        producerBeam ? producerBeam->size() : 0,
        candidatesForOperand.size() -
            (producerBeam ? std::min(producerBeam->size(), beamWidth) : 1));

    result.push_back(std::move(candidatesForOperand));
  }

  return result;
}

std::vector<TTNNLayoutAttr> MemoryLayoutPropagation::generateReshardCandidates(
    RankedTensorType tensorType, TTNNLayoutAttr currentLayout,
    bool exploreInterleavedToSharded, int64_t maxGridVolume) {
  // Generate reshard candidates targeting sharded layouts.
  // getShardedLayoutsForTensorTypeAndScalarType only returns sharded layouts,
  // so the output is always sharded regardless of the source layout.
  if (!tensorTypePossibleLayouts) {
    return {};
  }

  // For sharded source layouts: generate sharded-to-sharded reshards (always).
  // For interleaved source layouts: generate interleaved-to-sharded reshards
  // only when exploreInterleavedToSharded is true (caller decides).
  if (!exploreInterleavedToSharded &&
      (!currentLayout.getMemLayout() ||
       !isShardedMemoryLayout(currentLayout.getMemLayout().getValue()))) {
    return {};
  }

  Type scalarElementType = currentLayout.getScalarElementType();

  // Look up all sharded layouts for this tensor type and scalar type.
  auto tensorTypeIter = tensorTypePossibleLayouts->find(tensorType);
  if (tensorTypeIter == tensorTypePossibleLayouts->end()) {
    return {};
  }
  auto scalarIter = tensorTypeIter->second.find(scalarElementType);
  if (scalarIter == tensorTypeIter->second.end()) {
    return {};
  }

  std::vector<TTNNLayoutAttr> allSharded =
      getShardedLayoutsForTensorTypeAndScalarType(
          *tensorTypePossibleLayouts, tensorType, scalarElementType);

  // Filter out the current layout (no-op reshard) and non-sharded (defensive).
  std::vector<TTNNLayoutAttr> filtered;
  for (const auto &layout : allSharded) {
    if (layout == currentLayout) {
      continue;
    }
    if (!layout.getMemLayout() ||
        !isShardedMemoryLayout(layout.getMemLayout().getValue())) {
      continue;
    }
    if (static_cast<int64_t>(layout.getGrid().getGridVolume()) >
        maxGridVolume) {
      continue;
    }
    filtered.push_back(layout);
  }

  // Dedup by (memLayout, gridShape): layouts with the same sharding strategy
  // and grid shape differ only in shard dimensions, which the backend
  // recomputes. Keeping one per key avoids redundant validation calls.
  using Key = std::pair<TensorMemoryLayout, llvm::SmallVector<int64_t, 2>>;
  llvm::SmallVector<Key> seenKeys;
  std::vector<TTNNLayoutAttr> deduped;
  for (const auto &layout : filtered) {
    TensorMemoryLayout memLayout = layout.getMemLayout().getValue();
    auto gridShape = layout.getGrid().getShape();
    Key key{memLayout, {gridShape.begin(), gridShape.end()}};
    bool seen = false;
    for (const auto &existing : seenKeys) {
      if (existing == key) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      seenKeys.push_back(key);
      deduped.push_back(layout);
    }
  }

  // Stratified selection: take top-N per sharding type (by grid volume
  // descending), then merge. This ensures each sharding type gets
  // representation — without this, high-volume types (e.g., block_sharded)
  // would crowd out lower-volume types (e.g., height_sharded) when
  // candidates compete for a single global budget.
  llvm::DenseMap<TensorMemoryLayout, std::vector<TTNNLayoutAttr>> buckets;
  for (const auto &layout : deduped) {
    buckets[layout.getMemLayout().getValue()].push_back(layout);
  }
  for (auto &[type, layouts] : buckets) {
    std::sort(layouts.begin(), layouts.end(),
              [](const TTNNLayoutAttr &a, const TTNNLayoutAttr &b) {
                return a.getGrid().getGridVolume() >
                       b.getGrid().getGridVolume();
              });
    if (layouts.size() > maxReshardCandidatesPerType) {
      layouts.resize(maxReshardCandidatesPerType);
    }
  }
  deduped.clear();
  for (const auto &[type, layouts] : buckets) {
    deduped.insert(deduped.end(), layouts.begin(), layouts.end());
  }

  // Final sort for deterministic ordering.
  std::sort(deduped.begin(), deduped.end(),
            [](const TTNNLayoutAttr &a, const TTNNLayoutAttr &b) {
              return a.getGrid().getGridVolume() > b.getGrid().getGridVolume();
            });

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  generated {0} reshard candidates for {1}", deduped.size(),
               currentLayout);
  for ([[maybe_unused]] auto &layout : deduped) {
    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer, "\t\t{}", layout);
  }

  return deduped;
}

Operation *
MemoryLayoutPropagation::getProducerForOperandIdx(Operation *op,
                                                  size_t tensorOperandIdx) {
  size_t tensorIdx = 0;
  for (auto operand : op->getOperands()) {
    if (!mlir::isa<RankedTensorType>(operand.getType())) {
      continue;
    }
    if (tensorIdx == tensorOperandIdx) {
      return operand.getDefiningOp();
    }
    ++tensorIdx;
  }
  return nullptr;
}

void MemoryLayoutPropagation::consolidateBeam() {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "consolidateBeam: starting backward pass for {0} ops in beam",
               beamState.size());

  // Collect ops in IR (topological) order.
  SmallVector<Operation *> opsInOrder;
  func->walk([&](Operation *op) {
    if (beamState.count(op)) {
      opsInOrder.push_back(op);
    }
  });

  // Process in reverse topological order. This guarantees all consumers of an
  // op are resolved (have finalChoice entries) before the op itself is visited.
  for (auto it = opsInOrder.rbegin(); it != opsInOrder.rend(); ++it) {
    Operation *op = *it;
    if (!beamState.count(op) || beamState[op].empty()) {
      continue;
    }

    // Collect consumers of this op that are tracked in beamState.
    SmallVector<Operation *> consumers;
    for (auto result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (beamState.count(user)) {
          consumers.push_back(user);
        }
      }
    }

    if (consumers.empty()) {
      // Sink op (no consumers in beam): use best candidate from forward pass.
      finalChoice[op] = 0;
    } else if (consumers.size() == 1) {
      // Single consumer: follow its back-pointer for this producer.
      Operation *consumer = consumers[0];
      size_t consumerIdx = finalChoice[consumer];
      const BeamCandidate &consumerChosen = beamState[consumer][consumerIdx];

      // Find which tensor operand of consumer connects to this op.
      for (size_t opIdx = 0;
           opIdx < consumerChosen.producerCandidateIndices.size(); ++opIdx) {
        if (getProducerForOperandIdx(consumer, opIdx) == op) {
          size_t prodIdx = consumerChosen.producerCandidateIndices[opIdx];
          finalChoice[op] = prodIdx < beamState[op].size() ? prodIdx : 0;
          break;
        }
      }
      if (!finalChoice.count(op)) {
        finalChoice[op] = 0;
      }
    } else {
      // Fork point: all consumers are guaranteed resolved (reverse topo order).
      finalChoice[op] = resolveForForkPoint(op, consumers);

      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "consolidateBeam: fork at {0} resolved to candidate {1}",
                   op->getName(), finalChoice[op]);
      observer->onForkResolved(op, finalChoice[op], consumers);

      // Patch reshardLayouts for consumers that assumed a different producer
      // candidate. Without this, applyToIR won't insert a ToMemoryConfigOp
      // and the consumer gets a mismatched input layout.
      size_t chosenK = finalChoice[op];
      for (Operation *user : consumers) {
        size_t userChosenIdx = finalChoice[user];
        if (userChosenIdx >= beamState[user].size()) {
          continue;
        }
        BeamCandidate &userChosen = beamState[user][userChosenIdx];

        // Find which tensor operand connects to this fork producer.
        for (size_t opIdx = 0;
             opIdx < userChosen.producerCandidateIndices.size(); ++opIdx) {
          if (getProducerForOperandIdx(user, opIdx) != op) {
            continue;
          }

          size_t assumedK = userChosen.producerCandidateIndices[opIdx];
          if (assumedK == chosenK) {
            break;
          }

          // Consumer assumed a different producer candidate.
          // Record a reshard so applyToIR inserts a ToMemoryConfigOp.
          userChosen.reshardLayouts[opIdx] = userChosen.inputLayouts[opIdx];
          TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                       "consolidateBeam: fork reshard needed for {0} operand "
                       "{1} (assumed producer candidate {2}, chosen {3})",
                       user->getName(), opIdx, assumedK, chosenK);
          break;
        }
      }
    }
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "consolidateBeam: backward pass complete");
}

size_t MemoryLayoutPropagation::resolveForForkPoint(
    Operation *forkOp, llvm::ArrayRef<Operation *> consumers) {
  const auto &forkBeam = beamState[forkOp];
  size_t bestK = 0;
  int bestFreeCount = -1;

  for (size_t k = 0; k < forkBeam.size(); ++k) {
    int freeCount = 0;
    // All consumers are guaranteed resolved (reverse topo order).
    for (Operation *user : consumers) {
      size_t userChosenIdx = finalChoice[user];
      if (userChosenIdx >= beamState[user].size()) {
        continue;
      }
      const BeamCandidate &userChosen = beamState[user][userChosenIdx];
      // Check if this consumer's chosen candidate used producer candidate k.
      for (size_t opIdx = 0; opIdx < userChosen.producerCandidateIndices.size();
           ++opIdx) {
        if (getProducerForOperandIdx(user, opIdx) == forkOp) {
          if (userChosen.producerCandidateIndices[opIdx] == k) {
            ++freeCount;
          }
          break;
        }
      }
    }
    if (freeCount > bestFreeCount ||
        (freeCount == bestFreeCount &&
         forkBeam[k].score > forkBeam[bestK].score)) {
      bestFreeCount = freeCount;
      bestK = k;
    }
  }
  return bestK;
}

TTNNLayoutAttr
MemoryLayoutPropagation::getDRAMInterleavedFallback(Operation *op) {
  if (op->getNumResults() == 0) {
    return nullptr;
  }
  auto tensorType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!tensorType) {
    return nullptr;
  }
  auto currentLayout =
      mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
  if (!currentLayout) {
    return nullptr;
  }

  // If the op already has an L1 layout it was pinned by an earlier pass
  // (workaround or lowering) that knows what the backend kernel requires.
  if (currentLayout.hasL1BufferType()) {
    return currentLayout;
  }

  return currentLayout.withBufferType(BufferType::DRAM)
      .withMemoryLayout(TensorMemoryLayout::Interleaved)
      .withTensorShape(tensorType.getShape());
}

//===----------------------------------------------------------------------===//
// IR Transformation
//===----------------------------------------------------------------------===//

void MemoryLayoutPropagation::insertReturnDramSpills() {
  func->walk([&](func::ReturnOp returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      Value operand = returnOp.getOperand(i);
      auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType) {
        continue;
      }
      auto layout =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
      if (!layout || !layout.hasL1BufferType()) {
        continue;
      }

      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp) {
        continue;
      }

      TTNNLayoutAttr dramLayout =
          layout.withBufferType(BufferType::DRAM)
              .withMemoryLayout(TensorMemoryLayout::Interleaved)
              .withTensorShape(tensorType.getShape());
      insertReshardOp(returnOp, i, dramLayout);

      // insertReshardOp places the new op right before returnOp. Move it to
      // right after the defining op so it doesn't end up after non-hoistable
      // ops (e.g. mesh_shard) which would break trace hoisting.
      Operation *spillOp = returnOp.getOperand(i).getDefiningOp();
      if (spillOp && spillOp != definingOp) {
        spillOp->moveAfter(definingOp);
      }

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "Inserted to_memory_config to DRAM for func.return "
                   "operand {0}",
                   i);
    }
  });
}

void MemoryLayoutPropagation::applyToIR() {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "applyToIR: applying configs for {0} ops in beam state",
               beamState.size());
  // First pass: apply op configs.
  func->walk([&](Operation *op) {
    const BeamCandidate *chosen = getChosenCandidate(op);
    if (!chosen) {
      return;
    }
    applyOpConfig(op, *chosen);
  });

  // Second pass: insert reshard ops.
  func->walk([&](Operation *op) {
    const BeamCandidate *chosen = getChosenCandidate(op);
    if (!chosen) {
      return;
    }
    for (const auto &[operandIdx, reshardLayout] : chosen->reshardLayouts) {
      insertReshardOp(op, operandIdx, reshardLayout);
    }
  });

  fixupConvDeallocate(func);
  insertReturnDramSpills();

  // Third pass: update function return types.
  updateFunctionReturnTypes();
}

void MemoryLayoutPropagation::applyOpConfig(Operation *op,
                                            const BeamCandidate &candidate) {
  TTNNLayoutAttr chosenLayout = getOutputLayoutForResult(candidate, 0);
  if (!chosenLayout) {
    return;
  }

  // D2MSubgraphOp: apply chosen layout to result(s), output buffer(s),
  // and D2M subgraph function body.
  if (auto dispatchOp = dyn_cast<D2MSubgraphOp>(op)) {
    d2m_optimizer_utils::applyChosenLayoutToD2MSubgraphOp(
        dispatchOp, chosenLayout, deviceGrid);

    // Attach L1 usage annotation for spill management.
    if (chosenLayout.hasL1BufferType() &&
        candidate.validationResult.isSuccess() &&
        candidate.validationResult.outputL1Usage > 0) {
      OpBuilder builder(op->getContext());
      op->setAttr(
          "ttnn.output_l1_usage",
          builder.getI64IntegerAttr(candidate.validationResult.outputL1Usage));
    }
    return;
  }

  // Update all tensor results. For single-output ops this iterates once.
  // For multi-output ops (e.g. SplitQueryKeyValueAndSplitHeads), each result
  // gets its own layout from outputLayouts.
  for (auto result : op->getResults()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (!tensorType) {
      continue;
    }
    TTNNLayoutAttr resultLayout =
        getOutputLayoutForResult(candidate, result.getResultNumber());
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
    result.setType(newTensorType);
  }

  // Op-level attribute updates use result 0's layout (chosenLayout).

  // Update layout attribute for ops that have layout interface.
  if (auto opWithLayoutIF = mlir::dyn_cast<TTNNLayoutOpInterface>(op)) {
    opWithLayoutIF.setLayoutAttr(
        LayoutAttr::get(op->getContext(), chosenLayout.getLayout()));
  }

  // Update output data type attribute.
  if (auto dtypeOp = mlir::dyn_cast<TTNNDtypeOpInterface>(op)) {
    ttcore::DataTypeAttr newDataTypeAttr =
        ttcore::DataTypeAttr::get(op->getContext(), chosenLayout.getDataType());
    dtypeOp.setDtypeAttr(newDataTypeAttr);
  }

  // Handle existing ToLayoutOp memory config alignment.
  if (isa<ttnn::ToLayoutOp>(op)) {
    ttnn::ToLayoutOp toLayoutOp = llvm::cast<ttnn::ToLayoutOp>(op);
    toLayoutOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
        op->getContext(), chosenLayout.getMemLayout(),
        ttnn::BufferTypeAttr::get(op->getContext(),
                                  chosenLayout.getBufferType()),
        utils::createShardSpecIfNeeded(chosenLayout, deviceGrid)));
  }

  applyOpSpecificAttrs(op, candidate);

  // Attach L1 usage annotation for spill management.
  if (chosenLayout.hasL1BufferType() &&
      candidate.validationResult.isSuccess() &&
      candidate.validationResult.outputL1Usage > 0) {
    OpBuilder builder(op->getContext());
    op->setAttr(
        "ttnn.output_l1_usage",
        builder.getI64IntegerAttr(candidate.validationResult.outputL1Usage));
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "L1 annotation: {0} @{1} -> outputL1Usage={2} bytes, "
                 "bufType={3}, memLayout={4}",
                 op->getName(), op->getLoc(),
                 candidate.validationResult.outputL1Usage,
                 chosenLayout.getBufferType(), chosenLayout.getMemLayout());
  }
}

void MemoryLayoutPropagation::insertReshardOp(Operation *consumerOp,
                                              size_t operandIndex,
                                              TTNNLayoutAttr reshardLayout) {
  Value operand = consumerOp->getOperand(operandIndex);
  auto producerTensorType = mlir::cast<RankedTensorType>(operand.getType());

  // Skip if the memory config transition would be a no-op.
  // Check buffer type, memory layout, and (for sharded layouts) grid.
  if (auto producerLayout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          producerTensorType.getEncoding())) {
    bool sameBufferType =
        producerLayout.getBufferType() == reshardLayout.getBufferType();
    bool sameMemLayout =
        producerLayout.getMemLayout() == reshardLayout.getMemLayout();

    if (sameBufferType && sameMemLayout) {
      // For sharded layouts, also require matching grids.
      bool bothSharded =
          isShardedMemoryLayout(producerLayout.getMemLayout().getValue()) &&
          isShardedMemoryLayout(reshardLayout.getMemLayout().getValue());
      if (!bothSharded || producerLayout.getGrid() == reshardLayout.getGrid()) {
        return;
      }
    }
  }

  // Build the output layout by taking the producer's current layout and
  // applying the target buffer type, memory layout, and grid.
  TTNNLayoutAttr producerLayout =
      utils::getLayoutAttrFromTensor(producerTensorType);
  TTNNLayoutAttr outputLayout =
      producerLayout.withBufferType(reshardLayout.getBufferType())
          .withMemoryLayout(reshardLayout.getMemLayout())
          .withGrid(producerTensorType.getShape(), reshardLayout.getGrid())
          .withShardShape(reshardLayout.getScalarShardShape());
  RankedTensorType newTensorType =
      utils::RankedTensorTypeFactory::create(producerTensorType, outputLayout);

  MemoryConfigAttr outputMemConfigAttr = MemoryConfigAttr::get(
      consumerOp->getContext(), reshardLayout.getMemLayout(),
      BufferTypeAttr::get(consumerOp->getContext(),
                          reshardLayout.getBufferType()),
      utils::createShardSpecIfNeeded(reshardLayout, deviceGrid));

  OpBuilder builder(consumerOp);
  Location loc = ttmlir::utils::appendLocationSuffix(consumerOp->getLoc(),
                                                     "_mem_reconfig");

  ToMemoryConfigOp memoryReconfigOp = builder.create<ToMemoryConfigOp>(
      loc, newTensorType, operand, outputMemConfigAttr);

  consumerOp->setOperand(operandIndex, memoryReconfigOp->getResult(0));

  // Annotate L1 output usage so L1SpillManagement can track this op.
  if (outputLayout.hasL1BufferType()) {
    uint64_t l1Usage =
        utils::getPerCoreL1Usage(outputLayout, deviceGrid.getGridVolume());
    OpBuilder attrBuilder(memoryReconfigOp->getContext());
    memoryReconfigOp->setAttr("ttnn.output_l1_usage",
                              attrBuilder.getI64IntegerAttr(l1Usage));
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "Inserted memory reconfig op: {0}", memoryReconfigOp);
}

void MemoryLayoutPropagation::updateFunctionReturnTypes() {
  SmallVector<Type> funcResultTypes;

  func->walk([&](Operation *op) {
    if (op->getNumResults() == 0) {
      if (auto funcReturn = dyn_cast<func::ReturnOp>(op)) {
        funcResultTypes.append(funcReturn.getOperandTypes().begin(),
                               funcReturn.getOperandTypes().end());
      }
    }
  });

  FunctionType funcType = func.getFunctionType();
  FunctionType newFuncType = FunctionType::get(
      func.getContext(), funcType.getInputs(), funcResultTypes);
  func.setType(newFuncType);
}

} // namespace mlir::tt::ttnn
