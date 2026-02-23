// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagation.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>

namespace mlir::tt::ttnn {

LayoutPropagation::LayoutPropagation(
    func::FuncOp func, ttcore::GridAttr deviceGrid,
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
    const TensorTypeLayoutsMap *tensorTypePossibleLayouts, size_t beamWidth)
    : func(func), deviceGrid(deviceGrid), legalConfigs(legalConfigs),
      tensorTypePossibleLayouts(tensorTypePossibleLayouts),
      beamWidth(beamWidth) {}

/// Returns true for ops that cannot consume sharded L1 inputs.
/// These ops either hang or produce incorrect results with sharded input.
/// See https://github.com/tenstorrent/tt-mlir/issues/7145
static bool opForbidsShardedInputs(Operation *op) {
  return isa<ConcatenateHeadsOp>(op);
}

void LayoutPropagation::run() {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "LayoutPropagation::run() starting for func {0}",
               func.getName());

  size_t opIndex = 0;
  // Forward pass: propagate layouts in topological (IR) order.
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
    // Skip ops whose operands all derive from constant/parameter arguments.
    // These ops (e.g., BFP8 typecast on weights) will be re-hoisted into
    // const_eval functions. Promoting their output to L1 would cause the
    // const_eval to return L1 tensors that starve other ops of L1 budget.
    bool allFromConstEval =
        op->getNumOperands() > 0 &&
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
                 "[op {0}] Processing {1} @{2}, legalConfigs={3}",
                 opIndex, op->getName(), op->getLoc(),
                 legalConfigs.find(op)->second.size());

    beamState[op] = processOp(op);

    if (!beamState[op].empty()) {
      const auto &best = beamState[op][0];
      TTMLIR_DEBUG(
          ttmlir::LogComponent::GreedyOptimizer,
          "[op {0}] -> chosen: bufType={1}, memLayout={2}, "
          "coreCount={3}, isSharded={4}, isL1={5}, reshard={6}",
          opIndex, best.config.outputLayout.getBufferType(),
          best.config.outputLayout.getMemLayout(), best.score.coreCount,
          best.score.isSharded, best.score.isL1, best.score.requiresReshard);
    }
    ++opIndex;
  });

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "LayoutPropagation: processed {0} ops with beamWidth={1}",
               opIndex, beamWidth);

  // Backward pass: consolidate beam at fork points (only for K > 1).
  if (beamWidth > 1) {
    consolidateBeam();
  }

  // Apply resolved configs to IR.
  applyToIR();
}

llvm::SmallVector<BeamCandidate, 0>
LayoutPropagation::processOp(Operation *op) {
  // Step 1: Build input candidate sets (one set per operand).
  std::vector<std::vector<InputCandidate>> inputCandidateSets =
      getInputCandidateSets(op);

  // Step 2: Get output hints.
  auto it = legalConfigs.find(op);
  assert(it != legalConfigs.end());
  const std::vector<OpConfig> &configs = it->second;
  OutputHints outputHints = getOutputHints(op, configs);

  // Log search space dimensions.
  size_t crossProductSize = outputHints.hints.size();
  for (const auto &ics : inputCandidateSets) {
    crossProductSize *= ics.size();
  }
  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  processOp {0}: inputSets={1}, outputHints={2}, "
               "fallbackHints={3}, crossProduct={4}",
               op->getName(), inputCandidateSets.size(),
               outputHints.hints.size(), outputHints.fallbackHints.size(),
               crossProductSize);

  // Log output hints detail.
  for (size_t hi = 0; hi < outputHints.hints.size(); ++hi) {
    const auto &h = outputHints.hints[hi];
    if (h.outputLayout) {
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "    hint[{0}]: buf={1} mem={2} attemptL1Shard={3}",
                   hi, h.outputLayout.getBufferType(),
                   h.outputLayout.getMemLayout(),
                   outputHints.attemptL1Sharding);
    } else {
      TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                   "    hint[{0}]: NULL (backend decides) attemptL1Shard={1}",
                   hi, outputHints.attemptL1Sharding);
    }
  }

  // Step 3: Cross-product evaluation.
  llvm::SmallVector<BeamCandidate, 0> candidates;

  // Helper: check if a layout is sharded.
  auto isSharded = [](TTNNLayoutAttr layout) {
    if (!layout) {
      return false;
    }
    auto memLayout = layout.getMemLayout();
    return memLayout && isShardedMemoryLayout(memLayout.getValue());
  };

  // Helper: try a single hint against an input combination, collect candidate.
  // Returns true if the result is sharded.
  auto tryHint = [&](const OpConfig &hint, size_t hintIdx,
                     const std::vector<TTNNLayoutAttr> &inputLayouts,
                     bool anyReshard,
                     const llvm::SmallVector<size_t> &producerCandidateIndices,
                     const llvm::DenseMap<size_t, TTNNLayoutAttr>
                         &reshardLayouts) -> bool {
    auto result = op_constraint_validation::validateOperation(
        op, inputLayouts, hint);
    if (result.isSuccess()) {
      BeamCandidate candidate;
      candidate.config =
          OpConfig(result.actualOutputLayout, hint.opSpecificAttrs);
      candidate.score = scoreCandidate(op, hint, result, anyReshard);
      candidate.validationResult = result;
      candidate.inputLayouts = inputLayouts;
      candidate.producerCandidateIndices = producerCandidateIndices;
      candidate.reshardLayouts = reshardLayouts;
      candidates.push_back(std::move(candidate));

      TTMLIR_TRACE(
          ttmlir::LogComponent::GreedyOptimizer,
          "    VALID candidate for {0}: hint[{1}] outBuf={2} "
          "outMem={3} score(L1={4},sharded={5},"
          "reshard={6},cores={7},l1use={8})",
          op->getName(), hintIdx,
          candidate.config.outputLayout.getBufferType(),
          candidate.config.outputLayout.getMemLayout(),
          candidate.score.isL1, candidate.score.isSharded,
          candidate.score.requiresReshard, candidate.score.coreCount,
          candidate.score.outputL1Usage);

      return isSharded(result.actualOutputLayout);
    }

    // Log validation failures.
    llvm::StringRef hintBuf = "null";
    llvm::StringRef hintMem = "null";
    std::string hintBufStr, hintMemStr;
    if (hint.outputLayout) {
      llvm::raw_string_ostream bufOS(hintBufStr);
      bufOS << hint.outputLayout.getBufferType();
      hintBuf = hintBufStr;
      llvm::raw_string_ostream memOS(hintMemStr);
      memOS << hint.outputLayout.getMemLayout();
      hintMem = hintMemStr;
    }
    std::string inputDesc;
    llvm::raw_string_ostream inputOS(inputDesc);
    for (size_t ii = 0; ii < inputLayouts.size(); ++ii) {
      if (ii > 0) {
        inputOS << ", ";
      }
      inputOS << inputLayouts[ii].getBufferType() << "/"
              << inputLayouts[ii].getMemLayout();
    }
    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "    FAILED validation for {0}: hint[{1}] "
                 "outBuf={2} outMem={3} inputs=[{4}] reshard={5}",
                 op->getName(), hintIdx, hintBuf, hintMem, inputDesc,
                 anyReshard);
    return false;
  };

  size_t numOperandSets = inputCandidateSets.size();

  // If no tensor operands (e.g., constant-like ops), just try output hints.
  if (numOperandSets == 0) {
    std::vector<TTNNLayoutAttr> emptyInputs;
    llvm::SmallVector<size_t> emptyProducerIndices;
    llvm::DenseMap<size_t, TTNNLayoutAttr> emptyReshards;
    bool gotSharded = false;
    for (size_t hi = 0; hi < outputHints.hints.size(); ++hi) {
      if (tryHint(outputHints.hints[hi], hi, emptyInputs, false,
                  emptyProducerIndices, emptyReshards)) {
        gotSharded = true;
      }
    }
    // Try fallback hints if primary didn't produce a sharded result.
    if (!gotSharded) {
      for (size_t fi = 0; fi < outputHints.fallbackHints.size(); ++fi) {
        tryHint(outputHints.fallbackHints[fi],
                outputHints.hints.size() + fi, emptyInputs, false,
                emptyProducerIndices, emptyReshards);
      }
    }
  } else {
    // Iterate cross-product of input candidates using index vector.
    llvm::SmallVector<size_t> indices(numOperandSets, 0);
    bool done = false;

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

      // Try primary output hints with this input combination.
      bool gotSharded = false;
      for (size_t hi = 0; hi < outputHints.hints.size(); ++hi) {
        if (tryHint(outputHints.hints[hi], hi, inputLayouts, anyReshard,
                    producerCandidateIndices, reshardLayouts)) {
          gotSharded = true;
        }
      }

      // Try fallback hints only if primary didn't produce a sharded result.
      if (!gotSharded && !outputHints.fallbackHints.empty()) {
        TTMLIR_TRACE(
            ttmlir::LogComponent::GreedyOptimizer,
            "    Primary hints non-sharded for {0}, trying {1} fallback "
            "hints",
            op->getName(), outputHints.fallbackHints.size());
        for (size_t fi = 0; fi < outputHints.fallbackHints.size(); ++fi) {
          tryHint(outputHints.fallbackHints[fi],
                  outputHints.hints.size() + fi, inputLayouts, anyReshard,
                  producerCandidateIndices, reshardLayouts);
        }
      }

      // Advance the index vector (odometer-style).
      for (int i = static_cast<int>(numOperandSets) - 1; i >= 0; --i) {
        ++indices[i];
        if (indices[i] < inputCandidateSets[i].size()) {
          break;
        }
        indices[i] = 0;
        if (i == 0) {
          done = true;
        }
      }
    }
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "  processOp {0}: {1} valid candidates from cross-product",
               op->getName(), candidates.size());

  // Step 4: Sort by score descending, keep top-K.
  std::sort(candidates.begin(), candidates.end(),
            [](const BeamCandidate &a, const BeamCandidate &b) {
              return a.score > b.score;
            });

  if (candidates.size() > beamWidth) {
    candidates.resize(beamWidth);
  }

  // Log all kept beam candidates for this op.
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    const auto &c = candidates[ci];
    std::string inputDesc;
    llvm::raw_string_ostream inputOS(inputDesc);
    for (size_t ii = 0; ii < c.inputLayouts.size(); ++ii) {
      if (ii > 0) {
        inputOS << ", ";
      }
      inputOS << c.inputLayouts[ii].getBufferType() << "/"
              << c.inputLayouts[ii].getMemLayout();
    }
    TTMLIR_TRACE(
        ttmlir::LogComponent::GreedyOptimizer,
        "  BEAM[{0}] {1}: outBuf={2} outMem={3} "
        "score(L1={4},sharded={5},reshard={6},"
        "cores={7},l1use={8}) inputs=[{9}]",
        ci, op->getName(), c.config.outputLayout.getBufferType(),
        c.config.outputLayout.getMemLayout(), c.score.isL1,
        c.score.isSharded, c.score.requiresReshard, c.score.coreCount,
        c.score.outputL1Usage, inputDesc);
  }

  // Fallback: if no valid candidate found, use DRAM interleaved.
  if (candidates.empty()) {
    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "No valid candidate for op {0} @{1}, falling back to DRAM "
                 "interleaved.",
                 op->getName(), op->getLoc());

    TTNNLayoutAttr dramLayout = getDRAMInterleavedFallback(op);
    if (dramLayout) {
      BeamCandidate fallback;
      fallback.config = OpConfig(dramLayout);
      fallback.score = LayoutScore(); // Lowest possible score.
      candidates.push_back(std::move(fallback));
    }
  }

  return candidates;
}

std::vector<std::vector<LayoutPropagation::InputCandidate>>
LayoutPropagation::getInputCandidateSets(Operation *op) {
  std::vector<std::vector<InputCandidate>> result;

  for (auto operand : op->getOperands()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType) {
      continue;
    }
    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    if (!currentLayout) {
      continue;
    }

    std::vector<InputCandidate> candidatesForOperand;

    // Get the producer op's resolved layout from beam state.
    Operation *producerOp = operand.getDefiningOp();
    if (producerOp && beamState.count(producerOp)) {
      const auto &producerBeam = beamState[producerOp];
      for (size_t k = 0; k < producerBeam.size() && k < beamWidth; ++k) {
        InputCandidate ic;
        ic.layout = producerBeam[k].config.outputLayout;
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

    // Add L1-interleaved fallback when all producer candidates are sharded.
    // Without this, ops that can't consume sharded inputs (reshape, permute,
    // concatenate_heads) fall to DRAM, and ops that *can* (matmul) lose the
    // option of an L1-interleaved input that may score better.
    if (producerOp && beamState.count(producerOp)) {
      bool allSharded = !candidatesForOperand.empty() &&
                        llvm::all_of(candidatesForOperand,
                                     [](const InputCandidate &ic) {
                                       auto ml = ic.layout.getMemLayout();
                                       return ml &&
                                              isShardedMemoryLayout(
                                                  ml.getValue());
                                     });
      if (allSharded) {
        TTNNLayoutAttr l1Interleaved =
            currentLayout.withBufferType(BufferType::L1)
                .withMemoryLayout(TensorMemoryLayout::Interleaved);
        InputCandidate ic;
        ic.layout = l1Interleaved;
        ic.producerCandidateIndex = 0;
        ic.isReshard = true;
        candidatesForOperand.push_back(ic);

        TTMLIR_TRACE(
            ttmlir::LogComponent::GreedyOptimizer,
            "  operand: added L1-interleaved reshard for {0} ({1} "
            "producer candidates)",
            op->getName(), candidatesForOperand.size() - 1);
      }
    }

    // For ops that forbid sharded inputs, remove any sharded candidates
    // that slipped through from the producer beam. Guarantee at least one
    // interleaved candidate remains.
    if (opForbidsShardedInputs(op)) {
      candidatesForOperand.erase(
          std::remove_if(candidatesForOperand.begin(),
                         candidatesForOperand.end(),
                         [](const InputCandidate &ic) {
                           auto memLayout = ic.layout.getMemLayout();
                           return memLayout &&
                                  isShardedMemoryLayout(memLayout.getValue());
                         }),
          candidatesForOperand.end());
      if (candidatesForOperand.empty()) {
        InputCandidate ic;
        ic.layout = currentLayout.withBufferType(BufferType::DRAM)
                        .withMemoryLayout(TensorMemoryLayout::Interleaved);
        ic.producerCandidateIndex = 0;
        ic.isReshard = true;
        candidatesForOperand.push_back(ic);
      }
    }

    // Add reshard candidates if applicable.
    // Skip reshards for operands derived from constant/parameter arguments.
    // These will be re-hoisted into const_eval — L1 reshards would make the
    // const_eval return L1, occupying L1 for the lifetime of the tensor.
    bool isFromConstEvalChain = ttcore::valueTracesToConstantArgs(operand);
    if (shouldExploreReshards(op) && !isFromConstEvalChain) {
      // Generate reshard candidates from each unique producer candidate layout.
      // For beam search, the producer may have sharded layouts that differ from
      // the initial IR layout. We explore sharded-to-sharded reshards for each.
      llvm::SmallVector<TTNNLayoutAttr> layoutsToExplore;
      layoutsToExplore.push_back(currentLayout);
      for (const auto &ic : candidatesForOperand) {
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

      for (const auto &baseLayout : layoutsToExplore) {
        std::vector<TTNNLayoutAttr> reshardCandidates =
            generateReshardCandidates(tensorType, baseLayout);
        for (const auto &reshardLayout : reshardCandidates) {
          // Only add if different from already-present candidates.
          bool alreadyPresent = false;
          for (const auto &existing : candidatesForOperand) {
            if (existing.layout == reshardLayout) {
              alreadyPresent = true;
              break;
            }
          }
          if (!alreadyPresent) {
            InputCandidate ic;
            ic.layout = reshardLayout;
            // Back-point to the first producer candidate (greedy: always 0).
            ic.producerCandidateIndex = 0;
            ic.isReshard = true;
            candidatesForOperand.push_back(ic);
          }
        }
      }
    }

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "  operand {0}: {1} candidates (fromProducer={2}, "
                 "reshards={3})",
                 result.size(), candidatesForOperand.size(),
                 producerOp && beamState.count(producerOp)
                     ? beamState[producerOp].size()
                     : 0,
                 candidatesForOperand.size() -
                     (producerOp && beamState.count(producerOp)
                          ? std::min(beamState[producerOp].size(), beamWidth)
                          : 1));
    result.push_back(std::move(candidatesForOperand));
  }

  return result;
}

std::vector<TTNNLayoutAttr>
LayoutPropagation::generateReshardCandidates(
    RankedTensorType tensorType, TTNNLayoutAttr currentLayout) {
  // Only generate sharded-to-sharded reshard candidates. Resharding from
  // sharded to interleaved (DRAM or L1) almost always hurts performance.
  if (!tensorTypePossibleLayouts) {
    return {};
  }

  // Only explore reshards from sharded source layouts.
  if (!currentLayout.getMemLayout() ||
      !isShardedMemoryLayout(currentLayout.getMemLayout().getValue())) {
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
    filtered.push_back(layout);
  }

  // Deduplicate by (memLayout, gridShape) key -- keep the first occurrence.
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

  // Sort by grid volume (product of grid dims) descending -- more cores first.
  std::sort(deduped.begin(), deduped.end(),
            [](const TTNNLayoutAttr &a, const TTNNLayoutAttr &b) {
              auto aShape = a.getGrid().getShape();
              auto bShape = b.getGrid().getShape();
              int64_t aVol = 1, bVol = 1;
              for (int64_t d : aShape) {
                aVol *= d;
              }
              for (int64_t d : bShape) {
                bVol *= d;
              }
              return aVol > bVol;
            });

  // Keep top 8 candidates.
  constexpr size_t kMaxReshardCandidates = 8;
  if (deduped.size() > kMaxReshardCandidates) {
    deduped.resize(kMaxReshardCandidates);
  }

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "generateReshardCandidates: {0} sharded-to-sharded candidates "
               "for tensor type with shape [{1}]",
               deduped.size(), tensorType.getShape().size());

  return deduped;
}

Operation *LayoutPropagation::getProducerForOperandIdx(
    Operation *op, size_t tensorOperandIdx) {
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

void LayoutPropagation::consolidateBeam() {
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

  // Initialize: all ops default to candidate 0 (best from forward pass).
  for (auto *op : opsInOrder) {
    finalChoice[op] = 0;
  }

  // Track which fork points have been resolved.
  llvm::DenseSet<Operation *> resolvedForks;

  // Process in reverse order (backward pass).
  for (auto it = opsInOrder.rbegin(); it != opsInOrder.rend(); ++it) {
    Operation *op = *it;
    if (!beamState.count(op) || beamState[op].empty()) {
      continue;
    }

    size_t chosenIdx = finalChoice[op];
    if (chosenIdx >= beamState[op].size()) {
      chosenIdx = 0;
    }
    const BeamCandidate &chosen = beamState[op][chosenIdx];

    // Follow back pointers to set producer choices.
    for (size_t i = 0; i < chosen.producerCandidateIndices.size(); ++i) {
      Operation *producer = getProducerForOperandIdx(op, i);
      if (!producer || !beamState.count(producer)) {
        continue;
      }

      // Check if producer is a fork point (multiple consumers in beamState).
      bool isFork = false;
      for (Operation *user : producer->getResult(0).getUsers()) {
        if (user != op && beamState.count(user)) {
          isFork = true;
          break;
        }
      }

      if (isFork) {
        // Fork point: resolve once (first consumer to reach it sets it).
        if (!resolvedForks.contains(producer)) {
          finalChoice[producer] = resolveForForkPoint(producer);
          resolvedForks.insert(producer);
          TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                       "consolidateBeam: fork at {0} resolved to candidate {1}",
                       producer->getName(), finalChoice[producer]);
        }
      } else {
        // Single consumer: follow back pointer directly.
        size_t prodIdx = chosen.producerCandidateIndices[i];
        if (prodIdx < beamState[producer].size()) {
          finalChoice[producer] = prodIdx;
        }
      }
    }
  }

  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "consolidateBeam: resolved {0} fork points",
               resolvedForks.size());
}

size_t LayoutPropagation::resolveForForkPoint(Operation *forkOp) {
  const auto &forkBeam = beamState[forkOp];
  size_t bestK = 0;
  int bestFreeCount = -1;

  for (size_t k = 0; k < forkBeam.size(); ++k) {
    int freeCount = 0;
    for (Operation *user : forkOp->getResult(0).getUsers()) {
      if (!beamState.count(user)) {
        continue;
      }
      size_t userChosenIdx = finalChoice.count(user) ? finalChoice[user] : 0;
      if (userChosenIdx >= beamState[user].size()) {
        continue;
      }
      const BeamCandidate &userChosen = beamState[user][userChosenIdx];
      // Check if this consumer's chosen candidate used producer candidate k.
      for (size_t opIdx = 0;
           opIdx < userChosen.producerCandidateIndices.size(); ++opIdx) {
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

TTNNLayoutAttr LayoutPropagation::getDRAMInterleavedFallback(Operation *op) {
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
  return currentLayout.withBufferType(BufferType::DRAM)
      .withMemoryLayout(TensorMemoryLayout::Interleaved);
}

//===----------------------------------------------------------------------===//
// IR Transformation
//===----------------------------------------------------------------------===//

void LayoutPropagation::applyToIR() {
  TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
               "applyToIR: applying configs for {0} ops in beam state",
               beamState.size());
  // First pass: apply op configs (result types, DPS operands, op-specific
  // attrs, L1 usage annotation).
  func->walk([&](Operation *op) {
    if (!beamState.count(op)) {
      return;
    }

    const auto &beam = beamState[op];
    if (beam.empty()) {
      return;
    }

    size_t chosenIdx = finalChoice.count(op) ? finalChoice[op] : 0;
    if (chosenIdx >= beam.size()) {
      chosenIdx = 0;
    }
    applyOpConfig(op, beam[chosenIdx]);
  });

  // Second pass: insert reshard ops for edges that require memory
  // reconfiguration.
  func->walk([&](Operation *op) {
    if (!beamState.count(op)) {
      return;
    }

    const auto &beam = beamState[op];
    if (beam.empty()) {
      return;
    }

    size_t chosenIdx = finalChoice.count(op) ? finalChoice[op] : 0;
    if (chosenIdx >= beam.size()) {
      chosenIdx = 0;
    }
    const BeamCandidate &chosen = beam[chosenIdx];
    for (const auto &[operandIdx, reshardLayout] : chosen.reshardLayouts) {
      insertReshardOp(op, operandIdx, reshardLayout);
    }
  });

  // Fixup: disable deallocate_activation for conv2d/conv_transpose2d ops
  // whose input has multiple users, preventing use-after-free. This mirrors
  // the tryDisableDeallocateActivation logic in DFShardingPolicy.cpp.
  func->walk([&](Operation *op) {
    auto disableDeallocIfMultiUser = [](auto convOp) {
      auto config = convOp.getConv2dConfigAttr();
      if (!config || !config.getDeallocateActivation() ||
          !config.getDeallocateActivation().getValue()) {
        return;
      }
      Value input = convOp.getInput();
      if (!input.hasOneUse()) {
        convOp.setConv2dConfigAttr(
            config.withDeallocateActivation(false));
        TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                     "Disabled deallocate_activation for conv2d with "
                     "multi-use input: {}",
                     ttmlir::opToString(convOp));
      }
    };

    if (auto conv2d = dyn_cast<ttnn::Conv2dOp>(op)) {
      disableDeallocIfMultiUser(conv2d);
    } else if (auto convT = dyn_cast<ttnn::ConvTranspose2dOp>(op)) {
      disableDeallocIfMultiUser(convT);
    }
  });

  // Insert to_memory_config to DRAM for func.return operands that are in L1.
  // The caller expects DRAM tensors, so any L1 outputs must be spilled.
  func->walk([&](func::ReturnOp returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      Value operand = returnOp.getOperand(i);
      auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType) {
        continue;
      }
      auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          tensorType.getEncoding());
      if (!layout || !layout.hasL1BufferType()) {
        continue;
      }

      // Build DRAM interleaved target layout.
      TTNNLayoutAttr dramLayout =
          layout.withBufferType(BufferType::DRAM)
              .withMemoryLayout(TensorMemoryLayout::Interleaved);
      insertReshardOp(returnOp, i, dramLayout);

      TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                   "Inserted to_memory_config to DRAM for func.return "
                   "operand {0}",
                   i);
    }
  });

  // Third pass: update function return types.
  updateFunctionReturnTypes();
}

void LayoutPropagation::applyOpConfig(Operation *op,
                                       const BeamCandidate &candidate) {
  TTNNLayoutAttr chosenLayout = candidate.config.outputLayout;
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
    ttcore::DataTypeAttr newDataTypeAttr = ttcore::DataTypeAttr::get(
        op->getContext(), chosenLayout.getDataType());
    dtypeOp.setDtypeAttr(newDataTypeAttr);
  }

  // Update DPS operand (EmptyOp).
  if (isa<mlir::DestinationStyleOpInterface>(op)) {
    BufferType bufferType = chosenLayout.getBufferType();
    TensorMemoryLayoutAttr tensorMemoryLayoutAttr =
        chosenLayout.getMemLayout();

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
  // Handle existing ToLayoutOp memory config alignment.
  else if (isa<ttnn::ToLayoutOp>(op)) {
    ttnn::ToLayoutOp toLayoutOp = llvm::cast<ttnn::ToLayoutOp>(op);
    toLayoutOp.setMemoryConfigAttr(ttnn::MemoryConfigAttr::get(
        op->getContext(), chosenLayout.getMemLayout(),
        ttnn::BufferTypeAttr::get(op->getContext(),
                                  chosenLayout.getBufferType()),
        utils::createShardSpecIfNeeded(chosenLayout, deviceGrid)));
  }

  // Set op-specific configurations (Conv2d, Matmul).
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<ttnn::Conv2dOp>([&](ttnn::Conv2dOp convOp) {
        if (std::holds_alternative<ttnn::Conv2dAttrs>(
                candidate.config.opSpecificAttrs)) {
          ttnn::Conv2dAttrs conv2dAttrs =
              std::get<ttnn::Conv2dAttrs>(candidate.config.opSpecificAttrs);
          if (conv2dAttrs.conv2dConfig.has_value()) {
            convOp.setConv2dConfigAttr(conv2dAttrs.conv2dConfig.value());
          }
          if (conv2dAttrs.deviceComputeKernelConfig.has_value()) {
            convOp.setComputeConfigAttr(
                conv2dAttrs.deviceComputeKernelConfig.value());
          }
        }
      })
      .Case<ttnn::ConvTranspose2dOp>(
          [&](ttnn::ConvTranspose2dOp convOp) {
            if (std::holds_alternative<ttnn::Conv2dAttrs>(
                    candidate.config.opSpecificAttrs)) {
              ttnn::Conv2dAttrs conv2dAttrs =
                  std::get<ttnn::Conv2dAttrs>(
                      candidate.config.opSpecificAttrs);
              if (conv2dAttrs.conv2dConfig.has_value()) {
                convOp.setConv2dConfigAttr(
                    conv2dAttrs.conv2dConfig.value());
              }
            }
          })
      .Case<ttnn::MatmulOp, ttnn::LinearOp>([&](auto matmulOp) {
        if (std::holds_alternative<ttnn::MatmulAttrs>(
                candidate.config.opSpecificAttrs)) {
          ttnn::MatmulAttrs matmulAttrs = std::get<ttnn::MatmulAttrs>(
              candidate.config.opSpecificAttrs);
          if (matmulAttrs.matmulProgramConfig.has_value()) {
            auto programConfig = matmulAttrs.matmulProgramConfig.value();
            matmulOp.setMatmulProgramConfigAttr(programConfig);
            // Workaround for tt-metal issue #35060.
            bool hasFusedActivation =
                llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
                    .template Case<
                        MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                        MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
                        [](auto config) {
                          return config.getFusedActivation() != nullptr;
                        })
                    .Default([](mlir::Attribute) { return false; });
            if (hasFusedActivation) {
              matmulOp.removeActivationAttr();
            }
          }
        }
      })
      .Default([](Operation *) {});

  // Attach L1 usage annotation for Pass 2 (spill management).
  if (chosenLayout.hasL1BufferType() &&
      candidate.validationResult.isSuccess() &&
      candidate.validationResult.outputL1Usage > 0) {
    OpBuilder builder(op->getContext());
    op->setAttr("ttnn.output_l1_usage",
                builder.getI64IntegerAttr(
                    candidate.validationResult.outputL1Usage));
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "L1 annotation: {0} @{1} -> outputL1Usage={2} bytes, "
                 "bufType={3}, memLayout={4}",
                 op->getName(), op->getLoc(),
                 candidate.validationResult.outputL1Usage,
                 chosenLayout.getBufferType(),
                 chosenLayout.getMemLayout());
  }
}

void LayoutPropagation::insertReshardOp(Operation *consumerOp,
                                         size_t operandIndex,
                                         TTNNLayoutAttr reshardLayout) {
  Value operand = consumerOp->getOperand(operandIndex);
  auto producerTensorType =
      mlir::cast<RankedTensorType>(operand.getType());

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

  TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
               "Inserted memory reconfig op: {0}", memoryReconfigOp);
}

void LayoutPropagation::updateFunctionReturnTypes() {
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
