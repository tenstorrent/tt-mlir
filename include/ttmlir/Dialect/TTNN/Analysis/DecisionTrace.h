// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_DECISIONTRACE_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_DECISIONTRACE_H

#include "ttmlir/Dialect/TTNN/Analysis/L1SpillObserver.h"
#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagationObserver.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::tt::ttnn {

/// Convert a TTNNLayoutAttr to a human-readable string like
/// "L1/height_sharded/8x4" or "DRAM/interleaved". Returns "null" for null
/// layouts.
std::string layoutToString(TTNNLayoutAttr layout);

//===----------------------------------------------------------------------===//
// Trace record structs
//===----------------------------------------------------------------------===//

/// One (input combo x output hint) evaluation attempt.
struct EvaluationRecord {
  std::string hint;
  std::vector<std::string> inputs;
  bool valid = false;
  std::string failureReason;

  // Score breakdown (only populated when valid).
  bool isL1 = false;
  bool isSharded = false;
  uint64_t inputDramBytes = 0;
  bool requiresReshard = false;
  int64_t coreCount = 0;
  uint64_t outputL1Usage = 0;

  // Actual output layout from backend (only populated when valid).
  std::string output;
};

/// Per-operand input candidate summary.
struct InputCandidateSetRecord {
  size_t operandIndex = 0;
  size_t fromProducerBeam = 0;
  size_t fromReshard = 0;
  std::vector<std::string> candidates;
};

/// One beam survivor entry.
struct BeamEntryRecord {
  size_t rank = 0;
  std::string outputLayout;

  // Score breakdown.
  bool isL1 = false;
  bool isSharded = false;
  uint64_t inputDramBytes = 0;
  bool requiresReshard = false;
  int64_t coreCount = 0;
  uint64_t outputL1Usage = 0;
};

/// Output hints summary for an op.
struct OutputHintsRecord {
  size_t primaryCount = 0;
  size_t fallbackCount = 0;
  bool attemptL1Sharding = true;
};

/// All decision data for one op.
struct OpDecisionRecord {
  size_t opIndex = 0;
  std::string opName;
  std::string opLocation;

  std::vector<InputCandidateSetRecord> inputCandidateSets;
  OutputHintsRecord outputHints;
  size_t crossProductSize = 0;
  std::vector<EvaluationRecord> evaluations;
  std::vector<BeamEntryRecord> beam;
  bool usedDramFallback = false;
  bool isInplace = false;
};

/// Producer->consumer edge record.
struct EdgeRecord {
  size_t producerOpIndex = 0;
  size_t consumerOpIndex = 0;
  size_t operandIndex = 0;
  bool hasReshard = false;
  std::string reshardLayout;
};

/// Backward pass fork resolution record.
struct ForkResolutionRecord {
  std::string opName;
  std::string opLocation;
  size_t opIndex = 0;
  size_t chosenCandidateIndex = 0;
  size_t numConsumers = 0;
  std::vector<size_t> consumerOpIndices;
};

/// Final layout choice for one op.
struct FinalChoiceRecord {
  size_t opIndex = 0;
  std::string opName;
  std::string chosenLayout;
};

/// One event in the L1 spill management timeline.
struct SpillEventRecord {
  size_t position = 0;
  std::string opName;
  std::string action; // "dead_removal", "live_added", "oom",
                      // "demotion_success", "demotion_failed",
                      // "eviction", "self_spill", "revalidation"
  uint64_t occupiedL1Before = 0;
  uint64_t occupiedL1After = 0;
  uint64_t opL1Usage = 0;
  std::string victimName;
  std::string details;
};

/// L1 spill management trace data.
struct SpillManagementTrace {
  uint64_t budget = 0;
  size_t scheduleSize = 0;
  size_t totalSpills = 0;
  uint64_t finalOccupied = 0;
  size_t finalLiveTensors = 0;
  std::vector<SpillEventRecord> events;
};

//===----------------------------------------------------------------------===//
// DecisionTrace -- top-level container
//===----------------------------------------------------------------------===//

/// Top-level decision trace container that collects the full decision tree
/// of the greedy/beam-search layout propagation pass and L1 spill management.
/// Serializable to JSON.
class DecisionTrace {
public:
  /// Serialize the entire trace to a JSON value.
  llvm::json::Value toJSON() const;

  /// Write the JSON to a file. Returns true on success.
  bool writeToFile(llvm::StringRef path) const;

  /// Write full trace to <dir>/<funcName>_decision_trace.json.
  static bool writeTraceForFunc(llvm::StringRef dir, llvm::StringRef funcName,
                                const DecisionTrace &trace);

  /// Read-merge-write spill section into existing trace file.
  static bool mergeSpillTrace(llvm::StringRef dir, llvm::StringRef funcName,
                              const DecisionTrace &spillTrace);

  /// Serialize SpillManagementTrace to JSON Object (shared by toJSON + merge).
  static llvm::json::Object
  spillManagementToJSON(const SpillManagementTrace &spill);

  // Public data -- populated by observer instrumentation.
  std::string functionName;
  size_t beamWidth = 0;
  size_t totalOps = 0;

  std::vector<OpDecisionRecord> forwardPass;

  std::vector<EdgeRecord> edges;

  struct BackwardPass {
    std::vector<ForkResolutionRecord> forkResolutions;
  } backwardPass;

  std::vector<FinalChoiceRecord> finalChoices;

  SpillManagementTrace spillManagement;
};

//===----------------------------------------------------------------------===//
// DecisionTraceObserver -- implements both observer interfaces
//===----------------------------------------------------------------------===//

/// Concrete observer that implements both LayoutPropagationObserver and
/// L1SpillObserver, populating a single DecisionTrace data container.
class DecisionTraceObserver : public LayoutPropagationObserver,
                              public L1SpillObserver {
public:
  // LayoutPropagationObserver overrides.
  void onStart(llvm::StringRef funcName, size_t beamWidth) override;
  void onOpSetup(Operation *op,
                 const std::vector<std::vector<InputCandidate>> &inputSets,
                 const OutputHints &hints, size_t crossProductSize) override;
  void onEvaluation(Operation *op, const OpConfig &hint, size_t hintIdx,
                    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts, bool valid,
                    const BeamCandidate *candidate,
                    llvm::StringRef failureReason) override;
  void onBeamResult(Operation *op, llvm::ArrayRef<BeamCandidate> beam,
                    bool usedDramFallback) override;
  void onForkResolved(Operation *producer, size_t chosenIdx,
                      llvm::ArrayRef<Operation *> consumers) override;
  void onEdge(Operation *producer, Operation *consumer, size_t operandIdx,
              bool hasReshard, TTNNLayoutAttr reshardLayout) override;
  void onFinalChoice(Operation *op, size_t opIndex,
                     const BeamCandidate &chosen) override;
  void onEnd(size_t totalOps) override;
  void onInplaceOp(const InplaceOpInfo &info) override;

  // L1SpillObserver overrides.
  void onSpillStart(llvm::StringRef funcName, uint64_t budget,
                    size_t scheduleSize) override;
  void onDeadRemoval(Operation *op, int64_t pos,
                     uint64_t occupiedAfter) override;
  void onLiveAdded(Operation *op, int64_t pos, uint64_t opL1Usage,
                   int64_t lastUse, uint64_t occupiedAfter) override;
  void onOOM(Operation *op, int64_t pos, uint64_t occupiedL1) override;
  void onDemotion(Operation *op, int64_t pos, bool success,
                  uint64_t newL1Usage) override;
  void onEviction(Operation *victim, int64_t pos, uint64_t freedBytes) override;
  void onSelfSpill(Operation *op, int64_t pos) override;
  void onRevalidationCascade(Operation *changed, Operation *consumer,
                             bool outputChanged) override;
  void onSpillEnd(size_t totalSpills, uint64_t finalOccupied,
                  size_t liveTensors) override;

  const DecisionTrace *getDecisionTrace() const override { return &trace; }

  /// Mutable access for merging spill data into an existing trace.
  DecisionTrace &getTrace() { return trace; }

private:
  DecisionTrace trace;

  /// Map from op to its index in trace.forwardPass. Built from onBeamResult
  /// calls (one per processed op). Owned by the observer, not by
  /// LayoutPropagation.
  llvm::DenseMap<Operation *, size_t> opToTraceIndex;

  /// Current op record being built (set by onOpSetup, used by onEvaluation).
  OpDecisionRecord *currentOpRecord = nullptr;

  /// L1 occupied before current spill event (for before/after tracking).
  uint64_t spillOccupiedBefore = 0;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_DECISIONTRACE_H
