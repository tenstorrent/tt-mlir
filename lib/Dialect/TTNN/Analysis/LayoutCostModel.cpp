// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LayoutCostModel.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstdint>

namespace mlir::tt::ttnn {

namespace {

// Relative bandwidth / penalty constants for the analytical proxy.
// TODO: source BW_L1 : BW_DRAM from system_desc. Decode is weight-streaming
// bound, so the memory term dominates and these relative values rank correctly.
constexpr double kBwDram = 1.0;
constexpr double kBwL1 = 8.0;
constexpr double kDramPenalty = kBwL1 / kBwDram; // any DRAM leg pays DRAM BW

/// Historical default: rank by the local lexicographic `LayoutScore`, with
/// `preferCandidate` as the tiebreak. Lifted verbatim from the previous inline
/// beam comparator so the default build is byte-for-byte unchanged.
class HeuristicCostModel final : public LayoutCostModel {
public:
  bool better(Operation *op, const BeamCandidate &a,
              const BeamCandidate &b) const override {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return preferCandidate(op, a, b);
  }
};

/// Accumulated analytical-time cost: minimize estimated graph time = sum of
/// per-op roofline cost + reshard-edge cost along the producer chain. Reads
/// only data the constraint query already produced (coreCount/residency/bytes)
/// -- no getOpRuntime. Lower accumulatedCost wins.
class AnalyticalTimeCostModel final : public LayoutCostModel {
public:
  void annotate(
      Operation *op, BeamCandidate &candidate,
      llvm::ArrayRef<const BeamCandidate *> producerChoices) const override {
    double cost = nodeCost(candidate);
    // Path cost so far: each candidate commits to one producer per operand.
    for (const BeamCandidate *producer : producerChoices) {
      if (producer) {
        cost += producer->accumulatedCost;
      }
    }
    // Reshard edges (one runtime op each): producer output -> required input.
    for (const auto &entry : candidate.reshardLayouts) {
      size_t operandIdx = entry.first;
      const BeamCandidate *producer =
          operandIdx < producerChoices.size() ? producerChoices[operandIdx]
                                              : nullptr;
      cost += reshardCost(producer, entry.second);
    }
    candidate.accumulatedCost = cost;
  }

  bool better(Operation *, const BeamCandidate &a,
              const BeamCandidate &b) const override {
    return a.accumulatedCost < b.accumulatedCost;
  }

private:
  // Per-op roofline (memory term) from the constraint result. coreCount is
  // op-aware via RuleBook::adjustScore (e.g. RMSNorm is input-driven).
  // TODO: add the compute term (FLOPs/(cores*peak)) for compute-bound ops.
  static double nodeCost(const BeamCandidate &candidate) {
    const LayoutScore &s = candidate.score;
    double cores = static_cast<double>(std::max<int64_t>(s.coreCount, 1));
    return static_cast<double>(s.inputDramBytes) / (cores * kBwDram) +
           static_cast<double>(s.outputL1Usage) / (cores * kBwL1);
  }

  // One reshard op per edge; cost = bytes / BW(slowest leg). Bytes proxied by
  // the producer's output footprint; a DRAM leg (producer or target in DRAM)
  // pays the DRAM penalty.
  static double reshardCost(const BeamCandidate *producer,
                            TTNNLayoutAttr targetLayout) {
    if (!producer) {
      return 0.0;
    }
    double bytes = static_cast<double>(producer->score.outputL1Usage);
    bool dramLeg = !producer->score.isL1 || !targetLayout.hasL1BufferType();
    return bytes * (dramLeg ? kDramPenalty : 1.0);
  }
};

} // namespace

std::unique_ptr<LayoutCostModel>
createLayoutCostModel(LayoutCostModelKind kind) {
  switch (kind) {
  case LayoutCostModelKind::Heuristic:
    return std::make_unique<HeuristicCostModel>();
  case LayoutCostModelKind::AnalyticalTime:
    return std::make_unique<AnalyticalTimeCostModel>();
  }
  llvm_unreachable("unknown LayoutCostModelKind");
}

} // namespace mlir::tt::ttnn
