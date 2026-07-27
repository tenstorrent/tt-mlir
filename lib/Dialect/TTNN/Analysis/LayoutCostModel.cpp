// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LayoutCostModel.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstdint>

namespace mlir::tt::ttnn {

namespace {

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

/// Accumulated analytical-time cost: minimize estimated graph time (in
/// microseconds) = sum of per-op roofline cost + reshard-edge cost along the
/// producer chain. Reads only data the constraint query already produced
/// (coreCount/residency/bytes) plus shape-derived matmul tile-muls -- no
/// getOpRuntime. Lower accumulatedCost wins.
class AnalyticalTimeCostModel final : public LayoutCostModel {
public:
  explicit AnalyticalTimeCostModel(roofline::HwSpec hw) : hw(hw) {}

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
  // Time (us) to move `bytes` through aggregate DRAM bandwidth (shared across
  // cores) or per-core L1 bandwidth (scaled by the number of cores).
  double dramUs(uint64_t bytes) const {
    return hw.dramBandwidthBytesPerSec
               ? static_cast<double>(bytes) * 1e6 /
                     static_cast<double>(hw.dramBandwidthBytesPerSec)
               : 0.0;
  }
  double l1Us(uint64_t bytes, double cores) const {
    uint64_t perCore = hw.perCoreL1BandwidthBytesPerSec();
    return perCore ? static_cast<double>(bytes) * 1e6 /
                         (cores * static_cast<double>(perCore))
                   : 0.0;
  }

  // Per-op roofline in us: max(compute, memory). Compute is the matmul
  // tile-mul time at the candidate's core count (0 for non-matmul ops, so they
  // are memory-bound). Memory is DRAM-input read (incl. streamed weights) plus
  // the output write (L1 across cores, or aggregate DRAM). max() means dropping
  // cores raises the compute term on compute-bound ops -- so the model will not
  // trade away matmul parallelism for fewer reshards.
  double nodeCost(const BeamCandidate &candidate) const {
    const LayoutScore &s = candidate.score;
    double cores = static_cast<double>(std::max<int64_t>(s.coreCount, 1));

    double computeUs = 0.0;
    if (s.tileMuls != 0 && hw.aiclkHz != 0) {
      computeUs = static_cast<double>(s.tileMuls * hw.cyclesPerTileMatmul) *
                  1e6 / (cores * static_cast<double>(hw.aiclkHz));
    }

    double outWriteUs = s.isL1 ? l1Us(s.outputBytes, cores)
                               : dramUs(s.outputBytes);
    double memUs = dramUs(s.inputDramBytes) + outWriteUs;

    return std::max(computeUs, memUs);
  }

  // One reshard op per edge: time to move the producer's output through the
  // slowest leg. A DRAM leg (producer or target in DRAM) pays aggregate DRAM
  // bandwidth; an L1->L1 reshard moves through L1 across the producer's cores.
  // Bytes use the producer's buffer-agnostic footprint (nonzero even for
  // DRAM-resident producers, unlike outputL1Usage).
  double reshardCost(const BeamCandidate *producer,
                     TTNNLayoutAttr targetLayout) const {
    if (!producer) {
      return 0.0;
    }
    uint64_t bytes = producer->score.outputBytes;
    bool dramLeg = !producer->score.isL1 || !targetLayout.hasL1BufferType();
    if (dramLeg) {
      return dramUs(bytes);
    }
    double cores = static_cast<double>(std::max<int64_t>(producer->score.coreCount, 1));
    return l1Us(bytes, cores);
  }

  roofline::HwSpec hw;
};

} // namespace

std::unique_ptr<LayoutCostModel>
createLayoutCostModel(LayoutCostModelKind kind, roofline::HwSpec hwSpec) {
  switch (kind) {
  case LayoutCostModelKind::Heuristic:
    return std::make_unique<HeuristicCostModel>();
  case LayoutCostModelKind::AnalyticalTime:
    return std::make_unique<AnalyticalTimeCostModel>(hwSpec);
  }
  llvm_unreachable("unknown LayoutCostModelKind");
}

} // namespace mlir::tt::ttnn
