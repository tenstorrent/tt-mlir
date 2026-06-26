// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LayoutCostModel.h"

#include "llvm/Support/ErrorHandling.h"

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

} // namespace

std::unique_ptr<LayoutCostModel>
createLayoutCostModel(LayoutCostModelKind kind) {
  switch (kind) {
  case LayoutCostModelKind::Heuristic:
    return std::make_unique<HeuristicCostModel>();
  }
  llvm_unreachable("unknown LayoutCostModelKind");
}

} // namespace mlir::tt::ttnn
