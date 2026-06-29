// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTCOSTMODEL_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTCOSTMODEL_H

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Utils/Roofline.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

#include <memory>

namespace mlir::tt::ttnn {

/// Selects how layout candidates are ranked on the beam path.
enum class LayoutCostModelKind {
  /// Local lexicographic `LayoutScore` heuristic (historical default).
  Heuristic,
  /// Accumulated analytical-time cost: per-op roofline (from the constraint
  /// query's coreCount/residency) plus reshard `bytes / BW(slowest leg)`,
  /// summed along the producer chain. Minimizes estimated time, so it trades
  /// reshards against op parallelism instead of ranking a local tuple.
  AnalyticalTime,
};

/// Strategy for ranking layout candidates during memory-layout propagation.
///
/// The beam path funnels all candidate comparison through a single `better(...)`
/// call, so swapping the cost model swaps the optimization objective without
/// changing the beam machinery. The default `Heuristic` model reproduces the
/// historical local `LayoutScore` ordering exactly.
class LayoutCostModel {
public:
  virtual ~LayoutCostModel() = default;

  /// Annotate a freshly-evaluated candidate with any model-specific data used
  /// for later comparison (e.g. an accumulated path cost). Called once per
  /// candidate after `score`, `producerCandidateIndices` and `reshardLayouts`
  /// are populated. `producerChoices[i]` is the resolved producer candidate for
  /// tensor operand `i` (nullptr for block args / constants). The default
  /// implementation does nothing — the heuristic compares `score` directly.
  virtual void
  annotate(Operation *op, BeamCandidate &candidate,
           llvm::ArrayRef<const BeamCandidate *> producerChoices) const {}

  /// Strict-weak ordering: returns true if candidate `a` is strictly better
  /// than `b` for `op`. This is the only comparator the beam uses.
  virtual bool better(Operation *op, const BeamCandidate &a,
                      const BeamCandidate &b) const = 0;
};

/// Create the cost model for the given kind. `hwSpec` supplies the hardware
/// constants the AnalyticalTime model needs for its real-time roofline (ignored
/// by the Heuristic model).
std::unique_ptr<LayoutCostModel>
createLayoutCostModel(LayoutCostModelKind kind, roofline::HwSpec hwSpec);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LAYOUTCOSTMODEL_H
