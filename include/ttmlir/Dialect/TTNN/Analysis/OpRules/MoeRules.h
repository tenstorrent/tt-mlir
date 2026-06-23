// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MOERULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MOERULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// MoE ops: moe_compute and its prepare_moe_compute_* weight-prep ops.
//
// These ops have rigid layouts: the prepare ops emit a fixed bank-permuted
// bfp4 sharded weight layout, and moe_compute's input layouts are pinned by
// the workaround pass (drain-core indices/scores) while its output layouts are
// device-derived by TTNNDeduceMoEComputeLayouts. The kernel rejects any other
// sharding, so probing sharded candidates crashes the constraint query. Use
// NULL output hint only and skip reshard exploration; the deduce pass and the
// workaround pass remain the layout authority.
//===----------------------------------------------------------------------===//

struct MoeRuleBook : OpRuleBook {
  bool shouldExploreReshards() const override;
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MOERULES_H
