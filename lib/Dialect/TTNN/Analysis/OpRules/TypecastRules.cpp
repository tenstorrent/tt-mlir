// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TypecastRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

namespace mlir::tt::ttnn {

OutputHints TypecastRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> & /*legalConfigs*/) const {
  return layout_filter_utils::nullHintOnly();
}

} // namespace mlir::tt::ttnn
