// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

struct OpConfig {
  // Desired output layout for the op.
  //
  TTNNLayoutAttr outputLayout;

  // Op specific configuration. E.g. Conv2dConfigAttr for Conv2dOp.
  //
  Attribute config;

  OpConfig() = default;
  OpConfig(TTNNLayoutAttr outputLayout) : outputLayout(outputLayout) {}
  OpConfig(TTNNLayoutAttr outputLayout, Attribute config)
      : outputLayout(outputLayout), config(config) {}

  bool operator==(const OpConfig &other) const {
    return outputLayout == other.outputLayout && config == other.config;
  }

  void dump() const {
    if (outputLayout) {
      outputLayout.dump();
    }
    if (config) {
      config.dump();
    }
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
