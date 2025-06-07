// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

struct OpConfig {
  using Attributes = llvm::SmallVector<Attribute, 4>;

  // Desired output layout for the op.
  TTNNLayoutAttr outputLayout;

  // E.g. Conv2dConfigAttr for Conv2dOp.
  Attributes opSpecificAttrs;

  OpConfig() = default;
  OpConfig(TTNNLayoutAttr outputLayout) : outputLayout(outputLayout) {}
  OpConfig(TTNNLayoutAttr outputLayout, Attribute opSpecificAttr)
      : outputLayout(outputLayout) {
    opSpecificAttrs.push_back(opSpecificAttr);
  }
  OpConfig(TTNNLayoutAttr outputLayout, Attributes opSpecificAttrs)
      : outputLayout(outputLayout), opSpecificAttrs(opSpecificAttrs) {}

  bool operator==(const OpConfig &other) const {
    return outputLayout == other.outputLayout &&
           opSpecificAttrs == other.opSpecificAttrs;
  }

  void dump() const {
    if (outputLayout) {
      outputLayout.dump();
    }
    if (!opSpecificAttrs.empty()) {
      for (const auto &opSpecificAttr : opSpecificAttrs) {
        opSpecificAttr.dump();
      }
    }
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
