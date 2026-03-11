// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYLAYOUTPROPAGATION_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYLAYOUTPROPAGATION_H

#include "mlir/Pass/PassRegistry.h"

#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include <string>

namespace mlir::tt::ttnn {

// Pipeline options: extends tablegen-generated Options with complex fields
// that need custom CLI parsers (not supported by tablegen).
struct TTNNGreedyLayoutPropagationPipelineOptions {
  int64_t maxLegalLayouts = 64;
  bool rowMajorEnabled = false;
  int64_t beamWidth = 8;
  bool enableL1ShardingLayouts = true;
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout;
  llvm::StringMap<Conv2dConfigOverrideParams> overrideConv2dConfig;
  bool enableDecisionTrace = false;
  std::string decisionTraceDir = "ttrt-artifacts/decision_trace";
  bool enableCompileTimeStats = false;
};

std::unique_ptr<::mlir::Pass> createTTNNGreedyLayoutPropagation(
    TTNNGreedyLayoutPropagationPipelineOptions options);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYLAYOUTPROPAGATION_H
