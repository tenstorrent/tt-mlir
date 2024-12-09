// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_OPTIMIZER_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_OPTIMIZER_H

#include <mlir/Pass/PassRegistry.h>

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// TTNNOptimizer
//===----------------------------------------------------------------------===//
struct TTNNOptimizerOptions {
  llvm::StringMap<InputLayoutOverrideParams> overrideInputLayout =
      llvm::StringMap<InputLayoutOverrideParams>();
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout =
      llvm::StringMap<OutputLayoutOverrideParams>();
  bool memoryLayoutAnalysisEnabled = false;
  MemoryLayoutAnalysisPolicyType memoryLayoutAnalysisPolicy =
      MemoryLayoutAnalysisPolicyType::DFSharding;
  bool memReconfigEnabled = false;
  int64_t maxLegalLayouts = 64;
};

std::unique_ptr<::mlir::Pass> createTTNNOptimizer();
std::unique_ptr<::mlir::Pass> createTTNNOptimizer(TTNNOptimizerOptions options);

//===----------------------------------------------------------------------===//
// TTNNOptimizer Registration
//===----------------------------------------------------------------------===//
inline void registerTTNNOptimizer() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createTTNNOptimizer(); });
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_OPTIMIZER_H
