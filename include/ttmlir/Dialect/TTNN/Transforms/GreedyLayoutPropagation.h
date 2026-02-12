// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYLAYOUTPROPAGATION_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYLAYOUTPROPAGATION_H

#include "mlir/Pass/PassRegistry.h"

#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

namespace mlir::tt::ttnn {

struct TTIRToTTNNDevicePipelineOptions;

//===----------------------------------------------------------------------===//
// TTNNGreedyLayoutPropagation
//===----------------------------------------------------------------------===//
struct TTNNGreedyLayoutPropagationOptions {
  int64_t maxLegalLayouts = 64;
  bool rowMajorEnabled = false;
  llvm::StringMap<InsertMemReconfigParams> insertMemReconfig =
      llvm::StringMap<InsertMemReconfigParams>();
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout =
      llvm::StringMap<OutputLayoutOverrideParams>();
  llvm::StringMap<Conv2dConfigOverrideParams> overrideConv2dConfig =
      llvm::StringMap<Conv2dConfigOverrideParams>();

  TTNNGreedyLayoutPropagationOptions() = default;
};

std::unique_ptr<::mlir::Pass> createTTNNGreedyLayoutPropagation();
std::unique_ptr<::mlir::Pass>
createTTNNGreedyLayoutPropagation(TTNNGreedyLayoutPropagationOptions options);

//===----------------------------------------------------------------------===//
// TTNNGreedyLayoutPropagation Registration
//===----------------------------------------------------------------------===//
inline void registerTTNNGreedyLayoutPropagation() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTTNNGreedyLayoutPropagation();
  });
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYLAYOUTPROPAGATION_H
