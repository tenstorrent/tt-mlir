// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDPOLICY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"

namespace mlir::tt::ttnn {

class L1InterleavedPolicy : public MemoryLayoutAnalysisPolicy {
public:
  L1InterleavedPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<TensorConfigAttr>>
          &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : MemoryLayoutAnalysisPolicy(rootOp, l1ChainConfigs, legalLayouts,
                                   schedule, usableL1CacheSize) {}

  void run() final;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDPOLICY_H
