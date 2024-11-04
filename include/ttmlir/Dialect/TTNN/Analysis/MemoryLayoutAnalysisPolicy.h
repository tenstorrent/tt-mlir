// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPOLICY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"

namespace mlir::tt::ttnn {

class MemoryLayoutAnalysisPolicy {
protected:
  Operation *rootOp;
  std::vector<L1ChainConfig> *l1ChainConfigs;
  llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> legalLayouts;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> *schedule;
  unsigned usableL1CacheSize = 0;

public:
  virtual ~MemoryLayoutAnalysisPolicy() {};

  MemoryLayoutAnalysisPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : rootOp(rootOp), l1ChainConfigs(&l1ChainConfigs),
        legalLayouts(legalLayouts), schedule(&schedule),
        usableL1CacheSize(usableL1CacheSize) {}

  virtual void run() = 0;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPOLICY_H
