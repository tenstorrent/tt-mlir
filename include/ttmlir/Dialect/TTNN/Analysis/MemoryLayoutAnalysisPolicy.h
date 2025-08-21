// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPOLICY_H

#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::tt::ttnn {

class MemoryLayoutAnalysisPolicy {
protected:
  Operation *rootOp;
  std::vector<L1ChainConfig> *l1ChainConfigs;
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> *schedule;
  // usableL1CacheSize is previously scaled by a value between 0.0
  // and 1.0, where 1.0 means that the entire L1 cache can be used by
  // ops. This cap is set by a flag in the pipeline options.
  unsigned usableL1CacheSize = 0;
  ttcore::DeviceAttr deviceAttr;

public:
  virtual ~MemoryLayoutAnalysisPolicy() {};

  MemoryLayoutAnalysisPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : rootOp(rootOp), l1ChainConfigs(&l1ChainConfigs),
        legalConfigs(legalConfigs), schedule(&schedule),
        usableL1CacheSize(usableL1CacheSize) {}

  virtual void run() = 0;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPOLICY_H
