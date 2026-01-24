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
  ttcore::DeviceAttr deviceAttr;

public:
  virtual ~MemoryLayoutAnalysisPolicy() {};

  MemoryLayoutAnalysisPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule)
      : rootOp(rootOp), l1ChainConfigs(&l1ChainConfigs),
        legalConfigs(legalConfigs), schedule(&schedule) {}

  virtual void run() = 0;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPOLICY_H
