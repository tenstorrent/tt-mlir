// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <vector>

namespace mlir::tt::ttnn {

struct L1InterleavedAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalL1InterleavedConfigs;
  llvm::DenseMap<Operation *, OpConfig> currentConfigs;
  func::FuncOp funcOp;
  unsigned usableL1CacheSize = 0;

  L1InterleavedAnalysisInput() : legalL1InterleavedConfigs(), funcOp() {}

  L1InterleavedAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<OpConfig>>
          &legalL1InterleavedConfigs, // only tiled rn, but could have row-major
      const llvm::DenseMap<Operation *, OpConfig> &currentConfigs,
      const func::FuncOp &funcOp, unsigned usableL1CacheSize)
      : legalL1InterleavedConfigs(legalL1InterleavedConfigs),
        currentConfigs(currentConfigs), funcOp(funcOp),
        usableL1CacheSize(usableL1CacheSize) {}

  bool operator==(const L1InterleavedAnalysisInput &rhs) const {
    return legalL1InterleavedConfigs == rhs.legalL1InterleavedConfigs &&
           currentConfigs == rhs.currentConfigs && funcOp == rhs.funcOp &&
           usableL1CacheSize == rhs.usableL1CacheSize;
  }

  bool operator!=(const L1InterleavedAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct L1InterleavedAnalysisResult {
  llvm::DenseMap<Operation *, OpConfig> upgradedConfigs;

  L1InterleavedAnalysisResult() : upgradedConfigs() {}

  L1InterleavedAnalysisResult(
      const llvm::DenseMap<Operation *, OpConfig> &upgradedConfigs)
      : upgradedConfigs(upgradedConfigs) {}
};

// Analysis that runs after spillToDRAM to try upgrading DRAM operations
// to L1 interleaved when:
// 1. The operation has exactly one user
// 2. That user is the immediate next operation in the schedule
// 3. L1 memory constraints are satisfied
//
class L1InterleavedAnalysis : public TTNNAnalysis<L1InterleavedAnalysisInput,
                                                  L1InterleavedAnalysisResult> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override { return false; }

  // Check if operation has L1 interleaved layout available
  bool hasL1InterleavedLegalLayout(Operation *op) const;

  // Get L1 interleaved layouts for operation (both tiled and row-major if
  // available after LegalLayoutsAnalysis)
  std::vector<OpConfig> getL1InterleavedLayoutConfigs(Operation *op) const;

  bool outputsDRAMLayout(Operation *op) const;

  bool outputsL1Layout(Operation *op) const;

  // Check if operation has exactly one user that is immediate next in schedule
  bool hasImmediateConsumer(Operation *op) const;

  // Check if originally tiled or row-major to choose correct upgrade path
  bool isTiledTensorLayout(Operation *op) const;

  // Check if upgrading operation to L1 interleaved if safe, return updated
  // layout
  llvm::Expected<TTNNLayoutAttr> checkUpgradeToL1Interleaved(
      Operation *consumerOp, const OpConfig &consumerConfig,
      const Operation *upgradedProducerOp,
      const TTNNLayoutAttr upgradedProducerLayout) const;

public:
  L1InterleavedAnalysis(Operation *op) : TTNNAnalysis(op) {}
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDANALYSIS_H
