// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDFALLBACKANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDFALLBACKANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <vector>

namespace mlir::tt::ttnn {

struct L1InterleavedFallbackAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalL1InterleavedConfigs;
  llvm::DenseMap<Operation *, OpConfig> currentConfigs;
  func::FuncOp funcOp;
  unsigned usableL1CacheSize = 0;

  L1InterleavedFallbackAnalysisInput()
      : legalL1InterleavedConfigs(), funcOp() {}

  L1InterleavedFallbackAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<OpConfig>>
          &legalL1InterleavedConfigs,
      const llvm::DenseMap<Operation *, OpConfig> &currentConfigs,
      const func::FuncOp &funcOp, unsigned usableL1CacheSize)
      : legalL1InterleavedConfigs(legalL1InterleavedConfigs),
        currentConfigs(currentConfigs), funcOp(funcOp),
        usableL1CacheSize(usableL1CacheSize) {}

  bool operator==(const L1InterleavedFallbackAnalysisInput &rhs) const {
    return legalL1InterleavedConfigs == rhs.legalL1InterleavedConfigs &&
           currentConfigs == rhs.currentConfigs && funcOp == rhs.funcOp &&
           usableL1CacheSize == rhs.usableL1CacheSize;
  }

  bool operator!=(const L1InterleavedFallbackAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct L1InterleavedFallbackAnalysisResult {
  llvm::DenseMap<Operation *, OpConfig> upgradedConfigs;

  L1InterleavedFallbackAnalysisResult() : upgradedConfigs() {}

  L1InterleavedFallbackAnalysisResult(
      const llvm::DenseMap<Operation *, OpConfig> &upgradedConfigs)
      : upgradedConfigs(upgradedConfigs) {}
};

// Analysis that runs after spillToDRAM to try upgrading DRAM operations
// to L1 interleaved when:
// 1. The operation has exactly one user.
// 2. That user is the immediate next operation in the schedule.
// 3. L1 memory constraints are satisfied.
//
class L1InterleavedFallbackAnalysis
    : public TTNNAnalysis<L1InterleavedFallbackAnalysisInput,
                          L1InterleavedFallbackAnalysisResult> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override { return false; }

  // Check if operation has L1 interleaved layout available.
  bool hasL1InterleavedLegalLayout(Operation *op) const;

  // Get L1 interleaved layouts for operation (both tiled and row-major if
  // available after LegalOpLayoutsAnalysis).
  std::vector<OpConfig> getL1InterleavedLayoutConfigs(Operation *op) const;

  // Check if op's first user is immediate next op in schedule.
  bool hasImmediateConsumer(Operation *op) const;

  // Check if operation has a return op user.
  bool hasReturnOpUser(Operation *op) const;

  // Check if conv2D uses matmul in tt-metal,
  // reference: ttnn::operations::conv::use_matmul_for_1x1_conv
  // (1x1 kernel, stride=1, padding=0, dilation=1)
  bool isConv2DConvertibleToMatMul(Operation *op);

  // Check if a reshape operation should be skipped based on tt-metal TTNN
  // optimization rules. Returns true if the operation should be skipped, false
  // otherwise. reference:
  // ttnn::operations::data_movement::ReshapeViewOperation::invoke
  // Parameters:
  // - reshapeOp: The reshape operation to analyze
  // - isUserOp: true if this is a user reshape check, false for direct reshape
  // TODO(bmalesevic,#5086): replace to dynamic check when tt-metal fixed
  bool checkReshapeSkip(Operation *reshapeOperation, bool isUserOp) const;

  // Try to upgrade an operation to L1 interleaved layout by testing available
  // L1 configurations and selecting the first one that passes validation.
  void tryUpgradeToL1Interleaved(Operation *op);

  // Check if upgrading operation to L1 interleaved is safe via single-level
  // recursive validation of producer-consumer chain. Parameters support the
  // recursive check:
  // - upgradedProducerOp & upgradedProducerLayout: when non-null, represents
  //   the hypothetical upgraded producer we need to validate against.
  // - This allows checking if the consumer can handle the upgraded layout
  //   before committing to the upgrade.
  llvm::Expected<TTNNLayoutAttr> checkUpgradeToL1Interleaved(
      Operation *consumerOp, const OpConfig &consumerConfig,
      const Operation *upgradedProducerOp,
      const TTNNLayoutAttr upgradedProducerLayout) const;

public:
  L1InterleavedFallbackAnalysis(Operation *op) : TTNNAnalysis(op) {}
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDFALLBACKANALYSIS_H
