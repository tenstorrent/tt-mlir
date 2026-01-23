// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H

#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"

namespace mlir::tt::ttnn {

// The goal of this policy is to always solve simple fork-joins if that is
// possible. Fork-join is considered to be simple if there is no need for DRAM
// spill in its execution. Furthermore, if DRAM spill is necessary, this policy
// will not produce globally optimal solution.
//
class BFInterleavedPolicy : public MemoryLayoutAnalysisPolicy {
public:
  // In order to keep track of the L1 memory usage, we have to know two things
  // for each op:
  //    1. The L1 memory usage of each op's output tensor.
  //    2. The number of op's users currently relying on the op's output tensor.
  //       This is important for fork ops where the output tensor is used by
  //       multiple other ops.
  //
  struct OpL1MemUsage {
    uint64_t l1MemUsagePerUser;
    uint64_t numOfUnscheduledUsers;
  };

public:
  BFInterleavedPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule)
      : MemoryLayoutAnalysisPolicy(rootOp, l1ChainConfigs, legalConfigs,
                                   schedule) {}

  void run() final;

private:
  // Effective L1 cache size scaled by tensorL1UsageCap from module attribute.
  // Calculated once at the start of run() using utils::getTensorL1UsageCap().
  uint64_t usableL1CacheSize;

  // Check if the op is analyzable. Op is analyzable if it has at least one
  // legal config.
  bool isAnalyzable(Operation *op);

  // Iterate over all operands of the op that satisfy the analyzability
  // criterium defined by the isAnalyzable method. This is an abstraction
  // for the boilerplate code used in different places within the policy.
  //
  void walkOnAnalyzableOperands(Operation *op,
                                function_ref<void(Operation *)> callback);

  // Fetch op's DRAM layout from legalConfigs.
  bool hasDRAMBufferType(Operation *op);
  TTNNLayoutAttr getDRAMLayout(Operation *op);

  // Fetch op's L1 Interleaved layout from legalConfigs.
  bool hasL1BufferType(Operation *op);
  TTNNLayoutAttr getL1InterleavedLayout(Operation *op);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H
