// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"
#include <cstdint>

namespace mlir::tt::ttnn {

class BFInterleavedPolicy : public MemoryLayoutAnalysisPolicy {
public:
  // In order to keep track of the L1 memory usage, we have know two things
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
      const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
          &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : MemoryLayoutAnalysisPolicy(rootOp, l1ChainConfigs, legalLayouts,
                                   schedule, usableL1CacheSize) {}

  void run() final;

private:
  DeviceAttr deviceAttr;

  // Check if the op is analyzable. Op is analyzable if it has at least one
  // legal layout.
  bool isAnalyzable(Operation *op);

  // Iterate over all operands of the op that satisfy the analyzability
  // criterium defined by the isAnalyzable method. This is an abstraction
  // for the boilerplate code used in different places within the policy.
  //
  void walkOnAnalyzableOperands(Operation *op,
                                function_ref<void(Operation *)> callback);

  // Fetch op's DRAM layout from legalLayouts.
  bool hasDRAMBufferType(Operation *op);
  TTNNLayoutAttr getDRAMLayout(Operation *op);

  // Fetch op's L1 Interleaved layout from legalLayouts.
  bool hasL1BufferType(Operation *op);
  TTNNLayoutAttr getL1InterleavedLayout(Operation *op);

  size_t getAvailableL1CacheSize() const {
    // Figure out this const based on exec data, but will be replaced
    // with API.
    //
    constexpr float tensorL1UsageCap = 0.75;
    return tensorL1UsageCap * usableL1CacheSize;
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H
