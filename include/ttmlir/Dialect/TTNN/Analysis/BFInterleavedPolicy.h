// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H

#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"

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

  // This enum is used to determine the type of action required before
  // scheduling the next op.
  //
  // MergeL1ChainConfigsOp: Requirement for op to be of this type is to have L1
  //                        Interleaved output layout tensor. Action for this
  //                        type of op is to merge all L1ChainConfigs of its
  //                        operands and isert op's OpL1MemSpec in newly merged
  //                        L1ChainConfig.
  //
  // NewL1ChainConfigOp:    Requirement for op to be of this type is to have L1
  //                        Interleaved output layout tensor and non of its
  //                        operands belong to any L1ChainConfig. Action for
  //                        this type of op is to isert op's OpL1MemSpec in
  //                        DisjointL1ChainConfigsUnion which will result in
  //                        creating new L1ChainConfig.
  //
  // NoL1ChainConfigOp:     Requirement for op to be of this type is to either
  //                        be non-analyzable or its output layout tensor to
  //                        have DRAM buffer type. There are no action for this
  //                        type of op.
  //
  enum class NextOpType {
    MergeL1ChainConfigsOp,
    NewL1ChainConfigOp,
    NoL1ChainConfigOp
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
