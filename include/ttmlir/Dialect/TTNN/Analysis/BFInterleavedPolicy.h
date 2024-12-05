// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_BFINTERLEAVEDPOLICY_H

#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"

namespace mlir::tt::ttnn {

class BFInterleavedPolicy : public MemoryLayoutAnalysisPolicy {
public:
  struct OpMemSpec {
    TTNNLayoutAttr layout;
    // Minimum L1 memory usage required for scheduling the op
    // given the layouts of all the ops that are already scheduled.
    //
    uint64_t requiredL1Usage;
  };

  // This struct is holding information about the greedily choosen
  // configuration of the @baseOp: 1) layouts and 2) precedence.
  //
  // The @layouts represents the mapping between the op and its choosen
  // layout. All the ops that are included in the @layouts map must be
  // either @baseOp or its operand with legal L1 Interleaved output layout
  // at the moment of analyzing the @baseOp.
  //
  // The @precedence represents the order of the op's operands in which they
  // should be scheduled. Only op's operands that are included in the @layouts
  // map are included in the @precedence.
  //
  struct OpConfig {
    Operation *baseOp;
    llvm::DenseMap<Operation *, TTNNLayoutAttr> layouts;
    llvm::SmallVector<Operation *> precedence;
  };

  struct L1Usage {
    size_t outputL1Usage;
    size_t requiredL1Usage;
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
  // be
  //                        non-analyzable or its output layout tensor to have
  //                        DRAM buffer type. There are no action for this type
  //                        of op.
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
