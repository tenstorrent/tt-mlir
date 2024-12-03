// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_GREEDYL1INTERLEAVEDPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_GREEDYL1INTERLEAVEDPOLICY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

class GreedyL1InterleavedPolicy : public MemoryLayoutAnalysisPolicy {
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

public:
  GreedyL1InterleavedPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
          &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : MemoryLayoutAnalysisPolicy(rootOp, l1ChainConfigs, legalLayouts,
                                   schedule, usableL1CacheSize) {}

  /**
   * Retrieve the greedy OpConfig for the given base operation
   * and its opsL1Usage map.
   *
   * @param baseOp     The base operation for which the greedy configuration is
   * being determined.
   * @param opsL1Usage A map between the operation and its output L1 usage. All
   * operations included in the opsL1Usage map must be either the baseOp or its
   * operand with a legal L1 Interleaved output layout at the time of analyzing
   * the baseOp.
   * @return The greedy OpConfig for the baseOp.
   */
  OpConfig getGreedyConfig(Operation *baseOp,
                           llvm::DenseMap<Operation *, L1Usage> &opsL1Usage);

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

  // Precedence schedule map for each operation. It contains the order
  // in which operands need to be executed for each op.
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> precedenceMap;

  llvm::DenseSet<Operation *> visitedOps;
  void buildSchedule(mlir::Operation *op, func::FuncOp &func) {

    // Schedule all the precedents of the current operation
    //
    visitedOps.insert(op);
    for (Operation *precedent : precedenceMap[op]) {
      if (!visitedOps.count(precedent)) {
        buildSchedule(precedent, func);
      }
    }

    (*schedule)[func].push_back(op);
  }

  void constructSchedule(func::FuncOp &func) {
    func->walk([&](Operation *op) {
      if (op->hasTrait<mlir::OpTrait::ReturnLike>()) {
        Operation *outputOp = op->getOperand(0).getDefiningOp();
        buildSchedule(outputOp, func);
      }
    });
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_GREEDYL1INTERLEAVEDPOLICY_H
