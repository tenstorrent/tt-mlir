// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERGC
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct D2MInsertDstRegisterGCPass
    : public impl::D2MInsertDstRegisterGCBase<D2MInsertDstRegisterGCPass> {

  D2MInsertDstRegisterGCPass() = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCPass &pass) = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCOptions &options)
      : D2MInsertDstRegisterGCBase(options) {}

  std::unique_ptr<ColoringStrategy> createColoringStrategy() {
    std::string strategy = this->coloringStrategy.getValue();
    if (strategy == "greedy") {
      return std::make_unique<GreedyColoring>();
    }
    if (strategy == "pbqp") {
      // TODO(bnorris): Implement PBQP-based coloring strategy.
      llvm::errs() << "Warning: PBQP strategy not yet implemented, falling "
                      "back to Chaitin-Briggs\n";
      return std::make_unique<ChaitinBriggsColoring>();
    }
    return std::make_unique<ChaitinBriggsColoring>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) {
      return;
    }

    auto &liveness = getAnalysis<Liveness>();
    DstCapacityAnalysis dstCapacityAnalysis(func);

    uint32_t numColors = dstCapacityAnalysis.getMinDstCapacity();
    if (maxDstPhysicalSizeTiles > 0) {
      numColors = std::min(
          numColors, static_cast<uint32_t>(maxDstPhysicalSizeTiles.getValue()));
    }

    InterferenceGraph ig;
    for (auto &block : func.getBody()) {
      if (block.empty()) {
        continue;
      }
      auto liveIn = liveness.getLiveIn(&block);
      for (auto it1 = liveIn.begin(); it1 != liveIn.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != liveIn.end(); ++it2) {
          ig.addEdge(*it1, *it2);
        }
      }
    }

    if (ig.hasInterference()) {
      auto strategy = createColoringStrategy();
      auto coloring = strategy->colorGraph(ig, numColors);
      // TODO (Stage 4): Use coloring result to rewrite IR and insert
      // acquire/release ops.
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
