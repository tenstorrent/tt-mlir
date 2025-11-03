// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_DECOUPLECONSTFANOUTPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Clone definition op of value each of its uses and connect independent results
static void cloneDefPerUse(Value v, IRRewriter &rewriter) {
  llvm::SmallVector<OpOperand *> uses = llvm::to_vector(
      llvm::map_range(v.getUses(), [](OpOperand &use) { return &use; }));

  // Keep first use as is, clone for rest
  for (OpOperand *use : llvm::drop_begin(uses)) {
    rewriter.setInsertionPoint(use->getOwner());
    Operation *clone = rewriter.clone(*v.getDefiningOp());
    use->set(clone->getResult(0));
  }
}

template <typename T>
static void duplicateConstFedOps(func::FuncOp func, IRRewriter &rewriter) {
  SmallVector<T> targets;

  func.walk([&](T op) {
    Value src = op.getOperand();

    // Condition 1) source is constant feed
    if (!isa_and_nonnull<mlir::stablehlo::ConstantOp>(src.getDefiningOp())) {
      return;
    }

    // Condition 2) result has multiple uses
    if (op.getResult().getNumUses() < 2) {
      return;
    }
    targets.push_back(op);
  });

  for (auto op : targets) {
    rewriter.setInsertionPoint(op);
    cloneDefPerUse(op.getResult(), rewriter);
  }
}

// Returns true if any block argument carries explicit sharding (i.e. any dim
// has non-empty axes), as opposed to being fully replicated.
static bool hasExplicitlyShardedArguments(func::FuncOp funcOp) {
  for (BlockArgument arg : funcOp.getArguments()) {
    if (auto tensorSharding = mlir::sdy::getSharding(arg)) {
      for (const auto &dimSharding : tensorSharding.getDimShardings()) {
        // we consider [{?}] as sharded
        if (!dimSharding.getAxes().empty()) {
          return true;
        }
      }
    }
  }
  return false; // either no sharding attrs or all replicated
}

struct DecoupleConstFanoutPass
    : public impl::DecoupleConstFanoutPassBase<DecoupleConstFanoutPass> {
public:
  using impl::DecoupleConstFanoutPassBase<
      DecoupleConstFanoutPass>::DecoupleConstFanoutPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    IRRewriter rewriter(ctx);

    module.walk([&](func::FuncOp func) {
      // Skip functions without explicitly sharded arguments
      if (!hasExplicitlyShardedArguments(func)) {
        return;
      }

      // TODO(sshon): Right now we copy every constant-fed op with multiple
      // uses. But actually it is only necessary to duplicate those whose uses
      // are in sharding conflicts. But we don't have a way to detect sharding
      // conflict as we rely on sharding propagation pass from upstream. Once we
      // have a way to detect sharding conflict, we can limit the duplication
      // only to those ops with sharding conflicts.
      duplicateConstFedOps<::mlir::stablehlo::BroadcastInDimOp>(func, rewriter);
    });
  }
};
} // namespace mlir::tt::stablehlo
