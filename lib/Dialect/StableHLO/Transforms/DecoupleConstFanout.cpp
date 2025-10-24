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
  // Skip if use count is less than 2
  if (v.getNumUses() < 2) {
    return;
  }

  SmallVector<OpOperand *> uses;
  for (auto &use : v.getUses()) {
    uses.push_back(&use);
  }

  // Keep first use as is, clone for rest
  for (OpOperand *use : llvm::drop_begin(uses)) {
    Operation *def = v.getDefiningOp();
    rewriter.setInsertionPoint(use->getOwner());
    Operation *clone = rewriter.clone(*def);
    use->set(clone->getResult(0));
  }
}

template <typename T>
static void duplicateConstFedOps(func::FuncOp func, IRRewriter &rewriter) {
  SmallVector<T> targets;

  func.walk([&](T op) {
    Value src = op.getOperand();

    // Condition 1) source is constant feed or scalar (rank-0) feed
    bool srcIsConst = false;
    bool srcIsScalar = false;
    if (Operation *defOp = src.getDefiningOp()) {
      srcIsConst = isa<::mlir::stablehlo::ConstantOp>(defOp);
    }
    if (auto rankedType = dyn_cast<RankedTensorType>(src.getType())) {
      srcIsScalar = (rankedType.getRank() == 0);
    }

    if (!(srcIsConst || srcIsScalar)) {
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
      duplicateConstFedOps<::mlir::stablehlo::BroadcastInDimOp>(func, rewriter);
    });
  }
};
} // namespace mlir::tt::stablehlo
