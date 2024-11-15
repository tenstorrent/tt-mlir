// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <stablehlo/dialect/StablehloOps.h>

#include "ttmlir/Conversion/RedundantBroadcastElimination/RedundantBroadcastElimination.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_REDUNDANTBROADCASTELIMINATION
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

class RedundantBroadcastEliminationPass
    : public ttir::impl::RedundantBroadcastEliminationBase<
          RedundantBroadcastEliminationPass> {
public:
  using ttir::impl::RedundantBroadcastEliminationBase<
      RedundantBroadcastEliminationPass>::RedundantBroadcastEliminationBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](Operation *op) {
      if (mlir::isa<mlir::tt::ttir::BroadcastOp>(op)) {
        if (op->use_empty()) {
          return;
        }

        if (op->getResult(0).getType() == op->getOperand(0).getType()) {
          // This broadcast is redundant
          rewriter.replaceAllUsesWith(Value(op->getResult(0)),
                                      op->getOperand(0));
          rewriter.eraseOp(op);
        }
      }
    });
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createRedundantBroadcastEliminationPass() {
  return std::make_unique<RedundantBroadcastEliminationPass>();
}

} // namespace mlir::tt
