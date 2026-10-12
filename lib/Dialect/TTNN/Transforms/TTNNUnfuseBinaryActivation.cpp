// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNUNFUSEBINARYACTIVATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {
class TTNNUnfuseBinaryActivation
    : public impl::TTNNUnfuseBinaryActivationBase<
          TTNNUnfuseBinaryActivation> {
public:
  using impl::TTNNUnfuseBinaryActivationBase<
      TTNNUnfuseBinaryActivation>::TTNNUnfuseBinaryActivationBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    llvm::SmallVector<Operation *> fused;
    module->walk([&](Operation *op) {
      if (mlir::isa<AddOp, SubtractOp, MultiplyOp>(op) &&
          op->getAttr("post_activation")) {
        fused.push_back(op);
      }
    });
    for (Operation *op : fused) {
      auto act = mlir::cast<UnaryWithParamAttr>(op->getAttr("post_activation"));
      OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(op);
      Value result = op->getResult(0);
      Operation *unary = nullptr;
      switch (act.getOpType()) {
      case UnaryOpType::Relu6:
        unary = builder.create<Relu6Op>(op->getLoc(), result.getType(), result);
        break;
      case UnaryOpType::Relu:
        unary = builder.create<ReluOp>(op->getLoc(), result.getType(), result);
        break;
      default:
        op->emitError("unsupported fused post activation");
        signalPassFailure();
        return;
      }
      result.replaceAllUsesExcept(unary->getResult(0), unary);
      op->removeAttr("post_activation");
    }
  }
};
} // namespace
} // namespace mlir::tt::ttnn
