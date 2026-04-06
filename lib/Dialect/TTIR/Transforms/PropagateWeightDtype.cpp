// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRPROPAGATEWEIGHTDTYPE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Trace the weight operand backward through allowed ops (TM, CCL, Broadcast,
// MeshShard, MeshPartition, Typecast, RepeatInterleave) to find the originating
// func arg. If it carries "ttcore.weight_dtype", set it on the consumer op.
static void resolveAndSetWeightDtype(mlir::Value weight,
                                     mlir::Operation *consumerOp) {
  // Trace backward through allowed ops to find the originating func arg.
  mlir::Value source = weight;
  while (auto *op = source.getDefiningOp()) {
    if (op->hasTrait<TensorManipulation::Trait>() ||
        op->hasTrait<CCL::Trait>() ||
        mlir::isa<BroadcastOp, MeshShardOp, MeshPartitionOp, TypecastOp,
                  RepeatInterleaveOp>(op)) {
      source = op->getOperand(0);
    } else {
      TTMLIR_DEBUG(ttmlir::LogComponent::General,
                   "PropagateWeightDtype: unable to trace through op '{0}', "
                   "stopping backward walk",
                   op->getName().getStringRef());
      break;
    }
  }

  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(source);
  if (!blockArg) {
    return;
  }

  auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      blockArg.getOwner()->getParentOp());
  if (!funcOp) {
    return;
  }

  auto dtypeAttr = funcOp.getArgAttrOfType<mlir::StringAttr>(
      blockArg.getArgNumber(), "ttcore.weight_dtype");
  if (dtypeAttr) {
    consumerOp->setAttr("ttcore.weight_dtype", dtypeAttr);
  }
}

} // namespace

class TTIRPropagateWeightDtype
    : public impl::TTIRPropagateWeightDtypeBase<TTIRPropagateWeightDtype> {
public:
  using impl::TTIRPropagateWeightDtypeBase<
      TTIRPropagateWeightDtype>::TTIRPropagateWeightDtypeBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([](Operation *op) {
      if (auto matmulOp = mlir::dyn_cast<MatmulOp>(op)) {
        resolveAndSetWeightDtype(matmulOp.getB(), op);
      } else if (auto linearOp = mlir::dyn_cast<LinearOp>(op)) {
        resolveAndSetWeightDtype(linearOp.getB(), op);
      } else if (auto sparseMatmulOp = mlir::dyn_cast<SparseMatmulOp>(op)) {
        resolveAndSetWeightDtype(sparseMatmulOp.getB(), op);
      }
    });
  }
};

} // namespace mlir::tt::ttir
