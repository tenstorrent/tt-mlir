// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_INSERTRESHARDSFORUNPROPAGATEDOPSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static bool anyResultHasSharding(mlir::Operation *op) {
  auto spv = op->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
      mlir::sdy::TensorShardingAttr::name);
  return spv && !spv.getShardings().empty();
}

static void replicateOperandViaReshard(mlir::OpBuilder &builder,
                                       mlir::Operation *op,
                                       mlir::OpOperand &operand,
                                       mlir::StringAttr meshName) {
  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
  if (!tensorType) {
    return;
  }

  mlir::sdy::TensorShardingAttr replicated =
      shardy_utils::getClosedReplicatedTensorSdyShardingAttr(
          builder.getContext(), meshName, tensorType.getRank());

  builder.setInsertionPoint(op);
  auto reshard = builder.create<mlir::sdy::ReshardOp>(
      op->getLoc(), operand.get(), replicated);
  operand.set(reshard.getResult());
}

// For ops whose results were left unsharded by Shardy propagation, reshard
// any sharded operand to fully replicated. The reshard later lowers to an
// all_gather, so the consumer sees replicated operands and per-shard shapes
// match in UpdateGlobalToLocalShapes. See tt-xla#3643.
class InsertReshardsForUnpropagatedOpsPass
    : public impl::InsertReshardsForUnpropagatedOpsPassBase<
          InsertReshardsForUnpropagatedOpsPass> {
public:
  using impl::InsertReshardsForUnpropagatedOpsPassBase<
      InsertReshardsForUnpropagatedOpsPass>::
      InsertReshardsForUnpropagatedOpsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();

    if (shardy_utils::isGraphSolved(rootModule)) {
      return;
    }

    llvm::SmallVector<mlir::sdy::MeshOp> meshOps =
        shardy_utils::getMeshOps(rootModule);
    if (meshOps.empty()) {
      return;
    }
    if (meshOps.size() > 1) {
      rootModule.emitError("Pass currently only supports a single shardy mesh "
                           "op in the module");
      signalPassFailure();
      return;
    }
    mlir::sdy::MeshOp globalMeshOp = meshOps[0];
    mlir::StringAttr meshName = globalMeshOp.getSymNameAttr();

    mlir::OpBuilder builder(context);

    rootModule.walk([&](mlir::Operation *op) {
      // Restricted to stablehlo.reshape — the only op known to hit this
      // propagation gap. Other ops have custom sharding rules registered.
      if (!mlir::isa<mlir::stablehlo::ReshapeOp>(op)) {
        return;
      }
      if (op->getParentOfType<mlir::sdy::ManualComputationOp>()) {
        return;
      }
      if (anyResultHasSharding(op)) {
        return;
      }

      for (mlir::OpOperand &operand : op->getOpOperands()) {
        if (!mlir::isa<mlir::RankedTensorType>(operand.get().getType())) {
          continue;
        }
        mlir::sdy::TensorShardingAttr operandSharding =
            shardy_utils::getOperandShardingAttr(operand, globalMeshOp,
                                                 /*createIfMissing=*/false);
        if (!operandSharding || shardy_utils::isFullyReplicatedTensor(
                                    operandSharding, globalMeshOp)) {
          continue;
        }
        replicateOperandViaReshard(builder, op, operand, meshName);
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
