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

// Per-shard element count of `type` under `sharding`, or failure if any
// sharded dimension is not evenly divisible by its mesh axis size.
static mlir::FailureOr<int64_t>
localNumElements(mlir::sdy::MeshAttr meshAttr, mlir::RankedTensorType type,
                 mlir::sdy::TensorShardingAttr sharding) {
  mlir::FailureOr<mlir::RankedTensorType> localType =
      shardy_utils::populateShardedOutputType(meshAttr, type, sharding);
  if (mlir::failed(localType)) {
    return mlir::failure();
  }
  return localType->getNumElements();
}

// The reshape's single-result sharding, or a fully-replicated sharding when
// Shardy propagation left the result unannotated.
static mlir::sdy::TensorShardingAttr
getResultShardingOrReplicated(mlir::Operation *op,
                              mlir::sdy::MeshOp globalMeshOp) {
  if (auto spv = op->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
          mlir::sdy::TensorShardingAttr::name)) {
    if (!spv.getShardings().empty()) {
      return spv.getShardings().front();
    }
  }
  auto type = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  return shardy_utils::getClosedReplicatedTensorSdyShardingAttr(
      op->getContext(), globalMeshOp.getSymNameAttr(), type.getRank());
}

// A reshape is locally lowerable iff its operand and result shardings both
// localize successfully and yield the same per-shard element count. Shardy's
// conservative propagation can leave a reshape in two unrealizable states:
//   * the result is unannotated while a sharded operand shrinks per shard
//     (the original tt-xla#3643 gap), or
//   * the result is sharded on a flat dim that divides evenly while the
//     operand's factored dim does not (e.g. heads=30 split across model=4),
// both of which break the element-count check in UpdateGlobalToLocalShapes.
// Returns true when the reshape must be normalized to fully-replicated I/O.
static bool reshapeNeedsReplication(mlir::Operation *op,
                                    mlir::sdy::MeshOp globalMeshOp) {
  mlir::sdy::MeshAttr meshAttr = globalMeshOp.getMesh();

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }
  mlir::OpOperand &operand = op->getOpOperand(0);
  auto operandType =
      mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
  if (!operandType) {
    return false;
  }

  mlir::FailureOr<int64_t> resultElems = localNumElements(
      meshAttr, resultType, getResultShardingOrReplicated(op, globalMeshOp));
  mlir::FailureOr<int64_t> operandElems = localNumElements(
      meshAttr, operandType,
      shardy_utils::getOperandShardingAttr(operand, globalMeshOp,
                                           /*createIfMissing=*/false));

  // Either side failing to localize (non-divisible sharded dim) or a per-shard
  // element-count mismatch means the reshape cannot be lowered as-is.
  if (mlir::failed(resultElems) || mlir::failed(operandElems)) {
    return true;
  }
  return *resultElems != *operandElems;
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

// For reshapes that Shardy cannot lower as-is, reshard any sharded operand to
// fully replicated and drop the unrealizable result sharding. The reshard later
// lowers to an all_gather, so the reshape runs on replicated data and per-shard
// shapes match in UpdateGlobalToLocalShapes. This covers two cases: a result
// left unannotated while a sharded operand shrinks per shard (tt-xla#3643), and
// a result sharded on a flat dim that divides evenly while the operand's
// factored dim does not (e.g. heads=30 split across model=4, tt-xla#5148). See
// reshapeNeedsReplication for the per-shard element-count check.
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
      if (!reshapeNeedsReplication(op, globalMeshOp)) {
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

      // Drop the unrealizable result sharding so the reshape result is
      // replicated to match its now-replicated operands. Downstream
      // InsertExplicitReshards reconciles sharded consumers via collectives.
      op->removeAttr(mlir::sdy::TensorShardingAttr::name);
    });
  }
};

} // namespace mlir::tt::stablehlo
