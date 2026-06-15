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
// sharded dimension is not evenly divisible by its mesh axis size. A null
// sharding means the value is replicated, so the per-shard count equals the
// global element count.
static mlir::FailureOr<int64_t>
localNumElements(mlir::sdy::MeshAttr meshAttr, mlir::RankedTensorType type,
                 mlir::sdy::TensorShardingAttr sharding) {
  if (!sharding) {
    return type.getNumElements();
  }
  mlir::FailureOr<mlir::RankedTensorType> localType =
      shardy_utils::populateShardedOutputType(meshAttr, type, sharding);
  if (mlir::failed(localType)) {
    return mlir::failure();
  }
  return localType->getNumElements();
}

// A reshape is unrealizable when its result demands a finer sharding than its
// operand can locally supply. Shardy's conservative propagation leaves such a
// reshape in three states this pass must normalize to replicated I/O:
//   * the result is unannotated while a sharded operand shrinks per shard
//     (the original tt-xla#3643 gap),
//   * the result carries a sharding that cannot be localized because a sharded
//     dimension is not divisible by its mesh axis (e.g. heads=30 across
//     model=4), or
//   * the result is sharded *more* than the operand, so the operand's
//     per-shard data is too coarse to produce it locally -- e.g. a replicated
//     (1x3616x30x128) operand reshaped to a (3616x3840) result sharded on the
//     merged dim (tt-xla#5148). This breaks the element-count check in
//     UpdateGlobalToLocalShapes.
// A result that is *equally or less* sharded than the operand is a valid
// reshape (downstream InsertExplicitReshards gathers any extra operand
// sharding), so it is left untouched -- a mere per-shard element-count
// difference in that direction is not a failure.
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

  mlir::sdy::TensorShardingAttr operandSharding =
      shardy_utils::getOperandShardingAttr(operand, globalMeshOp,
                                           /*createIfMissing=*/false);
  bool operandSharded =
      operandSharding &&
      !shardy_utils::isFullyReplicatedTensor(operandSharding, globalMeshOp);

  // Result Shardy never annotated: reshard the sharded operand to replicated so
  // per-shard shapes match downstream (tt-xla#3643). Nothing to do when the
  // operand is already replicated.
  auto spv = op->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
      mlir::sdy::TensorShardingAttr::name);
  if (!spv || spv.getShardings().empty()) {
    return operandSharded;
  }

  // Annotated result: a result sharding that cannot localize (non-divisible
  // sharded dim) is unrealizable outright.
  mlir::FailureOr<int64_t> resultElems =
      localNumElements(meshAttr, resultType, spv.getShardings().front());
  if (mlir::failed(resultElems)) {
    return true;
  }

  // Otherwise intervene only when the result is sharded *more* than the operand
  // (smaller per-shard count): the operand cannot locally produce it. A null
  // operand sharding localizes to the full (replicated) count.
  mlir::FailureOr<int64_t> operandElems =
      localNumElements(meshAttr, operandType, operandSharding);
  if (mlir::failed(operandElems)) {
    return true;
  }
  return *resultElems < *operandElems;
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

// For reshapes that Shardy cannot lower as-is (see reshapeNeedsReplication),
// reshard any sharded operand to fully replicated and drop the unrealizable
// result sharding. The reshard later lowers to an all_gather, so the reshape
// runs on replicated data and per-shard shapes match in
// UpdateGlobalToLocalShapes; downstream InsertExplicitReshards reconciles
// sharded consumers. Covers both the unannotated-result case (tt-xla#3643) and
// non-divisible factored dims (e.g. heads=30 split across model=4,
// tt-xla#5148).
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
