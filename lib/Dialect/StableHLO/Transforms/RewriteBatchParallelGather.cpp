// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REWRITEBATCHPARALLELGATHERPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// Returns the identity broadcast dim if `v` matches
// `broadcast_in_dim(iota(dim = 0), broadcast_dims = [k])` where the iota has
// exactly one axis of size `expectedSize`.  Returns std::nullopt otherwise.
static std::optional<int64_t>
matchIotaPassThrough(mlir::Value v, int64_t expectedSize) {
  auto bcast = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return std::nullopt;
  }
  auto iota = bcast.getOperand().getDefiningOp<mlir::stablehlo::IotaOp>();
  if (!iota || iota.getIotaDimension() != 0) {
    return std::nullopt;
  }
  auto iotaTy = mlir::cast<mlir::RankedTensorType>(iota.getType());
  if (iotaTy.getRank() != 1 || iotaTy.getDimSize(0) != expectedSize) {
    return std::nullopt;
  }
  auto bdims = bcast.getBroadcastDimensions();
  if (bdims.size() != 1) {
    return std::nullopt;
  }
  return bdims[0];
}

struct RewriteBatchParallelGatherPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::GatherOp> {
  using mlir::OpRewritePattern<
      mlir::stablehlo::GatherOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::GatherOp gatherOp,
                  mlir::PatternRewriter &rewriter) const override {
    // (a) Only rewrite inside a flattened composite group.
    if (!gatherOp->hasAttr(utils::kReoutlineGroupAttr)) {
      return mlir::failure();
    }

    // (b) Idempotent: skip if already rewritten.
    auto dn = gatherOp.getDimensionNumbers();
    if (!dn.getOperandBatchingDims().empty()) {
      return mlir::failure();
    }

    // (c) Operand must have a non-trivial Shardy sharding on at least one dim.
    auto moduleOp = gatherOp->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::sdy::MeshOp> meshOps =
        shardy_utils::getMeshOps(moduleOp);
    if (meshOps.size() != 1) {
      return mlir::failure();
    }
    mlir::sdy::MeshOp meshOp = meshOps[0];

    mlir::sdy::TensorShardingAttr operandSharding =
        shardy_utils::getOperandShardingAttr(
            gatherOp.getOperation()->getOpOperand(0), meshOp);
    if (!operandSharding ||
        shardy_utils::isFullyReplicatedTensor(operandSharding)) {
      return mlir::failure();
    }

    // (d) start_indices must be a concatenate along index_vector_dim with
    //     exactly one operand per entry in start_index_map.
    auto concat = gatherOp.getStartIndices()
                      .getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    if (!concat) {
      return mlir::failure();
    }
    if (static_cast<int64_t>(concat.getDimension()) !=
        dn.getIndexVectorDim()) {
      return mlir::failure();
    }
    auto startIndexMap = dn.getStartIndexMap();
    if (concat.getInputs().size() != startIndexMap.size()) {
      return mlir::failure();
    }

    auto operandTy =
        mlir::cast<mlir::RankedTensorType>(gatherOp.getOperand().getType());
    auto sliceSizes = gatherOp.getSliceSizes();
    auto collapsedSliceDims = dn.getCollapsedSliceDims();
    auto dimShardings = operandSharding.getDimShardings();

    // (e) For each axis in startIndexMap, detect iota pass-through slots.
    llvm::SmallVector<size_t> keepSlots;         // slots to keep in concat
    llvm::SmallVector<int64_t> newStartIndexMap;
    llvm::SmallVector<int64_t> newCollapsedSliceDims;
    // Paired vectors — same index = one (operand_batching, start_indices_batching) pair.
    llvm::SmallVector<std::pair<int64_t, int64_t>> batchingPairs;

    for (auto [slot, axis] : llvm::enumerate(startIndexMap)) {
      // Candidate axis must be collapsed with slice_size == 1.
      if (!llvm::is_contained(collapsedSliceDims, axis) ||
          sliceSizes[axis] != 1) {
        keepSlots.push_back(slot);
        newStartIndexMap.push_back(axis);
        continue;
      }
      // Candidate axis must be sharded on the operand.
      if (static_cast<size_t>(axis) >= dimShardings.size() ||
          dimShardings[axis].getAxes().empty()) {
        keepSlots.push_back(slot);
        newStartIndexMap.push_back(axis);
        continue;
      }
      // Candidate concat slot must be iota(axis_size).broadcast(single_dim).
      auto maybeBdim = matchIotaPassThrough(concat.getInputs()[slot],
                                            operandTy.getDimSize(axis));
      if (!maybeBdim) {
        keepSlots.push_back(slot);
        newStartIndexMap.push_back(axis);
        continue;
      }
      batchingPairs.push_back({axis, *maybeBdim});
    }

    if (batchingPairs.empty() || keepSlots.empty()) {
      // Either nothing to batch, or everything would become batching (degenerate).
      return mlir::failure();
    }

    // Sort batching pairs by operand axis so operand_batching_dims is ascending.
    llvm::sort(batchingPairs,
               [](const auto &a, const auto &b) { return a.first < b.first; });

    llvm::SmallVector<int64_t> operandBatchingDims;
    llvm::SmallVector<int64_t> startIndicesBatchingDims;
    for (auto [obdim, sibdim] : batchingPairs) {
      operandBatchingDims.push_back(obdim);
      startIndicesBatchingDims.push_back(sibdim);
    }

    // Rebuild collapsed_slice_dims without the newly batchable axes.
    for (int64_t axis : collapsedSliceDims) {
      if (!llvm::is_contained(operandBatchingDims, axis)) {
        newCollapsedSliceDims.push_back(axis);
      }
    }

    // Collect ops to erase BEFORE any rewrites so we can check use counts.
    llvm::SmallVector<mlir::Operation *> toErase;
    for (size_t slot = 0; slot < startIndexMap.size(); ++slot) {
      if (llvm::is_contained(keepSlots, slot)) {
        continue;
      }
      mlir::Value slotVal = concat.getInputs()[slot];
      if (auto bc =
              slotVal.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        toErase.push_back(bc.getOperation());
        if (auto iota =
                bc.getOperand().getDefiningOp<mlir::stablehlo::IotaOp>()) {
          toErase.push_back(iota.getOperation());
        }
      }
    }

    // Build trimmed start_indices value.
    mlir::Value newIndices;
    if (keepSlots.size() == 1) {
      newIndices = concat.getInputs()[keepSlots.front()];
    } else {
      llvm::SmallVector<mlir::Value> newConcatInputs;
      for (size_t slot : keepSlots) {
        newConcatInputs.push_back(concat.getInputs()[slot]);
      }
      auto newConcat = rewriter.create<mlir::stablehlo::ConcatenateOp>(
          concat.getLoc(), newConcatInputs, concat.getDimension());
      // Copy discardable attributes (reoutline.group, sdy.sharding_per_value).
      for (auto namedAttr : concat->getAttrs()) {
        newConcat->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
      newIndices = newConcat.getResult();
    }

    // Build new dimension numbers.
    auto newDn = mlir::stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offsetDims=*/dn.getOffsetDims(),
        /*collapsedSliceDims=*/newCollapsedSliceDims,
        /*operandBatchingDims=*/operandBatchingDims,
        /*startIndicesBatchingDims=*/startIndicesBatchingDims,
        /*startIndexMap=*/newStartIndexMap,
        /*indexVectorDim=*/dn.getIndexVectorDim());

    // Create new gather; result type is unchanged.
    auto newGather = rewriter.create<mlir::stablehlo::GatherOp>(
        gatherOp.getLoc(), gatherOp.getType(), gatherOp.getOperand(),
        newIndices, newDn,
        rewriter.getDenseI64ArrayAttr(
            llvm::SmallVector<int64_t>(sliceSizes.begin(), sliceSizes.end())),
        gatherOp.getIndicesAreSortedAttr());

    // Copy all discardable attributes from the old gather (reoutline.group,
    // reoutline.seed, reoutline.comp_attrs, sdy.sharding_per_value, etc.).
    for (auto namedAttr : gatherOp->getAttrs()) {
      newGather->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.replaceOp(gatherOp, newGather.getResult());

    // Erase the old concat (now use-empty after gather replacement).
    if (concat->use_empty()) {
      rewriter.eraseOp(concat.getOperation());
    }

    // Erase dead iota/broadcast producers so the reoutline group stays
    // contiguous (outer bc first so iota becomes use-empty, then iota).
    for (mlir::Operation *op : toErase) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }

    return mlir::success();
  }
};

struct RewriteBatchParallelGatherPass
    : public impl::RewriteBatchParallelGatherPassBase<
          RewriteBatchParallelGatherPass> {
  using impl::RewriteBatchParallelGatherPassBase<
      RewriteBatchParallelGatherPass>::RewriteBatchParallelGatherPassBase;

  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteBatchParallelGatherPattern>(&getContext());
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::stablehlo
