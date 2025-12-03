// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REGISTERCUSTOMSHARDINGRULEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static constexpr llvm::StringLiteral sdpaTargetName =
    "tt.scaled_dot_product_attention";

static mlir::sdy::OpShardingRuleAttr
getScatterShardingRule(mlir::stablehlo::ScatterOp scatterOp) {
  mlir::Operation::operand_range inputs = scatterOp.getInputs();
  mlir::Operation::operand_range updates = scatterOp.getUpdates();
  mlir::Value indices = scatterOp.getScatterIndices();

  if (!llvm::hasSingleElement(inputs) || !llvm::hasSingleElement(updates)) {
    scatterOp->emitError(
        "Scatter operation has multiple input or update tensors. This is not "
        "supported.");
    return mlir::sdy::OpShardingRuleAttr();
  }

  RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(inputs.front().getType());
  RankedTensorType updateType =
      llvm::dyn_cast<RankedTensorType>(updates.front().getType());
  RankedTensorType indicesType =
      llvm::dyn_cast<RankedTensorType>(indices.getType());

  if (!inputType || !updateType || !indicesType) {
    scatterOp->emitError(
        "Scatter operation has unranked tensor types. This is not supported.");
    return mlir::sdy::OpShardingRuleAttr();
  }

  const int64_t inputRank = inputType.getRank();
  const int64_t indicesRank = indicesType.getRank();
  const int64_t updateRank = updateType.getRank();

  auto dimNums = scatterOp.getScatterDimensionNumbers();
  ArrayRef<int64_t> scatterDimsToOperandDims =
      dimNums.getScatterDimsToOperandDims();
  ArrayRef<int64_t> insertedWindowDims = dimNums.getInsertedWindowDims();

  // Return null if input is not sharded along scatter_dims_to_operand_dims.
  // This would fallback to default sharding rule in
  // op_sharding_rule_registry.cc.
  bool isShardedAlongScatterDims = false;
  if (sdy::TensorShardingAttr inputSharding =
          mlir::sdy::getSharding(inputs.front())) {
    ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
        inputSharding.getDimShardings();
    for (int64_t scatterDim : scatterDimsToOperandDims) {
      if (scatterDim < static_cast<int64_t>(dimShardings.size()) &&
          !dimShardings[scatterDim].getAxes().empty()) {
        isShardedAlongScatterDims = true;
        break;
      }
    }
  }

  if (!isShardedAlongScatterDims) {
    // Return null to fallback to default sharding rule.
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Check input and update have the same rank.
  if (inputRank != updateRank) {
    scatterOp->emitError(
        "Custom sharding rule not implemented for scatter "
        "operations where input and update tensors have different ranks.");
    return mlir::sdy::OpShardingRuleAttr();
  }

  sdy::OpShardingRuleBuilder builder(scatterOp);

  // Input, updates, and result need to be sharded together.
  // Replicate if dimension is in scatter_dims_to_operand_dims.
  // Replicate if dimension is in inserted_window_dims.
  // Shard otherwise.
  for (int64_t inputDim = 0; inputDim < inputRank; inputDim++) {
    bool isScatterDimsToOperandDim =
        llvm::is_contained(scatterDimsToOperandDims, inputDim);
    bool isInsertedWindowDim = llvm::is_contained(insertedWindowDims, inputDim);

    if (isScatterDimsToOperandDim || isInsertedWindowDim) {
      // Dimension is in scatter_dims_to_operand_dims or inserted_window_dims -
      // MUST REPLICATE.
      builder.addFactor({inputDim, sdy::kNullDim,
                         inputDim}, // [input_dim, indices_dim, updates_dim]
                        {inputDim}, // result_dim
                        inputType.getDimSize(inputDim),
                        mlir::sdy::FactorType::kNeedReplication);
    } else {
      // Dimension is NOT in scatter_dims_to_operand_dims or
      // inserted_window_dims - CAN SHARD. Shard input, updates, and result
      // together.
      builder.addFactor({inputDim, sdy::kNullDim,
                         inputDim}, // [input_dim, indices_dim, updates_dim]
                        {inputDim}, // result_dim
                        inputType.getDimSize(inputDim),
                        sdy::FactorType::kPassThrough // Can be sharded.
      );
    }
  }

  // Replicate all scatter_indices dimensions.
  for (int64_t indicesDim = 0; indicesDim < indicesRank; indicesDim++) {
    builder.addFactor({sdy::kNullDim, indicesDim,
                       sdy::kNullDim}, // [input_dim, indices_dim, updates_dim]
                      {sdy::kNullDim}, // Doesn't appear in result.
                      indicesType.getDimSize(indicesDim),
                      sdy::FactorType::kNeedReplication);
  }

  return builder.build();
}

static mlir::sdy::OpShardingRuleAttr
getSDPAShardingRule(mlir::stablehlo::CustomCallOp op) {
  auto qType = llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  auto kType = llvm::dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  auto vType = llvm::dyn_cast<RankedTensorType>(op.getOperand(2).getType());
  auto outType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());

  // SDPA requires Q/K/V and output to have identical shapes.
  if (qType.getShape() != kType.getShape() ||
      qType.getShape() != vType.getShape() ||
      qType.getShape() != outType.getShape()) {
    return mlir::sdy::OpShardingRuleAttr();
  }

  ArrayRef<int64_t> shape = qType.getShape();

  // SDPA is assumed to operate on 4D tensors: [B, H, S, D]
  // If the shape does not match, conservatively fallback to a pointwise rule.
  if (shape.size() != 4) {
    return mlir::sdy::OpShardingRuleBuilder::buildPointwise(op);
  }

  // SDPA can shard batch (B) and head (H) dimensions freely.
  // Sequence (S) and hidden (D) dimensions generally require replication,
  // unless a more advanced distributed attention algorithm is implemented.
  //
  // Dimension assignment:
  // dim 0 -> Batch
  // dim 1 -> Head
  // dim 2 -> Sequence length
  // dim 3 -> Hidden size
  auto getFactorType = [&](int64_t dim) -> mlir::sdy::FactorType {
    if (dim == 0 || dim == 1) {
      // Allow sharding
      return mlir::sdy::FactorType::kPassThrough;
    }
    // Disallow sharding, require replication
    return mlir::sdy::FactorType::kNeedReplication;
  };

  // Build the final sharding rule:
  // - Pass-through on B/H dims
  // - Replication on S/D dims
  return mlir::sdy::OpShardingRuleBuilder(op)
      .addPointwise(shape, getFactorType)
      .build();
}

template <typename OpTy>
struct StablehloShardingModel
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          StablehloShardingModel<OpTy>, OpTy> {

  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation *op) const {
    if (auto scatterOp = dyn_cast::cast<mlir::stablehlo::ScatterOp>(op)) {
      return getScatterShardingRule(scatterOp);
    }
    return mlir::sdy::OpShardingRuleBuilder::buildPointwise(op);
  }

  bool shouldKeepOutputShardingsDivisible(mlir::Operation *) const {
    return true;
  }
};

struct StablehloCustomCallShardingModel
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          StablehloCustomCallShardingModel, ::mlir::stablehlo::CustomCallOp> {

  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation *op) const {
    auto custom = llvm::cast<mlir::stablehlo::CustomCallOp>(op);
    assert(custom && "StablehloCustomCallShardingModel must be attached to "
                     "CustomCallOp only");
    return getCustomCallShardingRule(custom);
  }

  bool shouldKeepOutputShardingsDivisible(mlir::Operation *) const {
    return true;
  }

private:
  mlir::sdy::OpShardingRuleAttr
  getCustomCallShardingRule(mlir::stablehlo::CustomCallOp op) const {
    llvm::StringRef target = op.getCallTargetName();

    auto shardOpFunc = customCallShardingRules.lookup(target);
    if (shardOpFunc) {
      return shardOpFunc(op);
    }

    op.getOperation()->emitWarning()
        << "StableHLO CustomCallOp sharding rule is not defined for target '"
        << target << "'";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Map from CustomCall target names to their corresponding sharding rule
  // functions.
  llvm::DenseMap<llvm::StringRef, std::function<mlir::sdy::OpShardingRuleAttr(
                                      mlir::stablehlo::CustomCallOp)>>
      customCallShardingRules = {
          {sdpaTargetName, getSDPAShardingRule},
      };
};

class RegisterCustomShardingRulePass
    : public impl::RegisterCustomShardingRulePassBase<
          RegisterCustomShardingRulePass> {
public:
  using impl::RegisterCustomShardingRulePassBase<
      RegisterCustomShardingRulePass>::RegisterCustomShardingRulePassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();

    context->loadDialect<mlir::stablehlo::StablehloDialect>();
    // Register for stablehlo.CustomCallOp
    mlir::stablehlo::CustomCallOp::attachInterface<
        StablehloCustomCallShardingModel>(*context);
    mlir::stablehlo::ScatterOp::attachInterface<
        StablehloShardingModel<mlir::stablehlo::ScatterOp>>(*context);
  }
};

} // namespace mlir::tt::stablehlo
