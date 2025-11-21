// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REGISTERCUSTOMSHARDINGRULEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static constexpr llvm::StringLiteral sdpaTargetName =
    "tt.scaled_dot_product_attention";

static constexpr llvm::StringLiteral pagedSdpaDecodeTargetName =
    "tt.paged_scaled_dot_product_attention_decode";

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

static mlir::sdy::OpShardingRuleAttr
getPagedSdpaDecodeShardingRule(mlir::stablehlo::CustomCallOp op) {
  // We assume the StableHLO custom call has the following operands:
  //  0: query        [batch, num_users, num_heads, ...]
  //  1: key          [num_blocks_total, num_heads, block_size, head_size]
  //  2: value        [num_blocks_total, num_heads, block_size, head_size]
  //  3+: optional operands (page_table, attention_mask, cur_pos_tensor,
  //  attention_sink, ...)
  //
  // Sharding rule:
  // - query / key / value: shard along num_heads dimension
  // - everything else: replicated
  // - results: inherit num_heads sharding (same dim as in lowering)

  auto qType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
  auto kType = llvm::cast<RankedTensorType>(op.getOperand(1).getType());
  auto vType = llvm::cast<RankedTensorType>(op.getOperand(2).getType());
  auto outType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

  // We currently expect the output to have the same logical layout as Q.
  // (e.g., [1, num_users, num_heads, head_size])
  if (qType.getShape() != outType.getShape()) {
    op.getOperation()->emitWarning()
        << "Paged SDPA decode: Q and output shapes must match.";
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto checkQKVLayout = [](RankedTensorType ty) -> bool {
    if (ty.getRank() != 4) {
      return false;
    }
    return true;
  };

  if (!checkQKVLayout(qType) || !checkQKVLayout(kType) ||
      !checkQKVLayout(vType)) {
    // If K/V layout is not what we expect, be conservative.
    op.getOperation()->emitWarning()
        << "Paged SDPA decode: K/V layouts are unexpected, q rank: "
        << qType.getRank() << ", key rank: " << kType.getRank()
        << ", value rank: " << vType.getRank();
    return mlir::sdy::OpShardingRuleBuilder::buildPointwise(op);
  }

  const int64_t qHeadDim = 2;  // Q / Out: [1, U, H, D]
  const int64_t kvHeadDim = 1; // K / V:   [B, H, S, D]
  const int64_t outHeadDim = 2;
  auto qShape = qType.getShape();
  int64_t headSize = qShape[qHeadDim];

  mlir::sdy::OpShardingRuleBuilder builder(op);
  SmallVector<int64_t> headOperandDims(op.getNumOperands(),
                                       mlir::sdy::kNullDim);
  SmallVector<int64_t> headResultDims(op.getNumResults(), mlir::sdy::kNullDim);

  // 0: Q
  headOperandDims[0] = qHeadDim;
  // 1: K
  headOperandDims[1] = kvHeadDim;
  // 2: V
  headOperandDims[2] = kvHeadDim;
  // result 0: output
  headResultDims[0] = outHeadDim;

  builder.addFactor(headOperandDims, headResultDims, headSize,
                    mlir::sdy::FactorType::kPassThrough);
  return builder.build();
}

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
          {pagedSdpaDecodeTargetName, getPagedSdpaDecodeShardingRule},
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
  }
};

} // namespace mlir::tt::stablehlo
