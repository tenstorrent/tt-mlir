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

  constexpr int64_t batchDim = 0;
  constexpr int64_t headDim = 1;
  constexpr int64_t seqDim = 2;
  constexpr int64_t hiddenDim = 3;

  sdy::OpShardingRuleBuilder builder(op);

  // 1) Allow passthrough sharding on batch and head dimensions.
  //
  //    This is equivalent to a pointwise rule restricted to dims 0 and 1:
  //    we want to propagate whatever sharding we have on [B, H] as-is.
  builder.addPointwiseIf(shape, [&](int64_t dim) {
    return dim == batchDim || dim == headDim;
  });

  // 2) Tie the sequence dimension across Q/K/V/output.
  //
  //    Internally SDPA does matmuls over [S, D] in different ways, but as long
  //    Q, K, V and the output all agree on how the sequence dimension is
  //    sharded, we can allow sharding along S and only introduce CCL if some
  //    operand disagrees.
  //
  //    This factor says:
  //      - operand 0 (Q)  @ dim 2
  //      - operand 1 (K)  @ dim 2
  //      - operand 2 (V)  @ dim 2
  //      - result         @ dim 2
  //    all share the same sharding factor of size shape[seqDim].
  {
    llvm::SmallVector<int64_t, 3> operandSeqDims = {
        seqDim,  // Q
        seqDim,  // K
        seqDim   // V
    };
    builder.addFactor(operandSeqDims,
                      /*resultDim=*/seqDim,
                      /*dimSize=*/shape[seqDim],
                      /*factorType=*/sdy::FactorType::kPassThrough);
  }

  // 3) Keep the hidden dimension conservative for now.
  //
  //    Hidden (D) is a contraction dimension of the internal matmuls. Fully
  //    supporting sharding on D would require modeling partial sums and
  //    all-reduces explicitly. Until we model that properly, we require
  //    replication on D.
  {
    llvm::SmallVector<int64_t, 3> operandHiddenDims = {
        mlir::sdy::kNullDim,  // Q
        mlir::sdy::kNullDim,  // K
        mlir::sdy::kNullDim   // V
    };
    builder.addFactor(operandHiddenDims,
                      /*resultDim=*/hiddenDim,
                      /*dimSize=*/shape[hiddenDim],
                      /*factorType=*/sdy::FactorType::kNeedReplication);
  }
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

    auto it = customCallShardingRules.find(target);
    if (it != customCallShardingRules.end()) {
      return it->second(op);
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
  }
};

} // namespace mlir::tt::stablehlo
