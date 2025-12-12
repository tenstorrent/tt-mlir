// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "llvm/Support/Error.h"

#include <sstream>
#include <vector>

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

      mlir::ModuleOp module = getOperation();
      MLIRContext* ctx = module.getContext();
      
      auto parsePriorityString = [](llvm::StringRef priorityStr) 
          -> std::vector<int64_t> {
        std::vector<int64_t> result;
        llvm::SmallVector<llvm::StringRef> parts;
        priorityStr.split(parts, ',');
        for (auto part : parts) {
          int64_t val;
          if (part.getAsInteger(10, val)) {
            result.push_back(-1);
          } else {
            result.push_back(val);
          }
        }
        return result;
      };
      
      module.walk([&](mlir::func::FuncOp funcOp) {
        for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
          auto shardingAttr = funcOp.getArgAttrOfType<sdy::TensorShardingAttr>(
              i, "sdy.sharding");
          if (!shardingAttr)
            continue;
          
          std::vector<int64_t> userPriorities;
          if (auto priorityAttr = funcOp.getArgAttrOfType<mlir::StringAttr>(
                  i, "ttcore.sdy_priority")) {
            userPriorities = parsePriorityString(priorityAttr.getValue());
          }
          
          int64_t defaultPriority = 0;
          if (userPriorities.empty()) {
            continue;
          }
          
          SmallVector<sdy::DimensionShardingAttr> newDimShardings;
          size_t dimIdx = 0;
          for (auto dimSharding : shardingAttr.getDimShardings()) {
            if (!dimSharding.emptyAxes()) {
              // 사용자 priority가 있으면 해당 dimension의 priority 사용
              // 없거나 -1이면 default priority 사용
              int64_t priority = defaultPriority;
              if (dimIdx < userPriorities.size() && 
                  userPriorities[dimIdx] >= 0) {
                priority = userPriorities[dimIdx];
              }
              newDimShardings.push_back(sdy::DimensionShardingAttr::get(
                  ctx,
                  dimSharding.getAxes(),
                  dimSharding.getIsClosed(),
                  std::make_optional(priority)
              ));
            } else {
              newDimShardings.push_back(dimSharding);
            }
            ++dimIdx;
          }
          
          auto newShardingAttr = sdy::TensorShardingAttr::get(
              ctx,
              shardingAttr.getMeshOrRef(),
              newDimShardings,
              shardingAttr.getReplicatedAxes(),
              shardingAttr.getUnreducedAxes()
          );
          
          funcOp.setArgAttr(i, "sdy.sharding", newShardingAttr);
        }
      });
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();

    context->loadDialect<mlir::stablehlo::StablehloDialect>();
    // Register for stablehlo.CustomCallOp
    mlir::stablehlo::CustomCallOp::attachInterface<
        StablehloCustomCallShardingModel>(*context);
  }
};

} // namespace mlir::tt::stablehlo
