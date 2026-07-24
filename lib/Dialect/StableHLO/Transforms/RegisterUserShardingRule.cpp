// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REGISTERUSERSHARDINGRULEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class RegisterUserShardingRulePass
    : public impl::RegisterUserShardingRulePassBase<
          RegisterUserShardingRulePass> {
public:
  using impl::RegisterUserShardingRulePassBase<
      RegisterUserShardingRulePass>::RegisterUserShardingRulePassBase;

  void runOnOperation() final {
    getOperation().walk([&](mlir::stablehlo::CustomCallOp op) {
      auto frontendAttrs = op->getAttrOfType<mlir::DictionaryAttr>(
          gspmd_utils::kFrontendAttributesAttr);
      if (!frontendAttrs) {
        return;
      }

      // If attribute is defined but empty, fall back to Shardy's default
      // handling (replication)
      auto ruleStrAttr = frontendAttrs.getAs<mlir::StringAttr>(
          sharding_utils::kXlaSdyCustomShardingRuleAttr);
      if (!ruleStrAttr || ruleStrAttr.getValue().trim().empty()) {
        return;
      }

      auto rule = mlir::dyn_cast_or_null<mlir::sdy::OpShardingRuleAttr>(
          mlir::parseAttribute(ruleStrAttr.getValue(), &getContext()));
      if (!rule) {
        op.emitError() << "failed to parse '"
                       << sharding_utils::kXlaSdyCustomShardingRuleAttr
                       << "' frontend attribute as an #sdy.op_sharding_rule";
        signalPassFailure();
        return;
      }

      op->setAttr(mlir::sdy::kShardingRuleAttr, rule);
    });
  }
};

} // namespace mlir::tt::stablehlo
