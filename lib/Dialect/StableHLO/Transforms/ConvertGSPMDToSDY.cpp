// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#include "shardy/dialect/sdy/ir/dialect.h"
#pragma clang diagnostic pop

namespace mlir::tt::stablehlo {

#define GEN_PASS_DEF_CONVERTGSPMDTOSDYPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Define the sharding group custom call target name (based on user's code snippet)
inline constexpr llvm::StringRef kShardingGroupCustomCallTargetName = "ShardingGroup";

/// Helper function to extract sdy.sharding attribute from a value
static mlir::sdy::TensorShardingAttr getSharding(mlir::Value value) {
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    // For block arguments, look at the parent operation's argument attributes
    if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      if (auto argAttrDict = funcOp.getArgAttrDict(blockArg.getArgNumber())) {
        return argAttrDict.getAs<mlir::sdy::TensorShardingAttr>(
            mlir::sdy::TensorShardingAttr::name);
      }
    }
    return nullptr;
  }

  // For operation results, look at the operation's result attributes
  if (auto opResult = llvm::dyn_cast<mlir::OpResult>(value)) {
    mlir::Operation *op = opResult.getOwner();

    // First check for direct sdy.sharding attribute
    if (auto shardingPerValue = op->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
            mlir::sdy::TensorShardingAttr::name)) {
      auto shardings = shardingPerValue.getShardings();
      if (opResult.getResultNumber() < shardings.size()) {
        return shardings[opResult.getResultNumber()];
      }
    }

    // Check for sharding in mhlo.frontend_attributes.xla.sdy.sharding
    if (auto frontendAttrs = op->getAttrOfType<mlir::DictionaryAttr>("mhlo.frontend_attributes")) {
      if (auto shardingStr = frontendAttrs.getAs<mlir::StringAttr>("xla.sdy.sharding")) {
        std::string shardingValue = shardingStr.getValue().str();

        // Handle sdy.sharding_per_value format: "#sdy.sharding_per_value<[<...>]>"
        if (shardingValue.find("#sdy.sharding_per_value") != std::string::npos) {
          // Extract the first sharding from the per_value format
          size_t startPos = shardingValue.find("[<@");
          size_t endPos = shardingValue.find(">]>");
          if (startPos != std::string::npos && endPos != std::string::npos) {
            // Extract just the first sharding: "<@mesh, [{}, {"_axis_0"}, {}, {}]>"
            std::string firstSharding = shardingValue.substr(startPos + 1, endPos - startPos);

            // Parse this as a TensorShardingAttr using MLIR's parser
            mlir::MLIRContext *context = op->getContext();
            std::string parseStr = "#sdy.sharding" + firstSharding;

            auto shardingAttr = mlir::parseAttribute(parseStr, context);
            if (auto tensorSharding = llvm::dyn_cast_if_present<mlir::sdy::TensorShardingAttr>(shardingAttr)) {
              return tensorSharding;
            }
          }
        }
      }
    }
  }

  return nullptr;
}

namespace {

/// Pattern to convert GSPMD sharding custom calls to SDY sharding constraints
class SdyCustomCallPattern : public OpRewritePattern<mlir::stablehlo::CustomCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() == mlir::tt::gspmd_utils::kShardingCustomCallTargetName) {
      return rewriteShardingCustomCall(op, rewriter);
    }

    if (op.getCallTargetName() == kShardingGroupCustomCallTargetName) {
      return rewriteShardingGroupCustomCall(op, rewriter);
    }

    return rewriter.notifyMatchFailure(
        op, "expected CustomCallOp with xla.sdy target name.");
  }

private:
  LogicalResult rewriteShardingCustomCall(mlir::stablehlo::CustomCallOp op,
                                          PatternRewriter &rewriter) const {
    if (op->getNumResults() != 1) {
      op.emitError() << "expected CustomCallOp with exactly one result";
      return failure();
    }

    mlir::sdy::TensorShardingAttr sharding = getSharding(op->getResult(0));
    if (!sharding) {
      op.emitError() << "expected CustomCallOp with a sharding attribute";
      return failure();
    }

    // Replace the custom call with a sharding constraint
    rewriter.replaceOpWithNewOp<mlir::sdy::ShardingConstraintOp>(
        op, op.getInputs().front(), sharding);

    return success();
  }

  LogicalResult rewriteShardingGroupCustomCall(mlir::stablehlo::CustomCallOp op,
                                               PatternRewriter &rewriter) const {
    // For now, treat sharding group similar to regular sharding
    // This can be extended based on specific requirements
    return rewriteShardingCustomCall(op, rewriter);
  }
};

class ConvertGSPMDToSDYPass : public impl::ConvertGSPMDToSDYPassBase<ConvertGSPMDToSDYPass> {
public:
  using impl::ConvertGSPMDToSDYPassBase<ConvertGSPMDToSDYPass>::ConvertGSPMDToSDYPassBase;

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add the pattern to convert GSPMD custom calls to SDY sharding constraints
    patterns.add<SdyCustomCallPattern>(context);

    // Apply the patterns greedily
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::stablehlo