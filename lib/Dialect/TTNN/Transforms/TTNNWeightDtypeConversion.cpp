// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/WeightDtypeParser.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWEIGHTDTYPECONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Resolve the effective dtype for a weight value. Checks per-arg
// "ttcore.weight_dtype" string attributes on the function arguments the weight
// traces to. Per-arg annotations take priority over the global fallback.
static std::optional<ttcore::DataType>
resolveWeightDtype(mlir::Value weight,
                   std::optional<ttcore::DataType> globalDtype) {
  // Walk backwards through defining ops to collect all values in the producer
  // graph, then keep only the func arguments (block arguments) the weight
  // originates from.
  auto useDefChain = ttmlir::utils::getUseDefChain(weight);
  auto blockArgs =
      ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());

  // Weight doesn't trace to any func argument.
  if (blockArgs.empty()) {
    return globalDtype;
  }

  // Navigate from block argument -> owning block -> parent op to reach the
  // FuncOp, which stores per-argument attributes.
  mlir::Block *argOwner = blockArgs.front().getOwner();
  auto funcOp =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(argOwner->getParentOp());
  if (!funcOp) {
    return globalDtype;
  }

  // Check each func argument for a "ttcore.weight_dtype" attribute.
  // NOTE: If multiple block arguments have conflicting annotations, we pick
  // the first one. In practice a weight traces to a single function argument,
  // so we don't expect conflicts.
  for (auto blockArg : blockArgs) {
    if (auto dtypeStrAttr = funcOp.getArgAttrOfType<mlir::StringAttr>(
            blockArg.getArgNumber(), "ttcore.weight_dtype")) {
      auto perArgDtype = ttcore::DataTypeStringToEnum(dtypeStrAttr.getValue());
      if (perArgDtype) {
        return perArgDtype;
      }
    }
  }

  return globalDtype;
}

// Template pattern to rewrite Matmul/Linear operations to use a specified
// weight dtype. This pattern matches ops where the weight (B operand) is
// bf16/f32, and inserts a typecast operation to convert it to the target dtype.
//
// The effective dtype is resolved per-op: if the weight traces to a function
// argument with a "ttcore.weight_dtype" attribute, that per-arg dtype takes
// priority. Otherwise the global targetDtype (from the pass option) is used.
template <typename OpTy>
class WeightDtypeConversionPattern : public mlir::OpRewritePattern<OpTy> {
public:
  WeightDtypeConversionPattern(mlir::MLIRContext *ctx,
                               std::optional<ttcore::DataType> targetDtype)
      : mlir::OpRewritePattern<OpTy>(ctx), targetDtype(targetDtype) {}

  mlir::LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {
    // Get the weight operand (B operand).
    auto weight = op.getB();

    // Check if weight traces to constant/parameter arguments.
    if (!ttcore::valueTracesToConstantArgs(weight)) {
      return mlir::failure();
    }

    // Resolve effective dtype: per-arg annotation takes priority over global.
    auto effectiveDtype = resolveWeightDtype(weight, targetDtype);
    // No per-arg annotation and no global dtype.
    if (!effectiveDtype) {
      return mlir::failure();
    }

    ttcore::DataType dtype = *effectiveDtype;
    auto weightType = weight.getType();

    // Check if weight is already the target dtype or is convertible (bf16/f32).
    mlir::Type elType = weightType.getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == dtype) {
        return mlir::failure();
      }
      if (tileType.getDataType() != ttcore::DataType::BFloat16 &&
          tileType.getDataType() != ttcore::DataType::Float32) {
        return mlir::failure();
      }
    } else {
      if (!mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
        return mlir::failure();
      }
      // Already the target dtype — no conversion needed.
      if ((dtype == ttcore::DataType::BFloat16 &&
           mlir::isa<mlir::BFloat16Type>(elType)) ||
          (dtype == ttcore::DataType::Float32 &&
           mlir::isa<mlir::Float32Type>(elType))) {
        return mlir::failure();
      }
    }

    // Create new tensor type with target element type.
    auto newWeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType, dtype);

    // Insert typecast operation to convert weight to target dtype.
    auto typecastOp = rewriter.create<TypecastOp>(
        op.getLoc(), newWeightType, weight,
        ttcore::DataTypeAttr::get(rewriter.getContext(), dtype));

    // Update op to use the typecast result.
    rewriter.modifyOpInPlace(op, [&]() {
      // B is operand 1.
      op.getBMutable().assign(typecastOp.getResult());
    });

    return mlir::success();
  }

private:
  std::optional<ttcore::DataType> targetDtype;
};

class TTNNWeightDtypeConversionPass
    : public impl::TTNNWeightDtypeConversionBase<
          TTNNWeightDtypeConversionPass> {
public:
  using impl::TTNNWeightDtypeConversionBase<
      TTNNWeightDtypeConversionPass>::TTNNWeightDtypeConversionBase;

  static ttcore::DataType weightDtypeToDataType(WeightDtype wd) {
    switch (wd) {
    case WeightDtype::BFP_BFloat8:
      return ttcore::DataType::BFP_BFloat8;
    case WeightDtype::BFP_BFloat4:
      return ttcore::DataType::BFP_BFloat4;
    default:
      llvm_unreachable("Invalid WeightDtype for conversion");
    }
  }

  void runOnOperation() final {
    // Resolve global target dtype (std::nullopt if not specified).
    std::optional<ttcore::DataType> globalDtype;
    if (targetDtype != WeightDtype::None) {
      globalDtype = weightDtypeToDataType(targetDtype);
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<WeightDtypeConversionPattern<MatmulOp>,
                 WeightDtypeConversionPattern<LinearOp>>(&getContext(),
                                                         globalDtype);

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
