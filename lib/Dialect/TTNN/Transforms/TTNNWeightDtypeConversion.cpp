// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/WeightDtypeParser.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWEIGHTDTYPECONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Check if the given DataType is a legal per-arg weight dtype override.
static bool isLegalWeightDtype(ttcore::DataType dtype) {
  return dtype == ttcore::DataType::BFP_BFloat8 ||
         dtype == ttcore::DataType::BFP_BFloat4 ||
         dtype == ttcore::DataType::BFloat16;
}

// Template pattern to rewrite Matmul/Linear operations to use a specified
// weight dtype. This pattern matches ops where the weight (B operand) is
// bf16/f32, and inserts a typecast operation to convert it to the target dtype.
//
// The effective dtype is resolved per-op: if the op carries a
// "ttcore.weight_dtype" discardable attribute (propagated from function args
// by the TTIRPropagateWeightDtype pass), that per-op dtype takes priority.
// Otherwise the global targetDtype (from the pass option) is used.
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

    // Resolve effective dtype: per-op annotation takes priority over global.
    std::optional<ttcore::DataType> effectiveDtype = targetDtype;
    if (auto dtypeStrAttr = op->template getAttrOfType<mlir::StringAttr>(
            "ttcore.weight_dtype")) {
      auto perOpDtype = ttcore::DataTypeStringToEnum(dtypeStrAttr.getValue());
      if (perOpDtype && isLegalWeightDtype(*perOpDtype)) {
        effectiveDtype = perOpDtype;
      } else {
        op.emitWarning("ignoring invalid ttcore.weight_dtype \"")
            << dtypeStrAttr.getValue()
            << "\"; legal values are: bfp_bf8, bfp_bf4, bf16";
      }
    }

    // No per-op annotation and no global dtype.
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
    auto typecastOp = TypecastOp::create(
        rewriter, op.getLoc(), newWeightType, weight,
        ttcore::DataTypeAttr::get(rewriter.getContext(), dtype));

    // Update op to use the typecast result and clean up the attribute.
    rewriter.modifyOpInPlace(op, [&]() {
      op.getBMutable().assign(typecastOp.getResult());
      op->removeAttr("ttcore.weight_dtype");
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

  // Maps the global pipeline WeightDtype enum to a DataType. Only block
  // formats (bfp_bf8, bfp_bf4) are supported as global overrides because
  // models already arrive from frontends in bf16 — there is no need for a
  // global bf16 override. Per-tensor overrides (via ttcore.weight_dtype op
  // attribute) do support bf16 to let individual weights opt out of a
  // global block-format conversion.
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
                 WeightDtypeConversionPattern<LinearOp>,
                 WeightDtypeConversionPattern<SparseMatmulOp>>(&getContext(),
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
