// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWEIGHTBFP8CONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Pattern to rewrite Conv2d operations with prepared weights to use BFP8.
// This pattern matches Conv2d ops where the weight is produced by
// PrepareConv2dWeightsOp, and modifies operations to use BFP8 by setting
// weights_dtype in Conv2dConfigAttr for PrepareConv2dWeightsOp,
// PrepareConv2dBiasOp (if present), and Conv2dOp.
class Conv2dBFP8WeightsPattern : public mlir::OpRewritePattern<Conv2dOp> {
public:
  using OpRewritePattern<Conv2dOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Conv2dOp conv2dOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isConvertible(conv2dOp)) {
      return failure();
    }

    // Get the PrepareConv2dWeightsOp.
    auto prepareWeightsOp =
        conv2dOp.getWeight().getDefiningOp<PrepareConv2dWeightsOp>();
    assert(prepareWeightsOp && "isConvertible should have checked this");

    // Create BFP8 data type.
    auto bfp8DataType = ttcore::DataType::BFP_BFloat8;

    // Update PrepareConv2dWeightsOp's conv2d_config to set weights_dtype to
    // bfp_bf8.
    Conv2dConfigAttr prepareWeightsConfig =
        prepareWeightsOp.getConv2dConfigAttr()
            ? prepareWeightsOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::get(rewriter.getContext());

    prepareWeightsConfig = prepareWeightsConfig.withWeightsDtype(bfp8DataType);

    rewriter.modifyOpInPlace(prepareWeightsOp, [&]() {
      prepareWeightsOp.setConv2dConfigAttr(prepareWeightsConfig);
    });

    // Update PrepareConv2dBiasOp's conv2d_config if bias is present.
    if (conv2dOp.getBias()) {
      auto prepareBiasOp =
          conv2dOp.getBias().getDefiningOp<PrepareConv2dBiasOp>();
      if (prepareBiasOp) {
        Conv2dConfigAttr prepareBiasConfig =
            prepareBiasOp.getConv2dConfigAttr()
                ? prepareBiasOp.getConv2dConfigAttr()
                : Conv2dConfigAttr::get(rewriter.getContext());

        prepareBiasConfig = prepareBiasConfig.withWeightsDtype(bfp8DataType);

        rewriter.modifyOpInPlace(prepareBiasOp, [&]() {
          prepareBiasOp.setConv2dConfigAttr(prepareBiasConfig);
        });
      }
    }

    // Update Conv2d's conv2d_config to set weights_dtype to bfp_bf8.
    Conv2dConfigAttr conv2dConfig =
        conv2dOp.getConv2dConfigAttr()
            ? conv2dOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::get(rewriter.getContext());

    conv2dConfig = conv2dConfig.withWeightsDtype(bfp8DataType);

    rewriter.modifyOpInPlace(
        conv2dOp, [&]() { conv2dOp.setConv2dConfigAttr(conv2dConfig); });

    return mlir::success();
  }

private:
  bool isConvertible(Conv2dOp conv2dOp) const {
    // Get the weight operand.
    auto weight = conv2dOp.getWeight();

    // Check if weight is produced by PrepareConv2dWeightsOp.
    auto prepareOp = weight.getDefiningOp<PrepareConv2dWeightsOp>();
    if (!prepareOp) {
      return false;
    }

    // Check if already converted to BFP8.
    if (prepareOp.getConv2dConfigAttr()) {
      auto config = prepareOp.getConv2dConfigAttr();
      if (config.getWeightsDtype() &&
          config.getWeightsDtype().value() == ttcore::DataType::BFP_BFloat8) {
        return false;
      }
    }

    return true;
  }
};

// Template pattern to rewrite Matmul/Linear operations to use BFP8 weights.
// This pattern matches ops where the weight (B operand) is bf16/f32,
// and inserts a typecast operation to convert it to BFP8.
template <typename OpTy>
class MatmulLinearBFP8WeightsPattern : public mlir::OpRewritePattern<OpTy> {
public:
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {
    // Get the weight operand (B operand).
    auto weight = op.getB();

    // Check if weight traces to constant/parameter arguments.
    if (!ttcore::valueTracesToConstantArgs(weight)) {
      return mlir::failure();
    }

    auto weightType = mlir::cast<mlir::RankedTensorType>(weight.getType());

    // Check if the tensor has a TTNN layout encoding.
    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(weightType.getEncoding());
    if (!currentLayout) {
      return mlir::failure(); // Not a TTNN tensor.
    }

    // Check if weight is already BFP8 or is convertible (bf16/f32).
    mlir::Type elType = weightType.getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == ttcore::DataType::BFP_BFloat8) {
        return mlir::failure(); // Already converted.
      }
      // Check if it's bf16 or f32 tile type.
      if (tileType.getDataType() != ttcore::DataType::BFloat16 &&
          tileType.getDataType() != ttcore::DataType::Float32) {
        return mlir::failure(); // Only convert float types.
      }
    } else {
      // Check if weight is bf16 or f32 scalar type.
      if (!mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
        return mlir::failure(); // Only convert float types.
      }
    }

    // Create BFP8 data type.
    auto bfp8DataType = ttcore::DataType::BFP_BFloat8;

    // Create new tensor type with BFP8 element type using
    // RankedTensorTypeFactory.
    auto bfp8WeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType, bfp8DataType);

    // Insert typecast operation to convert weight to BFP8.
    auto typecastOp = rewriter.create<TypecastOp>(
        op.getLoc(), bfp8WeightType, weight,
        ttcore::DataTypeAttr::get(rewriter.getContext(), bfp8DataType));

    // Update op to use the typecast result.
    rewriter.modifyOpInPlace(op, [&]() {
      op.getBMutable().assign(typecastOp.getResult()); // B is operand 1.
    });

    return mlir::success();
  }
};

class TTNNWeightBFP8ConversionPass
    : public impl::TTNNWeightBFP8ConversionBase<TTNNWeightBFP8ConversionPass> {
public:
  using impl::TTNNWeightBFP8ConversionBase<
      TTNNWeightBFP8ConversionPass>::TTNNWeightBFP8ConversionBase;

  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns
        .add<Conv2dBFP8WeightsPattern, MatmulLinearBFP8WeightsPattern<MatmulOp>,
             MatmulLinearBFP8WeightsPattern<LinearOp>>(&getContext());

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
