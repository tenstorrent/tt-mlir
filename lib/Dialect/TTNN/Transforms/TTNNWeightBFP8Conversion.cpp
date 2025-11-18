// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWEIGHTBFP8CONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Pattern to rewrite Conv2d operations with prepared weights to use BFP8.
// This pattern matches Conv2d ops where the weight is produced by
// PrepareConv2dWeightsOp, and modifies both operations to use BFP8:
// 1. Sets output_dtype on PrepareConv2dWeightsOp to bfp_bf8
// 2. Sets weights_dtype in Conv2dConfigAttr to bfp_bf8
class Conv2dBFP8WeightsPattern : public mlir::OpRewritePattern<Conv2dOp> {
public:
  using OpRewritePattern<Conv2dOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Conv2dOp conv2dOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isConvertible(conv2dOp)) {
      return failure();
    }

    // Get the PrepareConv2dWeightsOp
    auto prepareOp =
        conv2dOp.getWeight().getDefiningOp<PrepareConv2dWeightsOp>();
    assert(prepareOp && "isConvertible should have checked this");

    // Create BFP8 data type
    auto bfp8DataType = ttcore::DataType::BFP_BFloat8;

    // Update PrepareConv2dWeightsOp to set output_dtype to bfp_bf8
    // and update the result type to reflect BFP8
    rewriter.modifyOpInPlace(prepareOp, [&]() {
      prepareOp.setOutputDtypeAttr(
          ttcore::DataTypeAttr::get(rewriter.getContext(), bfp8DataType));

      // Update the result type to use BFP8 tile type
      auto currentResultType =
          mlir::cast<mlir::RankedTensorType>(prepareOp.getResult().getType());

      // Create BFP8 tile element type
      mlir::Type bfp8TileType = ttcore::TileType::get(
          rewriter.getContext(), ttcore::TileType::getDefaultShape(),
          bfp8DataType);

      // Update the TTNNLayoutAttr encoding with the new element type
      auto currentLayout =
          mlir::cast<TTNNLayoutAttr>(currentResultType.getEncoding());
      auto newLayout = currentLayout.withElementType(
          bfp8TileType, currentResultType.getShape());

      // Create new tensor type with BFP8 element type and updated encoding
      auto newResultType = mlir::RankedTensorType::get(
          currentResultType.getShape(), bfp8TileType, newLayout);

      // Set the new type on the result
      prepareOp.getResult().setType(newResultType);
    });

    // Update Conv2d's conv2d_config to set weights_dtype to bfp_bf8
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
    // Get the weight operand
    mlir::Value weight = conv2dOp.getWeight();

    // Check if weight is produced by PrepareConv2dWeightsOp
    auto prepareOp = weight.getDefiningOp<PrepareConv2dWeightsOp>();
    if (!prepareOp) {
      return false;
    }

    // Check if already converted to BFP8
    if (prepareOp.getOutputDtype()) {
      auto outputDtype = prepareOp.getOutputDtype().value();
      if (outputDtype == ttcore::DataType::BFP_BFloat8) {
        return false;
      }
    }

    return true;
  }
};

// Pattern to rewrite Matmul operations to use BFP8 weights.
// This pattern matches Matmul ops where the weight (B operand) is bf16/f32,
// and inserts a typecast operation to convert it to BFP8.
class MatmulBFP8WeightsPattern : public mlir::OpRewritePattern<MatmulOp> {
public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatmulOp matmulOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Get the weight operand (B operand)
    mlir::Value weight = matmulOp.getB();
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
    if (!weightType) {
      return failure();
    }

    // Check if the tensor has a TTNN layout encoding
    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(weightType.getEncoding());
    if (!currentLayout) {
      return failure(); // Not a TTNN tensor
    }

    // Check if weight is already BFP8 or is convertible (bf16/f32)
    mlir::Type elType = weightType.getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == ttcore::DataType::BFP_BFloat8) {
        return failure(); // Already converted
      }
      // Check if it's bf16 or f32 tile type
      if (tileType.getDataType() != ttcore::DataType::BFloat16 &&
          tileType.getDataType() != ttcore::DataType::Float32) {
        return failure(); // Only convert float types
      }
    } else {
      // Check if weight is bf16 or f32 scalar type
      if (!mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
        return failure(); // Only convert float types
      }
    }

    // Create BFP8 data type
    auto bfp8DataType = ttcore::DataType::BFP_BFloat8;
    mlir::Type bfp8TileType = ttcore::TileType::get(
        rewriter.getContext(), ttcore::TileType::getDefaultShape(),
        bfp8DataType);

    // Update the TTNNLayoutAttr encoding with the new element type
    auto newLayout =
        currentLayout.withElementType(bfp8TileType, weightType.getShape());

    // Create new tensor type with BFP8 element type and updated encoding
    auto bfp8WeightType = mlir::RankedTensorType::get(weightType.getShape(),
                                                      bfp8TileType, newLayout);

    // Insert typecast operation to convert weight to BFP8
    auto typecastOp = rewriter.create<TypecastOp>(
        matmulOp.getLoc(), bfp8WeightType, weight,
        ttcore::DataTypeAttr::get(rewriter.getContext(), bfp8DataType));

    // Update matmul to use the typecast result
    rewriter.modifyOpInPlace(matmulOp, [&]() {
      matmulOp.setOperand(1, typecastOp.getResult()); // B is operand 1
    });

    return mlir::success();
  }
};

// Pattern to rewrite Linear operations to use BFP8 weights.
// This pattern matches Linear ops where the weight (B operand) is bf16/f32,
// and inserts a typecast operation to convert it to BFP8.
class LinearBFP8WeightsPattern : public mlir::OpRewritePattern<LinearOp> {
public:
  using OpRewritePattern<LinearOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LinearOp linearOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Get the weight operand (B operand)
    mlir::Value weight = linearOp.getB();
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
    if (!weightType) {
      return failure();
    }

    // Check if the tensor has a TTNN layout encoding
    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(weightType.getEncoding());
    if (!currentLayout) {
      return failure(); // Not a TTNN tensor
    }

    // Check if weight is already BFP8 or is convertible (bf16/f32)
    mlir::Type elType = weightType.getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == ttcore::DataType::BFP_BFloat8) {
        return failure(); // Already converted
      }
      // Check if it's bf16 or f32 tile type
      if (tileType.getDataType() != ttcore::DataType::BFloat16 &&
          tileType.getDataType() != ttcore::DataType::Float32) {
        return failure(); // Only convert float types
      }
    } else {
      // Check if weight is bf16 or f32 scalar type
      if (!mlir::isa<mlir::BFloat16Type, mlir::Float32Type>(elType)) {
        return failure(); // Only convert float types
      }
    }

    // Create BFP8 data type
    auto bfp8DataType = ttcore::DataType::BFP_BFloat8;
    mlir::Type bfp8TileType = ttcore::TileType::get(
        rewriter.getContext(), ttcore::TileType::getDefaultShape(),
        bfp8DataType);

    // Update the TTNNLayoutAttr encoding with the new element type
    auto newLayout =
        currentLayout.withElementType(bfp8TileType, weightType.getShape());

    // Create new tensor type with BFP8 element type and updated encoding
    auto bfp8WeightType = mlir::RankedTensorType::get(weightType.getShape(),
                                                      bfp8TileType, newLayout);

    // Insert typecast operation to convert weight to BFP8
    auto typecastOp = rewriter.create<TypecastOp>(
        linearOp.getLoc(), bfp8WeightType, weight,
        ttcore::DataTypeAttr::get(rewriter.getContext(), bfp8DataType));

    // Update linear to use the typecast result
    rewriter.modifyOpInPlace(linearOp, [&]() {
      linearOp.setOperand(1, typecastOp.getResult()); // B is operand 1
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
    // Only run if the flag is enabled
    if (!experimentalBfp8Weights) {
      return;
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<Conv2dBFP8WeightsPattern, MatmulBFP8WeightsPattern,
                 LinearBFP8WeightsPattern>(&getContext());

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
