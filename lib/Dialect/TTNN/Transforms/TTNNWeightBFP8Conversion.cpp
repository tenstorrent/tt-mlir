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

    auto weightType = weight.getType();

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
    auto typecastOp = TypecastOp::create(
        rewriter, op.getLoc(), bfp8WeightType, weight,
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
    patterns.add<MatmulLinearBFP8WeightsPattern<MatmulOp>,
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
