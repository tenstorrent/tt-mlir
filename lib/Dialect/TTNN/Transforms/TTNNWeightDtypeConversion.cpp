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
#define GEN_PASS_DEF_TTNNWEIGHTDTYPECONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Template pattern to rewrite Matmul/Linear operations to use a specified
// weight dtype. This pattern matches ops where the weight (B operand) is
// bf16/f32, and inserts a typecast operation to convert it to the target dtype.
template <typename OpTy>
class WeightDtypeConversionPattern : public mlir::OpRewritePattern<OpTy> {
public:
  WeightDtypeConversionPattern(mlir::MLIRContext *ctx,
                               ttcore::DataType targetDtype)
      : mlir::OpRewritePattern<OpTy>(ctx), targetDtype(targetDtype) {}

  mlir::LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {
    // Get the weight operand (B operand).
    auto weight = op.getB();

    // Check if weight traces to constant/parameter arguments.
    if (!ttcore::valueTracesToConstantArgs(weight)) {
      return mlir::failure();
    }

    auto weightType = weight.getType();

    // Check if weight is already the target dtype or is convertible (bf16/f32).
    mlir::Type elType = weightType.getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == targetDtype) {
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

    // Create new tensor type with target element type.
    auto newWeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType, targetDtype);

    // Insert typecast operation to convert weight to target dtype.
    auto typecastOp = rewriter.create<TypecastOp>(
        op.getLoc(), newWeightType, weight,
        ttcore::DataTypeAttr::get(rewriter.getContext(), targetDtype));

    // Update op to use the typecast result.
    rewriter.modifyOpInPlace(op, [&]() {
      op.getBMutable().assign(typecastOp.getResult()); // B is operand 1.
    });

    return mlir::success();
  }

private:
  ttcore::DataType targetDtype;
};

class TTNNWeightDtypeConversionPass
    : public impl::TTNNWeightDtypeConversionBase<
          TTNNWeightDtypeConversionPass> {
public:
  using impl::TTNNWeightDtypeConversionBase<
      TTNNWeightDtypeConversionPass>::TTNNWeightDtypeConversionBase;

  void runOnOperation() final {
    if (targetDtype.empty()) {
      getOperation()->emitError(
          "target-dtype must be specified for weight dtype conversion pass");
      signalPassFailure();
      return;
    }

    auto dtype = ttcore::DataTypeStringToEnum(targetDtype);
    if (!dtype) {
      getOperation()->emitError("Invalid target-dtype: " + targetDtype);
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<WeightDtypeConversionPattern<MatmulOp>,
                 WeightDtypeConversionPattern<LinearOp>>(&getContext(), *dtype);

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
