// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tt {
namespace ttir_to_ttnn::utils {
ttnn::ReshapeOp generateReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                                ArrayRef<int64_t> newShape,
                                PatternRewriter &rewriter,
                                mlir::Location newLoc) {
  // With reshape op, the output layout changes due to new output shape, hence
  // we need to create a new output layout attribute with the new shape.
  RankedTensorType inputType = input.getType();
  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, newShape);

  llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
  return rewriter.create<ttnn::ReshapeOp>(
      newLoc, outputType, input, rewriter.getI32ArrayAttr(newShapeI32));
}

ttnn::ReshapeOp
generateNHWFlatten(mlir::TypedValue<mlir::RankedTensorType> input,
                   PatternRewriter &rewriter, mlir::Location newLoc) {
  llvm::ArrayRef<int64_t> shape = input.getType().getShape();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  llvm::SmallVector<int64_t> newShape = {1, 1, shape[0] * shape[1] * shape[2],
                                         shape[3]};
  return generateReshape(input, newShape, rewriter, newLoc);
}

ttnn::PermuteOp generatePermute(mlir::TypedValue<mlir::RankedTensorType> input,
                                ArrayRef<int64_t> permutation,
                                PatternRewriter &rewriter,
                                mlir::Location newLoc) {
  RankedTensorType inputType = input.getType();
  llvm::SmallVector<int64_t> outputShape =
      ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, outputShape);

  return rewriter.create<ttnn::PermuteOp>(
      newLoc, outputType, input, rewriter.getDenseI64ArrayAttr(permutation),
      /* pad_value */ mlir::FloatAttr());
}

ttnn::PadOp generatePad(mlir::TypedValue<mlir::RankedTensorType> input,
                        ArrayRef<int32_t> padding, PatternRewriter &rewriter,
                        mlir::Location newLoc) {
  RankedTensorType inputType = input.getType();
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();

  assert(padding.size() == inputShape.size() * 2 &&
         "Padding must have 2 values per dimension");
  auto indices = llvm::seq<size_t>(0, inputShape.size());
  llvm::SmallVector<int64_t> outputShape =
      llvm::to_vector(llvm::map_range(indices, [&](size_t i) {
        return inputShape[i] + padding[2 * i] + padding[2 * i + 1];
      }));

  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, outputShape);

  return rewriter.create<ttnn::PadOp>(
      newLoc, outputType, input, rewriter.getDenseI32ArrayAttr(padding),
      rewriter.getF32FloatAttr(0.0f), rewriter.getBoolAttr(true));
}

mlir::Value decomposeGroupNorm(
    mlir::TypedValue<mlir::RankedTensorType> input, mlir::Value weight,
    mlir::Value bias, uint32_t numGroups, llvm::APFloat epsilon,
    RankedTensorType resultType, mlir::Operation *deviceAnchorOp,
    PatternRewriter &rewriter, mlir::Location loc) {
  RankedTensorType inputType = input.getType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  assert(inputShape.size() == 4 &&
         "decomposeGroupNorm expects canonical [N, 1, H*W, C] input");

  const int64_t N = inputShape[0];
  const int64_t S = inputShape[2];
  const int64_t C = inputShape[3];
  const int64_t G = numGroups;
  const int64_t Cpg = C / G;

  // Reshape [N, 1, S, C] -> [N, S, G, Cpg] so per-group reductions happen over
  // a contiguous (S, Cpg) sub-tensor (reduce over dims 1 and 3).
  SmallVector<int64_t> groupedShape = {N, S, G, Cpg};
  mlir::Value grouped =
      generateReshape(
          input, groupedShape, rewriter,
          ttmlir::utils::appendLocationSuffix(loc, "_group_reshape"))
          .getResult();
  RankedTensorType groupedType =
      mlir::cast<RankedTensorType>(grouped.getType());

  SmallVector<int64_t> statsShape = {N, 1, G, 1};
  RankedTensorType statsType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, statsShape);
  ArrayAttr reduceDims = rewriter.getI32ArrayAttr({1, 3});

  // mean = mean(grouped, dims=[1,3], keep_dim=true)
  auto meanOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_mean"), statsType, grouped,
      /*keep_dim=*/true, reduceDims);

  // centered = grouped - mean
  auto centeredOp = rewriter.create<ttnn::SubtractOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_center"), groupedType, grouped,
      meanOp.getResult());

  // variance = mean(centered * centered, dims=[1,3], keep_dim=true)
  auto squaredOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_square"), groupedType,
      centeredOp.getResult(), centeredOp.getResult());
  auto varianceOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_variance"), statsType,
      squaredOp.getResult(), /*keep_dim=*/true, reduceDims);

  // inv_std = rsqrt(variance + eps)
  auto deviceOp = ttnn::utils::getOrInsertDevice(rewriter, deviceAnchorOp);
  auto epsTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_eps"), statsType,
      rewriter.getF32FloatAttr(epsilon.convertToFloat()), deviceOp.getResult());
  auto stabilizedOp = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_add_eps"), statsType,
      varianceOp.getResult(), epsTensor.getResult());
  auto invStdOp = rewriter.create<ttnn::RsqrtOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_rsqrt"), statsType,
      stabilizedOp.getResult());

  // normalized = centered * inv_std
  auto normalizedOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_normalize"), groupedType,
      centeredOp.getResult(), invStdOp.getResult());

  // Restore the original [N, 1, S, C] shape.
  mlir::Value result =
      generateReshape(
          mlir::cast<mlir::TypedValue<RankedTensorType>>(
              normalizedOp.getResult()),
          inputShape, rewriter,
          ttmlir::utils::appendLocationSuffix(loc, "_unreshape"))
          .getResult();

  // Per-channel affine params materialized as 1-D [C]; reshape to [1,1,1,C] so
  // the broadcast is explicit.
  SmallVector<int64_t> affineShape = {1, 1, 1, C};
  if (weight) {
    mlir::Value reshapedWeight =
        generateReshape(
            mlir::cast<mlir::TypedValue<RankedTensorType>>(weight), affineShape,
            rewriter,
            ttmlir::utils::appendLocationSuffix(loc, "_weight_reshape"))
            .getResult();
    result = rewriter
                 .create<ttnn::MultiplyOp>(
                     ttmlir::utils::appendLocationSuffix(loc, "_weight_mul"),
                     resultType, result, reshapedWeight)
                 .getResult();
  }
  if (bias) {
    mlir::Value reshapedBias =
        generateReshape(
            mlir::cast<mlir::TypedValue<RankedTensorType>>(bias), affineShape,
            rewriter, ttmlir::utils::appendLocationSuffix(loc, "_bias_reshape"))
            .getResult();
    result = rewriter
                 .create<ttnn::AddOp>(
                     ttmlir::utils::appendLocationSuffix(loc, "_bias_add"),
                     resultType, result, reshapedBias)
                 .getResult();
  }

  return result;
}
} // namespace ttir_to_ttnn::utils
} // namespace tt
} // namespace mlir
