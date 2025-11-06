// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <numeric>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

//===----------------------------------------------------------------------===//
// IndexOp decomposition
//===----------------------------------------------------------------------===//

// ANCHOR: decomposing_an_op_index_ttir_decompose_pattern
// This transformation adjusts IndexOp attributes so that `begin`, `end`, and
// `step` become arrays, where each array element corresponds to a dimension of
// the input tensor. For dimensions other than the sliced dimension, default
// values are used.
//
namespace {
struct IndexToSliceConversionPattern
    : public OpConversionPattern<ttir::IndexOp> {
  using OpConversionPattern<ttir::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType =
        ::mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!inputType || !inputType.hasRank()) {
      return failure();
    }

    int64_t rank = inputType.getRank();
    llvm::SmallVector<mlir::Attribute, 4> begins, ends, steps;

    for (int64_t i = 0; i < rank; ++i) {
      if (i == op.getDim()) {
        begins.push_back(rewriter.getI32IntegerAttr(adaptor.getBegin()));
        ends.push_back(rewriter.getI32IntegerAttr(adaptor.getEnd()));
        steps.push_back(rewriter.getI32IntegerAttr(adaptor.getStep()));
      } else {
        begins.push_back(rewriter.getI32IntegerAttr(0));
        ends.push_back(rewriter.getI32IntegerAttr(inputType.getDimSize(i)));
        steps.push_back(rewriter.getI32IntegerAttr(1));
      }
    }

    auto newOp = rewriter.create<ttir::SliceStaticOp>(
        op.getLoc(), op.getType(), adaptor.getInput(), adaptor.getOutput(),
        rewriter.getArrayAttr(begins), rewriter.getArrayAttr(ends),
        rewriter.getArrayAttr(steps));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};
} // namespace
// ANCHOR_END: decomposing_an_op_index_ttir_decompose_pattern

//===----------------------------------------------------------------------===//
// Convolution passes
//===----------------------------------------------------------------------===//

template <uint32_t NDims>
using PaddingMatrix = std::array<std::array<int64_t, 2>, NDims>;

template <uint32_t NDims>
static PaddingMatrix<NDims> getPaddingMatrix(ArrayRef<int64_t> padding) {
  assert(padding.size() >= 2 * NDims &&
         "padding must be at least 2 * NDims sized array");

  PaddingMatrix<NDims> paddingMatrix;

  for (uint32_t i = 0; i < 2 * NDims; i += 2) {
    paddingMatrix[i / 2] = {padding[i], padding[i + 1]};
  }
  return paddingMatrix;
}

namespace {
struct ConvolutionDecompositionPattern
    : public OpConversionPattern<ttir::ConvolutionOp> {
public:
  using OpConversionPattern<ttir::ConvolutionOp>::OpConversionPattern;

  //  All convolutions will have a batch and feature dimension, and the kernel
  //  will have an input and output feature dimension. The spatial dimensions
  //  can be
  // represented by non-negative integers.
  enum ConvolutionDimension { BATCH = -1, FEATURE = -2, INVALID_DIM = -3 };
  enum ConvolutionKernelDimension {
    INPUT_FEATURES = -1,
    OUTPUT_FEATURES = -2,
    INVALID_KERNEL_DIM = -3
  };

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override = 0;

protected:
  static bool isNDimensional(ttir::ConvolutionOp op, uint32_t numSpatialDims) {
    return op.getConvolutionLayout().getInputSpatialDimensions().size() ==
           numSpatialDims;
  }

  static bool isSupportedConv(ttir::ConvolutionOp op) {
    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getOutputSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");
    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getKernelSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");

    // Not currently supporting window reversal
    if (llvm::any_of(op.getWindowReversal(), ttmlir::utils::identity<bool>)) {
      return false;
    }

    return true;
  }

  // This function will generate the transpose indices needed to convert a
  // convolution input to a desired layout. The reason for the separate
  // function is to encapsulate the logic for constructuring the inputLayout.
  static llvm::SmallVector<int64_t>
  generateConvPermutation(ttir::ConvolutionOp op,
                          llvm::ArrayRef<int64_t> ttnnConvolutionLayout) {

    llvm::SmallVector<int64_t> inputLayout(ttnnConvolutionLayout.size(),
                                           ConvolutionDimension::INVALID_DIM);
    inputLayout[op.getConvolutionLayout().getInputBatchDimension()] =
        ConvolutionDimension::BATCH;
    inputLayout[op.getConvolutionLayout().getInputFeatureDimension()] =
        ConvolutionDimension::FEATURE;

    for (const auto [spatialCount, spatialDim] : llvm::enumerate(
             op.getConvolutionLayout().getInputSpatialDimensions())) {
      inputLayout[spatialDim] = spatialCount;
    }

    return ttmlir::utils::generatePermutation(llvm::ArrayRef(inputLayout),
                                              ttnnConvolutionLayout);
  }

  // This function will generate the transpose indices needed to convert a
  // convolution input to a desired layout. The reason for the separate
  // function is to encapsulate the logic for constructuring the kernelLayout.
  static llvm::SmallVector<int64_t> generateConvKernelPermutation(
      ttir::ConvolutionOp op,
      llvm::ArrayRef<int64_t> ttnnConvolutionKernelLayout) {

    llvm::SmallVector<int64_t> kernelLayout(
        ttnnConvolutionKernelLayout.size(),
        ConvolutionKernelDimension::INVALID_KERNEL_DIM);
    kernelLayout[op.getConvolutionLayout().getKernelOutputFeatureDimension()] =
        ConvolutionKernelDimension::OUTPUT_FEATURES;
    kernelLayout[op.getConvolutionLayout().getKernelInputFeatureDimension()] =
        ConvolutionKernelDimension::INPUT_FEATURES;

    for (const auto [spatialCount, spatialDim] : llvm::enumerate(
             op.getConvolutionLayout().getKernelSpatialDimensions())) {
      kernelLayout[spatialDim] = spatialCount;
    }

    return ttmlir::utils::generatePermutation(llvm::ArrayRef(kernelLayout),
                                              ttnnConvolutionKernelLayout);
  }
};
} // namespace

// Helper structure to hold sliced convolution inputs for batch group
// decomposition.
struct ConvolutionSlices {
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> weights;
  llvm::SmallVector<llvm::SmallVector<int64_t>> outputShapes;
};

// Helper function to slice inputs and weights for batch group decomposition.
static ConvolutionSlices
sliceForBatchGroups(ConversionPatternRewriter &rewriter, Location loc,
                    Value input, Value weight, Value bias,
                    mlir::tt::ttir::ConvolutionLayoutAttr convolutionLayout,
                    uint64_t groupCount, int64_t groupDimensionIndex) {
  ConvolutionSlices slices;

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> weightShape = weightType.getShape();

  int64_t kernelOutputFeatureDim =
      convolutionLayout.getKernelOutputFeatureDimension();

  int64_t inputSliceSize = inputShape[groupDimensionIndex] / groupCount;
  int64_t weightSliceSize = weightShape[kernelOutputFeatureDim] / groupCount;

  for (uint64_t i = 0; i < groupCount; ++i) {
    // Slice input.
    llvm::SmallVector<int32_t> inputBegins(inputShape.size(), 0);
    llvm::SmallVector<int32_t> inputEnds(inputShape.begin(), inputShape.end());
    llvm::SmallVector<int32_t> inputSteps(inputShape.size(), 1);
    inputBegins[groupDimensionIndex] = i * inputSliceSize;
    inputEnds[groupDimensionIndex] = (i + 1) * inputSliceSize;

    llvm::SmallVector<int64_t> inputSliceShape(inputShape.begin(),
                                               inputShape.end());
    inputSliceShape[groupDimensionIndex] = inputSliceSize;

    auto inputSlice = ttir::utils::createDPSOp<ttir::SliceStaticOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_inputSlice"),
        inputSliceShape, inputType.getElementType(), inputType.getEncoding(),
        input, rewriter.getI32ArrayAttr(inputBegins),
        rewriter.getI32ArrayAttr(inputEnds),
        rewriter.getI32ArrayAttr(inputSteps));
    slices.inputs.push_back(inputSlice);

    // Slice weight.
    llvm::SmallVector<int32_t> weightBegins(weightShape.size(), 0);
    llvm::SmallVector<int32_t> weightEnds(weightShape.begin(),
                                          weightShape.end());
    llvm::SmallVector<int32_t> weightSteps(weightShape.size(), 1);
    weightBegins[kernelOutputFeatureDim] = i * weightSliceSize;
    weightEnds[kernelOutputFeatureDim] = (i + 1) * weightSliceSize;

    llvm::SmallVector<int64_t> weightSliceShape(weightShape.begin(),
                                                weightShape.end());
    weightSliceShape[kernelOutputFeatureDim] = weightSliceSize;

    auto weightSlice = ttir::utils::createDPSOp<ttir::SliceStaticOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_weightSlice"),
        weightSliceShape, weightType.getElementType(), weightType.getEncoding(),
        weight, rewriter.getI32ArrayAttr(weightBegins),
        rewriter.getI32ArrayAttr(weightEnds),
        rewriter.getI32ArrayAttr(weightSteps));
    slices.weights.push_back(weightSlice);
  }

  return slices;
}

// A decomposition pattern that matches to a ttir.convolution op that does 1D
// convolution. Since that is not supported in ttnn, we reshape the inputs and
// the output to match a 2D ttir.convolution op. The expectation is that the new
// ttir.convolution op will be picked up by the ConvolutionToConv2dPattern and
// translated into ttir.conv2d op.
namespace {
struct Legalize1DConvolutionPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;
  constexpr static uint32_t NUM_SPATIAL_DIMS = 1;

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();
    }

    auto outputType =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    auto convolutionLayout = adaptor.getConvolutionLayoutAttr();

    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

    uint64_t batchGroupCount = adaptor.getBatchGroupCount();
    uint64_t featureGroupCount = adaptor.getFeatureGroupCount();

    // Prepare inputs/weights (slice if batchGroupCount > 1).
    llvm::SmallVector<Value> inputSlices;
    llvm::SmallVector<Value> weightSlices;
    llvm::SmallVector<llvm::SmallVector<int64_t>> outputSliceShapes;
    assert(featureGroupCount == 1 ||
           batchGroupCount == 1 &&
               "At least one of the group counts must be 1.");

    // Split (X, Y, Z) is defined as splitting X into Y groups along the Z
    // dimension. If batch_group_count > 1: lhses = split(lhs,
    // batch_group_count, input_batch_dimension). rhses = split(rhs,
    // batch_group_count, kernel_output_feature_dimension). results... =
    // convolution(lhses..., rhses..., ..., batch_group_count=1, ...). result =
    // concatenate(results, output_feature_dimension).
    if (batchGroupCount > 1) {
      auto slices = sliceForBatchGroups(
          rewriter, op.getLoc(), adaptor.getInput(), adaptor.getWeight(),
          adaptor.getBias(), convolutionLayout, batchGroupCount,
          convolutionLayout.getInputBatchDimension());
      inputSlices = std::move(slices.inputs);
      weightSlices = std::move(slices.weights);

      int64_t outputFeatureDim = convolutionLayout.getOutputFeatureDimension();
      int64_t outputSliceSize = outputShape[outputFeatureDim] / batchGroupCount;
      for (uint64_t i = 0; i < batchGroupCount; ++i) {
        llvm::SmallVector<int64_t> outputSliceShape(outputShape.begin(),
                                                    outputShape.end());
        outputSliceShape[outputFeatureDim] = outputSliceSize;
        outputSliceShapes.push_back(outputSliceShape);
      }
    } else {
      inputSlices.push_back(adaptor.getInput());
      weightSlices.push_back(adaptor.getWeight());
      outputSliceShapes.push_back(
          llvm::SmallVector<int64_t>(outputShape.begin(), outputShape.end()));
    }

    llvm::SmallVector<Value> results;
    for (size_t i = 0; i < inputSlices.size(); ++i) {
      Value result = convert1DConvTo2D(
          rewriter, op.getLoc(), inputSlices[i], weightSlices[i],
          outputSliceShapes[i], outputType.getElementType(),
          outputType.getEncoding(), adaptor, convolutionLayout);
      results.push_back(result);
    }

    if (batchGroupCount > 1) {
      int64_t outputFeatureDim = convolutionLayout.getOutputFeatureDimension();
      auto concatOp = ttir::utils::createDPSOp<ttir::ConcatOp>(
          rewriter, op.getLoc(), outputType, results, outputFeatureDim);
      rewriter.replaceOp(op, concatOp);
    } else {
      rewriter.replaceOp(op, results[0]);
    }

    return success();
  }

private:
  // Convert a 1D convolution to a 2D convolution by adding a dimension.
  Value convert1DConvTo2D(
      ConversionPatternRewriter &rewriter, Location loc, Value input,
      Value weight, llvm::ArrayRef<int64_t> expectedOutputShape,
      Type outputElementType, Attribute outputEncoding, OpAdaptor adaptor,
      mlir::tt::ttir::ConvolutionLayoutAttr convolutionLayout) const {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());

    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> weightShape = weightType.getShape();

    // Add dimension to shapes for 2D conversion.
    llvm::SmallVector<int64_t> conv2dInputShape(inputShape.begin(),
                                                inputShape.end());
    conv2dInputShape.push_back(1);
    llvm::SmallVector<int64_t> conv2dWeightShape(weightShape.begin(),
                                                 weightShape.end());
    conv2dWeightShape.push_back(1);
    llvm::SmallVector<int64_t> conv2dOutputShape(expectedOutputShape.begin(),
                                                 expectedOutputShape.end());
    conv2dOutputShape.push_back(1);

    // Reshape input and weight to 2D.
    ttir::ReshapeOp reshapeInput = createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshapeInput"),
        input, conv2dInputShape);
    ttir::ReshapeOp reshapeWeight = createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshapeWeight"),
        weight, conv2dWeightShape);

    mlir::DenseI64ArrayAttr conv2dOpWindowsStridesAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getWindowStridesAttr(), 1);
    mlir::DenseI64ArrayAttr conv2dOpPaddingAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getPaddingAttr(), 0);
    conv2dOpPaddingAttr =
        addIntegerToDenseArrayAttr(rewriter, conv2dOpPaddingAttr, 0);
    mlir::DenseI64ArrayAttr conv2dOpInputDilationAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getInputDilationAttr(), 1);
    mlir::DenseI64ArrayAttr conv2dOpWeightDilationAttr =
        addIntegerToDenseArrayAttr(rewriter, adaptor.getWeightDilationAttr(),
                                   1);
    mlir::DenseBoolArrayAttr conv2dOpWindowReversalAttr =
        addBooleanToDenseArrayAttr(rewriter, adaptor.getWindowReversalAttr(),
                                   false);

    // The additional spatial dimension is added at the end (3rd in 0 indexed
    // array).
    llvm::SmallVector<int64_t> conv2dInputSpatialDimensions(
        convolutionLayout.getInputSpatialDimensions());
    conv2dInputSpatialDimensions.push_back(3);

    llvm::SmallVector<int64_t> conv2dKernelSpatialDimensions(
        convolutionLayout.getKernelSpatialDimensions());
    conv2dKernelSpatialDimensions.push_back(3);

    llvm::SmallVector<int64_t> conv2dOutputSpatialDimensions(
        convolutionLayout.getOutputSpatialDimensions());
    conv2dOutputSpatialDimensions.push_back(3);

    auto new2dConvolutionOp = ttir::utils::createDPSOp<ttir::ConvolutionOp>(
        rewriter, loc, conv2dOutputShape, outputElementType, outputEncoding,
        reshapeInput, reshapeWeight, Value(), conv2dOpWindowsStridesAttr,
        conv2dOpPaddingAttr, conv2dOpInputDilationAttr,
        conv2dOpWeightDilationAttr, conv2dOpWindowReversalAttr,
        mlir::tt::ttir::ConvolutionLayoutAttr::get(
            getContext(), convolutionLayout.getInputBatchDimension(),
            convolutionLayout.getInputFeatureDimension(),
            conv2dInputSpatialDimensions,
            convolutionLayout.getKernelOutputFeatureDimension(),
            convolutionLayout.getKernelInputFeatureDimension(),
            conv2dKernelSpatialDimensions,
            convolutionLayout.getOutputBatchDimension(),
            convolutionLayout.getOutputFeatureDimension(),
            conv2dOutputSpatialDimensions),
        adaptor.getFeatureGroupCountAttr(), rewriter.getI64IntegerAttr(1));

    ttir::ReshapeOp reshapeOutput = createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshapeOutput"),
        new2dConvolutionOp, expectedOutputShape);

    return reshapeOutput;
  }

  ttir::ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                                  Value input,
                                  ::llvm::ArrayRef<int64_t> targetShape) const {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(targetShape));

    return ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, loc, targetShape, inputType.getElementType(),
        inputType.getEncoding(), input, shapeAttr);
  }

  mlir::DenseI64ArrayAttr
  addIntegerToDenseArrayAttr(ConversionPatternRewriter &rewriter,
                             mlir::DenseI64ArrayAttr denseArrayAttr,
                             uint64_t integerValue) const {
    llvm::SmallVector<int64_t, 4> newDenseArray(denseArrayAttr.asArrayRef());
    newDenseArray.push_back(integerValue);
    return rewriter.getDenseI64ArrayAttr(newDenseArray);
  }

  mlir::DenseBoolArrayAttr
  addBooleanToDenseArrayAttr(ConversionPatternRewriter &rewriter,
                             mlir::DenseBoolArrayAttr denseArrayAttr,
                             bool booleanValue) const {
    llvm::SmallVector<bool, 4> newDenseArray(denseArrayAttr.asArrayRef());
    newDenseArray.push_back(booleanValue);
    return rewriter.getDenseBoolArrayAttr(newDenseArray);
  }
};
} // namespace

namespace {
struct ConvolutionToConv2dPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;

  constexpr static uint32_t NUM_SPATIAL_DIMS = 2;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 0;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 1;

  // NHWC
  static inline const std::vector<int64_t> conv2dLayout = {
      ConvolutionDimension::BATCH,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
      ConvolutionDimension::FEATURE,
  };
  // OIHW
  static inline const std::vector<int64_t> conv2dKernelLayout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
  };
  // IOHW; for conv_transpose2d
  static inline const std::vector<int64_t> conv2dTransposeKernelLayout = {
      ConvolutionKernelDimension::INPUT_FEATURES,
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
  };

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();
    }

    uint64_t batchGroupCount = adaptor.getBatchGroupCount();
    uint64_t featureGroupCount = adaptor.getFeatureGroupCount();

    assert(batchGroupCount == 1 ||
           featureGroupCount == 1 &&
               "At least one of the group counts must be 1.");
    auto convLayoutAttr = op.getConvolutionLayoutAttr();
    auto outputType =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());

    // Prepare inputs/weights (slice if batchGroupCount > 1).
    llvm::SmallVector<Value> inputSlices;
    llvm::SmallVector<Value> weightSlices;
    llvm::SmallVector<llvm::SmallVector<int64_t>> outputSliceShapes;

    if (batchGroupCount > 1) {

      auto slices = sliceForBatchGroups(
          rewriter, op.getLoc(), adaptor.getInput(), adaptor.getWeight(),
          adaptor.getBias(), convLayoutAttr, batchGroupCount,
          convLayoutAttr.getInputBatchDimension());
      inputSlices = std::move(slices.inputs);
      weightSlices = std::move(slices.weights);

      llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
      int64_t outputBatchDim = convLayoutAttr.getOutputBatchDimension();
      int64_t outputFeatureDim = convLayoutAttr.getOutputFeatureDimension();
      int64_t outputBatchSliceSize = outputShape[outputBatchDim];
      int64_t outputFeatureSliceSize =
          outputShape[outputFeatureDim] / batchGroupCount;
      for (uint64_t i = 0; i < batchGroupCount; ++i) {
        llvm::SmallVector<int64_t> outputSliceShape(outputShape.begin(),
                                                    outputShape.end());
        outputSliceShape[outputBatchDim] = outputBatchSliceSize;
        outputSliceShape[outputFeatureDim] = outputFeatureSliceSize;
        outputSliceShapes.push_back(outputSliceShape);
      }
    } else {
      inputSlices.push_back(adaptor.getInput());
      weightSlices.push_back(adaptor.getWeight());
      llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
      outputSliceShapes.push_back(
          llvm::SmallVector<int64_t>(outputShape.begin(), outputShape.end()));
    }

    llvm::SmallVector<Value> results;
    for (size_t i = 0; i < inputSlices.size(); ++i) {
      Value result = createConv2dForSlice(rewriter, op, adaptor, convLayoutAttr,
                                          inputSlices[i], weightSlices[i],
                                          outputSliceShapes[i]);
      results.push_back(result);
    }

    Value finalResult;
    if (batchGroupCount > 1) {
      llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
      llvm::SmallVector<int64_t> outputLayoutMap(
          conv2dLayout.size(), ConvolutionDimension::INVALID_DIM);
      outputLayoutMap[convLayoutAttr.getOutputBatchDimension()] =
          ConvolutionDimension::BATCH;
      outputLayoutMap[convLayoutAttr.getOutputFeatureDimension()] =
          ConvolutionDimension::FEATURE;
      for (const auto [spatialCount, spatialDim] :
           llvm::enumerate(convLayoutAttr.getOutputSpatialDimensions())) {
        outputLayoutMap[spatialDim] = spatialCount;
      }
      auto outputShapeInConv2dLayout = ::ttmlir::utils::applyPermutation(
          outputShape,
          ttmlir::utils::generatePermutation(llvm::ArrayRef(outputLayoutMap),
                                             llvm::ArrayRef(conv2dLayout)));

      auto concatOutputType = RankedTensorType::get(outputShapeInConv2dLayout,
                                                    outputType.getElementType(),
                                                    outputType.getEncoding());

      // Concat on feature dimension in conv2d layout (which is dimension 3 for
      // NHWC).
      int64_t concatDim = std::find(conv2dLayout.begin(), conv2dLayout.end(),
                                    ConvolutionDimension::FEATURE) -
                          conv2dLayout.begin();
      finalResult = ttir::utils::createDPSOp<ttir::ConcatOp>(
          rewriter, op.getLoc(), concatOutputType, results, concatDim);
    } else {
      finalResult = results[0];
    }

    // Apply inverse permutation to restore original layout.
    llvm::SmallVector<int64_t> outputLayout(conv2dLayout.size(),
                                            ConvolutionDimension::INVALID_DIM);
    outputLayout[convLayoutAttr.getOutputBatchDimension()] =
        ConvolutionDimension::BATCH;
    outputLayout[convLayoutAttr.getOutputFeatureDimension()] =
        ConvolutionDimension::FEATURE;
    for (const auto [spatialCount, spatialDim] :
         llvm::enumerate(convLayoutAttr.getOutputSpatialDimensions())) {
      outputLayout[spatialDim] = spatialCount;
    }
    auto outputPermutation = ttmlir::utils::generatePermutation(
        llvm::ArrayRef(conv2dLayout), llvm::ArrayRef(outputLayout));

    rewriter.replaceOpWithNewOp<ttir::PermuteOp>(
        op, op.getResult().getType(), finalResult, adaptor.getOutput(),
        outputPermutation);

    return success();
  }

private:
  // Create a Conv2d or ConvTranspose2d operation for a single slice.
  Value createConv2dForSlice(
      ConversionPatternRewriter &rewriter, ttir::ConvolutionOp op,
      OpAdaptor adaptor, mlir::tt::ttir::ConvolutionLayoutAttr convLayoutAttr,
      Value input, Value weight, llvm::ArrayRef<int64_t> outputShape) const {

    bool isTransposed = ttir::utils::isTransposedConv(op);

    auto strideAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(adaptor.getWindowStrides()[SPATIAL_DIM_HEIGHT]),
        static_cast<int32_t>(adaptor.getWindowStrides()[SPATIAL_DIM_WIDTH]),
    });
    auto dilationAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT]),
        static_cast<int32_t>(adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH]),
    });

    // Padding is a list of 2-tuples, the order of the 2-tuples is in
    // most-significant spatial dimension first order For Conv2d the most
    // significant spatial dimension is the height, followed by the width.
    auto paddingMatrix =
        getPaddingMatrix<NUM_SPATIAL_DIMS>(adaptor.getPadding());
    auto paddingAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_WIDTH][0]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_HEIGHT][1]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_WIDTH][1]),
    });

    auto groupsAttr =
        rewriter.getI32IntegerAttr(adaptor.getFeatureGroupCount());

    llvm::SmallVector<int64_t> newOutputShape{
        outputShape[convLayoutAttr.getOutputBatchDimension()],
        outputShape[convLayoutAttr
                        .getOutputSpatialDimensions()[SPATIAL_DIM_HEIGHT]],
        outputShape[convLayoutAttr
                        .getOutputSpatialDimensions()[SPATIAL_DIM_WIDTH]],
        outputShape[convLayoutAttr.getOutputFeatureDimension()]};

    RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
    RankedTensorType outputType = inputType.clone(newOutputShape);

    auto permutation = generateConvPermutation(op, conv2dLayout);
    auto permuteOutputShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
    auto permutedInput = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(op.getLoc(), "_input"),
        permuteOutputShape, inputType.getElementType(), inputType.getEncoding(),
        input, permutation);

    Value permutedWeight = weight;
    // TTNN api handles reversing weights internally for transposed convolution.
    // So ttir.reverse op is ignored and its input is used as weight.
    if (auto reverseOp =
            permutedWeight.getDefiningOp<mlir::tt::ttir::ReverseOp>();
        isTransposed && reverseOp) {
      permutedWeight = reverseOp.getInput();
    }
    auto weightType = mlir::cast<RankedTensorType>(permutedWeight.getType());
    auto kernelPermutation = generateConvKernelPermutation(
        op, isTransposed ? conv2dTransposeKernelLayout : conv2dKernelLayout);
    auto weightOutputShape = ::ttmlir::utils::applyPermutation(
        weightType.getShape(), kernelPermutation);
    permutedWeight = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(op.getLoc(), "_weight"),
        weightOutputShape, weightType.getElementType(),
        weightType.getEncoding(), permutedWeight, kernelPermutation);

    // If bias is provided, it needs to be reshaped to match the
    // expected shape.
    Value biasValue = adaptor.getBias();
    if (biasValue) {
      auto biasType = mlir::cast<RankedTensorType>(biasValue.getType());
      auto biasPermutation = generateConvPermutation(op, conv2dLayout);
      auto biasOutputShape = ::ttmlir::utils::applyPermutation(
          biasType.getShape(), biasPermutation);
      SmallVector<int32_t> biasOutputShapeI32(biasOutputShape.begin(),
                                              biasOutputShape.end());
      biasValue = ttir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, ttmlir::utils::appendLocationSuffix(op.getLoc(), "_bias"),
          biasOutputShape, biasType.getElementType(), biasType.getEncoding(),
          biasValue, rewriter.getI32ArrayAttr(biasOutputShapeI32));
    }

    mlir::Value newConv;
    if (isTransposed) {
      // [TODO](mmanzoor) Verify the implementation of transposed convolution
      // for tt-xla. https://github.com/tenstorrent/tt-mlir/issues/3293
      // stablehlo.convolution/ttir.convolution op doesn't have output_padding
      // attribute. So Torch-MLIR adds output_padding with padding attribute for
      // transposed convolution during lowering.
      // https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Linear.cpp
      auto outputPaddingAttr = rewriter.getDenseI32ArrayAttr(
          {static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_HEIGHT][1] -
                                paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
           static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_WIDTH][1] -
                                paddingMatrix[SPATIAL_DIM_WIDTH][0])});
      // Recomputing padding attribute based on Torch-MLIR lowering of
      // conv_transposed2d op: [top, left, bottom, right].
      // https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Linear.cpp
      paddingAttr = rewriter.getDenseI32ArrayAttr({
          static_cast<int32_t>(
              (weightType.getShape()[SPATIAL_DIM_HEIGHT] - 1) *
                  adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT] -
              paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
          static_cast<int32_t>(
              (weightType.getShape()[SPATIAL_DIM_WIDTH] - 1) *
                  adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH] -
              paddingMatrix[SPATIAL_DIM_WIDTH][0]),
          static_cast<int32_t>(
              (weightType.getShape()[SPATIAL_DIM_HEIGHT] - 1) *
                  adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT] -
              paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
          static_cast<int32_t>(
              (weightType.getShape()[SPATIAL_DIM_WIDTH] - 1) *
                  adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH] -
              paddingMatrix[SPATIAL_DIM_WIDTH][0]),
      });
      // Input dilation (lhs dilation) is used for stride for transposed
      // convolution.
      auto inputDilationAttr = rewriter.getDenseI32ArrayAttr({
          static_cast<int32_t>(adaptor.getInputDilation()[SPATIAL_DIM_HEIGHT]),
          static_cast<int32_t>(adaptor.getInputDilation()[SPATIAL_DIM_WIDTH]),
      });
      newConv = ttir::utils::createDPSOp<ttir::ConvTranspose2dOp>(
          rewriter, op->getLoc(), outputType, Value(permutedInput),
          Value(permutedWeight), biasValue, inputDilationAttr, paddingAttr,
          outputPaddingAttr, dilationAttr, groupsAttr);
    } else {
      newConv = ttir::utils::createDPSOp<ttir::Conv2dOp>(
          rewriter, op.getLoc(), outputType, Value(permutedInput),
          Value(permutedWeight), biasValue, strideAttr, paddingAttr,
          dilationAttr, groupsAttr,
          /*flattenedCompatInfo=*/nullptr);
    }

    return newConv;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Reverse Pattern Matching
//===----------------------------------------------------------------------===//

// Decomposing Reverse Op into Gather Op.
// As soon as tenstorrent/tt-metal#16618 is finished, this decomposition can be
// removed.
namespace {
struct ReverseOpConversionPattern
    : public OpConversionPattern<ttir::ReverseOp> {
  using OpConversionPattern<ttir::ReverseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ArrayRef<int64_t> dimensions = adaptor.getDimensions();
    ArrayRef<int64_t> shape = op.getInput().getType().getShape();
    Value currentInput = adaptor.getInput();
    for (int32_t dim : dimensions) {
      SmallVector<int32_t> indices;
      for (int32_t i = shape[dim] - 1; i >= 0; i--) {
        indices.push_back(i);
      }

      auto tensorType =
          RankedTensorType::get({shape[dim]}, rewriter.getI32Type());

      auto denseAttr = DenseIntElementsAttr::get(tensorType, indices);

      Value reversedIndices =
          rewriter.create<ttir::ConstantOp>(op.getLoc(), tensorType, denseAttr);

      SmallVector<int64_t> offsetDims;
      for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); i++) {
        if (i != dim) {
          offsetDims.push_back(i);
        }
      }

      SmallVector<int64_t> sliceSizes(shape.begin(), shape.end());
      sliceSizes[dim] = 1;

      currentInput = ttir::utils::createDPSOp<ttir::GatherOp>(
          rewriter, op.getLoc(), op.getResult().getType(),
          /*input=*/currentInput,
          /*start_indices=*/reversedIndices,
          /*offset_dims=*/offsetDims,
          /*collapsed_slice_dims=*/SmallVector<int64_t>{dim},
          /*operand_batching_dims=*/SmallVector<int64_t>{},
          /*start_indices_batching_dims=*/SmallVector<int64_t>{},
          /*start_index_map=*/SmallVector<int64_t>{dim},
          /*index_vector_dim=*/1,
          /*slice_sizes=*/sliceSizes,
          /*indices_are_sorted=*/false);
    }

    // Skip reverse operations that are used by transposed convolutions, as
    // TTNN's conv_transpose2d handles weight reversal internally. Pattern must
    // not be applied to reverse operations that are used by transposed
    // convolutions because transposed convolution has to handle weight reversal
    // internally.

    rewriter.replaceUsesWithIf(op, currentInput, [&](OpOperand &operand) {
      auto convOp = dyn_cast<ttir::ConvolutionOp>(operand.getOwner());
      return !convOp || !ttir::utils::isTransposedConv(convOp);
    });

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Gather Pattern Matching
//===----------------------------------------------------------------------===//

namespace {
struct GatherToEmbeddingConversionPattern
    : public OpConversionPattern<ttir::GatherOp> {
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

  /**
   * Validates Gather Op constraints for embedding conversion
   *
   * Enforces constraints on Gather Op to ensure valid embedding
   * transformation:
   * - start indices tensor isn't 1D when we are indexing multiple dims
   * - operandBatchingDims and startIndicesBatchingDims are none
   * - sliceSizes are fullDim for dimensions we are not indexing
   * - for dimensions we are indexing, sliceSizes must fit into one of:
   *   - all sliceSizes are 1
   *   - all sliceSizes are fullDim except one which can be anything
   */

  LogicalResult checkBasicLegality(ttir::GatherOp op,
                                   PatternRewriter &rewriter) const {

    // Get input and start indices tensor shape.
    auto inputShape = op.getInput().getType().getShape();
    auto startIndicesShape = op.getStartIndices().getType().getShape();

    // Get attributes needed for embedding op pattern matching checks.
    auto sliceSizes = op.getSliceSizes();
    auto startIndexMap = op.getStartIndexMap();

    // Check if start indices tensor isn't 1D when we are indexing multiple
    // dimensions because of matmul restrictions.
    if (startIndexMap.size() > 1 && startIndicesShape.size() == 1) {
      return rewriter.notifyMatchFailure(
          op, "Did not satisfy startIndicesShape.size() > 1 when "
              "startIndexMap.size() > 1");
    }

    // Check if there are no batching dims.
    if (!op.getOperandBatchingDims().empty() ||
        !op.getStartIndicesBatchingDims().empty()) {
      return rewriter.notifyMatchFailure(op, "Did not satisfy batching = none");
    }

    // Check slice sizes conditions.
    size_t fullIndexedDims = 0;
    size_t partialIndexedDims = 0;
    size_t singletonIndexedDims = 0;

    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (llvm::is_contained(startIndexMap, i)) {
        if (inputShape[i] == 1) {
          singletonIndexedDims++;
        } else if (sliceSizes[i] == inputShape[i]) {
          fullIndexedDims++;
        } else if (sliceSizes[i] != 1) {
          partialIndexedDims++;
        }
      } else if (sliceSizes[i] != inputShape[i]) {
        return rewriter.notifyMatchFailure(
            op, "Did not satisfy sliceSizes[i] = inputShape[i] for dims not "
                "in startIndexMap");
      }
    }

    size_t remainingIndexedDims =
        startIndexMap.size() - fullIndexedDims - singletonIndexedDims;
    if (partialIndexedDims && (remainingIndexedDims != 1)) {
      return rewriter.notifyMatchFailure(
          op,
          "Did not satisfy slice conditions for dimensions in startIndexMap");
    }

    if (fullIndexedDims &&
        (remainingIndexedDims > 1 ||
         (remainingIndexedDims + singletonIndexedDims) == 0)) {
      return rewriter.notifyMatchFailure(
          op,
          "Did not satisfy slice conditions for dimensions in startIndexMap");
    }

    return success();
  }

  /**
   * Lowers Gather Op into Embedding Op (and applies Reshape and Permute Ops, if
   * necessary)
   *
   * There is no TTNN Gather support.
   * Gather Op is lowered into Embedding Op Op.
   * Torch embeddings are lowered into Gather Op.
   * Most models use Gather Op to implement simple embeddings.
   * If encountered more complicated Gather Op implementations, they can be
   * lowered into slice/ concat/ etc.
   *
   * Embedding Op expects:
   * - weights to be strictly 2D. We index the first dimension of weights, and
   * take slices from the full second dimension.
   * - input can be 1D or 2D
   * - output shape is the shape of input with the last dimension of the
   * weights appended
   *
   *  - Gather Op input becomes Embedding Op weights. Because it can have
   * any number and order of dimensions, it is permuted and reshaped
   * (flattened).
   *  - Gather Op startIndices becomes Embedding Op input. Because it can
   * have any number and order of dimensions, it is permuted and reshaped
   * (flattened).
   * - Embedding Op output needs to be reshaped to recover lost
   * dimensions and permuted as Gather Op output dimensions can be in any
   * order.
   */

  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // GatherOp can be used to implement embedding lookup, check for that case.
    LogicalResult err = checkBasicLegality(op, rewriter);
    if (not err.succeeded()) {
      return err;
    }

    auto inputShape = op.getInput().getType().getShape();
    auto sliceSizes = op.getSliceSizes();
    auto originalStartIndexMap = op.getStartIndexMap();

    // If there are indexed dims that have full slice size, we need to ignore
    // them and slice indices accordingly, which is why we note the
    // actualIndexedDim.
    int64_t actualIndexedDim = -1;

    // If there is an indexed dim with slice size > 1, but not full, we need to
    // expand start indices to contain the implied ones.
    bool needsExpansion = false;

    // Create startIndexMap without dims for which sliceSizes[dim] =
    // inputShape[dim]. If there are dims for which sliceSizes[dim] =
    // inputShape[dim] = 1, they are treated specially:
    // - if there is a partially indexed dim, they are removed
    // - if all other indexed dims are full, one of them is kept
    size_t fullIndexedDims = 0;
    bool partialIndexedDimExists = false;
    for (size_t i = 0; i < originalStartIndexMap.size(); ++i) {
      int64_t dim = originalStartIndexMap[i];
      if (sliceSizes[dim] == inputShape[dim]) {
        fullIndexedDims++;
      } else if (sliceSizes[dim] != 1) {
        partialIndexedDimExists = true;
      }
    }

    llvm::SmallVector<int64_t> startIndexMap;
    for (size_t i = 0; i < originalStartIndexMap.size(); ++i) {
      int64_t dim = originalStartIndexMap[i];
      if (inputShape[dim] == 1) {
        if (fullIndexedDims == originalStartIndexMap.size()) {
          startIndexMap.push_back(dim);
          actualIndexedDim = i;
          break;
        }
        if (partialIndexedDimExists || fullIndexedDims > 0) {
          continue;
        }
      } else if (sliceSizes[dim] == inputShape[dim]) {
        continue;
      }

      startIndexMap.push_back(dim);
      actualIndexedDim = i;
      if (sliceSizes[dim] != 1) {
        needsExpansion = true;
      }
    }
    auto numIndexingDims = startIndexMap.size();

    auto inputPermuted = permuteInput(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_permuteInput"),
        op.getInput(), startIndexMap);
    auto input = reshapeInput(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_reshapeInput"),
        inputPermuted, numIndexingDims);

    // If we are indexing multiple dims, we need to transform indices for the
    // new single (flattened) indexing dim. If the extra indexed dims are full,
    // we need to slice indices.
    auto startIndices = op.getStartIndices();
    if (numIndexingDims > 1) {
      op->emitWarning("End results might be incorrect when indexing multiple "
                      "dimensions of input because of typecast ops.");
      startIndices =
          flattenStartIndices(rewriter, inputPermuted.getType().getShape(), op);
    } else if (originalStartIndexMap.size() != numIndexingDims) {
      startIndices = sliceStartIndices(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op->getLoc(),
                                              "_sliceStartIndices"),
          op.getStartIndices(), op.getIndexVectorDim(), actualIndexedDim);
    }

    if (startIndices.getType().getShape().size() != 2 ||
        (needsExpansion && op.getIndexVectorDim() != 0)) {
      startIndices =
          reshapeStartIndices(rewriter,
                              ttmlir::utils::appendLocationSuffix(
                                  op->getLoc(), "_reshapeStartIndices"),
                              startIndices);
    }

    // If we are indexing a dim with slice size > 1, we need to expand indices
    // to gather all the rows, not just the first one.
    if (needsExpansion) {
      startIndices = expandStartIndices(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op->getLoc(),
                                              "_expandStartIndices"),
          startIndices, op.getIndexVectorDim(),
          sliceSizes[originalStartIndexMap[actualIndexedDim]]);
    }

    // Calculate a new shape for output: this is new start indices shape + last
    // dim of input shape.
    auto startIndicesShape = startIndices.getType().getShape();
    llvm::SmallVector<int64_t> newOutputShape(startIndicesShape.begin(),
                                              startIndicesShape.end());
    newOutputShape.push_back(input.getType().getShape()[1]);

    auto embeddingOutputType = mlir::RankedTensorType::get(
        newOutputShape, input.getType().getElementType(),
        input.getType().getEncoding());
    ttir::EmbeddingOp embeddingOp = ttir::utils::createDPSOp<ttir::EmbeddingOp>(
        rewriter, op.getLoc(), embeddingOutputType, startIndices, input);

    rewriter.replaceOp(op, reshapeAndPermuteOutput(rewriter, embeddingOp,
                                                   startIndexMap[0], op));
    return success();
  }

private:
  // In StableHLO, startIndexMap attribute refers to which dims of input
  // we are indexing (with startIndices). We need these dims to be
  // flattened together to be the first dim of transformed input (that is
  // weights for ttir.embedding). This helper makes these indexing dims
  // the first few dims of input.
  // Example: inputShape = [2, 3, 4, 5], startIndexMap = [1, 3] ->
  // permutedInputShape = [3, 5, 2, 4]
  static ttir::PermuteOp
  permuteInput(ConversionPatternRewriter &rewriter, Location loc,
               ::mlir::TypedValue<::mlir::RankedTensorType> input,
               ::llvm::ArrayRef<int64_t> startIndexMap) {
    auto inputType = input.getType();
    llvm::SmallVector<int64_t> inputPermutation(startIndexMap);
    inputPermutation.append(llvm::filter_to_vector(
        llvm::seq<int64_t>(inputType.getRank()), [&startIndexMap](int64_t idx) {
          return !llvm::is_contained(startIndexMap, idx);
        }));
    auto permutedInputShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), inputPermutation);
    return ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, loc, permutedInputShape, inputType.getElementType(),
        inputType.getEncoding(), input, inputPermutation);
  }

  // This helper flattens the indexing dims to be one, first dim of
  // transformed input, and all the other dims to be the second dim.
  // Example: permutedInputShape = [3, 5, 2, 4], numIndexingDims = 2 ->
  // newIputShape = [15, 8]
  static ttir::ReshapeOp
  reshapeInput(ConversionPatternRewriter &rewriter, Location loc,
               ::mlir::TypedValue<::mlir::RankedTensorType> input,
               size_t numIndexingDims) {
    auto inputShape = input.getType().getShape();
    assert(
        numIndexingDims <= inputShape.size() &&
        "Number of indexing dims can't be greater than number of input dims");
    llvm::SmallVector<int64_t> newInputShape{
        std::accumulate(inputShape.begin(),
                        inputShape.begin() + numIndexingDims, int64_t{1},
                        std::multiplies<>()),
        std::accumulate(inputShape.begin() + numIndexingDims, inputShape.end(),
                        int64_t{1}, std::multiplies<>())};
    return createReshapeOp(rewriter, loc, input, newInputShape);
  }

  // If we are indexing multiple dimes of input, we need to adjust start
  // indices to represent indices that index one flattened dimension.
  // - indexVectorDim represents in what dimension are indices, so first we
  // permute to make sure it is the last dimension
  // - matmul doesn't work with integers (which startIndices are when lowered
  // form SHLO), so a typecast is added
  // - then we add matmul to transform the indices
  // Example: indexingDimsSizes = [3, 5], startIndices[...] = (i, j) ->
  // startIndices[...] = 5 * i + j (because reshaped indexingDimSize is 15)
  static ttir::MatmulOp
  flattenStartIndices(ConversionPatternRewriter &rewriter,
                      ::llvm::ArrayRef<int64_t> inputShape, ttir::GatherOp op) {
    auto startIndices = op.getStartIndices();
    auto startIndicesType = startIndices.getType();
    auto numIndexingDims = op.getStartIndexMap().size();
    auto indexVectorDim = op.getIndexVectorDim();

    llvm::SmallVector<int64_t> startIndicesPermutation = llvm::filter_to_vector(
        llvm::seq<int64_t>(startIndicesType.getRank()),
        [&indexVectorDim](int64_t idx) { return idx != indexVectorDim; });
    startIndicesPermutation.push_back(indexVectorDim);

    auto permutedStartIndicesShape = ttmlir::utils::applyPermutation(
        startIndicesType.getShape(), startIndicesPermutation);
    auto startIndicesPermuted =
        ttir::utils::createDPSOp<ttir::PermuteOp>(
            rewriter,
            ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                                "_permuteStartIndices"),
            permutedStartIndicesShape, startIndicesType.getElementType(),
            startIndicesType.getEncoding(), startIndices,
            startIndicesPermutation)
            .getResult();

    // Typecast op because matmul needs float operands.
    auto typecastResultType = startIndicesPermuted.getType().clone(
        mlir::Float32Type::get(op.getContext()));
    ttir::TypecastOp typecastOp = ttir::utils::createDPSOp<ttir::TypecastOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_typecast"),
        typecastResultType, startIndicesPermuted);

    // Const op with correct strides to matmul indices with.
    llvm::SmallVector<float> strides(numIndexingDims);
    int dimensionOffset = 1;
    for (int i = numIndexingDims - 1; i >= 0; i--) {
      strides[i] = dimensionOffset;
      dimensionOffset *= inputShape[i];
    }
    auto tensorType =
        mlir::RankedTensorType::get({static_cast<long>(numIndexingDims), 1},
                                    mlir::Float32Type::get(op.getContext()));
    auto denseAttr =
        mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(strides));
    ttir::ConstantOp constantOp = rewriter.create<ttir::ConstantOp>(
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_constant"),
        tensorType, denseAttr);

    // Return matmul op that transforms indices.
    llvm::SmallVector<int64_t> matmulResultShape = permutedStartIndicesShape;
    matmulResultShape[matmulResultShape.size() - 1] = 1;
    auto matmulResultType = mlir::RankedTensorType::get(
        matmulResultShape, Float32Type::get(op.getContext()));

    return ttir::utils::createDPSOp<ttir::MatmulOp>(
        rewriter, op->getLoc(), matmulResultType, typecastOp.getResult(),
        constantOp);
  }

  // If startIndicesShape[indexVectorDim] > 1, but we are actually slicing only
  // one dim and gathering the other dims fully, we need to slice startIndices
  // to keep only the relevant indices. Example: inputShape = [3, 5],
  // startIndexMap = [0, 1], sliceSizes = [1, 5], startIndices = [[2, 1], [0,
  // 3]], indexVectorDim=1 -> startIndices = [[2], [0]]
  static ttir::SliceStaticOp
  sliceStartIndices(ConversionPatternRewriter &rewriter, Location loc,
                    ::mlir::TypedValue<::mlir::RankedTensorType> startIndices,
                    int64_t indexVectorDim, int64_t actualIndexedDim) {
    auto startIndicesType = startIndices.getType();
    auto startIndicesShape = startIndicesType.getShape();
    int64_t rank = startIndicesType.getRank();

    // Create begins, ends, and steps arrays for slicing
    llvm::SmallVector<int32_t> begins(rank, 0);
    llvm::SmallVector<int32_t> ends(startIndicesShape.begin(),
                                    startIndicesShape.end());
    llvm::SmallVector<int32_t> steps(rank, 1);

    begins[indexVectorDim] = actualIndexedDim;
    ends[indexVectorDim] = actualIndexedDim + 1;

    // Calculate the result shape
    llvm::SmallVector<int64_t> resultShape(startIndicesShape);
    resultShape[indexVectorDim] = 1;

    return ttir::utils::createDPSOp<ttir::SliceStaticOp>(
        rewriter, loc, resultShape, startIndicesType.getElementType(),
        startIndicesType.getEncoding(), startIndices,
        rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
        rewriter.getI32ArrayAttr(steps));
  }

  // Helper that reshapes start indices to reduce number of dims, as Embedding
  // Op input needs to be 2D.
  static ttir::ReshapeOp reshapeStartIndices(
      ConversionPatternRewriter &rewriter, Location loc,
      ::mlir::TypedValue<::mlir::RankedTensorType> startIndices) {
    auto startIndicesShape = startIndices.getType().getShape();
    llvm::SmallVector<int64_t, 2> newStartIndicesShape{
        1, std::accumulate(startIndicesShape.begin(), startIndicesShape.end(),
                           int64_t{1}, std::multiplies<>())};
    return createReshapeOp(rewriter, loc, startIndices, newStartIndicesShape);
  }

  // Helper that expands start indices along the index vector dimension when
  // sliceSizes[actualIndexedDim] > 1. This creates additional indices by
  // adding consecutive values to the original indices. Because of earlier
  // reshape, we know startIndices has shape [1, N]. Example: startIndices =
  // [[2, 1]], indexVectorDim=0, sliceSize=3 -> startIndices =
  // [[2, 1], [3, 2], [4, 3]]
  static ttir::AddOp expandStartIndices(ConversionPatternRewriter &rewriter,
                                        Location loc, Value startIndices,
                                        int64_t indexVectorDim,
                                        int64_t sliceSize) {
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());
    auto startIndicesShape = startIndicesType.getShape();

    // Create NxM matrix where each column is [0, 1, 2, ..., N-1].
    int32_t N = sliceSize;
    int32_t M = startIndicesShape[1];

    llvm::SmallVector<int32_t> matrixData(N * M);
    for (int32_t col = 0; col < M; ++col) {
      for (int32_t row = 0; row < N; ++row) {
        matrixData[row * M + col] = row;
      }
    }
    llvm::SmallVector<int64_t> expandedShape(startIndicesShape);
    expandedShape[0] = sliceSize;

    auto expandedType = mlir::RankedTensorType::get(
        expandedShape, startIndicesType.getElementType(),
        startIndicesType.getEncoding());
    auto offsetAttr =
        mlir::DenseElementsAttr::get(expandedType, llvm::ArrayRef(matrixData));
    auto offsetConstant = rewriter.create<ttir::ConstantOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_offsetConstant"),
        expandedType, offsetAttr);
    // Create broadcast dimensions - all dimensions map directly except the
    // expanded one.
    llvm::SmallVector<int64_t> broadcastDimensions = {sliceSize, 1};

    // Broadcast the original startIndices to the expanded shape.
    auto broadcastedStartIndices = ttir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(loc, "_broadcastStartIndices"),
        expandedType, startIndices,
        rewriter.getDenseI64ArrayAttr(broadcastDimensions));

    // Add the broadcasted tensors to get the final expanded indices.
    return ttir::utils::createDPSOp<ttir::AddOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(loc, "_expandedStartIndices"),
        expandedType, broadcastedStartIndices, offsetConstant);
  }

  // In output, dims other than offsetDims map to startIndices shape, and
  // offsetDims map to input slices. After ttir.embedding all offseDims are
  // flattened to the last dim of output. First we reshape that output to
  // recover lost dims, then we permute them so offset dims are where the
  // attribute states.
  // Example: expectedOutputShape = [2, 3, 4, 5], offsetDims = [1, 3]
  // -> embeddingOutputShape = [2, 4, 15] -reshape-> [2, 4, 3, 5] -permute-> [2,
  // 3, 4, 5]
  static ttir::PermuteOp
  reshapeAndPermuteOutput(ConversionPatternRewriter &rewriter,
                          ::mlir::TypedValue<::mlir::RankedTensorType> output,
                          int64_t indexedDim, ttir::GatherOp op) {
    auto expectedOutputType = op.getOutput().getType();
    auto expectedOutputShape = expectedOutputType.getShape();
    auto offsetDims = op.getOffsetDims();
    auto collapsedSliceDims = op.getCollapsedSliceDims();
    // Because of permuting input to put the indexing dims first, the output has
    // corresponding dims in front of (other) offsetDims, as well. When size of
    // these dims in output is not 1, we need to move them to their correct
    // spot.
    bool needsOffsetReordering = false;
    size_t numSmallerCollapsedDims = 0;

    if (op.getSliceSizes()[indexedDim] != 1) {
      needsOffsetReordering = true;
      numSmallerCollapsedDims =
          std::lower_bound(collapsedSliceDims.begin(), collapsedSliceDims.end(),
                           indexedDim) -
          collapsedSliceDims.begin();
    }

    llvm::SmallVector<int64_t> outputPermutation;
    if (needsOffsetReordering) {
      outputPermutation.push_back(
          offsetDims[indexedDim - numSmallerCollapsedDims]);
    }
    for (size_t i = 0; i < expectedOutputShape.size(); ++i) {
      if (!llvm::is_contained(offsetDims, i)) {
        outputPermutation.push_back(i);
      }
    }
    for (size_t i = 0; i < offsetDims.size(); ++i) {
      if (!(needsOffsetReordering &&
            i == static_cast<size_t>(indexedDim - numSmallerCollapsedDims))) {
        outputPermutation.push_back(offsetDims[i]);
      }
    }

    auto inverseOutputPermutation =
        ttmlir::utils::inversePermutation(outputPermutation);
    auto permutedOutputShape =
        ttmlir::utils::applyPermutation(expectedOutputShape, outputPermutation);

    auto reshapedOutput = createReshapeOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_reshapeOutput"),
        output, permutedOutputShape);

    return ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_permuteOutput"),
        expectedOutputShape, expectedOutputType.getElementType(),
        expectedOutputType.getEncoding(), reshapedOutput,
        inverseOutputPermutation);
  }

  static ttir::ReshapeOp
  createReshapeOp(PatternRewriter &rewriter, Location loc, Value input,
                  ::llvm::ArrayRef<int64_t> targetShape) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(targetShape));

    return ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, loc, targetShape, inputType.getElementType(),
        inputType.getEncoding(), input, shapeAttr);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
/*
Below is the implementation of the DotGeneralOp decomposition into MatmulOp,
ReshapeOp, and PermuteOp. The DotGeneralOp is a more general form of MatmulOp
where tensors can have arbitrary contract dimensions. Contract dimensions are
the ones along which multiplication happens (typically summed over during the
operation). Previously, DotGeneralOp only supported cases where it directly
mapped to a MatmulOp, which typically involves batch dimensions (e.g., [5, 6, 7]
x [5, 7, 6] where 5 is the batch dimension and multiplication happens along
dimension 7). This decomposition extends the support to more flexible tensor
shapes, such as [5, 6, 7] x [5, 6, 7], where the contract dimension is 6 (or 7)
in both tensors. This allows DotGeneralOp to handle cases beyond the typical
MatmulOp constraints, enabling more complex tensor operations.
*/

namespace {
struct DotGeneralToMatmulConversionPattern
    : public OpConversionPattern<ttir::DotGeneralOp> {
  using OpConversionPattern<ttir::DotGeneralOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check if the original op should be hoisted.
    bool shouldHoist = ttir::utils::hasShouldHoistAttr(op);

    Value lhs = adaptor.getLhs();
    auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
    int64_t lhsRank = lhsType.getRank();
    SmallVector<int64_t> lhsBatchDims(op.getBatchDimsLhs());
    SmallVector<int64_t> lhsContractDims(op.getContractDimsLhs());

    Value rhs = adaptor.getRhs();
    auto rhsType = mlir::cast<RankedTensorType>(rhs.getType());
    int64_t rhsRank = rhsType.getRank();
    SmallVector<int64_t> rhsBatchDims(op.getBatchDimsRhs());
    SmallVector<int64_t> rhsContractDims(op.getContractDimsRhs());

    Type elementType = lhsType.getElementType();
    Attribute encoding = lhsType.getEncoding();

    SmallVector<int64_t> lhsResultDims =
        getResultDims(lhsBatchDims, lhsContractDims, lhsRank);
    SmallVector<int64_t> rhsResultDims =
        getResultDims(rhsBatchDims, rhsContractDims, rhsRank);

    // Compute permutation for lhs and rhs to get the desired layout.
    // For lhs: (batch dims, result dims, contract dims)
    // For rhs: (batch dims, contract dims, result dims)

    SmallVector<int64_t> lhsPermutation =
        getPermutation(lhsBatchDims, lhsResultDims, lhsContractDims);
    SmallVector<int64_t> rhsPermutation =
        getPermutation(rhsBatchDims, rhsContractDims, rhsResultDims);

    // Apply these permutations to lhs and rhs.

    ttir::PermuteOp lhsPermute = createPermuteOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_permuteLhs"), lhs,
        lhsType, lhsPermutation, shouldHoist);
    ttir::PermuteOp rhsPermute = createPermuteOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_permuteRhs"), rhs,
        rhsType, rhsPermutation, shouldHoist);

    // Compute final shape for lhs and rhs.
    // for lhs (batch dims, prod(result dims), prod(contract dims))
    // for rhs (batch dims, prod(contract dims), prod(result dims))

    SmallVector<int64_t> lhsMatmulInputShape = computeMatmulInputShape(
        rewriter, lhsType, lhsBatchDims, lhsResultDims, lhsContractDims);
    SmallVector<int64_t> rhsMatmulInputShape = computeMatmulInputShape(
        rewriter, rhsType, rhsBatchDims, rhsContractDims, rhsResultDims);

    // Apply this reshape to lhs and rhs to adapt to matmul op.
    // For lhs: (batch dims, prod(result dims), prod(contract dims))
    // For rhs: (batch dims, prod(contract dims), prod(result dims))

    ttir::ReshapeOp lhsMatmulInput = createMatmulFinal(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeLhs"),
        lhsPermute, lhsType, lhsMatmulInputShape, shouldHoist);
    ttir::ReshapeOp rhsMatmulInput = createMatmulFinal(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeRhs"),
        rhsPermute, rhsType, rhsMatmulInputShape, shouldHoist);

    // Get shape of matmul op result.

    SmallVector<int64_t> matmulDestinationShape;
    for (auto dim : lhsBatchDims) {
      matmulDestinationShape.push_back(lhsType.getShape()[dim]);
    }
    matmulDestinationShape.push_back(
        computeProductOfDims(lhsType.getShape(), lhsResultDims));
    matmulDestinationShape.push_back(
        computeProductOfDims(rhsType.getShape(), rhsResultDims));

    // Perform matmul operation.
    auto matmulOp = ttir::utils::createDPSOp<ttir::MatmulOp>(
        rewriter, op.getLoc(), matmulDestinationShape, elementType, encoding,
        lhsMatmulInput, rhsMatmulInput);

    // Propagate the hoist attribute to the matmul op.
    if (shouldHoist) {
      ttir::utils::addShouldHoistAttr(matmulOp, rewriter);
    }

    // Reshape the result by unrolling the prod(lhsResultDims) to original
    // lhsResultDims and likewise for rhsResultDims.

    SmallVector<int64_t> resultShape;
    for (auto dim : lhsBatchDims) {
      resultShape.push_back(lhsType.getShape()[dim]);
    }
    for (auto dim : lhsResultDims) {
      resultShape.push_back(lhsType.getShape()[dim]);
    }
    for (auto dim : rhsResultDims) {
      resultShape.push_back(rhsType.getShape()[dim]);
    }

    llvm::SmallVector<int32_t> finalShapeI32(resultShape.begin(),
                                             resultShape.end());

    auto reshapeOutput = ttir::utils::replaceOpWithNewDPSOp<ttir::ReshapeOp>(
        rewriter, op, resultShape, elementType, encoding, matmulOp,
        rewriter.getI32ArrayAttr(finalShapeI32));

    reshapeOutput->setLoc(ttmlir::utils::appendLocationSuffix(
        reshapeOutput->getLoc(), "_reshapeOutput"));

    // Propagate the hoist attribute to the final reshape op.
    if (shouldHoist) {
      ttir::utils::addShouldHoistAttr(reshapeOutput, rewriter);
    }

    return success();
  }

private:
  SmallVector<int64_t> getResultDims(const SmallVector<int64_t> &batchDims,
                                     const SmallVector<int64_t> &contractDims,
                                     int64_t rank) const {

    SmallVector<int64_t> allDims;
    for (int64_t i = 0; i < rank; i++) {
      allDims.push_back(i);
    }

    // Remove batch and contract dims.

    for (size_t i = 0; i < batchDims.size(); i++) {
      for (size_t j = 0; j < allDims.size(); j++) {
        if (allDims[j] == batchDims[i]) {
          allDims.erase(allDims.begin() + j);
          break;
        }
      }
    }
    for (size_t i = 0; i < contractDims.size(); i++) {
      for (size_t j = 0; j < allDims.size(); j++) {
        if (allDims[j] == contractDims[i]) {
          allDims.erase(allDims.begin() + j);
          break;
        }
      }
    }

    return allDims;
  }

  SmallVector<int64_t> getPermutation(const SmallVector<int64_t> &batchDims,
                                      const SmallVector<int64_t> &dims1,
                                      const SmallVector<int64_t> &dims2) const {

    SmallVector<int64_t> permutation;
    permutation.append(batchDims);
    permutation.append(dims1);
    permutation.append(dims2);

    return permutation;
  }

  ttir::PermuteOp createPermuteOp(PatternRewriter &rewriter, Location loc,
                                  Value input, RankedTensorType inputType,
                                  const SmallVector<int64_t> &permutation,
                                  bool shouldHoist = false) const {

    SmallVector<int64_t> destinationShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), permutation);

    auto permuteOp = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, loc, destinationShape, inputType.getElementType(),
        inputType.getEncoding(), input, permutation);

    // Propagate the hoist attribute to the permute op.
    if (shouldHoist) {
      ttir::utils::addShouldHoistAttr(permuteOp, rewriter);
    }
    return permuteOp;
  }

  SmallVector<int64_t>
  computeMatmulInputShape(ConversionPatternRewriter &rewriter,
                          RankedTensorType tensorType,
                          const SmallVector<int64_t> &batchDims,
                          const SmallVector<int64_t> &contractDims,
                          const SmallVector<int64_t> &resultDims) const {

    SmallVector<int64_t> finalShape;

    // Add the batch dimensions.
    for (auto dim : batchDims) {
      finalShape.push_back(tensorType.getShape()[dim]);
    }

    // Add the result and contract product dimensions.
    finalShape.push_back(
        computeProductOfDims(tensorType.getShape(), contractDims));
    finalShape.push_back(
        computeProductOfDims(tensorType.getShape(), resultDims));

    return finalShape;
  }

  ttir::ReshapeOp createMatmulFinal(PatternRewriter &rewriter, Location loc,
                                    Value input, RankedTensorType type,
                                    const SmallVector<int64_t> &finalShape,
                                    bool shouldHoist = false) const {

    llvm::SmallVector<int32_t> finalShapeI32(finalShape.begin(),
                                             finalShape.end());

    auto reshapeOp = ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, loc, finalShape, type.getElementType(), type.getEncoding(),
        input, rewriter.getI32ArrayAttr(finalShapeI32));
    // Propagate the hoist attribute to the reshape op.
    if (shouldHoist) {
      ttir::utils::addShouldHoistAttr(reshapeOp, rewriter);
    }

    return reshapeOp;
  }

  int64_t computeProductOfDims(ArrayRef<int64_t> tensorShape,
                               ArrayRef<int64_t> dims) const {
    int64_t product = 1;
    for (auto dim : dims) {
      product *= tensorShape[dim];
    }
    return product;
  }
};
} // namespace

namespace {
struct PoolingToPool2dPattern : public OpConversionPattern<ttir::PoolingOp> {
public:
  using OpConversionPattern<ttir::PoolingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PoolingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<int64_t> spatialDimIndices =
        getIndicesOfElementsLargerThanOne(op.getWindowDimensions());
    size_t numSpatialDimIndices = spatialDimIndices.size();
    if (numSpatialDimIndices > 2) {
      return rewriter.notifyMatchFailure(
          op, "No decompositions for a pooling op with " +
                  std::to_string(numSpatialDimIndices) + " spatial dimensions");
    }

    LogicalResult legalityResult =
        canDecompose2DPoolingOp(op, rewriter, spatialDimIndices);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    switch (op.getPoolingMethod()) {
    case ttir::PoolingMethod::Max: {
      llvm::SmallVector<Value> outputs = rewritePool2d<ttir::MaxPool2dOp>(
          op, adaptor, rewriter, spatialDimIndices);
      rewriter.replaceOp(op, outputs);
      return success();
    }
    case ttir::PoolingMethod::Average: {
      llvm::SmallVector<Value> outputs = rewritePool2d<ttir::AvgPool2dOp>(
          op, adaptor, rewriter, spatialDimIndices);
      rewriter.replaceOp(op, outputs);
      return success();
    }
    case ttir::PoolingMethod::Sum: {
      llvm::SmallVector<Value> outputs =
          rewriteSumPool2d(op, adaptor, rewriter, spatialDimIndices);
      rewriter.replaceOp(op, outputs);
      return success();
    }
    }
  }

private:
  llvm::SmallVector<int64_t>
  getIndicesOfElementsLargerThanOne(llvm::ArrayRef<int64_t> input) const {
    llvm::SmallVector<int64_t, 2> result;
    for (size_t i = 0; i < input.size(); i++) {
      if (input[i] > 1) {
        result.push_back(i);
      }
    }
    return result;
  }

  LogicalResult
  canDecompose2DPoolingOp(ttir::PoolingOp op,
                          ConversionPatternRewriter &rewriter,
                          llvm::SmallVector<int64_t> spatialDimIndices) const {

    // Window dimensions must be 4 in length
    if (op.getWindowDimensions().size() != 4) {
      return rewriter.notifyMatchFailure(
          op, "Polling 2D op is only supported for 4D tensor.");
    }

    // Window strides must be 4 in length
    if (op.getWindowStrides().size() != 4) {
      return rewriter.notifyMatchFailure(
          op, "Polling 2D op is only supported for 4D tensor.");
    }

    // Operand rank(s) must be 4
    for (Value operand : op.getInputs()) {
      auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
      if (operandType.getRank() != 4) {
        return rewriter.notifyMatchFailure(
            op, "Polling 2D op is only supported for 4D tensor.");
      }
    }

    // Window dimensions will have two or less than two non 1 elements;
    // representing the kernel size for max pooling operation.
    size_t numSpatialDimIndices = spatialDimIndices.size();
    if (numSpatialDimIndices > 2) {
      return rewriter.notifyMatchFailure(op, "Rank of kernel_size for " +
                                                 op.getOperationName() +
                                                 " op is greater than 2.");
    }

    // Window strides will have two or less than two non 1 elements;
    // representing the strides for max pooling operation.
    llvm::SmallVector<int64_t> trueWindowStrideIndices =
        getIndicesOfElementsLargerThanOne(op.getWindowStrides());
    size_t windowStrideSize = trueWindowStrideIndices.size();
    if (windowStrideSize > 2) {
      return rewriter.notifyMatchFailure(op, "Rank of strides for " +
                                                 op.getOperationName() +
                                                 " is greater than 2.");
    }

    // Padding must be 8 in length
    if (op.getPadding().size() != 8) {
      return rewriter.notifyMatchFailure(
          op, "Number of elements in padding does not match with " +
                  op.getOperationName() + " op.");
    }

    return success();
  }

  // ttir.pooling op supports variadic inputs; so corresponding pooling op (max
  // pool or average pool) is created for each input along with input/output
  // permutation. The last leaf op(s) are returned back which will replace the
  // original op.
  template <typename PoolOpType>
  llvm::SmallVector<Value>
  rewritePool2d(ttir::PoolingOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter,
                llvm::SmallVector<int64_t> spatialDimIndices) const {

    const int64_t SPATIAL_H = -3;
    const int64_t SPATIAL_W = -2;
    const int64_t NON_SPATIAL = -1;

    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getInputs()[0].getType());
    assert(inputType.getRank() == 4 && "Input must be 4D tensor");
    std::vector<int64_t> desiredLayout(inputType.getRank(), NON_SPATIAL);
    desiredLayout[inputType.getRank() - 3] = SPATIAL_H;
    desiredLayout[inputType.getRank() - 2] = SPATIAL_W;

    int64_t nonSpatialCount = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(desiredLayout.size()); i++) {
      if (desiredLayout[i] == NON_SPATIAL) {
        desiredLayout[i] = nonSpatialCount;
        nonSpatialCount++;
      }
    }

    int64_t numWinDims = op.getWindowDimensions().size();
    // Using default indices for channel first tensor if window dimension
    // attribute does not contain two non 1 elements for kernel size.
    // [TODO] (mmanzoor) Add an option to distinguish channel first vs channel
    // last and support channel last default indices.
    // https://github.com/tenstorrent/tt-mlir/issues/2237
    spatialDimIndices =
        (spatialDimIndices.size() == 2)
            ? spatialDimIndices
            : llvm::SmallVector<int64_t>({numWinDims - 2, numWinDims - 1});

    std::vector<int64_t> currentLayout(inputType.getRank(), NON_SPATIAL);
    currentLayout[spatialDimIndices[0]] = SPATIAL_H;
    currentLayout[spatialDimIndices[1]] = SPATIAL_W;

    nonSpatialCount = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(currentLayout.size()); i++) {
      if (currentLayout[i] == NON_SPATIAL) {
        currentLayout[i] = nonSpatialCount;
        nonSpatialCount++;
      }
    }

    auto permutation = ttmlir::utils::generatePermutation(
        llvm::ArrayRef(currentLayout), llvm::ArrayRef(desiredLayout));
    auto inverseOfPermutation = ttmlir::utils::inversePermutation(permutation);

    auto kernelAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(op.getWindowDimensions()[spatialDimIndices[0]]),
        static_cast<int32_t>(op.getWindowDimensions()[spatialDimIndices[1]]),
    });

    auto strideAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(op.getWindowStrides()[spatialDimIndices[0]]),
        static_cast<int32_t>(op.getWindowStrides()[spatialDimIndices[1]]),
    });

    auto dilationAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(op.getWindowDilations()[spatialDimIndices[0]]),
        static_cast<int32_t>(op.getWindowDilations()[spatialDimIndices[1]]),
    });

    auto paddingAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(op.getPadding()[2 * spatialDimIndices[0]]), // top
        static_cast<int32_t>(op.getPadding()[2 * spatialDimIndices[1]]), // left
        static_cast<int32_t>(
            op.getPadding()[2 * spatialDimIndices[0] + 1]), // bottom
        static_cast<int32_t>(
            op.getPadding()[2 * spatialDimIndices[1] + 1]), // right
    });

    auto ceilModeAttr = rewriter.getBoolAttr(false);

    llvm::SmallVector<Value> outputs;
    for (size_t i = 0; i < adaptor.getInputs().size(); i++) {
      Value input = adaptor.getInputs()[i];
      Value originalOutput = adaptor.getOutputs()[i];
      RankedTensorType originalOutputTy =
          mlir::cast<RankedTensorType>(originalOutput.getType());
      // Apply input permutation.
      RankedTensorType inputTy = mlir::cast<RankedTensorType>(input.getType());
      auto inputPermuteShape =
          ::ttmlir::utils::applyPermutation(inputTy.getShape(), permutation);
      input = ttir::utils::createDPSOp<ttir::PermuteOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_permuteInput"),
          inputPermuteShape, inputTy.getElementType(), inputTy.getEncoding(),
          input, permutation);

      // Apply output permutation.
      auto resultPermuteShape = ::ttmlir::utils::applyPermutation(
          originalOutputTy.getShape(), permutation);
      PoolOpType newPool;
      if constexpr (std::is_same_v<PoolOpType, ttir::AvgPool2dOp>) {
        newPool = ttir::utils::createDPSOp<PoolOpType>(
            rewriter, op.getLoc(), resultPermuteShape,
            originalOutputTy.getElementType(), originalOutputTy.getEncoding(),
            input, kernelAttr, strideAttr, dilationAttr, paddingAttr,
            ceilModeAttr, /*count_include_pad=*/rewriter.getBoolAttr(true));
      } else if constexpr (std::is_same_v<PoolOpType, ttir::MaxPool2dOp>) {
        newPool = ttir::utils::createDPSOp<PoolOpType>(
            rewriter, op.getLoc(), resultPermuteShape,
            originalOutputTy.getElementType(), originalOutputTy.getEncoding(),
            input, kernelAttr, strideAttr, dilationAttr, paddingAttr,
            ceilModeAttr);
      } else {
        llvm_unreachable("Pool2dOp must be AvgPool2dOp or MaxPool2dOp");
      }
      // Applying the inverse of permutation to the output will restore the
      // tensor to the original layout.
      auto output = ttir::utils::createDPSOp<ttir::PermuteOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_permuteOutput"),
          originalOutputTy.getShape(), originalOutputTy.getElementType(),
          originalOutputTy.getEncoding(), newPool, inverseOfPermutation);
      outputs.push_back(output);
    }

    return outputs;
  }

  // tt-metal doesn't support sum pooling. Therefore, sum pooling is implemented
  // by performing 'average pooling' multiplied by 'kernel size'. If pooling op
  // has multiple inputs then multiple average pooling op will be created and
  // each will be multiplied with the kernel size. This will return last leaf
  // op(s) (multiply op) which will replace the original op.
  llvm::SmallVector<Value>
  rewriteSumPool2d(ttir::PoolingOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter,
                   llvm::SmallVector<int64_t> spatialDimIndices) const {
    // Create average pooling op.
    llvm::SmallVector<Value> avgPoolOutputs = rewritePool2d<ttir::AvgPool2dOp>(
        op, adaptor, rewriter, spatialDimIndices);

    // Calculate kernel size and create constant op.
    auto kernel = op.getWindowDimensions();
    int64_t kernelSize = std::accumulate(kernel.begin(), kernel.end(),
                                         int64_t{1}, std::multiplies<>());
    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op.getResult(0).getType());
    auto elementType = outputType.getElementType();
    mlir::Attribute constantValue;
    if (mlir::isa<mlir::FloatType>(elementType)) {
      constantValue = mlir::FloatAttr::get(elementType, kernelSize);
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      constantValue = mlir::IntegerAttr::get(elementType, kernelSize);
    } else {
      llvm_unreachable("Un-supported data type for sum pooling 2d op.");
    }

    mlir::DenseElementsAttr constantValueAttr =
        mlir::SplatElementsAttr::get(outputType, constantValue);
    auto constantOp = rewriter.create<ttir::ConstantOp>(
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_constant"),
        outputType, constantValueAttr);

    llvm::SmallVector<Value> sumPoolOutputs;
    // Multiply each average pooling op with kernel size.
    for (Value inputOp : avgPoolOutputs) {
      auto outputOp = ttir::utils::createDPSOp<ttir::MultiplyOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op->getLoc(), "_multiply"),
          outputType, inputOp, constantOp);
      sumPoolOutputs.push_back(outputOp);
    }

    return sumPoolOutputs;
  }
};
} // namespace

// The following pattern rewriter will replace a PoolingOp with a FullOp in the
// case where the pooling operation is applied to the result of a FullOp
namespace {
class PoolingToFullOp : public OpConversionPattern<ttir::PoolingOp> {
public:
  using OpConversionPattern<ttir::PoolingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PoolingOp op, ttir::PoolingOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t kernelSize = std::accumulate(op.getWindowDimensions().begin(),
                                         op.getWindowDimensions().end(), 1,
                                         std::multiplies<int64_t>());
    SmallVector<Value> newResults;
    // If all the inputs are constant ops with splat values then we can easily
    // cannonicalize this
    for (size_t i = 0; i < op.getInputs().size(); i++) {
      ttir::FullOp constant =
          dyn_cast_or_null<ttir::FullOp>(op.getInputs()[i].getDefiningOp());
      if (!constant) {
        return failure();
      }
      ttir::FullOp newConstant;

      std::variant<int64_t, float> constValue;
      std::variant<int64_t, float> newConstValue;

      constValue = isa<IntegerAttr>(constant.getFillValue())
                       ? dyn_cast<IntegerAttr>(constant.getFillValue())
                             .getValue()
                             .getSExtValue()
                       : dyn_cast<FloatAttr>(constant.getFillValue())
                             .getValue()
                             .convertToFloat();

      if (op.getPoolingMethod() == ttir::PoolingMethod::Max ||
          op.getPoolingMethod() == ttir::PoolingMethod::Average) {
        newConstValue = constValue;
      } else if (op.getPoolingMethod() == ttir::PoolingMethod::Sum) {
        // Handle variant multiplication correctly using std::visit
        newConstValue = std::visit(
            [kernelSize](auto &&arg) -> std::variant<int64_t, float> {
              return arg * kernelSize;
            },
            constValue);
      } else {
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "Unknown pooling method");
      }

      mlir::Attribute newConstValueAttr =
          std::holds_alternative<int64_t>(newConstValue)
              ? cast<mlir::Attribute>(IntegerAttr::get(
                    IntegerType::get(rewriter.getContext(), 32),
                    std::get<int64_t>(newConstValue)))
              : cast<mlir::Attribute>(
                    FloatAttr::get(Float32Type::get(rewriter.getContext()),
                                   std::get<float>(newConstValue)));

      newConstant = rewriter.create<ttir::FullOp>(
          op.getLoc(), op.getResult(i).getType(), newConstValueAttr);
      newResults.push_back(newConstant);
    }

    rewriter.replaceOp(
        op, ValueRange(ArrayRef<Value>(newResults.begin(), newResults.end())));
    return success();
  }
};
} // namespace

// IndexSelectOp is converted to a series of SliceStaticOp and potentially a
// ConcatOp if the sliced dimension is sliced multiple times. For example, if
// the input tensor is
//    [[[1, 2, 3],
//      [4, 5, 6],
//      [7, 8, 9],
//      [10, 11, 12],
//      [13, 14, 15],
//      [16, 17, 18]],
//     [[19, 20, 21],
//      [22, 23, 24],
//      [25, 26, 27],
//      [28, 29, 30],
//      [31, 32, 33],
//      [34, 35, 36]]],
//    shape = [2, 6, 3]
// and the IndexSelectOp is dim=1, begin=0, length=2, stride=4, the output
// tensor will be
//    [[[1, 2, 3],
//      [4, 5, 6],
//      [13, 14, 15],
//      [16, 17, 18]],
//     [[19, 20, 21],
//      [22, 23, 24],
//      [31, 32, 33],
//      [34, 35, 36]]],
//    shape = [2, 4, 3]
// In this case 2 slices are created and concatenated to form the output tensor.
// First slice has begins=[0, 0, 0], ends=[2, 2, 3], steps=[1, 1, 1], and the
// second slice has begins=[0, 4, 0], ends=[2, 6, 3], steps=[1, 1, 1].
namespace {
struct SelectToSliceConversionPattern
    : public OpConversionPattern<ttir::IndexSelectOp> {
public:
  using OpConversionPattern<ttir::IndexSelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::IndexSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getType());

    auto inputShape = inputType.getShape();

    int32_t dim =
        op.getDim() < 0 ? inputType.getRank() + op.getDim() : op.getDim();

    int32_t begin = op.getBegin();
    int32_t length = op.getLength();
    int32_t stride = op.getStride();

    int32_t inputDimSize = inputType.getShape()[dim];
    int32_t numSlices = (inputDimSize - begin + stride - 1) / stride;

    llvm::SmallVector<int32_t, 4> begins, ends, steps;
    for (int32_t i = 0; i < inputType.getRank(); ++i) {
      // Always slicing with step 1.
      steps.push_back(1);
      if (i == dim) {
        // Push placeholder values for now which will be updated later.
        begins.push_back(0);
        ends.push_back(0);
        continue;
      }

      // For non-sliced dimensions, begin=0, end=dimSize, step=1.
      begins.push_back(0);
      ends.push_back(inputType.getDimSize(i));
    }

    // Create a slice for each slice of the input tensor. The slices are then
    // concatenated. The slices are created by updating the begin and end values
    // for the sliced dimension.
    llvm::SmallVector<Value> slices;
    for (int32_t i = 0; i < numSlices; ++i) {
      int32_t newBegin = begin + i * stride;
      int32_t newEnd = std::min(newBegin + length, inputDimSize);

      // Make a copy of the input shape and update the dim size.
      llvm::SmallVector<int64_t> resultShape(inputShape);
      resultShape[dim] = newEnd - newBegin;

      begins[dim] = newBegin;
      ends[dim] = newEnd;

      auto newOp = ttir::utils::createDPSOp<ttir::SliceStaticOp>(
          rewriter, op.getLoc(), resultShape, inputType.getElementType(),
          inputType.getEncoding(), adaptor.getInput(),
          rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
          rewriter.getI32ArrayAttr(steps));
      slices.push_back(newOp);
    }

    assert(!slices.empty());
    if (slices.size() > 1) {
      auto concatOp = ttir::utils::createDPSOp<ttir::ConcatOp>(
          rewriter, op.getLoc(), outputType, slices, dim);
      rewriter.replaceOp(op, concatOp);
    } else {
      rewriter.replaceOp(op, slices[0]);
    }

    return success();
  }
};
} // namespace

/*
 * This pattern rewrites ArangeOp by forcing the arange_dimension to be
 * rightmost dimension of the output tensor. This is done by replacing the
 * ArangeOp with a new one that has this property, and then transposing out last
 * dimension to the dimension specified by the original ArangeOp, and also
 * inserting a reshape to match the rank of the intended output and broadcasts
 * to repeat the data along the other dimensions.
 *
 * The ArangeOp that is generated here will be equivalent to how ttnn::ArangeOp
 * behaves.
 */
namespace {
struct ArangeForceLastDimensionPattern
    : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op.getResult().getType());

    int64_t arangeDimension = adaptor.getArangeDimension();
    int64_t start = adaptor.getStart();
    int64_t end = adaptor.getEnd();
    int64_t step = adaptor.getStep();

    int64_t arangeLength = (end - start) / step;

    const llvm::SmallVector<int64_t, 1> requiredShape{arangeLength};
    ArrayRef<int64_t> ttnnShape(requiredShape);
    if (ttnnShape == outputType.getShape()) {
      return success();
    }

    RankedTensorType arangeOutputType = RankedTensorType::get(
        requiredShape, outputType.getElementType(), outputType.getEncoding());

    Value output =
        rewriter
            .create<ttir::ArangeOp>( // perform arange on the last dimension to
                                     // match how ttnn behaves
                op.getLoc(), arangeOutputType, start, end, step, 0)
            .getResult();

    std::vector<int64_t> outputShape = arangeOutputType.getShape().vec();

    // Must match up the rank of the output with the rank of the intended output
    // from the original arange, with the arangeDimension in the correct
    // position
    if (outputType.getRank() != static_cast<int64_t>(outputShape.size())) {
      std::vector<int64_t> reshapeShape;
      for (uint32_t i = 0; i < outputType.getRank(); i++) {
        i == arangeDimension ? reshapeShape.push_back(arangeLength)
                             : reshapeShape.push_back(1);
      }

      output = ttir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeOutput"),
          reshapeShape, outputType.getElementType(), outputType.getEncoding(),
          output,
          rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(
              reshapeShape.begin(), reshapeShape.end())));

      outputShape = std::move(reshapeShape);
    }

    // Must broadcast the rest of the dimensions.
    SmallVector<Attribute> broadcastDims;
    for (uint32_t i = 0; i < outputShape.size(); i++) {
      if (i != arangeDimension && outputShape[i] != outputType.getShape()[i]) {
        outputShape[i] = outputType.getShape()[i];
        broadcastDims.push_back(rewriter.getI64IntegerAttr(i));
      }
    }
    if (!broadcastDims.empty()) {
      RankedTensorType broadcastType = RankedTensorType::get(
          outputShape, outputType.getElementType(), outputType.getEncoding());

      auto inputShape =
          mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      output = ttir::utils::createDPSOp<ttir::BroadcastOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_broadcastOutput"),
          broadcastType, output, broadcastShape);

      assert(mlir::cast<RankedTensorType>(output.getType()).getShape() ==
                 outputType.getShape() &&
             "Output shape must match the shape of the input tensor");
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

// TTNN does not support reduction operation for logical and. So this reduction
// is performed by decomposing/converting into reduction product (ttnn.prod op).
// If ttnn.prod output is zero then reduce_and output is false; otherwise the
// output is true.
namespace {
struct ReductionAndPattern : public OpConversionPattern<ttir::ReduceAndOp> {
public:
  using OpConversionPattern<ttir::ReduceAndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType reduceOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    ttir::utils::replaceOpWithNewDPSOp<ttir::ProdOp>(
        rewriter, op, reduceOutputType, adaptor.getInput(), op.getKeepDim(),
        op.getDimArgAttr());

    return success();
  }
};
} // namespace

// TTNN does not support reduction operation for logical or. So this reduction
// is performed by decomposing/converting into reduction sum (ttnn.sum op).
// If ttnn.sum output is zero then reduce_or output is false; otherwise the
// output is true.
namespace {
struct ReductionOrPattern : public OpConversionPattern<ttir::ReduceOrOp> {
public:
  using OpConversionPattern<ttir::ReduceOrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType reduceOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    ttir::utils::replaceOpWithNewDPSOp<ttir::SumOp>(
        rewriter, op, reduceOutputType, adaptor.getInput(), op.getKeepDim(),
        op.getDimArgAttr());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// BatchNorm decomposition helpers
//===----------------------------------------------------------------------===//

namespace {
// Helper function that ensures input is in NCHW format by permuting and
// reshaping the input tensor. Returns the transformed value and the normalized
// shape.
static std::pair<mlir::Value, llvm::SmallVector<int64_t>>
normalizeToNCHW(mlir::Value input, uint64_t featureIndex,
                ConversionPatternRewriter &rewriter, mlir::Location loc) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::ArrayRef<int64_t> shape = inputType.getShape();
  mlir::Value newInput = input;
  llvm::SmallVector<int64_t> currentShape(shape.begin(), shape.end());

  // If feature index is not 1, permute the input tensor so that the feature
  // dimension is at index 1 (NCHW format).
  if (featureIndex != 1) {
    // Build permutation to move featureIndex to position 1.
    llvm::SmallVector<int64_t> permutation(currentShape.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[featureIndex], permutation[1]);
    llvm::SmallVector<int64_t> permutedShape = ttmlir::utils::applyPermutation(
        llvm::ArrayRef(currentShape), llvm::ArrayRef(permutation));

    newInput = ttir::utils::createDPSOp<mlir::tt::ttir::PermuteOp>(
        rewriter, loc, permutedShape, inputType.getElementType(),
        inputType.getEncoding(), newInput,
        rewriter.getDenseI64ArrayAttr(permutation));
    currentShape = permutedShape;
  }

  // Reshape to 4D NCHW if needed:
  // If rank is 5, flatten last two dimensions into one.
  // If rank is less than 4, unsqueeze trailing dimensions until rank is 4.
  int64_t rank = currentShape.size();
  if (rank == 5) {
    llvm::SmallVector<int64_t> reshapedShape = {
        currentShape[0], currentShape[1], currentShape[2],
        currentShape[3] * currentShape[4]};
    llvm::SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                                reshapedShape.end());
    newInput = ttir::utils::createDPSOp<mlir::tt::ttir::ReshapeOp>(
        rewriter, loc, reshapedShape, inputType.getElementType(),
        inputType.getEncoding(), newInput,
        rewriter.getI32ArrayAttr(reshapedShapeI32));
    currentShape = reshapedShape;
  } else if (rank < 4) {
    llvm::SmallVector<int64_t> reshapedShape(currentShape.begin(),
                                             currentShape.end());
    reshapedShape.append(4 - rank, 1);
    llvm::SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                                reshapedShape.end());
    newInput = ttir::utils::createDPSOp<mlir::tt::ttir::ReshapeOp>(
        rewriter, loc, reshapedShape, inputType.getElementType(),
        inputType.getEncoding(), newInput,
        rewriter.getI32ArrayAttr(reshapedShapeI32));
    currentShape = reshapedShape;
  }

  return {newInput, currentShape};
}

// Helper function to denormalize output back to original layout.
// Forward pass: originalShape -> [permute] -> shapeAfterPermute -> [reshape] ->
// normalizedShape Backward pass: normalizedShape -> [undo reshape] ->
// shapeAfterPermute -> [undo permute] -> originalShape
static mlir::Value denormalizeFromNCHW(mlir::Value output,
                                       llvm::ArrayRef<int64_t> originalShape,
                                       llvm::ArrayRef<int64_t> normalizedShape,
                                       uint64_t originalFeatureIndex,
                                       ConversionPatternRewriter &rewriter,
                                       mlir::Location loc) {
  auto outputType = mlir::cast<mlir::RankedTensorType>(output.getType());
  mlir::Value result = output;

  //  Undo reshape if ranks differ (in reverse order of forward pass)
  if (originalShape.size() != normalizedShape.size()) {
    // Compute the shape after permute but before reshape (the intermediate
    // state). This is what the tensor shape would be if we only applied
    // permutation.
    llvm::SmallVector<int64_t> shapeAfterPermute(originalShape.begin(),
                                                 originalShape.end());
    if (originalFeatureIndex != 1) {
      std::swap(shapeAfterPermute[1], shapeAfterPermute[originalFeatureIndex]);
    }

    llvm::SmallVector<int32_t> shapeAfterPermuteI32(shapeAfterPermute.begin(),
                                                    shapeAfterPermute.end());
    result = ttir::utils::createDPSOp<mlir::tt::ttir::ReshapeOp>(
        rewriter, loc, shapeAfterPermute, outputType.getElementType(),
        outputType.getEncoding(), result,
        rewriter.getI32ArrayAttr(shapeAfterPermuteI32));
  }

  // Step 2: Undo permutation if featureIndex != 1
  if (originalFeatureIndex != 1) {
    // The inverse permutation is the same as the forward permutation
    // (swapping dimensions 1 and featureIndex is its own inverse)
    llvm::SmallVector<int64_t> permutation(originalShape.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[1], permutation[originalFeatureIndex]);

    result = ttir::utils::createDPSOp<mlir::tt::ttir::PermuteOp>(
        rewriter, loc, originalShape, outputType.getElementType(),
        outputType.getEncoding(), result,
        rewriter.getDenseI64ArrayAttr(permutation));
  }

  return result;
}

// Helper function to check if input type is valid for BatchNorm weight tensors
static bool isValidBatchNormWeightType(RankedTensorType inputType) {
  if (inputType.getRank() == 1) {
    return true;
  }
  if (inputType.getRank() == 4) {
    auto shape = inputType.getShape();
    return shape[0] == 1 && shape[2] == 1 && shape[3] == 1;
  }
  return false;
}

// Helper function to reshape BatchNorm weight tensors from 1D to 4D [1, C, 1,
// 1]
static mlir::Value getBatchNorm4DTensor(PatternRewriter &rewriter, Location loc,
                                        mlir::Value batchNormInput) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(batchNormInput.getType());

  if (inputType.getRank() == 4) {
    return batchNormInput;
  }

  auto newShape = llvm::SmallVector<int64_t>{1, inputType.getDimSize(0), 1, 1};
  llvm::SmallVector<int32_t> shape32(newShape.begin(), newShape.end());
  auto shapeAttr = rewriter.getI32ArrayAttr(shape32);

  return ttir::utils::createDPSOp<ttir::ReshapeOp>(
      rewriter, loc, newShape, inputType.getElementType(),
      inputType.getEncoding(), batchNormInput, shapeAttr);
}

// Helper function to reshape BatchNorm weight tensors from 4D [1, C, 1, 1] to
// 1D [C]
static mlir::Value reshapeBatchNorm4DTo1D(PatternRewriter &rewriter,
                                          Location loc, mlir::Value input4D,
                                          RankedTensorType target1DType) {
  auto input4DType = mlir::cast<RankedTensorType>(input4D.getType());

  // If already 1D, return as-is
  if (input4DType.getRank() == 1) {
    return input4D;
  }

  // Extract the channel dimension from [1, C, 1, 1] -> [C]
  llvm::SmallVector<int32_t> shape1D = {
      static_cast<int32_t>(target1DType.getDimSize(0))};

  return ttir::utils::createDPSOp<ttir::ReshapeOp>(
      rewriter, loc, target1DType.getShape(), target1DType.getElementType(),
      target1DType.getEncoding(), input4D, rewriter.getI32ArrayAttr(shape1D));
}
} // namespace

//===----------------------------------------------------------------------===//
// BatchNorm decomposition patterns
//===----------------------------------------------------------------------===//

// This pattern reshapes the non input tensors of the BatchNormInferenceOp to 4D
// tensors, by adding additional dimensions of size 1 so that the only
// non-1 dimension is the second dimension. This is done so that the
// op is compatible with ttnn op call.
namespace {
struct BatchNormInferencePattern
    : public OpConversionPattern<ttir::BatchNormInferenceOp> {
public:
  using OpConversionPattern<ttir::BatchNormInferenceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormInferenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    llvm::ArrayRef<int64_t> originalShape = inputType.getShape();
    uint64_t featureIndex =
        adaptor.getDimensionAttr().getValue().getZExtValue();

    auto meanType = mlir::cast<RankedTensorType>(adaptor.getMean().getType());
    if (!isValidBatchNormWeightType(meanType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp mean must be 1D tensor");
    }

    auto varType =
        mlir::cast<RankedTensorType>(adaptor.getVariance().getType());
    if (!isValidBatchNormWeightType(varType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp var must be 1D or 4D tensor");
    }

    auto weightType =
        mlir::cast<RankedTensorType>(adaptor.getScale().getType());
    if (!isValidBatchNormWeightType(weightType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp weight must be 1D or 4D tensor");
    }

    auto biasType = mlir::cast<RankedTensorType>(adaptor.getOffset().getType());
    if (!isValidBatchNormWeightType(biasType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp bias must be 1D or 4D tensor");
    }

    // Normalize input to NCHW format
    auto [normalizedInput, normalizedShape] =
        normalizeToNCHW(adaptor.getOperand(), featureIndex, rewriter, loc);

    // Reshape weight tensors to 4D (existing logic for TTNN compatibility)
    mlir::Value mean4D = getBatchNorm4DTensor(rewriter, loc, adaptor.getMean());
    mlir::Value variance4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getVariance());
    mlir::Value scale4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getScale());
    mlir::Value offset4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getOffset());

    // Create output type with normalized shape
    auto originalOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto normalizedOutputType = RankedTensorType::get(
        normalizedShape, originalOutputType.getElementType(),
        originalOutputType.getEncoding());

    // After normalization, feature dimension is always at index 1 (NCHW)
    mlir::Type integerType = mlir::IntegerType::get(rewriter.getContext(), 32);
    IntegerAttr dimensionAttr = mlir::IntegerAttr::get(integerType, 1);

    // Create the BatchNorm op with normalized input and 4D weight tensors
    auto batchNormInferenceOp =
        ttir::utils::createDPSOp<mlir::tt::ttir::BatchNormInferenceOp>(
            rewriter, loc, normalizedOutputType, normalizedInput, scale4D,
            offset4D, mean4D, variance4D, adaptor.getEpsilonAttr(),
            dimensionAttr);

    // Denormalize output back to original layout
    mlir::Value result =
        denormalizeFromNCHW(batchNormInferenceOp.getResult(), originalShape,
                            normalizedShape, featureIndex, rewriter, loc);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// This pattern reshapes the non input tensors of the BatchNormTrainingOp to 4D
// tensors so that the resulting BatchNormTrainingOp is compatible with ttnn op
// call.
namespace {
struct BatchNormTrainingPattern
    : public OpConversionPattern<ttir::BatchNormTrainingOp> {
public:
  using OpConversionPattern<ttir::BatchNormTrainingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormTrainingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    llvm::ArrayRef<int64_t> originalShape = inputType.getShape();
    uint64_t featureIndex =
        adaptor.getDimensionAttr().getValue().getZExtValue();

    auto scaleType = mlir::cast<RankedTensorType>(adaptor.getScale().getType());
    if (!isValidBatchNormWeightType(scaleType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp scale must be 1D or 4D tensor");
    }

    auto offsetType =
        mlir::cast<RankedTensorType>(adaptor.getOffset().getType());
    if (!isValidBatchNormWeightType(offsetType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp offset must be 1D or 4D tensor");
    }

    auto meanType =
        mlir::cast<RankedTensorType>(adaptor.getRunningMean().getType());
    if (!isValidBatchNormWeightType(meanType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp running_mean must be 1D or 4D tensor");
    }

    auto varType =
        mlir::cast<RankedTensorType>(adaptor.getRunningVariance().getType());
    if (!isValidBatchNormWeightType(varType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp running_variance must be 1D or 4D tensor");
    }

    // Normalize input to NCHW format
    auto [normalizedInput, normalizedShape] =
        normalizeToNCHW(adaptor.getOperand(), featureIndex, rewriter, loc);

    // Reshape all weight tensors to 4D (for TTNN compatibility)
    mlir::Value scale4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getScale());
    mlir::Value offset4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getOffset());
    mlir::Value mean4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getRunningMean());
    mlir::Value variance4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getRunningVariance());

    // Create output types with normalized shape and 4D weight tensors
    auto originalOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto normalizedOutputType = RankedTensorType::get(
        normalizedShape, originalOutputType.getElementType(),
        originalOutputType.getEncoding());

    // running_mean and running_variance results should be 4D [1, C, 1, 1]
    auto mean4DType = mlir::cast<RankedTensorType>(mean4D.getType());
    auto variance4DType = mlir::cast<RankedTensorType>(variance4D.getType());

    // Create new empty tensors with normalized shapes for DPS outputs
    auto outputEmpty =
        rewriter.create<ttir::EmptyOp>(loc, normalizedOutputType).getResult();
    auto batchMeanEmpty =
        rewriter.create<ttir::EmptyOp>(loc, mean4DType).getResult();
    auto batchVarianceEmpty =
        rewriter.create<ttir::EmptyOp>(loc, variance4DType).getResult();

    // After normalization, feature dimension is always at index 1 (NCHW)
    mlir::Type integerType = mlir::IntegerType::get(rewriter.getContext(), 32);
    IntegerAttr dimensionAttr = mlir::IntegerAttr::get(integerType, 1);

    // Create new BatchNormTrainingOp with normalized input and all 4D weight
    // tensors
    auto batchNormTrainingOp = rewriter.create<ttir::BatchNormTrainingOp>(
        loc, TypeRange{normalizedOutputType, mean4DType, variance4DType},
        normalizedInput, scale4D, offset4D, mean4D, variance4D,
        ValueRange{outputEmpty, batchMeanEmpty, batchVarianceEmpty},
        adaptor.getEpsilonAttr(), dimensionAttr, adaptor.getMomentumAttr());

    // Denormalize the output (first result) back to original layout
    mlir::Value denormalizedOutput =
        denormalizeFromNCHW(batchNormTrainingOp.getResults()[0], originalShape,
                            normalizedShape, featureIndex, rewriter, loc);

    // Reshape batch_mean and batch_variance from 4D [1, C, 1, 1] back to 1D [C]
    auto originalMeanType =
        mlir::cast<RankedTensorType>(op.getBatchMean().getType());
    auto originalVarianceType =
        mlir::cast<RankedTensorType>(op.getBatchVariance().getType());

    mlir::Value reshapedMean = reshapeBatchNorm4DTo1D(
        rewriter, loc, batchNormTrainingOp.getResults()[1], originalMeanType);
    mlir::Value reshapedVariance = reshapeBatchNorm4DTo1D(
        rewriter, loc, batchNormTrainingOp.getResults()[2],
        originalVarianceType);

    // Replace with denormalized output and reshaped mean/variance results
    rewriter.replaceOp(
        op, ValueRange{denormalizedOutput, reshapedMean, reshapedVariance});

    return success();
  }
};
} // namespace

// Utility function to get scale and zero point for quantized types.
static std::pair<mlir::Value, mlir::Value>
getScaleAndZeroPoint(mlir::quant::QuantizedType elementType,
                     ConversionPatternRewriter &rewriter, mlir::Location loc) {
  // Per-tensor quantization.
  if (auto quantPerTensorType =
          mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
    // Create ttir::ConstantOp for scale.
    float scaleValue = quantPerTensorType.getScale();
    mlir::RankedTensorType scaleType =
        mlir::RankedTensorType::get({1}, rewriter.getF32Type());
    mlir::DenseFPElementsAttr scaleDenseAttr =
        mlir::DenseFPElementsAttr::get(scaleType, scaleValue);
    ttir::ConstantOp scaleConstant =
        rewriter.create<ttir::ConstantOp>(loc, scaleType, scaleDenseAttr);

    // Create ttir::ConstantOp for zero point.
    int32_t zeroPoint = static_cast<int32_t>(quantPerTensorType.getZeroPoint());
    mlir::RankedTensorType zeroPointType = mlir::RankedTensorType::get(
        {1}, IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed));
    mlir::DenseIntElementsAttr zeroPointDenseAttr =
        mlir::DenseIntElementsAttr::get(zeroPointType, zeroPoint);
    ttir::ConstantOp zeroPointConstant = rewriter.create<ttir::ConstantOp>(
        loc, zeroPointType, zeroPointDenseAttr);
    return {scaleConstant.getResult(), zeroPointConstant.getResult()};
  }

  // Per-axis quantization.
  if (auto quantPerAxisType =
          mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
              elementType)) {
    // Create ttir::ConstantOp for scale.
    SmallVector<float> scales(
        llvm::to_vector_of<float>(quantPerAxisType.getScales()));
    mlir::RankedTensorType scaleType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(scales.size())}, rewriter.getF32Type());
    mlir::DenseFPElementsAttr scaleDenseAttr =
        mlir::DenseFPElementsAttr::get(scaleType, scales);
    ttir::ConstantOp scaleConstant =
        rewriter.create<ttir::ConstantOp>(loc, scaleType, scaleDenseAttr);

    // Create ttir::ConstantOp for zero point.
    SmallVector<int32_t> zeroPoints(
        llvm::to_vector_of<int32_t>(quantPerAxisType.getZeroPoints()));
    mlir::RankedTensorType zeroPointType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(zeroPoints.size())},
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed));
    mlir::DenseIntElementsAttr zeroPointDenseAttr =
        mlir::DenseIntElementsAttr::get(zeroPointType, zeroPoints);
    ttir::ConstantOp zeroPointConstant = rewriter.create<ttir::ConstantOp>(
        loc, zeroPointType, zeroPointDenseAttr);
    return {scaleConstant.getResult(), zeroPointConstant.getResult()};
  }

  return {nullptr, nullptr};
}

// Utility function to get axis for quantized types.
static IntegerAttr getAxis(mlir::quant::QuantizedType elementType,
                           ConversionPatternRewriter &rewriter) {
  IntegerAttr axis;
  if (auto perAxisType =
          mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
              elementType)) {
    axis = rewriter.getI32IntegerAttr(perAxisType.getQuantizedDimension());
  }
  return axis;
}

// TTNN runtime requires scale and zero point to be treated as input operands
// to quantize and dequantize ops. This reduction creates constant ops for scale
// and zero point and populates the TTIR quantize/dequantize ops with these
// constants as inputs.
namespace {
template <typename QuantizeOpTy, typename QuantizeUnrolledOpTy>
class QuantizationOpConversionPatternBase
    : public OpConversionPattern<QuantizeOpTy> {
public:
  using OpConversionPattern<QuantizeOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QuantizeOpTy op,
                  typename OpConversionPattern<QuantizeOpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::quant::QuantizedType elementType = getQuantizedElementType(op);
    if (!elementType) {
      return failure();
    }
    auto [scale, zeroPoint] =
        getScaleAndZeroPoint(elementType, rewriter, op.getLoc());
    if (!scale) {
      return rewriter.notifyMatchFailure(
          op, "Failed to extract scale and zero point from quantized type.");
    }
    IntegerAttr axisAttr = getAxis(elementType, rewriter);
    mlir::Type quantizeOutputType =
        this->getTypeConverter()->convertType(op.getOutput().getType());

    rewriter.replaceOpWithNewOp<QuantizeUnrolledOpTy>(
        op, quantizeOutputType, adaptor.getInput(), scale, zeroPoint, axisAttr,
        adaptor.getOutput());
    return success();
  }

protected:
  virtual mlir::quant::QuantizedType
  getQuantizedElementType(QuantizeOpTy op) const = 0;
};

struct QuantizeOpPattern
    : public QuantizationOpConversionPatternBase<ttir::QuantizeOp,
                                                 ttir::QuantizeUnrolledOp> {
  using QuantizationOpConversionPatternBase::
      QuantizationOpConversionPatternBase;

protected:
  mlir::quant::QuantizedType
  getQuantizedElementType(ttir::QuantizeOp op) const override {
    mlir::RankedTensorType outputType = op.getOutput().getType();
    return mlir::dyn_cast<mlir::quant::QuantizedType>(
        outputType.getElementType());
  }
};

struct DequantizeOpPattern
    : public QuantizationOpConversionPatternBase<ttir::DequantizeOp,
                                                 ttir::DequantizeUnrolledOp> {
  using QuantizationOpConversionPatternBase::
      QuantizationOpConversionPatternBase;

protected:
  mlir::quant::QuantizedType
  getQuantizedElementType(ttir::DequantizeOp op) const override {
    mlir::RankedTensorType inputType = op.getInput().getType();
    return mlir::dyn_cast<mlir::quant::QuantizedType>(
        inputType.getElementType());
  }
};

struct RequantizeOpPattern : public OpConversionPattern<ttir::RequantizeOp> {
public:
  using OpConversionPattern<ttir::RequantizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RequantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType inputType = op.getInput().getType();
    mlir::RankedTensorType outputType = op.getOutput().getType();

    mlir::quant::QuantizedType inputElementType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(inputType.getElementType());
    mlir::quant::QuantizedType outputElementType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(outputType.getElementType());

    if (!inputElementType || !outputElementType) {
      return failure();
    }

    auto [inputScale, inputZeroPoint] =
        getScaleAndZeroPoint(inputElementType, rewriter, op.getLoc());
    if (!inputScale) {
      return rewriter.notifyMatchFailure(
          op,
          "Failed to extract input scale and zero point from quantized type.");
    }

    auto [outputScale, outputZeroPoint] =
        getScaleAndZeroPoint(outputElementType, rewriter, op.getLoc());
    if (!outputScale) {
      return rewriter.notifyMatchFailure(
          op,
          "Failed to extract output scale and zero point from quantized type.");
    }

    IntegerAttr axisAttr = getAxis(inputElementType, rewriter);
    mlir::Type requantizeOutputType =
        this->getTypeConverter()->convertType(op.getOutput().getType());

    rewriter.replaceOpWithNewOp<ttir::RequantizeUnrolledOp>(
        op, requantizeOutputType, adaptor.getInput(), inputScale,
        inputZeroPoint, outputScale, outputZeroPoint, axisAttr,
        adaptor.getOutput());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ScatterOp decomposition to ScatterInDimOp
//===----------------------------------------------------------------------===//

// This decomposition transforms TTIR ScatterOp into TTIR ScatterInDimOp.
// TTIR ScatterOp follows stablehlo specification. TTIR ScatterInDimOp follows
// PyTorch specification.
namespace {
class ScatterToScatterInDimPattern
    : public OpConversionPattern<ttir::ScatterOp> {
  using OpConversionPattern<ttir::ScatterOp>::OpConversionPattern;

  LogicalResult checkBasicLegality(ttir::ScatterOp &op,
                                   PatternRewriter &rewriter) const {
    auto input_batching_dims = op.getInputBatchingDims();
    auto scatter_indices_batching_dims = op.getScatterIndicesBatchingDims();
    if (!input_batching_dims.empty() ||
        !scatter_indices_batching_dims.empty()) {
      return rewriter.notifyMatchFailure(
          op, "ScatterInDim doesn't currently support scatter with batching "
              "dimensions");
    }

    // Validate update_window_dims and inserted_window_dims.
    ArrayRef<int32_t> updateWindowDims = op.getUpdateWindowDims();
    ArrayRef<int32_t> insertedWindowDims = op.getInsertedWindowDims();

    // Get update tensor rank and shape.
    RankedTensorType updateType = op.getUpdate().getType();
    int64_t updateRank = updateType.getRank();
    ArrayRef<int64_t> updateShape = updateType.getShape();

    // Get index tensor shape.
    RankedTensorType indexType = op.getScatterIndices().getType();
    ArrayRef<int64_t> indexShape = indexType.getShape();

    // Create array to track which dimensions are covered.
    llvm::SmallVector<bool> dimsCovered(updateRank, false);

    // Check update_window_dims.
    for (auto dim : updateWindowDims) {
      if (dim < 0 || dim >= updateRank) {
        return rewriter.notifyMatchFailure(
            op, "update_window_dims contains invalid dimension index");
      }
      dimsCovered[dim] = true;
    }

    // Check inserted_window_dims.
    for (auto dim : insertedWindowDims) {
      if (dim < 0 || dim >= updateRank) {
        return rewriter.notifyMatchFailure(
            op, "inserted_window_dims contains invalid dimension index");
      }
      if (dimsCovered[dim]) {
        return rewriter.notifyMatchFailure(
            op, "update_window_dims and inserted_window_dims have overlapping "
                "dimensions");
      }
      dimsCovered[dim] = true;
    }

    // Check that all dimensions are covered.
    for (int64_t i = 0; i < updateRank; ++i) {
      if (!dimsCovered[i]) {
        return rewriter.notifyMatchFailure(
            op, "ScatterInDim does not support window scatter.");
      }
    }

    // Check that scatter_dims_to_operand_dims is in order.
    ArrayRef<int32_t> scatterDimsToOperandDims =
        op.getScatterDimsToOperandDims();
    if (!llvm::is_sorted(scatterDimsToOperandDims)) {
      return rewriter.notifyMatchFailure(
          op,
          "scatter_dims_to_operand_dims must be in strictly increasing order.");
    }

    bool multiDimensionalScatter = scatterDimsToOperandDims.size() > 1;
    uint32_t indexVectorDim = op.getIndexVectorDim();

    // Checks that apply to multi dimensional scatter.

    if (multiDimensionalScatter &&
        indexVectorDim != static_cast<uint32_t>(indexShape.size() - 1)) {
      return rewriter.notifyMatchFailure(
          op, "TTIR multi-dimensional scatter currently only supports "
              "index_vector_dim being the last dimension");
    }

    if (multiDimensionalScatter && !updateWindowDims.empty()) {
      return rewriter.notifyMatchFailure(
          op, "TTIR multi-dimensional scatter requires update_window_dims to "
              "be empty");
    }

    // Checks that apply to single dimensional scatter.

    if (!multiDimensionalScatter && indexShape.size() > updateShape.size()) {
      return rewriter.notifyMatchFailure(
          op, "TTIR scatter requires indices.rank <= updates.rank. Please add "
              "support for rank promotion if needed.");
    }

    if (!multiDimensionalScatter && indexVectorDim != 1u) {
      return rewriter.notifyMatchFailure(
          op,
          "TTIR single dimensional scatter requires index_vector_dim to be 1");
    }

    if (!multiDimensionalScatter && scatterDimsToOperandDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "TTIR single dimensional scatter currently only supports "
              "scattering along dimension 0");
    }
    return success();
  }

  Value flattenTensor(PatternRewriter &rewriter, Location loc, Value tensor,
                      const std::string &suffix = "") const {
    RankedTensorType tensorType =
        mlir::cast<RankedTensorType>(tensor.getType());
    ArrayRef<int64_t> tensorShape = tensorType.getShape();

    // Calculate total number of elements (product of all dimensions).
    int64_t totalElements = 1;
    for (int64_t dim : tensorShape) {
      totalElements *= dim;
    }

    // Create new 1D shape.
    llvm::SmallVector<int64_t> flattenedShape = {totalElements};

    // Create location with optional suffix.
    Location reshapeLocation =
        suffix.empty() ? loc : ttmlir::utils::appendLocationSuffix(loc, suffix);

    // Reshape tensor to 1D.
    Value flattenedTensor =
        createReshapeOp(rewriter, reshapeLocation, tensor, flattenedShape);

    return flattenedTensor;
  }

  Value extractElementWiseScatterIndices(ttir::ScatterOp op,
                                         PatternRewriter &rewriter) const {
    // Indices need to match updates tensor.
    TypedValue<RankedTensorType> indexTensor = op.getScatterIndices();
    RankedTensorType updateType = op.getUpdate().getType();
    RankedTensorType indexType = indexTensor.getType();
    ArrayRef<int64_t> indexShape = indexType.getShape();
    ArrayRef<int64_t> updateShape = updateType.getShape();

    if (indexShape.size() < updateShape.size()) {
      // Need to reshape indices by appending 1s to the shape.
      llvm::SmallVector<int64_t> newShape(indexShape.begin(), indexShape.end());
      newShape.resize(updateShape.size(), 1);

      indexTensor =
          createReshapeOp(rewriter, op.getLoc(), indexTensor, newShape);
      indexType = mlir::cast<RankedTensorType>(indexTensor.getType());
      indexShape = newShape;
    }

    // Repeat along update_window_dims to match update tensor shape.
    ArrayRef<int32_t> updateWindowDims = op.getUpdateWindowDims();
    llvm::SmallVector<int64_t> repeatDims(indexShape.size(), 1);
    bool needsRepeat = false;

    // For each update_window_dim, set repeat factor to match update tensor
    // size.
    for (auto dimAttr : updateWindowDims) {
      int64_t dim = dimAttr;
      if (indexShape[dim] != updateShape[dim]) {
        repeatDims[dim] = updateShape[dim];
        needsRepeat = true;
      }
    }

    if (needsRepeat) {
      llvm::SmallVector<int64_t> targetIndexShape(updateShape.begin(),
                                                  updateShape.end());
      RankedTensorType targetIndexType =
          RankedTensorType::get(targetIndexShape, indexType.getElementType(),
                                indexType.getEncoding());
      auto repeatDimsAttr = rewriter.getDenseI64ArrayAttr(repeatDims);

      indexTensor = ttir::utils::createDPSOp<ttir::RepeatOp>(
          rewriter, op.getLoc(), targetIndexType, indexTensor, repeatDimsAttr);
    }

    return indexTensor;
  }

  Value extractMultiDimensionalScatterIndices(ttir::ScatterOp op,
                                              PatternRewriter &rewriter) const {
    // Last dimension of indices is index_vector_dim.
    TypedValue<RankedTensorType> indexTensor = op.getScatterIndices();
    RankedTensorType indexType = indexTensor.getType();
    ArrayRef<int64_t> indexShape = indexType.getShape();
    int64_t indexVectorDim = op.getIndexVectorDim();

    // Get the input tensor to determine its shape for stride calculation.
    TypedValue<RankedTensorType> inputTensor = op.getInput();
    RankedTensorType inputType = inputTensor.getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();

    // Number of dimensions being indexed.
    int64_t numIndexDims = indexShape[indexVectorDim];

    // Calculate strides for each dimension (product of subsequent dimensions).
    llvm::SmallVector<int64_t> strides(numIndexDims);
    for (int64_t i = 0; i < numIndexDims; ++i) {
      int64_t stride = 1;
      for (int64_t j = i + 1; j < numIndexDims; ++j) {
        stride *= inputShape[j];
      }
      strides[i] = stride;
    }

    Value flatIndices = nullptr;

    // Process each dimension.
    for (int64_t dim = 0; dim < numIndexDims; ++dim) {
      // Slice to get indices for this dimension.
      llvm::SmallVector<int32_t> begins(indexType.getRank(), 0);
      llvm::SmallVector<int32_t> ends(indexType.getShape().begin(),
                                      indexType.getShape().end());
      llvm::SmallVector<int32_t> steps(indexType.getRank(), 1);

      begins[indexVectorDim] = static_cast<int32_t>(dim);
      ends[indexVectorDim] = static_cast<int32_t>(dim + 1);

      // Calculate slice shape.
      llvm::SmallVector<int64_t> sliceShape(indexType.getShape());
      sliceShape[indexVectorDim] = 1;

      auto beginsAttr = rewriter.getI32ArrayAttr(begins);
      auto endsAttr = rewriter.getI32ArrayAttr(ends);
      auto stepsAttr = rewriter.getI32ArrayAttr(steps);

      Value dimensionIndices = ttir::utils::createDPSOp<ttir::SliceStaticOp>(
          rewriter,
          ttmlir::utils::appendLocationSuffix(
              op.getLoc(), "_dim_" + std::to_string(dim) + "_slice"),
          sliceShape, indexType.getElementType(), indexType.getEncoding(),
          indexTensor, beginsAttr, endsAttr, stepsAttr);

      // Multiply by stride if stride > 1.
      if (strides[dim] > 1) {
        auto scalarAttr =
            rewriter.getI32IntegerAttr(static_cast<int32_t>(strides[dim]));

        RankedTensorType dimIndexType = RankedTensorType::get(
            sliceShape, indexType.getElementType(), indexType.getEncoding());

        Value strideTensor = rewriter.create<ttir::FullOp>(
            ttmlir::utils::appendLocationSuffix(
                op.getLoc(), "_stride_" + std::to_string(dim)),
            dimIndexType, scalarAttr);

        dimensionIndices = ttir::utils::createDPSOp<ttir::MultiplyOp>(
            rewriter,
            ttmlir::utils::appendLocationSuffix(
                op.getLoc(), "_dim_" + std::to_string(dim) + "_stride_mul"),
            sliceShape, indexType.getElementType(), indexType.getEncoding(),
            dimensionIndices, strideTensor);
      }

      // Add to flat indices.
      if (flatIndices == nullptr) {
        flatIndices = dimensionIndices;
      } else {
        flatIndices = ttir::utils::createDPSOp<ttir::AddOp>(
            rewriter,
            ttmlir::utils::appendLocationSuffix(
                op.getLoc(), "_add_dim_" + std::to_string(dim)),
            sliceShape, indexType.getElementType(), indexType.getEncoding(),
            flatIndices, dimensionIndices);
      }
    }

    // Flatten the indices to 1D.
    Value flattenedIndices =
        flattenTensor(rewriter, op.getLoc(), flatIndices, "_indices_flatten");

    TT_assertv(flattenedIndices, "Expected valid flat indices tensor");
    return flattenedIndices;
  }

  static ttir::ReshapeOp
  createReshapeOp(PatternRewriter &rewriter, Location loc, Value input,
                  ::llvm::ArrayRef<int64_t> targetShape) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(targetShape));

    return ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, loc, targetShape, inputType.getElementType(),
        inputType.getEncoding(), input, shapeAttr);
  }

public:
  LogicalResult
  matchAndRewrite(ttir::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(op, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    TypedValue<RankedTensorType> inputTensor = op.getInput();
    TypedValue<RankedTensorType> updateTensor = op.getUpdate();
    TypedValue<RankedTensorType> outputTensor = op.getOutput();
    RankedTensorType inputType = inputTensor.getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();

    auto scatterDimsToOperandDims = op.getScatterDimsToOperandDims();

    // Check if single dimension scatter.
    if (scatterDimsToOperandDims.size() == 1) {
      // Single-dimensional scatter.
      int32_t dim = scatterDimsToOperandDims[0];

      // Process indices to match update tensor shape.
      Value finalIndexTensor = extractElementWiseScatterIndices(op, rewriter);

      auto dimAttr = rewriter.getI32IntegerAttr(dim);

      // Get the expected output type.
      auto outputType = op.getResult().getType();

      // Create ScatterInDimOp.
      rewriter.replaceOpWithNewOp<ttir::ScatterInDimOp>(
          op, outputType, inputTensor, finalIndexTensor, updateTensor,
          outputTensor, dimAttr);

      return success();
    }

    if (scatterDimsToOperandDims.size() > 1) {
      // Multi-dimensional scatter.
      int32_t dim =
          0; // Always scatter along dimension 0 for flattened tensors.

      // Extract multi-dimensional indices and flatten to 1D.
      Value finalIndexTensor =
          extractMultiDimensionalScatterIndices(op, rewriter);

      // Flatten input tensor to 1D.
      Value flattenedInput =
          flattenTensor(rewriter, op.getLoc(), inputTensor, "_input_flatten");

      // Flatten update tensor to 1D.
      Value flattenedUpdate =
          flattenTensor(rewriter, op.getLoc(), updateTensor, "_update_flatten");

      // Perform scatter operation on flattened tensors.
      auto dimAttr = rewriter.getI32IntegerAttr(dim);

      // Get flattened result type.
      RankedTensorType flattenedInputType =
          mlir::cast<RankedTensorType>(flattenedInput.getType());

      Value scatterResult = ttir::utils::createDPSOp<ttir::ScatterInDimOp>(
          rewriter, op.getLoc(), flattenedInputType.getShape(),
          flattenedInputType.getElementType(), flattenedInputType.getEncoding(),
          flattenedInput, finalIndexTensor, flattenedUpdate, dimAttr);

      // Reshape result back to original input shape.
      Value reshapedResult = createReshapeOp(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_result_reshape"),
          scatterResult, inputShape);

      rewriter.replaceOp(op, reshapedResult);
      return success();
    }

    return failure();
  }
};
} // namespace

// TTNN api supports product reduction along one or all dimensions. This
// decomposition will transform product reduction op to multiple reduction ops.
// Each op will perform reduction along one dimension only.
namespace {
struct ReductionProdPattern : public OpConversionPattern<ttir::ProdOp> {
public:
  using OpConversionPattern<ttir::ProdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ProdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimArg = op.getDimArg();
    if (!dimArg) {
      return failure();
    }

    uint64_t rank = op.getInput().getType().getRank();
    uint64_t dimArgSize = dimArg->size();
    if (dimArgSize == 1 || dimArgSize == rank) {
      return failure();
    }

    // Extract reduction dimensions.
    llvm::SmallVector<int32_t> reduceDims(
        llvm::map_to_vector(*dimArg, [](Attribute dim) -> int32_t {
          return mlir::cast<IntegerAttr>(dim).getInt();
        }));

    // Reduce dimensions are sorted in descending order to apply reduction on
    // higher dimension first. This helps to avoid modifying dimArg which will
    // be required in case of applying reduction on lower dimension first.
    llvm::sort(reduceDims, std::greater<>());

    Value runningProdOp = op.getInput();
    llvm::SmallVector<int64_t> shape{op.getInput().getType().getShape()};
    auto elementType = op.getInput().getType().getElementType();
    bool keepDim = op.getKeepDim();

    for (int dim : reduceDims) {
      mlir::ArrayAttr dimArg =
          rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(/*Size=*/1, dim));
      if (keepDim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }

      RankedTensorType outputType = RankedTensorType::get(shape, elementType);
      runningProdOp = ttir::utils::createDPSOp<ttir::ProdOp>(
          rewriter, op->getLoc(), outputType, runningProdOp,
          op.getKeepDimAttr(), dimArg);
    }

    rewriter.replaceOp(op, runningProdOp);
    return success();
  }
};
} // namespace

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<PoolingToPool2dPattern>(typeConverter, ctx);
  patterns.add<PoolingToFullOp>(typeConverter, ctx);
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<Legalize1DConvolutionPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);
  patterns.add<GatherToEmbeddingConversionPattern>(typeConverter, ctx);
  patterns.add<SelectToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<ArangeForceLastDimensionPattern>(typeConverter, ctx);
  patterns.add<DotGeneralToMatmulConversionPattern>(typeConverter, ctx);
  patterns.add<ReductionAndPattern>(typeConverter, ctx);
  patterns.add<ReductionOrPattern>(typeConverter, ctx);
  patterns.add<BatchNormInferencePattern>(typeConverter, ctx);
  patterns.add<BatchNormTrainingPattern>(typeConverter, ctx);
  patterns.add<QuantizeOpPattern>(typeConverter, ctx);
  patterns.add<DequantizeOpPattern>(typeConverter, ctx);
  patterns.add<RequantizeOpPattern>(typeConverter, ctx);
  patterns.add<ReductionProdPattern>(typeConverter, ctx);
  patterns.add<ScatterToScatterInDimPattern>(typeConverter, ctx);
  patterns.add<ReverseOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
