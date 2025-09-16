// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

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

    // Not currently support batch groups
    if (op.getBatchGroupCount() != 1) {
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

    // The shapes that the convolution currently operates with have are 3D, and
    // we need to add another dimension for it to match the conv2d signature, so
    // adding a dimension of size 1 to the end of input and output shapes.
    auto outputType =
        mlir::cast<RankedTensorType>(adaptor.getOutput().getType());
    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
    llvm::SmallVector<int64_t> conv2dOutputShape(outputShape);
    conv2dOutputShape.push_back(1);

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t, 4> reshapeInputShape(inputShape.begin(),
                                                    inputShape.end());
    reshapeInputShape.push_back(1);

    auto weightType =
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    llvm::ArrayRef<int64_t> weightShape = weightType.getShape();
    llvm::SmallVector<int64_t, 4> reshapeWeightShape(weightShape.begin(),
                                                     weightShape.end());
    reshapeWeightShape.push_back(1);

    ttir::ReshapeOp reshapeInput = createReshapeOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeInput"),
        adaptor.getInput(), reshapeInputShape);
    ttir::ReshapeOp reshapeWeight = createReshapeOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeWeight"),
        adaptor.getWeight(), reshapeWeightShape);

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

    auto convolutionLayout = adaptor.getConvolutionLayoutAttr();

    // The additional spatial dimension is added at the and (3rd in 0 indexed
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
        rewriter, op.getLoc(), conv2dOutputShape, outputType.getElementType(),
        outputType.getEncoding(), reshapeInput, reshapeWeight, Value(),
        conv2dOpWindowsStridesAttr, conv2dOpPaddingAttr,
        conv2dOpInputDilationAttr, conv2dOpWeightDilationAttr,
        conv2dOpWindowReversalAttr,
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
        adaptor.getFeatureGroupCountAttr(), adaptor.getBatchGroupCountAttr());

    ttir::ReshapeOp reshapeOutput = createReshapeOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeOutput"),
        new2dConvolutionOp, outputShape);

    rewriter.replaceOp(op, reshapeOutput);

    return success();
  }

private:
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

    auto convLayoutAttr = op.getConvolutionLayoutAttr();
    // [TODO](mmanzoor) Verify the implementation of transposed convolution for
    // tt-xla. https://github.com/tenstorrent/tt-mlir/issues/3293
    // Determine if the stablehlo.convolution op represents a regular or
    // transposed convolution, based on Torch-MLIR lowering patterns.
    // https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Linear.cpp
    bool isTransposed =
        convLayoutAttr.getKernelInputFeatureDimension() ==
            convLayoutAttr.getInputSpatialDimensions()[SPATIAL_DIM_WIDTH] &&
        convLayoutAttr.getKernelOutputFeatureDimension() ==
            convLayoutAttr.getInputSpatialDimensions()[SPATIAL_DIM_HEIGHT] &&
        convLayoutAttr.getInputSpatialDimensions() !=
            convLayoutAttr.getKernelSpatialDimensions() &&
        convLayoutAttr.getOutputSpatialDimensions() !=
            convLayoutAttr.getKernelSpatialDimensions();

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

    llvm::ArrayRef<int64_t> outputShape = op.getResult().getType().getShape();
    llvm::SmallVector<int64_t> newOutputShape{
        outputShape[adaptor.getConvolutionLayout().getOutputBatchDimension()],
        outputShape[adaptor.getConvolutionLayout()
                        .getOutputSpatialDimensions()[SPATIAL_DIM_HEIGHT]],
        outputShape[adaptor.getConvolutionLayout()
                        .getOutputSpatialDimensions()[SPATIAL_DIM_WIDTH]],
        outputShape[adaptor.getConvolutionLayout()
                        .getOutputFeatureDimension()]};

    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    RankedTensorType outputType = inputType.clone(newOutputShape);

    auto permutation = generateConvPermutation(op, conv2dLayout);
    auto permuteOutputShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
    auto input = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(op.getLoc(), "_input"),
        permuteOutputShape, inputType.getElementType(), inputType.getEncoding(),
        adaptor.getInput(), permutation);

    auto weight = adaptor.getWeight();
    // TTNN api handles reversing weights internally for transposed convolution.
    // So ttir.reverse op is ignored and its input is used as weight.
    if (auto reverseOp = weight.getDefiningOp<mlir::tt::ttir::ReverseOp>();
        isTransposed && reverseOp) {
      weight = reverseOp.getInput();
    }
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());
    auto kernelPermutation = generateConvKernelPermutation(
        op, isTransposed ? conv2dTransposeKernelLayout : conv2dKernelLayout);
    auto weightOutputShape = ::ttmlir::utils::applyPermutation(
        mlir::cast<RankedTensorType>(weight.getType()).getShape(),
        kernelPermutation);
    weight = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(op.getLoc(), "_weight"),
        weightOutputShape, weightType.getElementType(),
        weightType.getEncoding(), weight, kernelPermutation);

    // If bias is provided, it needs to be reshaped to match the expected shape
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
      // conv_transposed2d op.
      // https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Linear.cpp
      paddingAttr = rewriter.getDenseI32ArrayAttr({
          static_cast<int32_t>(
              (weightType.getShape()[SPATIAL_DIM_HEIGHT] - 1) *
                  adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT] -
              paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
          static_cast<int32_t>(
              (weightType.getShape()[SPATIAL_DIM_HEIGHT] - 1) *
                  adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT] -
              paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
          static_cast<int32_t>(
              (weightType.getShape()[SPATIAL_DIM_WIDTH] - 1) *
                  adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH] -
              paddingMatrix[SPATIAL_DIM_WIDTH][0]),
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
          rewriter, op->getLoc(), outputType, Value(input), Value(weight),
          biasValue, inputDilationAttr, paddingAttr, outputPaddingAttr,
          dilationAttr, groupsAttr);
    } else {
      newConv = ttir::utils::createDPSOp<ttir::Conv2dOp>(
          rewriter, op.getLoc(), outputType, Value(input), Value(weight),
          biasValue, strideAttr, paddingAttr, dilationAttr, groupsAttr,
          /*flattenedCompatInfo=*/nullptr);
    }

    // Applying the inverse of permutation to the output will restore the
    // tensor to the original layout.

    llvm::SmallVector<int64_t> outputLayout(conv2dLayout.size(),
                                            ConvolutionDimension::INVALID_DIM);
    outputLayout[op.getConvolutionLayout().getOutputBatchDimension()] =
        ConvolutionDimension::BATCH;
    outputLayout[op.getConvolutionLayout().getOutputFeatureDimension()] =
        ConvolutionDimension::FEATURE;
    for (const auto [spatialCount, spatialDim] : llvm::enumerate(
             op.getConvolutionLayout().getOutputSpatialDimensions())) {
      outputLayout[spatialDim] = spatialCount;
    }
    auto outputPermutation = ttmlir::utils::generatePermutation(
        llvm::ArrayRef(conv2dLayout), llvm::ArrayRef(outputLayout));

    rewriter.replaceOpWithNewOp<ttir::PermuteOp>(op, op.getResult().getType(),
                                                 newConv, adaptor.getOutput(),
                                                 outputPermutation);

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
   * - startIndexMap = collapsedSliceDims
   * - sliceSizes are fullDim for dimensions we are not indexing
   */

  LogicalResult checkBasicLegality(ttir::GatherOp op,
                                   PatternRewriter &rewriter) const {

    // Get input and start indices tensor shape.
    auto inputShape = op.getInput().getType().getShape();
    auto startIndicesShape = op.getStartIndices().getType().getShape();

    // Get attributes needed for embedding op pattern matching checks.
    auto sliceSizes = op.getSliceSizes();
    auto startIndexMap = op.getStartIndexMap();
    auto collapsedSliceDims = op.getCollapsedSliceDims();

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

    // Check if collapsed slice dims are exactly the dims we are indexing.
    if (collapsedSliceDims.size() != startIndexMap.size()) {
      return rewriter.notifyMatchFailure(
          op, "Did not satisfy startIndexMap = collapsedSliceDims");
    }
    for (size_t i = 0; i < startIndexMap.size(); i++) {
      if (startIndexMap[i] != collapsedSliceDims[i]) {
        return rewriter.notifyMatchFailure(
            op, "Did not satisfy startIndexMap = collapsedSliceDims");
      }
    }

    // Check if slice sizes are dim size for dims we are not indexing.
    int inputShapeIndex = 0;
    int startIndexMapIndex = 0;
    for (; inputShapeIndex < static_cast<int>(inputShape.size());
         inputShapeIndex++) {
      if (startIndexMapIndex < static_cast<int>(startIndexMap.size()) &&
          startIndexMap[startIndexMapIndex] == inputShapeIndex) {
        startIndexMapIndex++;
      } else if (sliceSizes[inputShapeIndex] != inputShape[inputShapeIndex]) {
        return rewriter.notifyMatchFailure(
            op, "Did not satisfy sliceSizes[i] = inputShape[i] for i not in "
                "startIndexMap");
      }
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

    auto numIndexingDims = op.getStartIndexMap().size();

    auto inputPermuted = permuteInput(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_permuteInput"),
        op.getInput(), op.getStartIndexMap());
    auto input = reshapeInput(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_reshapeInput"),
        inputPermuted, numIndexingDims);

    // If we are indexing multiple dims, we need to tranform indices for the new
    // single (flattened) indexing dim.
    auto startIndicesTransformed = op.getStartIndices();
    if (numIndexingDims > 1) {
      op->emitWarning("End results might be incorrect when indexing multiple "
                      "dimensions of input because of typecast ops.");
      startIndicesTransformed = transformStartIndices(
          rewriter, inputPermuted.getType().getShape(), op);
    }
    auto startIndices = startIndicesTransformed;
    if (startIndices.getType().getShape().size() >= 3) {
      startIndices =
          reshapeStartIndices(rewriter,
                              ttmlir::utils::appendLocationSuffix(
                                  op->getLoc(), "_reshapeStartIndices"),
                              startIndicesTransformed);
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

    rewriter.replaceOp(op, reshapeAndPermuteOutput(rewriter, op->getLoc(),
                                                   embeddingOp, op.getOutput(),
                                                   op.getOffsetDims()));
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
  transformStartIndices(ConversionPatternRewriter &rewriter,
                        ::llvm::ArrayRef<int64_t> inputShape,
                        ttir::GatherOp op) {
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

  // Helper that reshapes start indices to reduce number of dims, as Embedding
  // Op input can be 1D or 2D.
  static ttir::ReshapeOp reshapeStartIndices(
      ConversionPatternRewriter &rewriter, Location loc,
      ::mlir::TypedValue<::mlir::RankedTensorType> startIndices) {
    auto startIndicesShape = startIndices.getType().getShape();
    llvm::SmallVector<int64_t, 1> newStartIndicesShape{
        std::accumulate(startIndicesShape.begin(), startIndicesShape.end(),
                        int64_t{1}, std::multiplies<>())};
    return createReshapeOp(rewriter, loc, startIndices, newStartIndicesShape);
  }

  // In output, dims other than offsetDims map to startIndices shape, and
  // offsetDims map to input slices. After ttir.embedding all offseDims are
  // flattened to the last dim of output. First we reshape that output to
  // recover lost dims, then we permute them so offset dims are where the
  // attribute states.
  // Example: expectedOutputShape = [2, 3, 4, 5], offsetDims = [1, 3]
  // -> embeddingOutputShape = [2, 4, 15] -reshape-> [2, 4, 3, 5] -permute-> [2,
  // 3, 4, 5]
  static ttir::PermuteOp reshapeAndPermuteOutput(
      ConversionPatternRewriter &rewriter, Location loc,
      ::mlir::TypedValue<::mlir::RankedTensorType> output,
      ::mlir::TypedValue<::mlir::RankedTensorType> expectedOutput,
      ::llvm::ArrayRef<int64_t> offsetDims) {
    auto expectedOutputType = expectedOutput.getType();
    auto expectedOutputShape = expectedOutputType.getShape();

    llvm::SmallVector<int64_t> outputPermutation;
    size_t offsetDimsIndex = 0;
    for (size_t outputShapeIndex = 0;
         outputShapeIndex < expectedOutputShape.size(); outputShapeIndex++) {
      if (offsetDimsIndex < offsetDims.size() &&
          offsetDims[offsetDimsIndex] == static_cast<long>(outputShapeIndex)) {
        offsetDimsIndex++;
        continue;
      }
      outputPermutation.push_back(outputShapeIndex);
    }
    for (offsetDimsIndex = 0; offsetDimsIndex < offsetDims.size();
         offsetDimsIndex++) {
      outputPermutation.push_back(offsetDims[offsetDimsIndex]);
    }
    auto inverseOutputPermutation =
        ttmlir::utils::inversePermutation(outputPermutation);
    auto permutedOutputShape = ttmlir::utils::applyPermutation(
        expectedOutputType.getShape(), outputPermutation);

    auto reshapedOutput = createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshapeOutput"),
        output, permutedOutputShape);

    return ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_permuteOutput"),
        expectedOutputType.getShape(), expectedOutputType.getElementType(),
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

    // Check if the original op should be hoisted
    bool shouldHoist = op->hasAttr("tt.hoist");

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
      matmulOp->setAttr("tt.hoist", rewriter.getUnitAttr());
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
      reshapeOutput->setAttr("tt.hoist", rewriter.getUnitAttr());
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

    // propagate the hoist attribute if needed
    if (shouldHoist) {
      permuteOp->setAttr("tt.hoist", rewriter.getUnitAttr());
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
    // propagate the hoist attribute if needed
    if (shouldHoist) {
      reshapeOp->setAttr("tt.hoist", rewriter.getUnitAttr());
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

// This pattern reshapes the non input tensors of the BatchNormOp to 4D
// tensors, by adding additional dimensions of size 1 so that the only
// non-1 dimension is the second dimension. This is done so that the
// op is compatible with ttnn op call.
namespace {
struct BatchNormPattern : public OpConversionPattern<ttir::BatchNormOp> {
public:
  using OpConversionPattern<ttir::BatchNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto meanType = mlir::cast<RankedTensorType>(adaptor.getMean().getType());
    if (!getIsInputTypeValid(meanType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp mean must be 1D tensor");
    }
    mlir::Value mean_4d = get4DTensor(rewriter, op.getLoc(), adaptor.getMean());

    auto varType =
        mlir::cast<RankedTensorType>(adaptor.getVariance().getType());
    if (!getIsInputTypeValid(varType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp var must be 1D or 4D tensor");
    }
    mlir::Value variance_4d =
        get4DTensor(rewriter, op.getLoc(), adaptor.getVariance());

    auto weightType =
        mlir::cast<RankedTensorType>(adaptor.getScale().getType());
    if (!getIsInputTypeValid(weightType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp weight must be 1D or 4D tensor");
    }
    mlir::Value scale_4d =
        get4DTensor(rewriter, op.getLoc(), adaptor.getScale());

    auto biasType = mlir::cast<RankedTensorType>(adaptor.getOffset().getType());
    if (!getIsInputTypeValid(biasType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp bias must be 1D or 4D tensor");
    }
    mlir::Value offset_4d =
        get4DTensor(rewriter, op.getLoc(), adaptor.getOffset());

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::BatchNormOp>(
        rewriter, op, outputType, adaptor.getOperand(), scale_4d, offset_4d,
        mean_4d, variance_4d, adaptor.getEpsilonAttr(),
        adaptor.getDimensionAttr(), adaptor.getTrainingAttr());

    return success();
  }

private:
  bool getIsInputTypeValid(RankedTensorType inputType) const {
    if (inputType.getRank() == 1) {
      return true;
    }
    if (inputType.getRank() == 4) {
      auto shape = inputType.getShape();
      return shape[0] == 1 && shape[2] == 1 && shape[3] == 1;
    }
    return false;
  }

  mlir::Value get4DTensor(PatternRewriter &rewriter, Location loc,
                          mlir::Value batchNormInput) const {
    auto inputType =
        mlir::cast<mlir::RankedTensorType>(batchNormInput.getType());

    if (inputType.getRank() == 4) {
      return batchNormInput;
    }

    auto newShape =
        llvm::SmallVector<int64_t>{1, inputType.getDimSize(0), 1, 1};
    return createReshapeOp(rewriter, loc, batchNormInput, newShape);
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
  patterns.add<BatchNormPattern>(typeConverter, ctx);
  patterns.add<QuantizeOpPattern>(typeConverter, ctx);
  patterns.add<DequantizeOpPattern>(typeConverter, ctx);
  patterns.add<RequantizeOpPattern>(typeConverter, ctx);
  patterns.add<ReductionProdPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
