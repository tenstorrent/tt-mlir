// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceOpsRewritePattern.h"
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

    auto newOp = rewriter.create<ttir::SliceOp>(
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

    // Not currently supporting spatial dims other than 2 for the 1D case.
    if (op.getConvolutionLayout().getInputSpatialDimensions()[0] != 2) {
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
        rewriter, op.getLoc(), adaptor.getInput(), reshapeInputShape);
    ttir::ReshapeOp reshapeWeight = createReshapeOp(
        rewriter, op.getLoc(), adaptor.getWeight(), reshapeWeightShape);

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

    auto new2dConvolutionOp = ttmlir::utils::createDPSOp<ttir::ConvolutionOp>(
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

    ttir::ReshapeOp reshapeOutput =
        createReshapeOp(rewriter, op.getLoc(), new2dConvolutionOp, outputShape);

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

    return ttmlir::utils::createDPSOp<ttir::ReshapeOp>(
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

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();
    }

    auto strideHeightAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_HEIGHT]);
    auto strideWidthAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowStrides()[SPATIAL_DIM_WIDTH]);
    auto dilationHeightAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_HEIGHT]);
    auto dilationWidthAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWeightDilation()[SPATIAL_DIM_WIDTH]);

    // Padding is a list of 2-tuples, the order of the 2-tuples is in
    // most-significant spatial dimension first order For Conv2d the most
    // significant spatial dimension is the height, followed by the width.
    auto paddingMatrix =
        getPaddingMatrix<NUM_SPATIAL_DIMS>(adaptor.getPadding());
    auto paddingTopAttr =
        // paddingMatrix[SPATIAL_DIM_HEIGHT][0];
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_HEIGHT][0]);
    auto paddingBottomAttr =
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_HEIGHT][1]);
    auto paddingLeftAttr =
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_WIDTH][0]);
    auto paddingRightAttr =
        rewriter.getSI32IntegerAttr(paddingMatrix[SPATIAL_DIM_WIDTH][1]);

    auto groupsAttr =
        rewriter.getSI32IntegerAttr(adaptor.getFeatureGroupCount());

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
    auto input = ttmlir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, op.getLoc(), permuteOutputShape, inputType.getElementType(),
        inputType.getEncoding(), adaptor.getInput(), permutation);

    auto weightType =
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    auto kernelPermutation =
        generateConvKernelPermutation(op, conv2dKernelLayout);
    auto weightOutputShape = ::ttmlir::utils::applyPermutation(
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType()).getShape(),
        kernelPermutation);
    auto weight = ttmlir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, op.getLoc(), weightOutputShape, weightType.getElementType(),
        weightType.getEncoding(), adaptor.getWeight(), kernelPermutation);

    ttir::Conv2dOp newConv = ttmlir::utils::createDPSOp<ttir::Conv2dOp>(
        rewriter, op.getLoc(), outputType, Value(input), Value(weight),
        adaptor.getBias(), strideHeightAttr, strideWidthAttr,
        dilationHeightAttr, dilationWidthAttr, groupsAttr, paddingLeftAttr,
        paddingRightAttr, paddingTopAttr, paddingBottomAttr);

    // Applying the inverse of permutation to the output will restore the
    // tensor to the original layout.
    rewriter.replaceOpWithNewOp<ttir::PermuteOp>(
        op, op.getResult().getType(), newConv, adaptor.getOutput(),
        ttmlir::utils::inversePermutation(permutation));

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
   * Validates Gather op constraints for embedding conversion
   *
   * Enforces constraints on Gather operation to ensure valid embedding
   * transformation:
   * - Output tensor shape: Multi-dimensional with last dimension as embedding
   * size/ hiddenDim
   * - Slice sizes: Must be [1, hiddenDim], where hiddenDim matches last output
   * dimension
   * - Offset dimensions: Strictly [2]
   * - Collapsed slice dimensions: Strictly [0]
   * - Start indices shape: Must be compatible with output shape
   *   - startIndices.size() < output.size()
   *   - if startIndices.size() == output.size(), then startIndices[-1] == 1
   *   - Last dimension of start indices can be reduced by reshape op.
   *   - This is due to embedding weights requiring to have smaller size than
   * output shape
   *
   */

  LogicalResult checkBasicLegality(ttir::GatherOp op,
                                   PatternRewriter &rewriter) const {

    // variables for embedding pattern matching checks
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto shape = outputType.getShape();

    // start indices of the gather op:
    auto startIndices = op.getStartIndices();
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());

    // slice sizes of the gather op
    auto sliceSizes = op.getSliceSizes();
    auto offsetDims = op.getOffsetDims();
    // collapsed slice dims of the gather op
    auto collapsedSliceDims = op.getCollapsedSliceDims();

    if (shape.size() > 1) {
      auto hiddenDim = shape[shape.size() - 1];
      // check if sliceSizes has more than one element
      if (sliceSizes.size() <= 1) {
        return rewriter.notifyMatchFailure(op, "Did not satisfy sliceSizes");
      }
      // check if sliceSizes is [1, hiddenDim]
      if (sliceSizes[0] != 1 || sliceSizes[1] != hiddenDim) {
        return rewriter.notifyMatchFailure(op, "Did not satisfy sliceSizes");
      }
    }

    // check if offsetDims is [2]
    if (offsetDims.size() > 1 || offsetDims[0] != 2) {
      return rewriter.notifyMatchFailure(op, "Did not satisfy offsetDims");
    }

    // check if collapsedSliceDims is [0]
    if (collapsedSliceDims.size() > 1 || collapsedSliceDims[0] != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "Did not satisfy collapsedSliceDims");
    }

    // check if startIndices and output have same shape, if not, check if
    // reshape is possible can reshape startIndices to remove the last dimension
    // if it is 1
    if (shape.size() == startIndicesType.getShape().size() &&
        startIndicesType.getShape()[shape.size() - 1] != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "Did not satisfy startIndicesType");
    }

    return success();
  }

  /**
   * Lowers Gather Op into Embedding Op (and applies Reshape Op, if necessary)
   *
   * - There is no TTNN Gather support.
   *
   * - TTIR Gather Op is lowered into TTIR Embedding Op. Torch embeddings are
   * lowered into Gather Op. Most models use Gather Op to implement simple
   * embeddings.
   *
   * - If encountered more complicated Gather Op implementations, they can be
   * lowered into slice/ concat/ etc.
   *
   * - Start Indices of Gather Op are expected to match Weight for Embeddings.
   * startIndices.size() < weight.size() however, if
   * startIndices.size() == weight.size() && startIndices[-1] == 1,
   * we can apply Reshape Op to reduce the last dimension.
   *
   */

  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // GatherOp can be used to implement embedding lookup, check for that case
    LogicalResult err = checkBasicLegality(op, rewriter);
    if (not err.succeeded()) {
      return err;
    }
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto shape = outputType.getShape();

    // start indices of the gather op
    auto startIndices = op.getStartIndices();
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());

    // check if start indices need to be reshaped
    ::mlir::Value input = op.getStartIndices();
    if (shape.size() == startIndicesType.getShape().size() &&
        startIndicesType.getShape()[shape.size() - 1] == 1) {
      // reduce weight tensor dimension
      // insert reshape op to remove the last dimension of start indices
      // before gather/ embedding op
      std::vector<int64_t> newShapeI64(startIndicesType.getShape().begin(),
                                       startIndicesType.getShape().end() - 1);

      ttir::ReshapeOp reshapeOp =
          createReshapeOp(rewriter, op.getLoc(), startIndices, newShapeI64);

      assert(reshapeOp && "Failed to create reshape op");
      reshapeOp->moveBefore(op);
      input = reshapeOp.getResult();
    }

    // convert gather to embedding, use reshaped input if needed
    ttir::EmbeddingOp embeddingOp = rewriter.create<ttir::EmbeddingOp>(
        op.getLoc(), op.getResult().getType(), input, op.getOperands()[0],
        op.getOutput());

    assert(embeddingOp != nullptr && "Failed to create embedding op");
    rewriter.replaceOp(op, embeddingOp);

    return success();
  }

private:
  ttir::ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                                  Value input,
                                  ::llvm::ArrayRef<int64_t> targetShape) const {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(targetShape));

    return ttmlir::utils::createDPSOp<ttir::ReshapeOp>(
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

    ttir::PermuteOp lhsPermute =
        createPermuteOp(rewriter, op.getLoc(), lhs, lhsType, lhsPermutation);
    ttir::PermuteOp rhsPermute =
        createPermuteOp(rewriter, op.getLoc(), rhs, rhsType, rhsPermutation);

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
        rewriter, op.getLoc(), lhsPermute, lhsType, lhsMatmulInputShape);
    ttir::ReshapeOp rhsMatmulInput = createMatmulFinal(
        rewriter, op.getLoc(), rhsPermute, rhsType, rhsMatmulInputShape);

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
    auto matmulOp = ttmlir::utils::createDPSOp<ttir::MatmulOp>(
        rewriter, op.getLoc(), matmulDestinationShape, elementType, encoding,
        lhsMatmulInput, rhsMatmulInput);

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

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::ReshapeOp>(
        rewriter, op, resultShape, elementType, encoding, matmulOp,
        rewriter.getI32ArrayAttr(finalShapeI32));

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

  ttir::PermuteOp
  createPermuteOp(PatternRewriter &rewriter, Location loc, Value input,
                  RankedTensorType inputType,
                  const SmallVector<int64_t> &permutation) const {

    SmallVector<int64_t> destinationShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), permutation);

    return ttmlir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, loc, destinationShape, inputType.getElementType(),
        inputType.getEncoding(), input, permutation);
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

  ttir::ReshapeOp
  createMatmulFinal(PatternRewriter &rewriter, Location loc, Value input,
                    RankedTensorType type,
                    const SmallVector<int64_t> &finalShape) const {

    llvm::SmallVector<int32_t> finalShapeI32(finalShape.begin(),
                                             finalShape.end());

    return ttmlir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, loc, finalShape, type.getElementType(), type.getEncoding(),
        input, rewriter.getI32ArrayAttr(finalShapeI32));
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
      rewritePool2d<ttir::MaxPool2dOp>(op, adaptor, rewriter,
                                       spatialDimIndices);
      return success();
    }
    default: {
      return rewriter.notifyMatchFailure(
          op, "Failed to match pooling method: " +
                  stringifyPoolingMethod(op.getPoolingMethod()));
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
      return rewriter.notifyMatchFailure(
          op, "Rank of kernel_size for pooling 2D op is greater than 2.");
    }

    // Window strides will have two or less than two non 1 elements;
    // representing the strides for max pooling operation.
    llvm::SmallVector<int64_t> trueWindowStrideIndices =
        getIndicesOfElementsLargerThanOne(op.getWindowStrides());
    size_t windowStrideSize = trueWindowStrideIndices.size();
    if (windowStrideSize > 2) {
      return rewriter.notifyMatchFailure(
          op, "Rank of strides for pooling 2D is greater than 2.");
    }

    // Padding must be 8 in length
    if (op.getPadding().size() != 8) {
      return rewriter.notifyMatchFailure(
          op,
          "Number of elements in padding does not match with pooling 2D op.");
    }

    return success();
  }

  template <typename PoolOpType>
  void rewritePool2d(ttir::PoolingOp op, OpAdaptor adaptor,
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
    // [TODO] (mmanzoor) Add an option to distingush channel first vs channel
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

    auto kernelHeightAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowDimensions()[spatialDimIndices[0]]));
    auto kernelWidthAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowDimensions()[spatialDimIndices[1]]));

    auto strideHeightAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowStrides()[spatialDimIndices[0]]));

    auto strideWidthAttr = rewriter.getSI32IntegerAttr(
        static_cast<int32_t>(op.getWindowStrides()[spatialDimIndices[1]]));

    auto dilationHeightAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowDilations()[spatialDimIndices[0]]);
    auto dilationWidthAttr = rewriter.getSI32IntegerAttr(
        adaptor.getWindowDilations()[spatialDimIndices[1]]);
    auto ceilModeAttr = rewriter.getBoolAttr(false);

    auto paddingTopAttr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatialDimIndices[0]]);
    auto paddingBottomAttr = rewriter.getSI32IntegerAttr(
        op.getPadding()[2 * spatialDimIndices[0] + 1]);
    auto paddingLeftAttr =
        rewriter.getSI32IntegerAttr(op.getPadding()[2 * spatialDimIndices[1]]);
    auto paddingRightAttr = rewriter.getSI32IntegerAttr(
        op.getPadding()[2 * spatialDimIndices[1] + 1]);

    llvm::SmallVector<Value> outputs;
    for (Value input : adaptor.getInputs()) {
      RankedTensorType inputTy = mlir::cast<RankedTensorType>(input.getType());

      auto inputPermuteShape =
          ::ttmlir::utils::applyPermutation(inputTy.getShape(), permutation);
      input = ttmlir::utils::createDPSOp<ttir::PermuteOp>(
          rewriter, op.getLoc(), inputPermuteShape, inputTy.getElementType(),
          inputTy.getEncoding(), input, permutation);

      auto outputType = mlir::cast<RankedTensorType>(op.getResult(0).getType());
      auto newOutputShape =
          ::ttmlir::utils::applyPermutation(outputType.getShape(), permutation);

      auto newPool = ttmlir::utils::createDPSOp<PoolOpType>(
          rewriter, op.getLoc(), newOutputShape, outputType.getElementType(),
          outputType.getEncoding(), input, kernelHeightAttr, kernelWidthAttr,
          strideHeightAttr, strideWidthAttr, dilationHeightAttr,
          dilationWidthAttr, ceilModeAttr, paddingTopAttr, paddingBottomAttr,
          paddingLeftAttr, paddingRightAttr);

      // Applying the inverse of permutation to the output will restore the
      // tensor to the original layout.
      auto output = ttmlir::utils::createDPSOp<ttir::PermuteOp>(
          rewriter, op.getLoc(), outputType.getShape(),
          outputType.getElementType(), outputType.getEncoding(), newPool,
          inverseOfPermutation);

      outputs.push_back(output);
    }

    rewriter.replaceOp(op, outputs);
  }
};
} // namespace

// SelectOp is converted to a series of SliceOp and potentially a ConcatOp if
// the sliced dimension is sliced multiple times. For example, if the input
// tensor is
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
// and the SelectOp is dim=1, begin=0, length=2, stride=4, the output tensor
// will be
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
    : public OpConversionPattern<ttir::SelectOp> {
public:
  using OpConversionPattern<ttir::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SelectOp op, OpAdaptor adaptor,
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

      auto newOp = ttmlir::utils::createDPSOp<ttir::SliceOp>(
          rewriter, op.getLoc(), resultShape, inputType.getElementType(),
          inputType.getEncoding(), adaptor.getInput(),
          rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
          rewriter.getI32ArrayAttr(steps));
      slices.push_back(newOp);
    }

    assert(!slices.empty());
    if (slices.size() > 1) {
      auto concatOp = ttmlir::utils::createDPSOp<ttir::ConcatOp>(
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
 * behaves. The reason this pass is done in TTIR rather than generated when we
 * want to lower to TTNN is because in the future we will want to consteval the
 * ArangeOp, but have the option to not include repeated data in the constant
 * tensor and broadcast at runtime instead. Consteval will be implemented for
 * the TTIR dialect only and so this explication of the TMs implicit in ArangeOp
 * must be done in TTIR.
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
    int64_t arangeDimensionNegative = arangeDimension - outputType.getRank();
    int64_t start = adaptor.getStart();
    int64_t end = adaptor.getEnd();
    int64_t step = adaptor.getStep();

    int64_t arangeLength = (end - start) / step;

    const llvm::SmallVector<int64_t, 4> requiredShape{1, 1, 1, arangeLength};
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
                op.getLoc(), arangeOutputType, start, end, step, 3)
            .getResult();

    std::vector<int64_t> outputShape = arangeOutputType.getShape().vec();
    // Must transpose the output so that the data changes along the axis defined
    // by arangeDimension
    if (arangeDimensionNegative != -1) {
      std::vector<int64_t> transposeShape = outputShape;
      transposeShape[arangeDimensionNegative + transposeShape.size()] =
          arangeLength;
      transposeShape[arangeOutputType.getRank() - 1] = 1;
      RankedTensorType transposeType = RankedTensorType::get(
          transposeShape, arangeOutputType.getElementType(),
          arangeOutputType.getEncoding());

      output = ttmlir::utils::createDPSOp<ttir::TransposeOp>(
          rewriter, op.getLoc(), transposeType, output,
          arangeDimensionNegative + transposeShape.size(),
          arangeOutputType.getRank() - 1);

      outputShape = std::move(transposeShape);
    }

    // Must match up the rank of the output with the rank of the intended output
    // from the original arange, with the arangeDimension in the correct
    // position
    if (outputType.getRank() != static_cast<int64_t>(outputShape.size())) {
      std::vector<int64_t> reshapeShape;
      for (uint32_t i = 0; i < outputType.getRank(); i++) {
        i == arangeDimension ? reshapeShape.push_back(end)
                             : reshapeShape.push_back(1);
      }

      output = ttmlir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, op.getLoc(), reshapeShape, outputType.getElementType(),
          outputType.getEncoding(), output,
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

      output = ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
          rewriter, op.getLoc(), broadcastType, output, broadcastShape);

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

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::ProdOp>(
        rewriter, op, reduceOutputType, adaptor.getInput(), op.getKeepDim(),
        op.getDimArgAttr());

    return success();
  }
};
} // namespace

// tt-metal does not support 'keepdim' attribute for argmax op and always
// generate output with the same rank as input tensor. Decompose the op to
// argmax followed by reshape if keepdim = False.
namespace {
struct ArgMaxOpKeepDimConversionPattern
    : public OpConversionPattern<ttir::ArgMaxOp> {
public:
  using OpConversionPattern<ttir::ArgMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArgMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool keepDim = op.getKeepDim();
    if (keepDim) {
      return failure();
    }

    RankedTensorType inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getInput().getType()));
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getOutput().getType()));
    llvm::SmallVector<int64_t> outputShapeVec =
        ttnn::workarounds::decomposition::calculateNewReduceShape(
            inputType, op.getDimArg());
    RankedTensorType newOutputType = RankedTensorType::get(
        outputShapeVec, outputType.getElementType(), outputType.getEncoding());

    tensor::EmptyOp argMaxOutputTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputShapeVec, outputType.getElementType());

    ttir::ArgMaxOp newArgMaxOp = rewriter.create<mlir::tt::ttir::ArgMaxOp>(
        op->getLoc(), newOutputType, op.getInput(), argMaxOutputTensor,
        /*keepDim*/ true, op.getDimArgAttr());

    tensor::EmptyOp reshapeOutputTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputType.getShape(), outputType.getElementType());
    mlir::ArrayAttr shapeAttr = rewriter.getI32ArrayAttr(
        llvm::SmallVector<int32_t>(outputType.getShape()));
    rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        op, outputType, newArgMaxOp, reshapeOutputTensor, shapeAttr);

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

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::SumOp>(
        rewriter, op, reduceOutputType, adaptor.getInput(), op.getKeepDim(),
        op.getDimArgAttr());

    return success();
  }
};
} // namespace

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<PoolingToPool2dPattern>(typeConverter, ctx);
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<Legalize1DConvolutionPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);
  patterns.add<GatherToEmbeddingConversionPattern>(typeConverter, ctx);
  patterns.add<SelectToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<ArangeForceLastDimensionPattern>(typeConverter, ctx);
  patterns.add<DotGeneralToMatmulConversionPattern>(typeConverter, ctx);
  patterns.add<ReductionAndPattern>(typeConverter, ctx);
  patterns.add<ReductionOrPattern>(typeConverter, ctx);
  patterns.add<ArgMaxOpKeepDimConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
