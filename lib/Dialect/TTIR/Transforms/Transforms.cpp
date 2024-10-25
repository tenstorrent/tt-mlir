// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <string>
#include <utility>
#include <vector>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSLIDINGWINDOW2DFIXSHAPES
#define GEN_PASS_DEF_TTIRCONVOLUTIONTOCONV2D
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

std::vector<int64_t> collapseNHW(std::vector<int64_t> shape) {
  std::vector<int64_t> collapsed(shape.size(), 1);

  int64_t NHW = 1;
  for (uint32_t i = 0; i < shape.size() - 1; i++) {
    NHW *= shape[i];
  }
  collapsed[collapsed.size() - 2] = NHW;
  collapsed[collapsed.size() - 1] = shape[shape.size() - 1];
  return collapsed;
}

//===----------------------------------------------------------------------===//
// Sliding window pass
//===----------------------------------------------------------------------===//

template <typename T>
class UncollapsedSlidingWindow2dPatternRewriter : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                            Value input, ::llvm::ArrayRef<int64_t> shapei64,
                            ::mlir::ArrayAttr operandConstraints) const {
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output =
        rewriter.create<tensor::EmptyOp>(loc, shapei64, ty.getElementType());

    auto shape_attr = rewriter.getI32ArrayAttr(
        {static_cast<int32_t>(shapei64[0]), static_cast<int32_t>(shapei64[1]),
         static_cast<int32_t>(shapei64[2]), static_cast<int32_t>(shapei64[3])});
    return rewriter.create<ttir::ReshapeOp>(
        loc, output.getType(), input, output, shape_attr, operandConstraints);
  }

  MaxPool2dOp createMaxPool2dOp(PatternRewriter &rewriter, MaxPool2dOp op,
                                Value input, int32_t input_height,
                                int32_t input_width,
                                RankedTensorType new_result_type) const {
    auto output = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), new_result_type.getShape(),
        new_result_type.getElementType());

    auto input_height_attr = rewriter.getSI32IntegerAttr(input_height);
    auto input_width_attr = rewriter.getSI32IntegerAttr(input_width);

    MaxPool2dOp new_maxpool = rewriter.create<MaxPool2dOp>(
        op.getLoc(), new_result_type, input, output, op.getKernelHeightAttr(),
        op.getKernelWidthAttr(), op.getStrideHeightAttr(),
        op.getStrideWidthAttr(), op.getDilationHeightAttr(),
        op.getDilationWidthAttr(), op.getCeilModeAttr(),
        op.getPaddingLeftAttr(), op.getPaddingRightAttr(),
        op.getPaddingTopAttr(), op.getPaddingBottomAttr(),
        op.getOperandConstraints(), input_height_attr, input_width_attr);

    return new_maxpool;
  }

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
    ::llvm::ArrayRef<int64_t> input_shape =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType()).getShape();

    if (input_shape.size() != 4) {
      return failure();
    }

    if (input_shape[0] == 1 && input_shape[1] == 1) {
      return failure();
    }

    if (!llvm::isa<MaxPool2dOp>(op)) {
      return failure();
    }

    // By this point we are certain that the input tensor is not in the form (1,
    // 1, N*H*W, C) And so we must insert reshapes on the input/output

    std::vector<int64_t> new_input_shape = collapseNHW(input_shape);
    ::llvm::ArrayRef<int64_t> new_input_shape_array(new_input_shape);

    ReshapeOp input_reshape =
        createReshapeOp(rewriter, op.getLoc(), op.getInput(),
                        new_input_shape_array, op.getOperandConstraints());

    std::vector<int64_t> new_result_shape =
        collapseNHW(op.getResult().getType().getShape().vec());
    ::llvm::ArrayRef<int64_t> new_result_shape_array(new_result_shape);

    RankedTensorType new_result_type = RankedTensorType::get(
        new_result_shape_array, op.getResult().getType().getElementType(),
        op.getResult().getType().getEncoding());

    Operation *new_op = createMaxPool2dOp(
        rewriter, mlir::cast<MaxPool2dOp>(op), input_reshape,
        static_cast<int32_t>(input_shape[1]),
        static_cast<int32_t>(input_shape[2]), new_result_type);

    ReshapeOp output_reshape = createReshapeOp(
        rewriter, op.getLoc(), new_op->getResult(0),
        op.getResult().getType().getShape().vec(), op.getOperandConstraints());

    rewriter.replaceOp(op, output_reshape);
    return success();
  }
};

class TTIRSlidingWindow2dFixShapes
    : public impl::TTIRSlidingWindow2dFixShapesBase<
          TTIRSlidingWindow2dFixShapes> {
public:
  using impl::TTIRSlidingWindow2dFixShapesBase<
      TTIRSlidingWindow2dFixShapes>::TTIRSlidingWindow2dFixShapesBase;

  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<UncollapsedSlidingWindow2dPatternRewriter<MaxPool2dOp>>(
          &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Convolution passes
//===----------------------------------------------------------------------===//

using TransposeDims = std::tuple<int64_t, int64_t>;

template<uint32_t NDims>
using PaddingMatrix = std::array<std::array<int64_t, 2>, NDims>;

template<uint32_t NDims>
static PaddingMatrix<NDims> getPaddingMatrix(DenseIntElementsAttr paddingAttr) {
  PaddingMatrix<NDims> paddingMatrix;
  std::vector<int64_t> paddingFlattened(paddingAttr.value_begin<int64_t>(),
                                 paddingAttr.value_end<int64_t>());

  for (uint32_t i = 0; i < 2*NDims; i+=2) {
    paddingMatrix[i/2] = {paddingFlattened[i], paddingFlattened[i+1]};
  }
  return paddingMatrix;
}
/*
 * The following functions are used to generate the transpose operations needed
 * to convert a convolution operation to the specific op definitions for a
 * ConvNdOp for any N spatial dimensions.
 *
 * All convolutions will have a batch and feature dimension, and the kernel will
 * have an input and output feature dimension. The spatial dimensions can be
 * represented by non-negative integers.
 */
enum ConvolutionDimension { BATCH = -1, FEATURE = -2, INVALID_DIM = -3 };

enum ConvolutionKernelDimension {
  INPUT_FEATURES = -1,
  OUTPUT_FEATURES = -2,
  INVALID_KERNEL_DIM = -3
};

static tensor::EmptyOp generateTransposeDPSOutput(Value input, int64_t dim0,
                                                  int64_t dim1,
                                                  PatternRewriter &rewriter) {
  auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto output_shape = input_type.getShape().vec();
  std::swap(output_shape[dim0], output_shape[dim1]);

  auto output_type = RankedTensorType::get(
      output_shape, input_type.getElementType(), input_type.getEncoding());

  return rewriter.create<tensor::EmptyOp>(input.getLoc(), output_shape,
                                          output_type.getElementType());
}

static TransposeOp generateTranspose(Value input, int64_t dim0, int64_t dim1,
                                     PatternRewriter &rewriter,
                                     ::mlir::ArrayAttr operandConstraints) {
  auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto output_shape = input_type.getShape().vec();
  std::swap(output_shape[dim0], output_shape[dim1]);

  auto dim0_attr = rewriter.getSI32IntegerAttr(dim0);
  auto dim1_attr = rewriter.getSI32IntegerAttr(dim1);

  auto dps_output = generateTransposeDPSOutput(input, dim0, dim1, rewriter);
  return rewriter.create<TransposeOp>(input.getLoc(), dps_output.getType(),
                                      input, dps_output, dim0_attr, dim1_attr,
                                      operandConstraints);
}

static std::vector<TransposeDims> generateKernelTransposeIndices(
    ConvolutionOp op,
    const std::vector<int64_t> ttnn_convolution_kernel_layout) {
  std::vector<TransposeDims> transpose_indices;

  std::vector<int64_t> kernel_layout(
      ttnn_convolution_kernel_layout.size(),
      ConvolutionKernelDimension::INVALID_KERNEL_DIM);
  kernel_layout[op.getConvolutionLayout().getKernelOutputFeatureDimension()] =
      ConvolutionKernelDimension::OUTPUT_FEATURES;
  kernel_layout[op.getConvolutionLayout().getKernelInputFeatureDimension()] =
      ConvolutionKernelDimension::INPUT_FEATURES;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getKernelSpatialDimensions()) {
    kernel_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  const std::vector<int64_t> desired_kernel_layout =
      ttnn_convolution_kernel_layout;
  for (int64_t i = 0; i < static_cast<int64_t>(kernel_layout.size()); i++) {
    if (kernel_layout[i] != desired_kernel_layout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(kernel_layout.begin(), kernel_layout.end(),
                               desired_kernel_layout[i]) -
                     kernel_layout.begin();
      transpose_indices.push_back(std::make_tuple(dim0, dim1));
      std::swap(kernel_layout[dim0], kernel_layout[dim1]);
    }
  }

  return transpose_indices;
}

static std::vector<TransposeDims> generateInputTransposeIndices(
    ConvolutionOp op, const std::vector<int64_t> ttnn_convolution_layout) {
  std::vector<TransposeDims> transpose_indices;

  std::vector<int64_t> input_layout(ttnn_convolution_layout.size(),
                                    ConvolutionDimension::INVALID_DIM);
  input_layout[op.getConvolutionLayout().getInputBatchDimension()] =
      ConvolutionDimension::BATCH;
  input_layout[op.getConvolutionLayout().getInputFeatureDimension()] =
      ConvolutionDimension::FEATURE;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getInputSpatialDimensions()) {
    input_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  const std::vector<int64_t> desired_input_layout = ttnn_convolution_layout;
  for (int64_t i = 0; i < static_cast<int64_t>(input_layout.size()); i++) {
    if (input_layout[i] != desired_input_layout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(input_layout.begin(), input_layout.end(),
                               desired_input_layout[i]) -
                     input_layout.begin();
      transpose_indices.push_back(std::make_tuple(dim0, dim1));
      std::swap(input_layout[dim0], input_layout[dim1]);
    }
  }

  return transpose_indices;
}

/**
 * Although this function is mostly a clone of generateInputTransposeIndices,
 * its slightly different in that if the original Convolution op had the same
 * input and output layout, this function will generate the same transposes,
 * that were applied to the input but in reverse order. This makes optimizing
 * away the inserted transposes easier.
 */
static std::vector<TransposeDims> generateOutputTransposeIndices(
    ConvolutionOp op, const std::vector<int64_t> ttnn_convolution_layout) {
  std::vector<TransposeDims> transpose_indices;

  std::vector<int64_t> desired_output_layout(ttnn_convolution_layout.size(),
                                             ConvolutionDimension::INVALID_DIM);
  desired_output_layout[op.getConvolutionLayout().getOutputBatchDimension()] =
      ConvolutionDimension::BATCH;
  desired_output_layout[op.getConvolutionLayout().getOutputFeatureDimension()] =
      ConvolutionDimension::FEATURE;

  int64_t spatial_count = 0;
  for (int64_t spatial_dim :
       op.getConvolutionLayout().getOutputSpatialDimensions()) {
    desired_output_layout[spatial_dim] = spatial_count;
    spatial_count++;
  }

  std::vector<int64_t> output_layout = ttnn_convolution_layout;

  for (int64_t i = static_cast<int64_t>(desired_output_layout.size()) - 1;
       i >= 0; i--) {
    if (desired_output_layout[i] != output_layout[i]) {
      int64_t dim0 = i;
      int64_t dim1 = std::find(output_layout.begin(), output_layout.end(),
                               desired_output_layout[i]) -
                     output_layout.begin();
      transpose_indices.push_back(std::make_tuple(dim0, dim1));
      std::swap(output_layout[dim0], output_layout[dim1]);
    }
  }

  return transpose_indices;
}

static Value
generateTransposeSequence(Value input, PatternRewriter &rewriter,
                          std::vector<TransposeDims> transpose_indices,
                          ::mlir::ArrayAttr operandConstraints) {
  for (auto [dim0, dim1] : transpose_indices) {
    input = generateTranspose(input, dim0, dim1, rewriter, operandConstraints)
                .getResult();
  }

  return input;
}

class ConvolutionToConv2dPatternRewriter
    : public OpRewritePattern<ConvolutionOp> {
public:
  using OpRewritePattern<ConvolutionOp>::OpRewritePattern;

  constexpr static uint32_t numSpatialDims = 2;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 0;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 1;

  // NHWC
  const std::vector<int64_t> conv2d_layout = {ConvolutionDimension::BATCH, SPATIAL_DIM_HEIGHT, SPATIAL_DIM_WIDTH,
                                              ConvolutionDimension::FEATURE};
  // OIHW
  const std::vector<int64_t> conv2d_kernel_layout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES, SPATIAL_DIM_HEIGHT, SPATIAL_DIM_WIDTH};
  LogicalResult isConv2d(ConvolutionOp op) const {

    // Conv2d will have 2 spatial dimensions

    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getOutputSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");
    assert(op.getConvolutionLayout().getInputSpatialDimensions().size() ==
               op.getConvolutionLayout().getKernelSpatialDimensions().size() &&
           "Convolution input, output, and kernel must have the same number of "
           "spatial dimensions");

    if (op.getConvolutionLayout().getInputSpatialDimensions().size() != numSpatialDims) {
      return failure();
    }

    // Not currently supporting window reversal
    std::vector<bool> window_reversal(op.getWindowReversal().begin(),
                                      op.getWindowReversal().end());
    for (bool reversed : window_reversal) {
      if (reversed) {
        return failure();
      }
    }

    // Not currently support batch groups
    if (op.getBatchGroupCount() != 1) {
      return failure();
    }

    return success();
  }

  LogicalResult matchAndRewrite(ConvolutionOp op,
                                PatternRewriter &rewriter) const final {

    if (failed(isConv2d(op))) {
      return failure();
    }

    auto stride_height_attr =
        rewriter.getSI32IntegerAttr(op.getWindowStrides()[SPATIAL_DIM_HEIGHT]);
    auto stride_width_attr =
        rewriter.getSI32IntegerAttr(op.getWindowStrides()[SPATIAL_DIM_WIDTH]);
    auto dilation_height_attr =
        rewriter.getSI32IntegerAttr(op.getWeightDilation()[SPATIAL_DIM_HEIGHT]);
    auto dilation_width_attr =
        rewriter.getSI32IntegerAttr(op.getWeightDilation()[SPATIAL_DIM_WIDTH]);

    // Padding is a list of 2-tuples, the order of the 2-tuples is in most-significant spatial dimension first order
    // For Conv2d the most significant spatial dimension is the height, followed by the width.
    auto padding_matrix = getPaddingMatrix<numSpatialDims>(op.getPadding());
    auto padding_top_attr = rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_HEIGHT][0]);
    auto padding_bottom_attr = rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_HEIGHT][1]);
    auto padding_left_attr = rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_WIDTH][0]);
    auto padding_right_attr = rewriter.getSI32IntegerAttr(padding_matrix[SPATIAL_DIM_WIDTH][1]);

    auto groups_attr = rewriter.getSI32IntegerAttr(op.getFeatureGroupCount());

    auto output_shape = op.getResult().getType().getShape().vec();
    std::vector<int64_t> new_output_shape = {
        output_shape[op.getConvolutionLayout().getOutputBatchDimension()],
        output_shape[op.getConvolutionLayout().getOutputSpatialDimensions()[SPATIAL_DIM_HEIGHT]],
        output_shape[op.getConvolutionLayout().getOutputSpatialDimensions()[SPATIAL_DIM_WIDTH]],
        output_shape[op.getConvolutionLayout().getOutputFeatureDimension()]};

    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType =
        inputType.cloneWith(new_output_shape, inputType.getElementType());

    auto convDPSOutput = rewriter.create<tensor::EmptyOp>(
        op.getInput().getLoc(), new_output_shape, outputType.getElementType());

    auto input_transpose_indices =
        generateInputTransposeIndices(op, conv2d_layout);
    Value input = generateTransposeSequence(op.getInput(), rewriter,
                                            input_transpose_indices,
                                            op.getOperandConstraints());

    auto kernel_transpose_indices =
        generateKernelTransposeIndices(op, conv2d_kernel_layout);
    Value weight = generateTransposeSequence(op.getWeight(), rewriter,
                                             kernel_transpose_indices,
                                             op.getOperandConstraints());
    Conv2dOp new_conv = rewriter.create<ttir::Conv2dOp>(
        op.getLoc(), outputType, input, weight, op.getBias(), convDPSOutput,
        stride_height_attr, stride_width_attr, dilation_height_attr,
        dilation_width_attr, groups_attr, padding_left_attr, padding_right_attr,
        padding_top_attr, padding_bottom_attr, op.getOperandConstraints());

    auto output_transpose_indices =
        generateOutputTransposeIndices(op, conv2d_layout);
    Value output = generateTransposeSequence(new_conv.getResult(), rewriter,
                                             output_transpose_indices,
                                             op.getOperandConstraints());

    rewriter.replaceOp(op, output);

    return success();
  }
};

class TTIRConvolutionToConv2d
    : public impl::TTIRConvolutionToConv2dBase<TTIRConvolutionToConv2d> {
public:
  using impl::TTIRConvolutionToConv2dBase<
      TTIRConvolutionToConv2d>::TTIRConvolutionToConv2dBase;

  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<ConvolutionToConv2dPatternRewriter>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

} // namespace mlir::tt::ttir
