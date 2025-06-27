// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace tt {
namespace ttir {

#define GEN_PASS_DEF_TTIRCONV2DTOMATMULPASS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
// Pattern to convert Conv2D operations with 1x1 kernels to MatMul operations
// when applicable. If a bias is present, a Linear operation is used instead.
// Requirements: The kernel size must be 1x1, the stride must
// be 1 and the padding must be 0. Resulting transform:
//     Input of [N, C, H, W] reshaped to [N*H*W, C]
//     Weight vector of [O, I, H, W] permuted to [I, O, H, W] and reshaped to
//     [I*O*H, W] Matmul of [N*H*W, C] and [I*O*H, W] with output [N*H*W, W]
//     Reshape of [N*H*W, W] to original output tensor of [N, H, W, C]
class Conv2DToMatmulPattern : public OpRewritePattern<ttir::Conv2dOp> {
public:
  Conv2DToMatmulPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<ttir::Conv2dOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ttir::Conv2dOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType weightType =
        mlir::cast<RankedTensorType>(op.getWeight().getType());
    llvm::ArrayRef<int64_t> weightShape = weightType.getShape();

    // Check kernel size is 1x1.
    if (weightShape[2] != 1 || weightShape[3] != 1) {
      return failure();
    }

    // Check stride is 1.
    mlir::Attribute rawStrideAttr = op.getStride();
    if (mlir::IntegerAttr strideAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(rawStrideAttr)) {
      if (strideAttr.getInt() != 1) {
        return failure();
      }
    } else if (mlir::DenseI32ArrayAttr strideAttr =
                   mlir::dyn_cast<mlir::DenseI32ArrayAttr>(rawStrideAttr)) {
      llvm::ArrayRef<int32_t> strides = strideAttr.asArrayRef();
      if (strides.size() != 2 || strides[0] != 1 || strides[1] != 1) {
        return failure();
      }
    } else {
      return failure();
    }

    // Check padding is 0.
    mlir::Attribute rawPaddingAttr = op.getPadding();
    if (mlir::IntegerAttr paddingAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(rawPaddingAttr)) {
      if (paddingAttr.getInt() != 0) {
        return failure();
      }
    } else if (mlir::DenseI32ArrayAttr paddingArrayAttr =
                   mlir::dyn_cast<mlir::DenseI32ArrayAttr>(rawPaddingAttr)) {
      if (paddingArrayAttr.asArrayRef().size() != 4 ||
          llvm::any_of(paddingArrayAttr.asArrayRef(),
                       [](int32_t v) { return v != 0; })) {
        return failure();
      }
    } else {
      return failure();
    }

    // only groups = 1 is supported for now
    if (op.getGroups() != 1) {
      return failure();
    }

    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(op.getInput().getType());
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op.getResult().getType());

    // Input is OIHW.
    int64_t flattenedSpatialDim = inputShape[1] * inputShape[2];
    int64_t inputChannel = inputShape[3];
    int64_t outputChannel = weightShape[0];

    // Create reshaped input: [N*H*W, C].
    SmallVector<int64_t> reshapedInputShape = {
        inputShape[0] * flattenedSpatialDim, inputChannel};
    RankedTensorType reshapedInputType =
        RankedTensorType::get(reshapedInputShape, inputType.getElementType());

    // Create empty tensor for reshape output.
    Value emptyReshapedInput =
        rewriter.create<ttir::EmptyOp>(op.getLoc(), reshapedInputType);
    SmallVector<int32_t> reshapedInputShape32;
    for (int64_t dim : reshapedInputShape) {
      reshapedInputShape32.push_back(static_cast<int32_t>(dim));
    }
    mlir::ArrayAttr reshapedInputShapeAttr =
        rewriter.getI32ArrayAttr(reshapedInputShape32);

    Value reshapedInput = rewriter.create<ttir::ReshapeOp>(
        op.getLoc(), reshapedInputType, op.getInput(), emptyReshapedInput,
        reshapedInputShapeAttr);

    // Permute weights.
    SmallVector<int64_t> transposeWeightShape = {weightShape[1], weightShape[0],
                                                 1, 1};
    SmallVector<int64_t> permutation = {1, 0, 2, 3};
    RankedTensorType transposedWeightType = RankedTensorType::get(
        transposeWeightShape, weightType.getElementType());
    Value emptyTransposedWeight =
        rewriter.create<ttir::EmptyOp>(op.getLoc(), transposedWeightType);
    mlir::DenseI64ArrayAttr permAttr =
        rewriter.getDenseI64ArrayAttr(permutation);

    Value transposedWeight = rewriter.create<ttir::PermuteOp>(
        op.getLoc(), transposedWeightType, op.getWeight(),
        emptyTransposedWeight, permAttr);

    // Reshape weight to [I, O].
    SmallVector<int64_t> reshapedWeightShape = {inputChannel, outputChannel};
    RankedTensorType reshapedWeightType =
        RankedTensorType::get(reshapedWeightShape, weightType.getElementType());
    Value emptyReshapedWeight =
        rewriter.create<ttir::EmptyOp>(op.getLoc(), reshapedWeightType);
    SmallVector<int32_t> reshapedWeightShape32;
    for (int64_t dim : reshapedWeightShape) {
      reshapedWeightShape32.push_back(static_cast<int32_t>(dim));
    }
    mlir::ArrayAttr reshapedWeightShapeAttr =
        rewriter.getI32ArrayAttr(reshapedWeightShape32);

    Value reshapedWeight = rewriter.create<ttir::ReshapeOp>(
        op.getLoc(), reshapedWeightType, transposedWeight, emptyReshapedWeight,
        reshapedWeightShapeAttr);
    SmallVector<int64_t> matmulOutputShape = {
        inputShape[0] * flattenedSpatialDim, outputChannel};
    RankedTensorType matmulOutputType =
        RankedTensorType::get(matmulOutputShape, outputType.getElementType());

    Value emptyTensor =
        rewriter.create<ttir::EmptyOp>(op.getLoc(), matmulOutputType);

    // Create the matmul operation (or Linear if bias is present).
    Value matmulOrLinearResult;
    if (op.getBias()) {
      // reshape bias to 1D
      SmallVector<int64_t> biasShape = {outputChannel};
      RankedTensorType biasType = RankedTensorType::get(
          biasShape, op.getBias().getType().getElementType());
      Value emptyReshapedBias =
          rewriter.create<ttir::EmptyOp>(op.getLoc(), biasType);
      Value reshapedBias = rewriter.create<ttir::ReshapeOp>(
          op.getLoc(), biasType, op.getBias(), emptyReshapedBias,
          rewriter.getI32ArrayAttr({static_cast<int32_t>(biasShape[0])}));
      matmulOrLinearResult = rewriter.create<ttir::LinearOp>(
          op.getLoc(), matmulOutputType, reshapedInput, reshapedWeight,
          reshapedBias, emptyTensor);
    } else {
      matmulOrLinearResult = rewriter.create<ttir::MatmulOp>(
          op.getLoc(), matmulOutputType, reshapedInput, reshapedWeight,
          emptyTensor);
    }

    // Reshape matmul or linear result back to NHWC format
    SmallVector<int64_t> finalOutputShape = {inputShape[0], inputShape[1],
                                             inputShape[2], outputChannel};
    SmallVector<int32_t> finalOutputShape32;
    for (int64_t dim : finalOutputShape) {
      finalOutputShape32.push_back(static_cast<int32_t>(dim));
    }
    mlir::ArrayAttr finalOutputShapeAttr =
        rewriter.getI32ArrayAttr(finalOutputShape32);

    // Replace the original op with a new ReshapeOp in one step
    rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        op, outputType, matmulOrLinearResult, op.getOutput(),
        finalOutputShapeAttr);
    return success();
  }
};

struct TTIRConv2DToMatmulPass
    : public impl::TTIRConv2DToMatmulPassBase<TTIRConv2DToMatmulPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    RewritePatternSet patterns(context);
    patterns.add<Conv2DToMatmulPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace ttir
} // namespace tt
} // namespace mlir
