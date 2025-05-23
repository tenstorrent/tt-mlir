// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt {
namespace {
// Get the dimensions to broadcast.
//
// This function calculates the dimensions to broadcast. We assume that input
// and target shapes are broadcastable. For example if input shape is [4, 1, 3]
// and we want to broadcast to [1, 4, 5, 3], the function will return [0, 2]
// since we want to broadcast 0th and 2nd dimension of result shape.
static SmallVector<int64_t, 2> getBroadcastDims(ArrayRef<int64_t> inputShape,
                                                ArrayRef<int64_t> targetShape) {
  const int64_t sizeDiff = targetShape.size() - inputShape.size();
  assert(sizeDiff >= 0 && "targetShape cannot be smaller than inputShape!");

  // Create padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput;
  paddedInput.append(sizeDiff, 1); // Prepend with 1s
  paddedInput.append(inputShape.begin(), inputShape.end());

  // Find broadcast dimensions we want to broadcast along (including padding
  // dimensions).
  SmallVector<int64_t, 2> broadcastDims;
  for (const auto &it : llvm::enumerate(llvm::zip(paddedInput, targetShape))) {
    const size_t i = it.index();
    const auto &[inputDim, targetDim] = it.value();
    // Prepended dimensions are always broadcasted.
    if (i < static_cast<size_t>(sizeDiff) || inputDim != targetDim) {
      broadcastDims.push_back(i);
    }
  }

  return broadcastDims;
}

// Get the dimensions to collapse.
//
// This function calculates the dimensions to collapse. We assume that input
// and target shapes are broadcastable. linalg.broadcast requires that input
// tensor only contains dimensions that won't be broadcasted in input tensor.
// For example if input shape is [4, 1, 3] and we want to broadcast to [4, 5,
// 3], then we need to collapse the first dimension of input tensor to [4, 3].
// This function calculates the dimensions to collapse. In case above, we will
// return [[0], [1, 2]].
static SmallVector<SmallVector<int64_t, 2>, 2>
getCollapseDims(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> targetShape) {
  // Calculate the size difference.
  const size_t sizeDiff = targetShape.size() - inputShape.size();

  // Create the padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput(sizeDiff, 1);
  paddedInput.append(inputShape.begin(), inputShape.end());

  SmallVector<int64_t, 2> collapseDims;
  SmallVector<SmallVector<int64_t, 2>, 2> reassocIndexes;
  for (size_t i = sizeDiff; i < targetShape.size(); ++i) {
    const size_t inputDim = paddedInput[i];
    const size_t targetDim = targetShape[i];
    // Adjust the index to account for the prepended dimensions
    // that are not part of the input shape.
    collapseDims.push_back(i - sizeDiff);
    if (inputDim == targetDim) {
      reassocIndexes.push_back(collapseDims);
      collapseDims.clear();
    }
  }

  if (!collapseDims.empty()) {
    if (reassocIndexes.empty()) {
      reassocIndexes.push_back(collapseDims);
    } else {
      reassocIndexes.back().append(collapseDims.begin(), collapseDims.end());
    }
  }

  return reassocIndexes;
}
} // namespace

namespace {
// Conversion pattern of operations which have exactly 2 input and 1 output
// operands.
template <typename TTIROpTy, typename LinalgOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    RankedTensorType lhsType =
        cast<RankedTensorType>(adaptor.getLhs().getType());
    RankedTensorType rhsType =
        cast<RankedTensorType>(adaptor.getRhs().getType());

    // First, compute broadcasted shape from operands.

    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();

    SmallVector<int64_t> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(lhsShape, rhsShape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(op, "Operands are not broadcastable!");
    }

    // Rewrite inputs to target dims with broadcast and collapse shape ops, as
    // needed.
    SmallVector<Value, 2> inputs{adaptor.getLhs(), adaptor.getRhs()};
    SmallVector<Value, 2> broadcastedInputs;
    for (Value input : inputs) {
      auto inputRankedTensorType = dyn_cast<RankedTensorType>(input.getType());
      assert(inputRankedTensorType &&
             "Binary element-wise operations must be ranked tensor types!");

      // Insert and use a broadcast op if input does not perfectly match target
      // shape.
      SmallVector<int64_t, 2> broadcastDims =
          getBroadcastDims(inputRankedTensorType.getShape(), broadcastedShape);

      // If we need to broadcast along all dims, then we need to collapse to a
      // scalar via empty collapseDimGroups.
      SmallVector<SmallVector<int64_t, 2>, 2> collapseDimGroups =
          (broadcastDims.size() != broadcastedShape.size())
              ? getCollapseDims(inputRankedTensorType.getShape(),
                                broadcastedShape)
              : SmallVector<SmallVector<int64_t, 2>, 2>();

      if (!broadcastDims.empty()) {
        Value broadcastInput = input;
        // The broadcast op requires we actually collapse any dimensions with
        // size 1 we want to broadcast along.
        if (collapseDimGroups.size() !=
            inputRankedTensorType.getShape().size()) {
          broadcastInput = rewriter.create<tensor::CollapseShapeOp>(
              loc, input, collapseDimGroups);
        }
        auto initTensor = rewriter.create<ttir::EmptyOp>(
            loc, broadcastedShape, inputRankedTensorType.getElementType());
        auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
            loc, broadcastInput, initTensor.getResult(), broadcastDims);
        broadcastedInputs.push_back(broadcastOp.getResults().front());
      } else {
        broadcastedInputs.push_back(input);
      }
    }

    // Perform the actual op substitution, using broadcasted operands when
    // needed.
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    static_assert(ttir::utils::has_dps_trait_v<TTIROpTy>);
    auto outputs = adaptor.getOperands().take_back(op.getNumDpsInits());
    rewriter.replaceOpWithNewOp<LinalgOpTy>(op, resultTypes, broadcastedInputs,
                                            outputs);
    return success();
  }
};
} // namespace

namespace {
// General elementwise conversion pattern, without implicit broadcasting etc.
// Appropriate for unary ops, or other ops without broadcasting.
template <typename TTIROpTy, typename LinAlgOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    static_assert(ttir::utils::has_dps_trait_v<TTIROpTy>);
    auto inputs =
        ttir::utils::getDpsInputsFromAdaptor(adaptor, op.getNumDpsInits());
    auto outputs =
        ttir::utils::getDpsOutputsFromAdaptor(adaptor, op.getNumDpsInits());
    rewriter.replaceOpWithNewOp<LinAlgOpTy>(op, resultTypes, inputs, outputs);
    return success();
  }
};
} // namespace

namespace {
class TransposeOpConversionPattern
    : public OpConversionPattern<ttir::TransposeOp> {
public:
  using OpConversionPattern<ttir::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInput();
    const size_t permSize =
        dyn_cast<RankedTensorType>(input.getType()).getShape().size();
    SmallVector<int64_t> permutation;
    permutation.resize(permSize);
    for (size_t i = 0; i < permSize; i++) {
      permutation[i] = i;
    }

    const int64_t dim0 =
        (op.getDim0() < 0) ? op.getDim0() + permSize : op.getDim0();
    const int64_t dim1 =
        (op.getDim1() < 0) ? op.getDim1() + permSize : op.getDim1();

    permutation[dim1] = dim0;
    permutation[dim0] = dim1;
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, input, adaptor.getOutput(), permutation);
    return success();
  }
};
} // namespace

namespace {
class SoftmaxOpConversionPattern : public OpConversionPattern<ttir::SoftmaxOp> {
public:
  using OpConversionPattern<ttir::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInput();
    const size_t inputSize =
        dyn_cast<RankedTensorType>(input.getType()).getShape().size();
    const int32_t dimension = (op.getDimension() < 0)
                                  ? op.getDimension() + inputSize
                                  : op.getDimension();

    rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
        op, this->getTypeConverter()->convertType(op.getType()), input,
        adaptor.getOutput(), dimension);
    return success();
  }
};
} // namespace

namespace {
class EmptyOpConversionPattern : public OpConversionPattern<ttir::EmptyOp> {
public:
  using OpConversionPattern<ttir::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        /*dynamicSizes*/ ValueRange());
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.reshape operation
class ReshapeOpConversionPattern : public OpConversionPattern<ttir::ReshapeOp> {
public:
  using OpConversionPattern<ttir::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType, input,
                                                   output);

    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.permute operation
class PermuteOpConversionPattern : public OpConversionPattern<ttir::PermuteOp> {
public:
  using OpConversionPattern<ttir::PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    llvm::ArrayRef<int64_t> permutation = op.getPermutation();

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, input, adaptor.getOutput(), permutation);

    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.slice operation
class SliceOpConversionPattern : public OpConversionPattern<ttir::SliceOp> {
public:
  using OpConversionPattern<ttir::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    assert(inputType && "Input must be a ranked tensor type.");

    SmallVector<OpFoldResult> offsets, sizes, strides;

    ArrayAttr begins = op.getBegins();
    ArrayAttr ends = op.getEnds();
    ArrayAttr steps = op.getStep();

    assert(begins.size() == ends.size() && begins.size() == steps.size() &&
           "Invalid slice attributes.");

    for (unsigned i = 0; i < begins.size(); ++i) {
      const int32_t beginVal = llvm::cast<IntegerAttr>(begins[i]).getInt();
      const int32_t endVal = llvm::cast<IntegerAttr>(ends[i]).getInt();
      const int32_t stepVal = llvm::cast<IntegerAttr>(steps[i]).getInt();

      offsets.push_back(rewriter.getI64IntegerAttr(beginVal));

      // Calculate size: (end - begin) / step.
      int64_t size = (endVal - beginVal);
      if (stepVal != 0) {
        size = (size + stepVal - 1) / stepVal;
      }
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      strides.push_back(rewriter.getI64IntegerAttr(stepVal));
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, input, offsets, sizes, strides);

    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.concat operation
class ConcatOpConversionPattern : public OpConversionPattern<ttir::ConcatOp> {
public:
  using OpConversionPattern<ttir::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the dimension to concatenate along.
    int64_t dim = op.getDim();
    static_assert(ttir::utils::has_dps_trait_v<ttir::ConcatOp>);
    auto inputs =
        ttir::utils::getDpsInputsFromAdaptor(adaptor, op.getNumDpsInits());

    // Create a tensor.empty for the result.
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());

    // Insert each input tensor into the result tensor.
    Value result = emptyTensor;
    int64_t offset = 0;

    for (auto input : inputs) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      assert(inputType && "Input must be a ranked tensor type.");

      // Calculate offsets, sizes, and strides for this input
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (unsigned i = 0; i < inputType.getRank(); ++i) {
        if (i == dim) {
          offsets.push_back(rewriter.getI64IntegerAttr(offset));
          offset += inputType.getDimSize(i);
        } else {
          offsets.push_back(rewriter.getI64IntegerAttr(0));
        }
        sizes.push_back(rewriter.getI64IntegerAttr(inputType.getDimSize(i)));
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }

      result = rewriter.create<tensor::InsertSliceOp>(
          op.getLoc(), input, result, offsets, sizes, strides);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class EmbeddingOpConversionPattern
    : public OpConversionPattern<ttir::EmbeddingOp> {
public:
  using OpConversionPattern<ttir::EmbeddingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input tensor and weight
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto weightType = dyn_cast<RankedTensorType>(weight.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    if (!inputType || !weightType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "Input, weight, or result is not a ranked tensor");
    }

    // For embedding, we're gathering along dimension 0
    SmallVector<int64_t> dims{
        0}; // Always gather along dimension 0 for embedding
    auto dimsAttr = rewriter.getDenseI64ArrayAttr(dims);

    // tensor.gather requires integer indices, so we need to cast if the input
    // is not an integer type
    if (!inputType.getElementType().isIntOrIndex()) {
      // Create a new type with the same shape but i64 element type
      auto newInputType =
          RankedTensorType::get(inputType.getShape(), rewriter.getI64Type());

      // Convert the input tensor to integer type
      input =
          rewriter.create<arith::FPToSIOp>(op.getLoc(), newInputType, input);

      // Update inputType to reflect the new type
      inputType = dyn_cast<RankedTensorType>(input.getType());
    }

    // For tensor.gather, the last dimension of indices must match the length of
    // gather_dims If indices is 1D and gather_dims has one element, reshape
    // indices to add a dimension
    auto indicesShape = inputType.getShape();
    if (indicesShape.size() == 1 && dims.size() == 1) {
      // Create a new shape with an additional dimension of size 1
      SmallVector<int64_t> newShape(indicesShape.begin(), indicesShape.end());
      newShape.push_back(1);

      // Create a new type with the additional dimension
      auto reshapedType =
          RankedTensorType::get(newShape, inputType.getElementType());

      // Create a reshape operation to add the dimension
      input = rewriter.create<tensor::ExpandShapeOp>(
          op.getLoc(), reshapedType, input,
          ArrayRef<ReassociationIndices>{{0, 1}});
    }

    // Create the tensor.gather operation
    auto result = rewriter.create<tensor::GatherOp>(op.getLoc(), resultType,
                                                    weight, input, dimsAttr);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.constant operation
class ConstantOpConversionPattern
    : public OpConversionPattern<ttir::ConstantOp> {
public:
  using OpConversionPattern<ttir::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto newConstant =
        rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, value);

    rewriter.replaceOp(op, newConstant.getResult());
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.gather operation
class GatherOpConversionPattern : public OpConversionPattern<ttir::GatherOp> {
public:
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input and indices tensors
    Value input = adaptor.getInput();
    Value indices = adaptor.getStartIndices();

    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get gather operation attributes
    auto startIndexMap = op.getStartIndexMap();

    // Convert startIndexMap to a DenseI64ArrayAttr
    SmallVector<int64_t> dims(startIndexMap.begin(), startIndexMap.end());
    auto dimsAttr = DenseI64ArrayAttr::get(rewriter.getContext(), dims);

    // Convert indices to integer type if needed
    auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
    if (!indicesType) {
      return failure();
    }

    // Check if the indices tensor is already of integer type
    if (!indicesType.getElementType().isIntOrIndex()) {
      // Create a new type with the same shape but i64 element type
      auto newIndicesType =
          RankedTensorType::get(indicesType.getShape(), rewriter.getI64Type());

      // Convert the indices tensor to integer type
      indices = rewriter.create<arith::FPToSIOp>(op.getLoc(), newIndicesType,
                                                 indices);

      // Update indicesType to reflect the new type
      indicesType = dyn_cast<RankedTensorType>(indices.getType());
    }

    // For tensor.gather, the last dimension of indices must match the length of
    // gather_dims If indices is 1D and gather_dims has one element, reshape
    // indices to add a dimension
    auto indicesShape = indicesType.getShape();
    if (indicesShape.size() == 1 && dims.size() == 1) {
      // Create a new shape with an additional dimension of size 1
      SmallVector<int64_t> newShape(indicesShape.begin(), indicesShape.end());
      newShape.push_back(1);

      // Create a new type with the additional dimension
      auto reshapedType =
          RankedTensorType::get(newShape, indicesType.getElementType());

      // Create a reshape operation to add the dimension
      indices = rewriter.create<tensor::ExpandShapeOp>(
          op.getLoc(), reshapedType, indices,
          ArrayRef<ReassociationIndices>{{0, 1}});
    }

    // Create the tensor.gather operation
    auto result = rewriter.create<tensor::GatherOp>(op.getLoc(), resultType,
                                                    input, indices, dimsAttr);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void populateTTIRToLinalgPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  patterns.add<
      ElementwiseBinaryOpConversionPattern<ttir::AddOp, linalg::AddOp>,
      ElementwiseBinaryOpConversionPattern<ttir::MultiplyOp, linalg::MulOp>,
      ElementwiseBinaryOpConversionPattern<ttir::SubtractOp, linalg::SubOp>,
      ElementwiseBinaryOpConversionPattern<ttir::DivOp, linalg::DivOp>,
      ElementwiseBinaryOpConversionPattern<ttir::PowOp, linalg::PowFOp>,
      ElementwiseOpConversionPattern<ttir::AbsOp, linalg::AbsOp>,
      ElementwiseOpConversionPattern<ttir::SqrtOp, linalg::SqrtOp>,
      ElementwiseOpConversionPattern<ttir::RsqrtOp, linalg::RsqrtOp>,
      ElementwiseOpConversionPattern<ttir::ExpOp, linalg::ExpOp>,
      ElementwiseOpConversionPattern<ttir::LogOp, linalg::LogOp>,
      ElementwiseOpConversionPattern<ttir::CeilOp, linalg::CeilOp>,
      ElementwiseOpConversionPattern<ttir::FloorOp, linalg::FloorOp>,
      ElementwiseOpConversionPattern<ttir::TanhOp, linalg::TanhOp>,
      ElementwiseOpConversionPattern<ttir::ReciprocalOp, linalg::ReciprocalOp>,
      ElementwiseOpConversionPattern<ttir::NegOp, linalg::NegFOp>,
      TransposeOpConversionPattern, SoftmaxOpConversionPattern,
      EmptyOpConversionPattern, ReshapeOpConversionPattern,
      PermuteOpConversionPattern, SliceOpConversionPattern,
      ConcatOpConversionPattern, ConstantOpConversionPattern,
      EmbeddingOpConversionPattern, GatherOpConversionPattern>(typeConverter,
                                                               ctx);
}

} // namespace mlir::tt
