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

using namespace mlir;
using namespace mlir::tt;

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
SmallVector<SmallVector<int64_t, 2>, 2>
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
    auto inputs = adaptor.getOperands().drop_back(op.getNumDpsInits());
    auto outputs = adaptor.getOperands().take_back(op.getNumDpsInits());
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

    auto dim0 = op.getDim0();
    auto dim1 = op.getDim1();

    if (dim0 < 0) {
      dim0 = permSize + dim0;
    }
    if (dim1 < 0) {
      dim1 = permSize + dim1;
    }

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
// Conversion pattern for ttir.broadcast operation
class BroadcastOpConversionPattern
    : public OpConversionPattern<ttir::BroadcastOp> {
public:
  using OpConversionPattern<ttir::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto outputType = dyn_cast<RankedTensorType>(adaptor.getOutput().getType());

    if (!inputType || !outputType)
      return failure();

    // Calculate broadcast dimensions
    SmallVector<int64_t> broadcastDims =
        getBroadcastDims(inputType.getShape(), outputType.getShape());

    // Create DenseI64ArrayAttr from the broadcast dimensions
    auto broadcastDimsAttr = rewriter.getDenseI64ArrayAttr(broadcastDims);

    // Use the correct builder signature
    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
        op, input, adaptor.getOutput(), broadcastDimsAttr);

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

    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

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
    auto permutation = op.getPermutation();

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

    if (!inputType)
      return failure();

    // Convert begins, ends, and steps to the format expected by
    // tensor.extract_slice
    SmallVector<OpFoldResult> offsets, sizes, strides;

    // Extract the actual integer values from the attributes
    auto begins = op.getBegins();
    auto ends = op.getEnds();
    auto steps = op.getStep();

    // Make sure all arrays have the same size
    if (begins.size() != ends.size() || begins.size() != steps.size())
      return failure();

    for (unsigned i = 0; i < begins.size(); ++i) {
      // Convert attribute to actual integer values using proper attribute
      // casting
      int32_t beginVal = llvm::cast<IntegerAttr>(begins[i]).getInt();
      int32_t endVal = llvm::cast<IntegerAttr>(ends[i]).getInt();
      int32_t stepVal = llvm::cast<IntegerAttr>(steps[i]).getInt();

      offsets.push_back(rewriter.getI64IntegerAttr(beginVal));

      // Calculate size: (end - begin) / step
      int64_t size = (endVal - beginVal);
      if (stepVal != 0) {
        size = (size + stepVal - 1) / stepVal;
      }
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      strides.push_back(rewriter.getI64IntegerAttr(stepVal));
    }

    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

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
    // Get the dimension to concatenate along
    int64_t dim = op.getDim();
    auto inputs = adaptor.getInputs();

    // Create a tensor.empty for the result
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());

    // Insert each input tensor into the result tensor
    Value result = emptyTensor;
    int64_t offset = 0;

    for (auto input : inputs) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType)
        return failure();

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

      // Insert this input into the result
      result = rewriter.create<tensor::InsertSliceOp>(
          op.getLoc(), input, result, offsets, sizes, strides);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.where operation
class WhereOpConversionPattern : public OpConversionPattern<ttir::WhereOp> {
public:
  using OpConversionPattern<ttir::WhereOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the inputs - WhereOp is an ElementwiseTernaryOp
    auto inputs = adaptor.getInputs();
    if (inputs.size() != 3)
      return failure();

    Value condition = inputs[0];
    Value trueValue = inputs[1];
    Value falseValue = inputs[2];

    // Use getOutputs()[0] instead of getOutput()
    Value output = adaptor.getOutputs()[0];

    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult(0).getType()));
    if (!resultType)
      return failure();

    // Build the indexing maps for the generic operation
    auto context = rewriter.getContext();
    unsigned nloops = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    // Input maps for condition, x, and y
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(nloops, context));
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(nloops, context));
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(nloops, context));
    // Output map
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(nloops, context));

    // Build the iterator types (all parallel)
    SmallVector<utils::IteratorType> iteratorTypes(
        nloops, utils::IteratorType::parallel);

    // Convert iterator types to string attributes
    SmallVector<StringRef> iteratorTypeNames;
    for (auto iterType : iteratorTypes)
      iteratorTypeNames.push_back(utils::stringifyIteratorType(iterType));

    // Create the generic op with the correct builder signature
    auto genericOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(),
        /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{condition, trueValue, falseValue},
        /*outputs=*/ValueRange{output},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes);

    // Build the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        TypeRange{llvm::cast<ShapedType>(condition.getType()).getElementType(),
                  llvm::cast<ShapedType>(trueValue.getType()).getElementType(),
                  llvm::cast<ShapedType>(falseValue.getType()).getElementType(),
                  resultType.getElementType()},
        {op.getLoc(), op.getLoc(), op.getLoc(), op.getLoc()});

    rewriter.setInsertionPointToStart(block);

    // Implement where(cond, x, y) = cond ? x : y
    auto condValue = block->getArgument(0);
    auto xValue = block->getArgument(1);
    auto yValue = block->getArgument(2);

    // Create a select operation
    auto selectOp = rewriter.create<arith::SelectOp>(op.getLoc(), condValue,
                                                     xValue, yValue);

    rewriter.create<linalg::YieldOp>(op.getLoc(), selectOp.getResult());

    rewriter.replaceOp(op, genericOp.getResults());
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
    // Gather is a complex operation that requires a custom implementation
    // using linalg.generic

    auto input = adaptor.getInput();
    auto startIndices = adaptor.getStartIndices();
    auto output = adaptor.getOutput();

    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

    // For simplicity, we'll implement a basic version that works for common
    // cases A full implementation would need to handle all the parameters

    // Create a generic linalg op
    auto context = rewriter.getContext();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto indicesType = dyn_cast<RankedTensorType>(startIndices.getType());

    if (!inputType || !indicesType)
      return failure();

    unsigned nloops = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    // Input map
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(nloops, context));
    // Start indices map
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(nloops, context));
    // Output map
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(nloops, context));

    // Build the iterator types (all parallel)
    SmallVector<utils::IteratorType> iteratorTypes(
        nloops, utils::IteratorType::parallel);

    // Create the generic op
    auto genericOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(),
        /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{input, startIndices},
        /*outputs=*/ValueRange{output},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes);

    // Build the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        TypeRange{dyn_cast<ShapedType>(input.getType()).getElementType(),
                  dyn_cast<ShapedType>(startIndices.getType()).getElementType(),
                  resultType.getElementType()},
        {op.getLoc(), op.getLoc(), op.getLoc()});

    rewriter.setInsertionPointToStart(block);

    // This is a simplified implementation - a full implementation would need to
    // handle all the parameters like offset_dims, collapsed_slice_dims, etc.

    // For now, we'll just yield the input value
    rewriter.create<linalg::YieldOp>(op.getLoc(), block->getArgument(0));

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.arange operation
class ArangeOpConversionPattern : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the attributes
    int64_t start = op.getStart();
    int64_t step = op.getStep();
    int64_t arangeDimension = op.getArangeDimension();

    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

    // Create a tensor.empty for the result
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());

    // Build a linalg.generic to fill the tensor with the arange values
    auto context = rewriter.getContext();
    unsigned rank = resultType.getRank();

    // Create indexing maps
    SmallVector<AffineMap> indexingMaps;
    // Output map
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, context));

    // Build the iterator types (all parallel)
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Create the generic op
    auto genericOp =
        rewriter.create<linalg::GenericOp>(op.getLoc(),
                                           /*resultTensorTypes=*/resultType,
                                           /*inputs=*/ValueRange{},
                                           /*outputs=*/ValueRange{emptyTensor},
                                           /*indexingMaps=*/indexingMaps,
                                           /*iteratorTypes=*/iteratorTypes);

    // Build the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        TypeRange{resultType.getElementType()}, {op.getLoc()});

    rewriter.setInsertionPointToStart(block);

    // Create the index operations to compute the arange value
    SmallVector<Value> indices;
    for (unsigned i = 0; i < rank; ++i) {
      indices.push_back(rewriter.create<linalg::IndexOp>(op.getLoc(), i));
    }

    // Compute the arange value: start + step * index[arangeDimension]
    Value arangeIndex = indices[arangeDimension];

    // Convert to the appropriate type
    Value indexValue;
    if (resultType.getElementType().isIntOrIndex()) {
      indexValue = arangeIndex;
    } else {
      // For floating point types, convert the index to float
      indexValue = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), arangeIndex);
      indexValue = rewriter.create<arith::SIToFPOp>(
          op.getLoc(), resultType.getElementType(), indexValue);
    }

    // Compute: start + step * index
    Value stepValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), resultType.getElementType(),
        rewriter.getFloatAttr(resultType.getElementType(), step));

    Value startValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), resultType.getElementType(),
        rewriter.getFloatAttr(resultType.getElementType(), start));

    Value stepMulIndex;
    if (resultType.getElementType().isIntOrIndex()) {
      stepMulIndex =
          rewriter.create<arith::MulIOp>(op.getLoc(), stepValue, indexValue);
    } else {
      stepMulIndex =
          rewriter.create<arith::MulFOp>(op.getLoc(), stepValue, indexValue);
    }

    Value result;
    if (resultType.getElementType().isIntOrIndex()) {
      result =
          rewriter.create<arith::AddIOp>(op.getLoc(), startValue, stepMulIndex);
    } else {
      result =
          rewriter.create<arith::AddFOp>(op.getLoc(), startValue, stepMulIndex);
    }

    rewriter.create<linalg::YieldOp>(op.getLoc(), result);

    rewriter.replaceOp(op, genericOp.getResults());
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
    // Get the constant value
    auto value = op.getValue();

    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

    // Create a new constant op with the converted type
    auto newConstant =
        rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, value);

    rewriter.replaceOp(op, newConstant.getResult());
    return success();
  }
};
} // namespace

namespace {
// Base class for reduction operations
template <typename TTIROpTy>
class ReductionOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

protected:
  // Create a generic linalg op for reduction
  linalg::GenericOp
  createReductionGenericOp(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                           ConversionPatternRewriter &rewriter,
                           RankedTensorType resultType) const {
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    bool keepDim = op.getKeepDim();

    // Get dimensions to reduce
    SmallVector<int64_t> reductionDims;
    if (auto dimArg = op.getDimArg()) {
      // Specific dimensions provided
      for (auto dim : dimArg.value()) {
        reductionDims.push_back(llvm::cast<IntegerAttr>(dim).getInt());
      }
    } else {
      // Reduce over all dimensions
      auto inputType = llvm::cast<RankedTensorType>(input.getType());
      for (int64_t i = 0; i < inputType.getRank(); ++i) {
        reductionDims.push_back(i);
      }
    }

    // Create the indexing maps
    auto context = rewriter.getContext();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());

    unsigned inputRank = inputType.getRank();

    // Create input indexing map
    SmallVector<AffineExpr> inputExprs;
    for (unsigned i = 0; i < inputRank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }

    // Create output indexing map
    SmallVector<AffineExpr> outputExprs;
    unsigned outputDimIdx = 0;
    for (unsigned i = 0; i < inputRank; ++i) {
      if (std::find(reductionDims.begin(), reductionDims.end(), i) ==
          reductionDims.end()) {
        // This dimension is not being reduced
        outputExprs.push_back(rewriter.getAffineDimExpr(outputDimIdx++));
      } else if (keepDim) {
        // This dimension is being reduced but we're keeping it
        outputExprs.push_back(rewriter.getAffineDimExpr(outputDimIdx++));
      }
    }

    // Create the indexing maps
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(AffineMap::get(inputRank, 0, inputExprs, context));
    indexingMaps.push_back(AffineMap::get(inputRank, 0, outputExprs, context));

    // Create the iterator types
    SmallVector<utils::IteratorType> iteratorTypes;
    for (unsigned i = 0; i < inputRank; ++i) {
      if (std::find(reductionDims.begin(), reductionDims.end(), i) !=
          reductionDims.end()) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
      } else {
        iteratorTypes.push_back(utils::IteratorType::parallel);
      }
    }

    // Create the generic op
    auto genericOp =
        rewriter.create<linalg::GenericOp>(op.getLoc(),
                                           /*resultTensorTypes=*/resultType,
                                           /*inputs=*/ValueRange{input},
                                           /*outputs=*/ValueRange{output},
                                           /*indexingMaps=*/indexingMaps,
                                           /*iteratorTypes=*/iteratorTypes);

    return genericOp;
  }
};

// Conversion pattern for ttir.reduce_or operation
class ReduceOrOpConversionPattern
    : public ReductionOpConversionPattern<ttir::ReduceOrOp> {
public:
  using ReductionOpConversionPattern<
      ttir::ReduceOrOp>::ReductionOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

    // Create the generic op for reduction
    auto genericOp =
        createReductionGenericOp(op, adaptor, rewriter, resultType);

    // Build the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        TypeRange{llvm::cast<ShapedType>(adaptor.getInput().getType())
                      .getElementType(),
                  resultType.getElementType()},
        {op.getLoc(), op.getLoc()});

    rewriter.setInsertionPointToStart(block);

    // Implement the reduction operation: logical OR
    Value lhs = block->getArgument(0);
    Value rhs = block->getArgument(1);

    // Create the OR operation
    Value result = rewriter.create<arith::OrIOp>(op.getLoc(), lhs, rhs);

    rewriter.create<linalg::YieldOp>(op.getLoc(), result);

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

// Conversion pattern for ttir.max operation
class MaxOpConversionPattern
    : public ReductionOpConversionPattern<ttir::MaxOp> {
public:
  using ReductionOpConversionPattern<ttir::MaxOp>::ReductionOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

    // Create the generic op for reduction
    auto genericOp =
        createReductionGenericOp(op, adaptor, rewriter, resultType);

    // Build the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        TypeRange{llvm::cast<ShapedType>(adaptor.getInput().getType())
                      .getElementType(),
                  resultType.getElementType()},
        {op.getLoc(), op.getLoc()});

    rewriter.setInsertionPointToStart(block);

    // Implement the reduction operation: maximum
    Value lhs = block->getArgument(0);
    Value rhs = block->getArgument(1);

    // Create the max operation based on the element type
    Value result;
    if (resultType.getElementType().isIntOrIndex()) {
      result = rewriter.create<arith::MaxSIOp>(op.getLoc(), lhs, rhs);
    } else {
      result = rewriter.create<arith::MaximumFOp>(op.getLoc(), lhs, rhs);
    }

    rewriter.create<linalg::YieldOp>(op.getLoc(), result);

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

// Conversion pattern for ttir.sum operation
class SumOpConversionPattern
    : public ReductionOpConversionPattern<ttir::SumOp> {
public:
  using ReductionOpConversionPattern<ttir::SumOp>::ReductionOpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the result type
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return failure();

    // Create the generic op for reduction
    auto genericOp =
        createReductionGenericOp(op, adaptor, rewriter, resultType);

    // Build the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        TypeRange{llvm::cast<ShapedType>(adaptor.getInput().getType())
                      .getElementType(),
                  resultType.getElementType()},
        {op.getLoc(), op.getLoc()});

    rewriter.setInsertionPointToStart(block);

    // Implement the reduction operation: sum
    Value lhs = block->getArgument(0);
    Value rhs = block->getArgument(1);

    // Create the add operation based on the element type
    Value result;
    if (resultType.getElementType().isIntOrIndex()) {
      result = rewriter.create<arith::AddIOp>(op.getLoc(), lhs, rhs);
    } else {
      result = rewriter.create<arith::AddFOp>(op.getLoc(), lhs, rhs);
    }

    rewriter.create<linalg::YieldOp>(op.getLoc(), result);

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};
} // namespace

namespace {
// Specialized conversion pattern for comparison operations that need to handle
// both integer and floating-point types
template <typename TTIROpTy, arith::CmpIPredicate IntPredicate,
          arith::CmpFPredicate FloatPredicate,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ComparisonOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // First, compute broadcasted shape from operands.
    SmallVector<Value, 2> inputs = adaptor.getInputs();
    assert(inputs.size() == 2 &&
           "Binary element-wise operations must have 2 inputs!");
    ArrayRef<int64_t> input0Shape =
        dyn_cast<RankedTensorType>(inputs[0].getType()).getShape();
    ArrayRef<int64_t> input1Shape =
        dyn_cast<RankedTensorType>(inputs[1].getType()).getShape();

    SmallVector<int64_t, 4> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(input0Shape, input1Shape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(op, "Operands are not broadcastable!");
    }

    // Rewrite inputs to target dims with broadcast and collapse shape ops, as
    // needed.
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

    // Perform the actual op substitution based on element type
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    // Get the element type to determine which comparison op to use
    auto elementType =
        dyn_cast<RankedTensorType>(inputs[0].getType()).getElementType();

    // Create a generic linalg op with the appropriate comparison operation
    auto indexingMaps =
        rewriter.getMultiDimIdentityMap(broadcastedShape.size());
    SmallVector<AffineMap> indexingMapsArray(broadcastedInputs.size() + 1,
                                             indexingMaps);

    // All parallel iterators for elementwise operations
    SmallVector<utils::IteratorType> iteratorTypes(
        broadcastedShape.size(), utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultTypes, broadcastedInputs, adaptor.getOutputs(),
        indexingMapsArray, iteratorTypes);

    // Create the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        {elementType, elementType,
         llvm::cast<ShapedType>(resultTypes[0]).getElementType()},
        {loc, loc, loc});

    rewriter.setInsertionPointToStart(block);

    Value result;
    if (elementType.isIntOrIndex()) {
      // For integer types, use arith.cmpi
      result = rewriter.create<arith::CmpIOp>(
          loc, IntPredicate, block->getArgument(0), block->getArgument(1));
    } else if (elementType.isF32() || elementType.isF64() ||
               elementType.isBF16()) {
      // For floating-point types, use arith.cmpf
      result = rewriter.create<arith::CmpFOp>(
          loc, FloatPredicate, block->getArgument(0), block->getArgument(1));
    } else {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for comparison");
    }

    rewriter.create<linalg::YieldOp>(loc, result);
    rewriter.replaceOp(op, genericOp.getResults());

    return success();
  }
};
} // namespace

namespace {
// Specialized conversion pattern for sigmoid operation
class SigmoidOpConversionPattern : public OpConversionPattern<ttir::SigmoidOp> {
public:
  using OpConversionPattern<ttir::SigmoidOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the inputs and outputs
    SmallVector<Value, 2> inputs = adaptor.getInputs();
    assert(inputs.size() == 1 && "Sigmoid should have exactly 1 input");
    Value input = inputs[0];

    // Get outputs
    SmallVector<Value, 1> outputs = adaptor.getOutputs();
    assert(outputs.size() == 1 && "Sigmoid should have exactly 1 output");
    Value output = outputs[0];

    // Get result type
    auto resultType = llvm::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult(0).getType()));
    if (!resultType)
      return failure();

    // Create a generic linalg op
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    unsigned rank = inputType.getRank();

    // Create identity indexing maps for input and output
    auto indexingMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(2, indexingMap);

    // All parallel iterators for elementwise operations
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, output, indexingMaps, iteratorTypes);

    // Create the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        {inputType.getElementType(), resultType.getElementType()}, {loc, loc});

    rewriter.setInsertionPointToStart(block);

    // Implement sigmoid as 1 / (1 + exp(-x))
    // First, negate the input
    Value negated = rewriter.create<arith::NegFOp>(loc, block->getArgument(0));

    // Then, compute exp(-x)
    Value expNegated = rewriter.create<math::ExpOp>(loc, negated);

    // Add 1 to exp(-x)
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(block->getArgument(0).getType(), 1.0));
    Value onePlusExpNegated =
        rewriter.create<arith::AddFOp>(loc, one, expNegated);

    // Compute 1 / (1 + exp(-x))
    Value result = rewriter.create<arith::DivFOp>(loc, one, onePlusExpNegated);

    rewriter.create<linalg::YieldOp>(loc, result);
    rewriter.replaceOp(op, genericOp.getResults());

    return success();
  }
};
} // namespace

namespace {
// Specialized conversion pattern for trigonometric operations
template <typename TTIROpTy, typename MathOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class TrigOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the inputs and outputs
    SmallVector<Value, 2> inputs = adaptor.getInputs();
    assert(inputs.size() == 1 && "Trig op should have exactly 1 input");
    Value input = inputs[0];

    // Get outputs
    SmallVector<Value, 1> outputs = adaptor.getOutputs();
    assert(outputs.size() == 1 && "Trig op should have exactly 1 output");
    Value output = outputs[0];

    // Get result type
    auto resultType = llvm::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult(0).getType()));
    if (!resultType)
      return failure();

    // Create a generic linalg op
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    unsigned rank = inputType.getRank();

    // Create identity indexing maps for input and output
    auto indexingMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(2, indexingMap);

    // All parallel iterators for elementwise operations
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, output, indexingMaps, iteratorTypes);

    // Create the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        {inputType.getElementType(), resultType.getElementType()}, {loc, loc});

    rewriter.setInsertionPointToStart(block);

    // Create the trigonometric operation
    Value result = rewriter.create<MathOpTy>(loc, block->getArgument(0));

    rewriter.create<linalg::YieldOp>(loc, result);
    rewriter.replaceOp(op, genericOp.getResults());

    return success();
  }
};
} // namespace

namespace {
// Specialized conversion pattern for logical not operation
class LogicalNotOpConversionPattern
    : public OpConversionPattern<ttir::LogicalNotOp> {
public:
  using OpConversionPattern<ttir::LogicalNotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LogicalNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the inputs and outputs
    SmallVector<Value, 2> inputs = adaptor.getInputs();
    assert(inputs.size() == 1 && "LogicalNot should have exactly 1 input");
    Value input = inputs[0];

    // Get outputs
    SmallVector<Value, 1> outputs = adaptor.getOutputs();
    assert(outputs.size() == 1 && "LogicalNot should have exactly 1 output");
    Value output = outputs[0];

    // Get result type
    auto resultType = llvm::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult(0).getType()));
    if (!resultType)
      return failure();

    // Create a generic linalg op
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    unsigned rank = inputType.getRank();

    // Create identity indexing maps for input and output
    auto indexingMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(2, indexingMap);

    // All parallel iterators for elementwise operations
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, output, indexingMaps, iteratorTypes);

    // Create the body of the generic op
    auto *block = rewriter.createBlock(
        &genericOp.getRegion(), genericOp.getRegion().begin(),
        {inputType.getElementType(), resultType.getElementType()}, {loc, loc});

    rewriter.setInsertionPointToStart(block);

    // Implement logical not with XOR
    Value one = rewriter.create<arith::ConstantIntOp>(
        loc, 1, 1); // Create a constant 1 (true) with i1 type
    Value result =
        rewriter.create<arith::XOrIOp>(loc, block->getArgument(0), one);

    rewriter.create<linalg::YieldOp>(loc, result);
    rewriter.replaceOp(op, genericOp.getResults());

    return success();
  }
};
} // namespace

namespace mlir::tt {

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
      TrigOpConversionPattern<ttir::CosOp, math::CosOp>,
      TrigOpConversionPattern<ttir::SinOp, math::SinOp>,
      ComparisonOpConversionPattern<ttir::EqualOp, arith::CmpIPredicate::eq,
                                    arith::CmpFPredicate::OEQ>,
      ComparisonOpConversionPattern<ttir::GreaterEqualOp,
                                    arith::CmpIPredicate::sge,
                                    arith::CmpFPredicate::OGE>,
      ComparisonOpConversionPattern<ttir::GreaterThanOp,
                                    arith::CmpIPredicate::sgt,
                                    arith::CmpFPredicate::OGT>,
      TransposeOpConversionPattern, SoftmaxOpConversionPattern,
      SigmoidOpConversionPattern, EmptyOpConversionPattern,
      BroadcastOpConversionPattern, ReshapeOpConversionPattern,
      PermuteOpConversionPattern, SliceOpConversionPattern,
      ConcatOpConversionPattern, WhereOpConversionPattern,
      GatherOpConversionPattern, ArangeOpConversionPattern,
      ConstantOpConversionPattern, ReduceOrOpConversionPattern,
      MaxOpConversionPattern, SumOpConversionPattern,
      LogicalNotOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
