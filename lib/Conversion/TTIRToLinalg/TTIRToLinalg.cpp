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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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

// Get dimensions from the dim_arg attribute; if the attribute is not present or
// empty, return all dimensions
static SmallVector<int64_t> getDimsFromAttribute(Operation *op, int64_t rank) {
  if (auto dimAttr = op->getAttrOfType<ArrayAttr>("dim_arg")) {
    if (dimAttr.size() == 0) {
      // If dim_arg is present but empty, reduce along all dimensions
      SmallVector<int64_t> allDims(rank);
      std::iota(allDims.begin(), allDims.end(), 0);
      return allDims;
    }

    // Otherwise, use the provided dimensions
    SmallVector<int64_t> dims;
    for (auto dim : dimAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(dim)) {
        dims.push_back(intAttr.getInt());
      }
    }
    return dims;
  }

  // If no dim_arg attribute, reduce along all dimensions
  SmallVector<int64_t> allDims(rank);
  std::iota(allDims.begin(), allDims.end(), 0);
  return allDims;
}

// Get keep_dim attribute; return false is not present.
static bool getKeepDimFromAttribute(Operation *op) {
  if (auto keepDimAttr = op->getAttrOfType<BoolAttr>("keep_dim")) {
    return keepDimAttr.getValue();
  }
  return false;
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
class GatherOpConversionPattern : public OpConversionPattern<ttir::GatherOp> {
public:
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return decomposeToLinalg(op, adaptor, rewriter);
  }

private:
  LogicalResult decomposeToLinalg(ttir::GatherOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value startIndices = adaptor.getStartIndices();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto indicesType = cast<RankedTensorType>(startIndices.getType());
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    // Extract attributes
    auto offsetDims = op.getOffsetDims();
    auto collapsedSliceDims = op.getCollapsedSliceDims();
    auto startIndexMap = op.getStartIndexMap();
    auto indexVectorDim = op.getIndexVectorDim();

    // Create initial tensor for result
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Build indexing maps for the generic op
    auto resultRank = resultType.getRank();

    // Create identity map for output
    SmallVector<AffineExpr> outputExprs;
    for (int i = 0; i < resultRank; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap outputMap =
        AffineMap::get(resultRank, 0, outputExprs, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {outputMap};

    // All dimensions are parallel for gather
    SmallVector<utils::IteratorType> iteratorTypes(
        resultRank, utils::IteratorType::parallel);

    // Create the indexing logic using linalg.generic
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{}, ValueRange{initTensor}, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] is the current value in the output tensor

          // Get the current output indices
          SmallVector<Value> outputIndices;
          for (int i = 0; i < resultRank; ++i) {
            outputIndices.push_back(b.create<linalg::IndexOp>(loc, i));
          }

          // Build the input indices for the gather
          SmallVector<Value> inputIndices(inputType.getRank());

          // Initialize all indices to zero first to avoid null values
          for (int64_t i = 0; i < inputType.getRank(); ++i) {
            inputIndices[i] = b.create<arith::ConstantIndexOp>(loc, 0);
          }

          // Determine which output dimensions are batch dimensions
          SmallVector<int64_t> batchDims;
          for (int64_t i = 0; i < resultRank; ++i) {
            if (!llvm::is_contained(offsetDims, i)) {
              batchDims.push_back(i);
            }
          }

          // Extract indices from startIndices tensor for each dimension in
          // startIndexMap
          for (size_t i = 0; i < startIndexMap.size(); ++i) {
            SmallVector<Value> fullIndices;

            // Build indices based on the structure of the indices tensor
            if (indexVectorDim == 0 && indicesType.getRank() == 1) {
              // Special case: 1D indices tensor with index_vector_dim=0
              // For gathering, we use the first batch dimension (output dim 0)
              fullIndices.push_back(outputIndices[0]);
            } else if (indexVectorDim ==
                       static_cast<int64_t>(indicesType.getRank())) {
              // Index vector is implicit (size 1)
              // Use batch dimensions from output
              for (auto batchDim : batchDims) {
                fullIndices.push_back(outputIndices[batchDim]);
              }
            } else {
              // Normal case: index vector is at a specific dimension
              // Build indices from batch dimensions
              int batchIdx = 0;
              for (int64_t d = 0; d < indicesType.getRank(); ++d) {
                if (d == indexVectorDim) {
                  // This is the index vector dimension
                  fullIndices.push_back(
                      b.create<arith::ConstantIndexOp>(loc, i));
                } else {
                  // This is a batch dimension
                  if (static_cast<size_t>(batchIdx) < batchDims.size()) {
                    fullIndices.push_back(outputIndices[batchDims[batchIdx]]);
                    batchIdx++;
                  }
                }
              }
            }

            // Extract the index value
            Value idxValue =
                b.create<tensor::ExtractOp>(loc, startIndices, fullIndices);

            // Convert to index type if needed
            Value idx;
            if (idxValue.getType().isF32()) {
              // First convert f32 to i32
              Value i32Val =
                  b.create<arith::FPToSIOp>(loc, b.getI32Type(), idxValue);
              // Then convert i32 to index
              idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), i32Val);
            } else if (idxValue.getType().isInteger(32)) {
              // Direct cast from i32 to index
              idx =
                  b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxValue);
            } else if (idxValue.getType().isInteger(64)) {
              // Direct cast from i64 to index
              idx =
                  b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxValue);
            } else {
              // Already index type
              idx = idxValue;
            }

            inputIndices[startIndexMap[i]] = idx;
          }

          // Map offset dimensions from output to input
          // For each offset dimension in the output, find the corresponding
          // input dimension
          for (size_t i = 0; i < offsetDims.size(); ++i) {
            int64_t outputDim = offsetDims[i];

            // Find the corresponding input dimension
            // We need to skip over gathered dimensions and collapsed dimensions
            int64_t inputDim = 0;
            int64_t nonCollapsedCount = 0;

            // Count non-collapsed dimensions until we reach the i-th one
            while (inputDim < inputType.getRank() &&
                   static_cast<size_t>(nonCollapsedCount) <= i) {
              if (!llvm::is_contained(collapsedSliceDims, inputDim) &&
                  !llvm::is_contained(startIndexMap, inputDim)) {
                if (static_cast<size_t>(nonCollapsedCount) == i) {
                  inputIndices[inputDim] = outputIndices[outputDim];
                  break;
                }
                nonCollapsedCount++;
              }
              inputDim++;
            }
          }

          // Extract the value from input tensor
          Value extracted =
              b.create<tensor::ExtractOp>(loc, input, inputIndices);

          b.create<linalg::YieldOp>(loc, extracted);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class MatmulOpConversionPattern : public OpConversionPattern<ttir::MatmulOp> {
public:
  using OpConversionPattern<ttir::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "Operands or result is not a ranked tensor");
    }

    bool transposeA = op.getTransposeA();
    bool transposeB = op.getTransposeB();

    // Handle transposition if needed
    if (transposeA) {
      auto lhsShape = lhsType.getShape();
      SmallVector<int64_t> transposedShape;

      if (lhsShape.size() >= 2) {
        // For 2D+ tensors, transpose the last two dimensions
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          transposedShape.push_back(lhsShape[i]);
        }
        transposedShape.push_back(lhsShape[lhsShape.size() - 1]);
        transposedShape.push_back(lhsShape[lhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, lhsType.getElementType());

        // Create permutation attribute
        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 2));

        // Create transpose op
        lhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 lhs, permutation);
        lhsType = transposedType;
      }
    }

    if (transposeB) {
      auto rhsShape = rhsType.getShape();
      SmallVector<int64_t> transposedShape;

      if (rhsShape.size() >= 2) {
        // For 2D+ tensors, transpose the last two dimensions
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          transposedShape.push_back(rhsShape[i]);
        }
        transposedShape.push_back(rhsShape[rhsShape.size() - 1]);
        transposedShape.push_back(rhsShape[rhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, rhsType.getElementType());

        // Create permutation attribute
        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 2));

        // Create transpose op
        rhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 rhs, permutation);
        rhsType = transposedType;
      }
    }

    // Ensure both tensors are 3D for tosa.matmul
    unsigned lhsRank = lhsType.getRank();
    unsigned rhsRank = rhsType.getRank();

    // Convert to 3D tensors if needed
    Value lhs3D = lhs;
    Value rhs3D = rhs;
    RankedTensorType lhs3DType = lhsType;
    RankedTensorType rhs3DType = rhsType;

    // If LHS is 2D, reshape to 3D with batch size 1
    if (lhsRank == 2) {
      SmallVector<int64_t> newShape = {1, lhsType.getDimSize(0),
                                       lhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      // Create shape tensor for reshape - matching your original approach
      auto shapeType = mlir::tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {1, lhsType.getDimSize(0),
                                          lhsType.getDimSize(1)};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape LHS to 3D
      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    } else if (lhsRank > 3) {
      // For tensors with rank > 3, collapse all but the last two dimensions
      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < lhsRank - 2; ++i) {
        collapsedBatchSize *= lhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       lhsType.getShape()[lhsRank - 2],
                                       lhsType.getShape()[lhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      // Create shape tensor for reshape
      auto shapeType = mlir::tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {collapsedBatchSize,
                                          lhsType.getShape()[lhsRank - 2],
                                          lhsType.getShape()[lhsRank - 1]};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape LHS to 3D
      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    }

    // If RHS is 2D, reshape to 3D with batch size 1
    if (rhsRank == 2) {
      SmallVector<int64_t> newShape = {1, rhsType.getDimSize(0),
                                       rhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      // Create shape tensor for reshape
      auto shapeType = RankedTensorType::get({3}, rewriter.getI64Type());
      auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
      auto shapeOp =
          rewriter.create<arith::ConstantOp>(op.getLoc(), shapeType, shapeAttr);

      // Reshape RHS to 3D
      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    } else if (rhsRank > 3) {
      // For tensors with rank > 3, collapse all but the last two dimensions
      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < rhsRank - 2; ++i) {
        collapsedBatchSize *= rhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       rhsType.getShape()[rhsRank - 2],
                                       rhsType.getShape()[rhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      // Create shape tensor for reshape
      auto shapeType = RankedTensorType::get({3}, rewriter.getI64Type());
      auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
      auto shapeOp =
          rewriter.create<arith::ConstantOp>(op.getLoc(), shapeType, shapeAttr);

      // Reshape RHS to 3D
      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    }

    // Check if we need to broadcast batch dimensions
    if (lhs3DType.getShape()[0] != rhs3DType.getShape()[0]) {
      // We need to broadcast one of the inputs to match the other's batch
      // dimension
      if (lhs3DType.getShape()[0] == 1 && rhs3DType.getShape()[0] > 1) {
        // Use TOSA tile operation for broadcasting
        SmallVector<int64_t> multiples = {rhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({rhs3DType.getShape()[0],
                                              lhs3DType.getShape()[1],
                                              lhs3DType.getShape()[2]},
                                             lhs3DType.getElementType());

        // Create multiples using tosa.const for tile operation
        auto multiplesType = RankedTensorType::get({3}, rewriter.getI64Type());
        auto multiplesAttr =
            DenseIntElementsAttr::get(multiplesType, multiples);
        auto multiplesOp = rewriter.create<tosa::ConstOp>(
            op.getLoc(), multiplesType, multiplesAttr);

        lhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, lhs3D,
                                              multiplesOp);
        lhs3DType = cast<RankedTensorType>(lhs3D.getType());
      } else if (rhs3DType.getShape()[0] == 1 && lhs3DType.getShape()[0] > 1) {
        // Use TOSA tile operation for broadcasting
        SmallVector<int64_t> multiples = {lhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({lhs3DType.getShape()[0],
                                              rhs3DType.getShape()[1],
                                              rhs3DType.getShape()[2]},
                                             rhs3DType.getElementType());

        // Create multiples using tosa.const for tile operation
        auto multiplesType = RankedTensorType::get({3}, rewriter.getI64Type());
        auto multiplesAttr =
            DenseIntElementsAttr::get(multiplesType, multiples);
        auto multiplesOp = rewriter.create<tosa::ConstOp>(
            op.getLoc(), multiplesType, multiplesAttr);

        rhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, rhs3D,
                                              multiplesOp);
        rhs3DType = cast<RankedTensorType>(rhs3D.getType());
      }
    }

    // Now both tensors should have the same batch dimension
    auto matmulResultType =
        RankedTensorType::get({lhs3DType.getShape()[0], lhs3DType.getShape()[1],
                               rhs3DType.getShape()[2]},
                              resultType.getElementType());

    // Perform matrix multiplication using tosa.matmul
    Value matmulResult = rewriter.create<tosa::MatMulOp>(
        op.getLoc(), matmulResultType, lhs3D, rhs3D);

    // Reshape result back to original rank if needed
    if (resultType.getRank() != matmulResultType.getRank()) {
      // Create shape tensor for reshape
      auto shapeType = mlir::tosa::shapeType::get(rewriter.getContext(),
                                                  resultType.getRank());
      SmallVector<int64_t> shapeValues;
      for (auto dim : resultType.getShape()) {
        shapeValues.push_back(dim);
      }
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      // Reshape result
      matmulResult = rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), resultType, matmulResult, shapeOp.getResult());
    }

    rewriter.replaceOp(op, matmulResult);
    return success();
  }
};
} // namespace

// Helper function to create a chain of reduction operations for multiple
// dimensions
template <typename ReductionOp>
static Value createReductionOpChain(Value input, RankedTensorType resultType,
                                    ArrayRef<int64_t> dims, bool keepDim,
                                    Location loc,
                                    ConversionPatternRewriter &rewriter) {
  // Sort dimensions in descending order to avoid changing indices during
  // reduction
  SmallVector<int64_t> sortedDims(dims.begin(), dims.end());
  std::sort(sortedDims.begin(), sortedDims.end(), std::greater<int64_t>());

  Value result = input;
  auto inputType = cast<RankedTensorType>(input.getType());

  // For each dimension, create a reduction operation
  for (size_t i = 0; i < sortedDims.size(); ++i) {
    int64_t dim = sortedDims[i];

    // Create the axis attribute for this dimension
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(dim));

    // For the last dimension in our chain, use the final result type
    // For intermediate dimensions, calculate the intermediate shape
    RankedTensorType opResultType;
    if (i == sortedDims.size() - 1) {
      opResultType = resultType;
    } else {
      SmallVector<int64_t> shape(inputType.getShape().begin(),
                                 inputType.getShape().end());
      if (keepDim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
      opResultType = RankedTensorType::get(shape, inputType.getElementType());
    }

    // Create the reduction operation
    result = rewriter.create<ReductionOp>(loc, opResultType, result, axisAttr);
  }

  return result;
}

namespace {
class SumOpConversionPattern : public OpConversionPattern<ttir::SumOp> {
public:
  using OpConversionPattern<ttir::SumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create a chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceSumOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class MaxOpConversionPattern : public OpConversionPattern<ttir::MaxOp> {
public:
  using OpConversionPattern<ttir::MaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Get dimensions to reduce and keep_dim attribute
    SmallVector<int64_t> dims = getDimsFromAttribute(op, rank);
    bool keepDim = getKeepDimFromAttribute(op);

    // Create a chain of reduction operations
    Value result = createReductionOpChain<tosa::ReduceMaxOp>(
        input, resultType, dims, keepDim, op.getLoc(), rewriter);

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
      ConcatOpConversionPattern, ConstantOpConversionPattern>(typeConverter,
                                                              ctx);
}

void populateTTIRToTosaPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  patterns.add<MatmulOpConversionPattern, GatherOpConversionPattern,
               MaxOpConversionPattern, SumOpConversionPattern>(typeConverter,
                                                               ctx);
}

} // namespace mlir::tt
