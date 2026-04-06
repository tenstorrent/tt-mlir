// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"

#include "ttmlir/Conversion/TTIRToLinalg/EltwiseBinary.h"
#include "ttmlir/Conversion/TTIRToLinalg/EltwiseUnary.h"
#include "ttmlir/Conversion/TTIRToLinalg/Pooling.h"
#include "ttmlir/Conversion/TTIRToLinalg/Reduction.h"
#include "ttmlir/Conversion/TTIRToLinalg/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttir_to_linalg {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
namespace {
// Convert a tensor of floating-point values to a tensor of boolean values
// using comparison semantics (positive values are true, non-positive are
// false)--whereOp uses this pattern unfortunately.
static FailureOr<Value>
convertToBooleanTensorComparison(Value input, Location loc,
                                 ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return input;
  }

  // If it's already a boolean tensor, return it as is
  if (inputType.getElementType().isInteger(1)) {
    return input;
  }

  auto elementType = inputType.getElementType();
  if (!elementType.isF32()) {
    return failure();
  }

  // Create zero constant.
  SmallVector<int64_t> zeroShape(inputType.getRank(), 1);
  auto zeroType = RankedTensorType::get(zeroShape, elementType);
  DenseElementsAttr zeroAttr =
      DenseElementsAttr::get(zeroType, rewriter.getF32FloatAttr(0.0f));
  auto zeroConst = rewriter.create<tosa::ConstOp>(loc, zeroType, zeroAttr);

  // For comparison semantics: positive values are true, so we need: (input >
  // 0).
  auto boolType =
      RankedTensorType::get(inputType.getShape(), rewriter.getIntegerType(1));
  auto greaterThanZero =
      rewriter.create<tosa::GreaterOp>(loc, boolType, input, zeroConst);

  return greaterThanZero.getResult();
}

// Reshape a tensor by prepending 1s to match the target rank.
// This is useful for broadcasting parameters like weight and bias in LayerNorm.
// For example, if input has shape [64] and targetRank is 3 with numTrailingDims
// = 1, the result will have shape [1, 1, 64].
static Value reshapeByPrependingOnes(Value input, int64_t targetRank,
                                     int64_t numTrailingDims, Type elementType,
                                     Location loc,
                                     ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());

  SmallVector<int64_t> broadcastShape;
  for (int64_t i = 0; i < targetRank - numTrailingDims; ++i) {
    broadcastShape.push_back(1);
  }
  for (auto dim : inputType.getShape()) {
    broadcastShape.push_back(dim);
  }

  auto reshapedType = RankedTensorType::get(broadcastShape, elementType);
  auto shapeType =
      tosa::shapeType::get(rewriter.getContext(), broadcastShape.size());
  auto shapeAttr = rewriter.getIndexTensorAttr(broadcastShape);
  auto shapeOp = rewriter.create<tosa::ConstShapeOp>(loc, shapeType, shapeAttr);
  return rewriter.create<tosa::ReshapeOp>(loc, reshapedType, input,
                                          shapeOp.getResult());
}

} // namespace

//===----------------------------------------------------------------------===//
// TOSA Conversions Patterns
//===----------------------------------------------------------------------===//

namespace {
class WhereOpConversionPattern : public OpConversionPattern<ttir::WhereOp> {
public:
  using OpConversionPattern<ttir::WhereOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value condition = adaptor.getFirst();
    Value trueValue = adaptor.getSecond();
    Value falseValue = adaptor.getThird();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto conditionOrFailure =
        convertToBooleanTensorComparison(condition, op.getLoc(), rewriter);
    if (failed(conditionOrFailure)) {
      return rewriter.notifyMatchFailure(
          op, "Condition element type must be f32 or i1");
    }
    condition = *conditionOrFailure;

    auto result = rewriter.create<tosa::SelectOp>(
        op.getLoc(), resultType, condition, trueValue, falseValue);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ReshapeOpConversionPattern : public OpConversionPattern<ttir::ReshapeOp> {
public:
  using OpConversionPattern<ttir::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto newShape = resultType.getShape();
    SmallVector<int64_t> newShapeValues(newShape.begin(), newShape.end());
    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShapeValues);
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), resultType, adaptor.getInput(), shapeOp);

    rewriter.replaceOp(op, reshapeOp);

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

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    const size_t permSize = inputType.getShape().size();
    SmallVector<int32_t> permutation(permSize);
    for (size_t i = 0; i < permSize; i++) {
      permutation[i] = static_cast<int32_t>(i);
    }

    const int64_t dim0 =
        (op.getDim0() < 0) ? op.getDim0() + permSize : op.getDim0();
    const int64_t dim1 =
        (op.getDim1() < 0) ? op.getDim1() + permSize : op.getDim1();

    permutation[dim1] = static_cast<int32_t>(dim0);
    permutation[dim0] = static_cast<int32_t>(dim1);

    // Create TransposeOp directly with the permutation array
    auto result = rewriter.create<tosa::TransposeOp>(op.getLoc(), resultType,
                                                     input, permutation);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ConcatOpConversionPattern : public OpConversionPattern<ttir::ConcatOp> {
public:
  using OpConversionPattern<ttir::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputs = adaptor.getOperands();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // TOSA concat requires non-negative axis, so normalize negative dimensions.
    int64_t dim = normalizeDim(op.getDim(), resultType.getRank());

    // TOSA concat requires at least two inputs.
    if (inputs.size() < 2) {
      return failure();
    }

    // Concatenate all inputs at once using the final result type.
    Value result =
        rewriter.create<tosa::ConcatOp>(op.getLoc(), resultType, inputs, dim);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class BroadcastOpConversionPattern
    : public OpConversionPattern<ttir::BroadcastOp> {
public:
  using OpConversionPattern<ttir::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> targetShape = resultType.getShape();

    // Calculate broadcast dimensions - these are the dimensions we need to
    // broadcast along.
    SmallVector<int64_t, 2> broadcastDims =
        getBroadcastDims(inputShape, targetShape);

    // If no broadcasting needed, just use the input directly.
    if (broadcastDims.empty()) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // linalg.broadcast requires that input tensor only contains dimensions
    // that won't be broadcasted. We need to collapse any size-1 dimensions
    // that will be broadcasted.
    SmallVector<SmallVector<int64_t, 2>, 2> collapseDimGroups =
        (broadcastDims.size() != targetShape.size())
            ? getCollapseDims(inputShape, targetShape)
            : SmallVector<SmallVector<int64_t, 2>, 2>();

    Value broadcastInput = input;

    // The broadcast op requires we actually collapse any dimensions with
    // size 1 we want to broadcast along.
    if (collapseDimGroups.size() != inputShape.size()) {
      broadcastInput = rewriter.create<tensor::CollapseShapeOp>(
          loc, input, collapseDimGroups);
    }

    auto initTensor = rewriter.create<ttir::EmptyOp>(
        loc, targetShape, inputType.getElementType());
    auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, broadcastInput, initTensor.getResult(), broadcastDims);

    rewriter.replaceOp(op, broadcastOp.getResults().front());
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
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
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
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
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
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {1, rhsType.getDimSize(0),
                                          rhsType.getDimSize(1)};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

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
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      SmallVector<int64_t> shapeValues = {collapsedBatchSize,
                                          rhsType.getShape()[rhsRank - 2],
                                          rhsType.getShape()[rhsRank - 1]};
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

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

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

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

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

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
      auto shapeType =
          tosa::shapeType::get(rewriter.getContext(), resultType.getRank());
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

namespace {
class Conv2dOpConversionPattern : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
    Attribute strides = adaptor.getStride();
    Attribute dilations = adaptor.getDilation();
    Attribute padding = adaptor.getPadding();
    uint32_t groups = adaptor.getGroups();

    if (groups > 1) {
      return rewriter.notifyMatchFailure(
          op, "Grouped convolution is not supported yet.");
    }
    auto weightType = cast<RankedTensorType>(weight.getType());

    // TTIR uses (O,C,H,W) but TOSA uses (O,H,W,C) for weight.
    SmallVector<int32_t> permutation = {0, 2, 3, 1};
    auto weightShape = weightType.getShape();
    SmallVector<int64_t> transposedShape = ttmlir::utils::applyPermutation(
        weightShape, llvm::to_vector_of<int64_t>(permutation));

    auto transposedWeightType =
        RankedTensorType::get(transposedShape, weightType.getElementType());

    auto transposedWeight = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposedWeightType, weight, permutation);

    // Reshape bias from 4D (1,1,1,B) to 1D (B) for TOSA.
    // If bias is not provided, create a zero bias tensor.
    Value reshapedBias = nullptr;
    if (bias) {
      auto biasType = cast<RankedTensorType>(bias.getType());
      auto biasShape = biasType.getShape();
      if (biasShape.size() != 4) {
        return rewriter.notifyMatchFailure(op, "Bias must be 4D");
      }
      if (!(biasShape[0] == 1 && biasShape[1] == 1 && biasShape[2] == 1)) {
        return rewriter.notifyMatchFailure(
            op, "Bias must be 4D with shape (1,1,1,B)");
      }
      SmallVector<int64_t> reshapedBiasShape = {biasShape[3]};
      auto reshapedBiasType =
          RankedTensorType::get(reshapedBiasShape, biasType.getElementType());
      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 1);
      auto shapeAttr = rewriter.getIndexTensorAttr(reshapedBiasShape);
      auto shapeOp = rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType,
                                                         shapeAttr);

      reshapedBias = rewriter
                         .create<tosa::ReshapeOp>(op.getLoc(), reshapedBiasType,
                                                  bias, shapeOp.getResult())
                         .getResult();
    } else {
      int64_t outputChannels = weightShape[0];
      auto biasElementType =
          cast<RankedTensorType>(input.getType()).getElementType();
      auto biasType = RankedTensorType::get({outputChannels}, biasElementType);

      Attribute zeroAttr;
      if (isa<FloatType>(biasElementType)) {
        zeroAttr = DenseElementsAttr::get(
            biasType,
            APFloat::getZero(
                cast<FloatType>(biasElementType).getFloatSemantics()));
      } else if (isa<IntegerType>(biasElementType)) {
        zeroAttr = DenseElementsAttr::get(
            biasType,
            APInt::getZero(cast<IntegerType>(biasElementType).getWidth()));
      } else {
        return rewriter.notifyMatchFailure(op, "Unsupported bias element type");
      }
      reshapedBias = rewriter.create<tosa::ConstOp>(
          op.getLoc(), biasType, cast<DenseElementsAttr>(zeroAttr));
    }
    // Expand stride if it contains only one element.
    auto stridesResult = ttmlir::utils::getPairOfInteger<int32_t>(strides);
    if (!stridesResult) {
      return rewriter.notifyMatchFailure(
          op, "stride must be an integer or array attribute");
    }

    // Expand padding if it contains only one or two elements.
    auto paddingResult = ttmlir::utils::getQuadrupleOfInteger<int32_t>(padding);
    if (!paddingResult) {
      return rewriter.notifyMatchFailure(
          op, "padding must be an integer, 2-element, or 4-element array "
              "attribute");
    }
    // If padding is a single integer, expand it to 4 elements.
    // If padding is a 2-element array, it is (width, height) and expand it to 4
    // elements (width, height, width, height). If padding is a 4-element array,
    // TTIR uses (top, left, bottom, right) but TOSA uses (top, bottom, left,
    // right). Permutation of indices 1 and 2 is needed to match the padding
    // order. (width, height, width, height) -> (width, width, height, height)
    // (top, left, bottom, right) -> (top, bottom, left, right)

    auto [paddingTop, paddingLeft, paddingBottom, paddingRight] =
        *paddingResult;

    // Expand dilation if it contains only one element.
    auto dilationsResult = ttmlir::utils::getPairOfInteger<int32_t>(dilations);
    if (!dilationsResult) {
      return rewriter.notifyMatchFailure(
          op, "dilation must be an integer or array attribute");
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // Handle flattened input: unflatten and compute correct NHWC result type.
    auto flatInfo = op.getFlattenedCompatInfoAttr();
    RankedTensorType savedFlatResultType;
    if (flatInfo) {
      savedFlatResultType = resultType;
      input = unflattenInput(input, flatInfo, rewriter, op.getLoc());
      int64_t inH = flatInfo.getInputHeight();
      int64_t inW = flatInfo.getInputWidth();
      int64_t outH = (inH + paddingTop + paddingBottom -
                      dilationsResult->first * (weightShape[2] - 1) - 1) /
                         stridesResult->first +
                     1;
      int64_t outW = (inW + paddingLeft + paddingRight -
                      dilationsResult->second * (weightShape[3] - 1) - 1) /
                         stridesResult->second +
                     1;
      resultType = RankedTensorType::get({flatInfo.getBatchSize(), outH, outW,
                                          savedFlatResultType.getShape()[3]},
                                         savedFlatResultType.getElementType());
    }

    // Choose accumulator type based on result element type.
    Type accType;
    if (isa<FloatType>(resultType.getElementType())) {
      accType = rewriter.getF32Type();
    } else if (isa<IntegerType>(resultType.getElementType())) {
      accType = rewriter.getI32Type();
    } else {
      return rewriter.notifyMatchFailure(
          op, "Unsupported result element type for conv2d");
    }

    // Update padding and return shape to be used in the TOSA Conv2DOp.
    // input_height - 1 + pad_top + pad_bottom - (kernel_height - 1) *
    // dilation_y must be divisible by stride_y. input_width - 1 + pad_left +
    // pad_right - (kernel_width - 1) * dilation_x must be divisible by
    // stride_x. The padding values are updated to ensure this condition is met.
    int64_t inputHeight = cast<RankedTensorType>(input.getType()).getShape()[1];
    int64_t inputWidth = cast<RankedTensorType>(input.getType()).getShape()[2];
    int64_t kernelHeight = weightShape[2];
    int64_t kernelWidth = weightShape[3];

    paddingBottom += calculateExtraPadding(
        inputHeight, kernelHeight, stridesResult->first, paddingTop,
        paddingBottom, dilationsResult->first);
    paddingRight += calculateExtraPadding(
        inputWidth, kernelWidth, stridesResult->second, paddingLeft,
        paddingRight, dilationsResult->second);

    auto expandedStridesAttr = rewriter.getDenseI64ArrayAttr(
        {stridesResult->first, stridesResult->second});
    auto expandedDilationsAttr = rewriter.getDenseI64ArrayAttr(
        {dilationsResult->first, dilationsResult->second});
    auto expandedPaddingAttr = rewriter.getDenseI64ArrayAttr(
        {paddingTop, paddingBottom, paddingLeft, paddingRight});

    // Update return shape to be used in the TOSA Conv2DOp.
    // Height and width must be adjusted for extra padding.
    // Batch size and output channels are not changed.
    SmallVector<int64_t> resultShape = {
        resultType.getShape()[0],
        (inputHeight - 1 + paddingTop + paddingBottom -
         (kernelHeight - 1) * dilationsResult->first) /
                stridesResult->first +
            1,
        (inputWidth - 1 + paddingLeft + paddingRight -
         (kernelWidth - 1) * dilationsResult->second) /
                stridesResult->second +
            1,
        resultType.getShape()[3]};

    auto actualResultType =
        RankedTensorType::get(resultShape, resultType.getElementType());

    auto conv2dOp = rewriter.create<tosa::Conv2DOp>(
        op.getLoc(), actualResultType, input, transposedWeight.getResult(),
        reshapedBias, expandedPaddingAttr, expandedStridesAttr,
        expandedDilationsAttr, TypeAttr::get(accType));

    Value result = sliceResultToShape(conv2dOp.getResult(), resultType,
                                      rewriter, op.getLoc());

    // Flatten result back if input was flattened.
    if (flatInfo) {
      result =
          createTosaReshape(result, savedFlatResultType, rewriter, op.getLoc());
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
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto weightType = cast<RankedTensorType>(weight.getType());
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    int64_t inputRank = inputType.getRank();
    int64_t weightRank = weightType.getRank();
    int64_t resultRank = resultType.getRank();

    // Create empty output tensor.
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Input indexing map: project result dims to input dims.
    // result(d0, ..., d_{N-1}, d_N, ..., d_{N+E-1}) -> input(d0, ..., d_{N-1})
    // This lets linalg read input elements via an affine map rather than
    // scalar tensor.extract, enabling better tiling and vectorization.
    SmallVector<AffineExpr> inputExprs;
    for (int64_t i = 0; i < inputRank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap inputMap =
        AffineMap::get(resultRank, 0, inputExprs, rewriter.getContext());

    // Identity map for output.
    AffineMap outputMap =
        AffineMap::getMultiDimIdentityMap(resultRank, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};

    // All dimensions are parallel.
    SmallVector<utils::IteratorType> iteratorTypes(
        resultRank, utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{input}, ValueRange{initTensor},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] is the input element (index value) read via the affine map.
          // args[1] is the current output element.
          Value idxValue = args[0];

          // Convert the index value to index type.
          Value idx;
          if (idxValue.getType().isF32()) {
            Value i32Val =
                b.create<arith::FPToSIOp>(loc, b.getI32Type(), idxValue);
            idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), i32Val);
          } else {
            idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxValue);
          }

          // Build weight indices:
          // - Leading dims (all 1s in weight) are 0.
          // - Second-to-last dim is the extracted index.
          // - Last dim is the last result iteration index.
          SmallVector<Value> weightIndices;
          for (int64_t i = 0; i < weightRank - 2; ++i) {
            weightIndices.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
          }
          weightIndices.push_back(idx);
          weightIndices.push_back(
              b.create<linalg::IndexOp>(loc, resultRank - 1));

          // Extract the value from weight tensor.
          Value extracted =
              b.create<tensor::ExtractOp>(loc, weight, weightIndices);

          b.create<linalg::YieldOp>(loc, extracted);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
// ConcatenateHeads conversion pattern.
// This operation concatenates multiple heads of a multi-head attention tensor.
// Input shape: [batch_size, num_heads, sequence_size, head_size]
// Output shape: [batch_size, sequence_size, num_heads * head_size]
// It is equivalent to: permute([0, 2, 1, 3]) + reshape
class ConcatenateHeadsOpConversionPattern
    : public OpConversionPattern<ttir::ConcatenateHeadsOp> {
public:
  using OpConversionPattern<ttir::ConcatenateHeadsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ConcatenateHeadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    Type elementType = inputType.getElementType();

    // ConcatenateHeads expects exactly 4D input:
    // [batch_size, num_heads, sequence_size, head_size].
    if (inputType.getRank() != 4) {
      return rewriter.notifyMatchFailure(
          op, "ConcatenateHeads requires 4D input tensor.");
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // Input: [batch_size, num_heads, sequence_size, head_size]
    // Step 1: Transpose to [batch_size, sequence_size, num_heads, head_size]
    SmallVector<int32_t> permutation = {0, 2, 1, 3};
    SmallVector<int64_t> transposedShape = {inputShape[0], inputShape[2],
                                            inputShape[1], inputShape[3]};
    auto transposedType = RankedTensorType::get(transposedShape, elementType);

    auto transposeOp = rewriter.create<tosa::TransposeOp>(loc, transposedType,
                                                          input, permutation);

    // Step 2: Reshape to [batch_size, sequence_size, num_heads * head_size]
    ArrayRef<int64_t> outputShape = resultType.getShape();
    SmallVector<int64_t> newShapeValues(outputShape.begin(), outputShape.end());
    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), outputShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShapeValues);
    auto shapeOp = rewriter.create<tosa::ConstShapeOp>(loc, shapeType, attr);

    auto reshapeOp =
        rewriter.create<tosa::ReshapeOp>(loc, resultType, transposeOp, shapeOp);

    rewriter.replaceOp(op, reshapeOp);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Linalg Conversions Patterns
//===----------------------------------------------------------------------===//
namespace {
// Decomposes softmax into elementary operations that can be lowered through
// linalg-to-loops. linalg.softmax cannot be lowered directly because it
// implements AggregatedOpInterface rather than LinalgStructuredInterface.
//
// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
//
// Steps:
// 1. max = ReduceMax(input, dim) - for numerical stability
// 2. shifted = input - max (broadcast)
// 3. exp_vals = exp(shifted)
// 4. sum_exp = ReduceSum(exp_vals, dim)
// 5. result = exp_vals / sum_exp (broadcast)
class SoftmaxOpConversionPattern : public OpConversionPattern<ttir::SoftmaxOp> {
public:
  using OpConversionPattern<ttir::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elementType = inputType.getElementType();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // Normalize dimension to be positive.
    int64_t dim = normalizeDim(op.getDimension(), rank);

    // Create reduced shape (with dim size = 1 for broadcasting).
    SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape[dim] = 1;
    auto reducedType = RankedTensorType::get(reducedShape, elementType);

    // Step 1: Compute max along dimension for numerical stability.
    auto axisAttr = rewriter.getI32IntegerAttr(dim);
    Value maxVal =
        rewriter.create<tosa::ReduceMaxOp>(loc, reducedType, input, axisAttr);

    // Step 2: Subtract max from input (input - max).
    // tosa::SubOp handles broadcasting automatically.
    Value shifted = rewriter.create<tosa::SubOp>(loc, inputType, input, maxVal);

    // Step 3: Compute exp(shifted).
    Value expVals = rewriter.create<tosa::ExpOp>(loc, inputType, shifted);

    // Step 4: Compute sum of exp along dimension.
    Value sumExp =
        rewriter.create<tosa::ReduceSumOp>(loc, reducedType, expVals, axisAttr);

    // Step 5: Divide exp by sum (exp / sum).
    // Use reciprocal and multiply with broadcasting.
    Value reciprocal =
        rewriter.create<tosa::ReciprocalOp>(loc, reducedType, sumExp);

    // tosa::MulOp requires a shift tensor (0 for float ops).
    auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
    auto shiftAttr =
        DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    Value result = rewriter.create<tosa::MulOp>(loc, resultType, expVals,
                                                reciprocal, shift);

    rewriter.replaceOp(op, result);
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
// Conversion pattern for ttir.permute operation
class PermuteOpConversionPattern : public OpConversionPattern<ttir::PermuteOp> {
public:
  using OpConversionPattern<ttir::PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    Value input = adaptor.getInput();
    llvm::ArrayRef<int64_t> permutation = op.getPermutation();

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, input, output.getResult(), permutation);

    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.slice_static operation.
class SliceStaticOpConversionPattern
    : public OpConversionPattern<ttir::SliceStaticOp> {
public:
  using OpConversionPattern<ttir::SliceStaticOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SliceStaticOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op,
                                         "Input must be a ranked tensor type.");
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;

    ArrayAttr begins = op.getBegins();
    ArrayAttr ends = op.getEnds();
    ArrayAttr steps = op.getStep();

    if (!(begins.size() == ends.size() && begins.size() == steps.size())) {
      return rewriter.notifyMatchFailure(op, "Invalid slice attributes.");
    }

    for (unsigned i = 0; i < begins.size(); ++i) {
      const int32_t beginVal = llvm::cast<IntegerAttr>(begins[i]).getInt();
      const int32_t endVal = llvm::cast<IntegerAttr>(ends[i]).getInt();
      const int32_t stepVal = llvm::cast<IntegerAttr>(steps[i]).getInt();

      offsets.push_back(rewriter.getI64IntegerAttr(beginVal));

      // Calculate size: (end - begin + step - 1) / step
      int64_t size = (endVal - beginVal);
      if (stepVal != 0) {
        size = (size + stepVal - 1) / stepVal;
      }
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      strides.push_back(rewriter.getI64IntegerAttr(stepVal));
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // Create the extract_slice operation
    Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), resultType, input, offsets, sizes, strides);

    rewriter.replaceOp(op, extractedSlice);

    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.pad operation.
class PadOpConversionPattern : public OpConversionPattern<ttir::PadOp> {
public:
  using OpConversionPattern<ttir::PadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op,
                                         "Input must be a ranked tensor type.");
    }

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // Get padding attribute: format is [dim0_low, dim0_high, dim1_low,
    // dim1_high, ...]
    ArrayRef<int32_t> paddingArray = op.getPadding();
    int64_t rank = inputType.getRank();
    if (static_cast<int64_t>(paddingArray.size()) != 2 * rank) {
      return rewriter.notifyMatchFailure(op, "Padding size must be 2 * rank.");
    }

    // Extract low and high padding for each dimension.
    SmallVector<OpFoldResult> lowPad, highPad;
    for (int64_t i = 0; i < rank; ++i) {
      lowPad.push_back(rewriter.getIndexAttr(paddingArray[2 * i]));
      highPad.push_back(rewriter.getIndexAttr(paddingArray[2 * i + 1]));
    }

    // Get the padding value and create a constant.
    float padValue = op.getValue().convertToFloat();
    Type elementType = inputType.getElementType();
    Value padConstant;
    if (isa<FloatType>(elementType)) {
      padConstant = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getFloatAttr(elementType, padValue));
    } else {
      padConstant = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(elementType, static_cast<int64_t>(padValue)));
    }

    // Create tensor::PadOp and replace.
    rewriter.replaceOpWithNewOp<tensor::PadOp>(op, resultType, input, lowPad,
                                               highPad, padConstant);
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
    auto value = adaptor.getValue();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    if (!valueType) {
      return rewriter.notifyMatchFailure(
          op, "Value type must be a ranked tensor type.");
    }

    ElementsAttr convertedValue;

    // If types already match (e.g., after HoistCPUOps conversion), use value
    // directly.
    if (valueType == resultType) {
      convertedValue = value;
    } else if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
      // Use getFromRawBuffer to reinterpret the raw data with the new type.
      // This handles signedness conversions (si32/ui32 -> i32) which have the
      // same bit width.
      convertedValue = DenseElementsAttr::getFromRawBuffer(
          resultType, denseAttr.getRawData());
    } else if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
      convertedValue = DenseResourceElementsAttr::get(
          resultType, resourceAttr.getRawHandle());
    } else {
      return rewriter.notifyMatchFailure(
          op, "Expected DenseElementsAttr or DenseResourceElementsAttr");
    }

    auto newConstant = rewriter.create<arith::ConstantOp>(
        op.getLoc(), resultType, convertedValue);

    rewriter.replaceOp(op, newConstant.getResult());
    return success();
  }
};
} // namespace

namespace {
// Template conversion pattern for constant fill operations (zeros, ones).
template <typename TTIROpTy, int64_t FillValue>
class NamedFillOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    DenseElementsAttr fillAttr =
        createDenseElementsAttr(resultType, static_cast<double>(FillValue));
    if (!fillAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for constant fill");
    }

    auto constOp =
        rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, fillAttr);

    rewriter.replaceOp(op, constOp.getResult());
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.full operation.
class FullOpConversionPattern : public OpConversionPattern<ttir::FullOp> {
public:
  using OpConversionPattern<ttir::FullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // Extract numeric value as double.
    Attribute fillValue = adaptor.getFillValue();
    double value;
    if (auto floatAttr = dyn_cast<FloatAttr>(fillValue)) {
      value = floatAttr.getValue().convertToDouble();
    } else if (auto intAttr = dyn_cast<IntegerAttr>(fillValue)) {
      value = static_cast<double>(intAttr.getInt());
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported fill_value type");
    }

    // Create DenseElementsAttr with appropriate element type.
    DenseElementsAttr fillAttr = createDenseElementsAttr(resultType, value);
    if (!fillAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for full fill");
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, fillAttr);
    return success();
  }
};
} // namespace

namespace {
// Conversion pattern for ttir.arange operation.
// Generates a tensor with evenly spaced values within a given interval.
class ArangeOpConversionPattern : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));

    // ArangeForceLastDimensionPattern ensures arange is always 1D.
    if (resultType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Arange must be 1D after decomposition");
    }

    int64_t start = adaptor.getStart();
    int64_t step = adaptor.getStep();
    Type elementType = resultType.getElementType();

    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);

    AffineMap outputMap = rewriter.getDimIdentityMap();
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{}, ValueRange{initTensor},
        SmallVector<AffineMap>{outputMap}, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value idx = b.create<linalg::IndexOp>(loc, 0);

          // Compute: start + idx * step
          Value result;
          if (isa<FloatType>(elementType)) {
            Value idxFloat =
                b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
            Value idxFP = b.create<arith::SIToFPOp>(loc, elementType, idxFloat);
            Value startVal = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elementType, static_cast<double>(start)));
            Value stepVal = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elementType, static_cast<double>(step)));
            Value scaled = b.create<arith::MulFOp>(loc, idxFP, stepVal);
            result = b.create<arith::AddFOp>(loc, startVal, scaled);
          } else {
            Value idxInt = b.create<arith::IndexCastOp>(loc, elementType, idx);
            Value startVal = b.create<arith::ConstantOp>(
                loc, b.getIntegerAttr(elementType, start));
            Value stepVal = b.create<arith::ConstantOp>(
                loc, b.getIntegerAttr(elementType, step));
            Value scaled = b.create<arith::MulIOp>(loc, idxInt, stepVal);
            result = b.create<arith::AddIOp>(loc, startVal, scaled);
          }

          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
// Decomposes LayerNorm into elementary operations that can be lowered through
// linalg-to-loops.
//
// layer_norm(x, weight, bias, epsilon) =
//   ((x - mean(x)) / sqrt(var(x) + epsilon)) * weight + bias
//
// Steps:
// 1. mean = ReduceSum(input, dims) / num_elements
// 2. centered = input - mean (broadcast)
// 3. variance = ReduceSum(centered^2, dims) / num_elements
// 4. inv_std = rsqrt(variance + epsilon)
// 5. normalized = centered * inv_std (broadcast)
// 6. if weight: result = normalized * weight
// 7. if bias: result = result + bias
class LayerNormOpConversionPattern
    : public OpConversionPattern<ttir::LayerNormOp> {
public:
  using OpConversionPattern<ttir::LayerNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elementType = inputType.getElementType();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    // Get normalized_shape to determine which dimensions to reduce over.
    // normalized_shape specifies the shape of the dimensions to normalize,
    // which are always the last N dimensions of the input.
    ArrayRef<int64_t> normalizedShape = op.getNormalizedShape();
    int64_t numNormDims = normalizedShape.size();

    // The reduction dimensions are the last numNormDims dimensions.
    SmallVector<int64_t> reductionDims;
    for (int64_t i = rank - numNormDims; i < rank; ++i) {
      reductionDims.push_back(i);
    }

    // Calculate number of elements being reduced.
    int64_t numElements = 1;
    for (int64_t dim : reductionDims) {
      numElements *= inputType.getShape()[dim];
    }

    // Create reduced shape (with reduced dims = 1 for broadcasting).
    SmallVector<int64_t> reducedShape(inputType.getShape());
    for (int64_t dim : reductionDims) {
      reducedShape[dim] = 1;
    }
    auto reducedType = RankedTensorType::get(reducedShape, elementType);

    // Step 1: Compute mean = sum(input) / num_elements.
    // Use reduction op chain to sum over all reduction dimensions.
    Value sum = createReductionOpChain<tosa::ReduceSumOp>(
        input, reducedType, reductionDims, /*keepDim=*/true, loc, rewriter);

    // Create constant for division by num_elements.
    DenseElementsAttr numElementsAttr =
        createDenseElementsAttr(reducedType, static_cast<double>(numElements));
    if (!numElementsAttr) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type for layer norm");
    }
    Value numElementsConst =
        rewriter.create<tosa::ConstOp>(loc, reducedType, numElementsAttr);
    Value reciprocalN =
        rewriter.create<tosa::ReciprocalOp>(loc, reducedType, numElementsConst);

    // mean = sum * (1/N)
    auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
    auto shiftAttr =
        DenseElementsAttr::get(shiftType, rewriter.getI8IntegerAttr(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    Value mean =
        rewriter.create<tosa::MulOp>(loc, reducedType, sum, reciprocalN, shift);

    // Step 2: centered = input - mean (tosa broadcasts automatically).
    Value centered = rewriter.create<tosa::SubOp>(loc, inputType, input, mean);

    // Step 3: Compute variance = mean(centered^2).
    // First compute centered^2.
    Value centeredSquared =
        rewriter.create<tosa::MulOp>(loc, inputType, centered, centered, shift);

    // Sum of squared differences.
    Value sumSquared = createReductionOpChain<tosa::ReduceSumOp>(
        centeredSquared, reducedType, reductionDims, /*keepDim=*/true, loc,
        rewriter);

    // variance = sumSquared * (1/N)
    Value variance = rewriter.create<tosa::MulOp>(loc, reducedType, sumSquared,
                                                  reciprocalN, shift);

    // Step 4: Add epsilon for numerical stability.
    float epsilon = op.getEpsilon().convertToFloat();
    DenseElementsAttr epsilonAttr =
        createDenseElementsAttr(reducedType, static_cast<double>(epsilon));
    Value epsilonConst =
        rewriter.create<tosa::ConstOp>(loc, reducedType, epsilonAttr);
    Value variancePlusEps =
        rewriter.create<tosa::AddOp>(loc, reducedType, variance, epsilonConst);

    // Step 5: inv_std = rsqrt(variance + epsilon).
    Value invStd =
        rewriter.create<tosa::RsqrtOp>(loc, reducedType, variancePlusEps);

    // Step 6: normalized = centered * inv_std (tosa broadcasts automatically).
    Value normalized =
        rewriter.create<tosa::MulOp>(loc, resultType, centered, invStd, shift);

    // Step 7: Apply weight (gamma) if present.
    // Weight and bias need to be reshaped to match the input rank for TOSA ops.
    // They have shape [normalized_shape], need to prepend 1s to match input
    // rank.
    Value result = normalized;
    if (adaptor.getWeight()) {
      Value reshapedWeight = reshapeByPrependingOnes(
          adaptor.getWeight(), rank, numNormDims, elementType, loc, rewriter);
      result = rewriter.create<tosa::MulOp>(loc, resultType, result,
                                            reshapedWeight, shift);
    }

    // Step 8: Apply bias (beta) if present.
    if (adaptor.getBias()) {
      Value reshapedBias = reshapeByPrependingOnes(
          adaptor.getBias(), rank, numNormDims, elementType, loc, rewriter);
      result =
          rewriter.create<tosa::AddOp>(loc, resultType, result, reshapedBias);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class SqueezeOpConversionPattern : public OpConversionPattern<ttir::SqueezeOp> {
public:
  using OpConversionPattern<ttir::SqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    int64_t dim = normalizeDim(op.getDim(), inputType.getRank());

    auto inputShape = inputType.getShape();
    SmallVector<int64_t> newShape;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (i != dim) {
        newShape.push_back(inputShape[i]);
      }
    }

    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShape);
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(op.getLoc(), resultType,
                                                      input, shapeOp);

    rewriter.replaceOp(op, reshapeOp);

    return success();
  }
};
} // namespace

namespace {
class UnsqueezeOpConversionPattern
    : public OpConversionPattern<ttir::UnsqueezeOp> {
public:
  using OpConversionPattern<ttir::UnsqueezeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    SmallVector<int64_t> newShape(resultType.getShape());

    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShape);
    auto shapeOp =
        rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, resultType, input,
                                                 shapeOp);

    return success();
  }
};
} // namespace

namespace {
class MeshShardOpConversionPattern
    : public OpConversionPattern<ttir::MeshShardOp> {
public:
  using OpConversionPattern<ttir::MeshShardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MeshShardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle identity MeshShard ops - these are no-ops that just pass
    // through the input.
    if (adaptor.getShardType() != ttcore::MeshShardType::Identity) {
      return rewriter.notifyMatchFailure(
          op, "Only identity MeshShard ops are supported.");
    }

    // Identity MeshShard is a no-op, just forward the input.
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};
} // namespace

namespace {
class ClampScalarOpConversionPattern
    : public OpConversionPattern<ttir::ClampScalarOp> {
public:
  using OpConversionPattern<ttir::ClampScalarOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ClampScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto elementType = resultType.getElementType();
    TypedAttr minAttr, maxAttr;

    if (isa<FloatType>(elementType)) {
      minAttr = rewriter.getFloatAttr(
          elementType, ttmlir::utils::attributeToDouble(op.getMin()));
      maxAttr = rewriter.getFloatAttr(
          elementType, ttmlir::utils::attributeToDouble(op.getMax()));
    } else if (isa<IntegerType>(elementType)) {
      minAttr = rewriter.getIntegerAttr(
          elementType,
          static_cast<int64_t>(ttmlir::utils::attributeToDouble(op.getMin())));
      maxAttr = rewriter.getIntegerAttr(
          elementType,
          static_cast<int64_t>(ttmlir::utils::attributeToDouble(op.getMax())));
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type for clamp");
    }

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, resultType, input, minAttr,
                                               maxAttr);
    return success();
  }
};
} // namespace

namespace {
class LinearOpConversionPattern : public OpConversionPattern<ttir::LinearOp> {
public:
  using OpConversionPattern<ttir::LinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();
    Value bias = adaptor.getBias();

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
      if (lhsShape.size() >= 2) {
        SmallVector<int64_t> transposedShape;
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          transposedShape.push_back(lhsShape[i]);
        }
        transposedShape.push_back(lhsShape[lhsShape.size() - 1]);
        transposedShape.push_back(lhsShape[lhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, lhsType.getElementType());

        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < lhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(lhsShape.size() - 2));

        lhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 lhs, permutation);
        lhsType = transposedType;
      }
    }

    if (transposeB) {
      auto rhsShape = rhsType.getShape();
      if (rhsShape.size() >= 2) {
        SmallVector<int64_t> transposedShape;
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          transposedShape.push_back(rhsShape[i]);
        }
        transposedShape.push_back(rhsShape[rhsShape.size() - 1]);
        transposedShape.push_back(rhsShape[rhsShape.size() - 2]);

        auto transposedType =
            RankedTensorType::get(transposedShape, rhsType.getElementType());

        SmallVector<int32_t> permutation;
        for (size_t i = 0; i < rhsShape.size() - 2; ++i) {
          permutation.push_back(static_cast<int32_t>(i));
        }
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 1));
        permutation.push_back(static_cast<int32_t>(rhsShape.size() - 2));

        rhs = rewriter.create<tosa::TransposeOp>(op.getLoc(), transposedType,
                                                 rhs, permutation);
        rhsType = transposedType;
      }
    }

    // Ensure both tensors are 3D for tosa.matmul
    unsigned lhsRank = lhsType.getRank();
    unsigned rhsRank = rhsType.getRank();

    Value lhs3D = lhs;
    Value rhs3D = rhs;
    RankedTensorType lhs3DType = lhsType;
    RankedTensorType rhs3DType = rhsType;

    // If LHS is 2D, reshape to 3D with batch size 1
    if (lhsRank == 2) {
      SmallVector<int64_t> newShape = {1, lhsType.getDimSize(0),
                                       lhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    } else if (lhsRank > 3) {
      // Check for dynamic dimensions in batch dimensions.
      for (uint32_t i = 0; i < lhsRank - 2; ++i) {
        if (lhsType.isDynamicDim(i)) {
          return rewriter.notifyMatchFailure(
              op, "Dynamic batch dimensions not supported in LinearOp");
        }
      }

      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < lhsRank - 2; ++i) {
        collapsedBatchSize *= lhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       lhsType.getShape()[lhsRank - 2],
                                       lhsType.getShape()[lhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      lhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, lhs,
                                               shapeOp.getResult());
      lhs3DType = newType;
    }

    // If RHS is 2D, reshape to 3D with batch size 1
    if (rhsRank == 2) {
      SmallVector<int64_t> newShape = {1, rhsType.getDimSize(0),
                                       rhsType.getDimSize(1)};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    } else if (rhsRank > 3) {
      // Check for dynamic dimensions in batch dimensions.
      for (uint32_t i = 0; i < rhsRank - 2; ++i) {
        if (rhsType.isDynamicDim(i)) {
          return rewriter.notifyMatchFailure(
              op, "Dynamic batch dimensions not supported in LinearOp");
        }
      }

      int64_t collapsedBatchSize = 1;
      for (uint32_t i = 0; i < rhsRank - 2; ++i) {
        collapsedBatchSize *= rhsType.getShape()[i];
      }

      SmallVector<int64_t> newShape = {collapsedBatchSize,
                                       rhsType.getShape()[rhsRank - 2],
                                       rhsType.getShape()[rhsRank - 1]};
      auto newType = RankedTensorType::get(newShape, rhsType.getElementType());

      auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
      auto attr = rewriter.getIndexTensorAttr(newShape);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      rhs3D = rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, rhs,
                                               shapeOp.getResult());
      rhs3DType = newType;
    }

    // Check if we need to broadcast batch dimensions
    if (lhs3DType.getShape()[0] != rhs3DType.getShape()[0]) {
      if (lhs3DType.getShape()[0] == 1 && rhs3DType.getShape()[0] > 1) {
        SmallVector<int64_t> multiples = {rhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({rhs3DType.getShape()[0],
                                              lhs3DType.getShape()[1],
                                              lhs3DType.getShape()[2]},
                                             lhs3DType.getElementType());

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

        lhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, lhs3D,
                                              multiplesOp);
        lhs3DType = cast<RankedTensorType>(lhs3D.getType());
      } else if (rhs3DType.getShape()[0] == 1 && lhs3DType.getShape()[0] > 1) {
        SmallVector<int64_t> multiples = {lhs3DType.getShape()[0], 1, 1};
        auto newType = RankedTensorType::get({lhs3DType.getShape()[0],
                                              rhs3DType.getShape()[1],
                                              rhs3DType.getShape()[2]},
                                             rhs3DType.getElementType());

        auto shapeType = tosa::shapeType::get(rewriter.getContext(), 3);
        auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
        auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, multiplesAttr);

        rhs3D = rewriter.create<tosa::TileOp>(op.getLoc(), newType, rhs3D,
                                              multiplesOp);
        rhs3DType = cast<RankedTensorType>(rhs3D.getType());
      }
    }

    // Perform matrix multiplication
    auto matmulResultType =
        RankedTensorType::get({lhs3DType.getShape()[0], lhs3DType.getShape()[1],
                               rhs3DType.getShape()[2]},
                              resultType.getElementType());

    Value matmulResult = rewriter.create<tosa::MatMulOp>(
        op.getLoc(), matmulResultType, lhs3D, rhs3D);

    // Reshape result back to original rank if needed
    if (resultType.getRank() != matmulResultType.getRank()) {
      auto shapeType =
          tosa::shapeType::get(rewriter.getContext(), resultType.getRank());
      SmallVector<int64_t> shapeValues;
      for (auto dim : resultType.getShape()) {
        shapeValues.push_back(dim);
      }
      auto attr = rewriter.getIndexTensorAttr(shapeValues);
      auto shapeOp =
          rewriter.create<tosa::ConstShapeOp>(op.getLoc(), shapeType, attr);

      matmulResult = rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), resultType, matmulResult, shapeOp.getResult());
    }

    // If bias is provided, add it to the result
    if (bias) {
      auto biasType = cast<RankedTensorType>(bias.getType());

      // Reshape bias to match result rank if needed by prepending 1s
      if (biasType.getRank() < resultType.getRank()) {
        SmallVector<int64_t> newBiasShape;
        int64_t rankDiff = resultType.getRank() - biasType.getRank();
        for (int64_t i = 0; i < rankDiff; ++i) {
          newBiasShape.push_back(1);
        }
        for (auto dim : biasType.getShape()) {
          newBiasShape.push_back(dim);
        }

        auto reshapedBiasType =
            RankedTensorType::get(newBiasShape, biasType.getElementType());
        auto shapeType =
            tosa::shapeType::get(rewriter.getContext(), newBiasShape.size());
        auto shapeAttr = rewriter.getIndexTensorAttr(newBiasShape);
        auto shapeOp = rewriter.create<tosa::ConstShapeOp>(
            op.getLoc(), shapeType, shapeAttr);
        bias = rewriter.create<tosa::ReshapeOp>(op.getLoc(), reshapedBiasType,
                                                bias, shapeOp.getResult());
      }

      matmulResult = rewriter.create<tosa::AddOp>(op.getLoc(), resultType,
                                                  matmulResult, bias);
    }

    rewriter.replaceOp(op, matmulResult);
    return success();
  }
};
} // namespace

namespace {
class RepeatOpConversionPattern : public OpConversionPattern<ttir::RepeatOp> {
public:
  using OpConversionPattern<ttir::RepeatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "Result type must be a ranked tensor type.");
    }

    auto repeatDimensions = op.getRepeatDimensions();
    SmallVector<int64_t> multiples(repeatDimensions.begin(),
                                   repeatDimensions.end());

    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), multiples.size());
    auto multiplesAttr = rewriter.getIndexTensorAttr(multiples);
    auto multiplesOp = rewriter.create<tosa::ConstShapeOp>(
        op.getLoc(), shapeType, multiplesAttr);

    rewriter.replaceOpWithNewOp<tosa::TileOp>(op, resultType,
                                              adaptor.getInput(), multiplesOp);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  populateTTIRToLinalgEltwiseUnaryPatterns(ctx, patterns, typeConverter);
  populateTTIRToLinalgEltwiseBinaryPatterns(ctx, patterns, typeConverter);
  populateTTIRToLinalgPoolingPatterns(ctx, patterns, typeConverter);
  populateTTIRToLinalgReductionPatterns(ctx, patterns, typeConverter);

  patterns
      .add<SoftmaxOpConversionPattern, EmptyOpConversionPattern,
           PermuteOpConversionPattern, SliceStaticOpConversionPattern,
           PadOpConversionPattern, ConstantOpConversionPattern,
           NamedFillOpConversionPattern<ttir::ZerosOp, 0>,
           NamedFillOpConversionPattern<ttir::OnesOp, 1>,
           FullOpConversionPattern, ArangeOpConversionPattern,
           MeshShardOpConversionPattern, ConcatenateHeadsOpConversionPattern>(
          typeConverter, ctx);
}

void populateTTIRToTosaPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  populateTTIRToTosaEltwiseUnaryPatterns(ctx, patterns, typeConverter);
  populateTTIRToTosaEltwiseBinaryPatterns(ctx, patterns, typeConverter);
  populateTTIRToTosaPoolingPatterns(ctx, patterns, typeConverter);
  populateTTIRToTosaReductionPatterns(ctx, patterns, typeConverter);

  patterns.add<BroadcastOpConversionPattern, MatmulOpConversionPattern,
               LinearOpConversionPattern, ClampScalarOpConversionPattern,
               EmbeddingOpConversionPattern, LayerNormOpConversionPattern,
               SqueezeOpConversionPattern, UnsqueezeOpConversionPattern,
               Conv2dOpConversionPattern>(typeConverter, ctx);

  // Special operations
  patterns.add<WhereOpConversionPattern, ReshapeOpConversionPattern,
               TransposeOpConversionPattern, ConcatOpConversionPattern,
               RepeatOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt::ttir_to_linalg
