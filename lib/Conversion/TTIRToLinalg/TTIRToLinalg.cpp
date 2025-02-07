// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

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
        auto initTensor = rewriter.create<tensor::EmptyOp>(
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
    rewriter.replaceOpWithNewOp<LinalgOpTy>(op, resultTypes, broadcastedInputs,
                                            adaptor.getOutputs());
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
      ElementwiseBinaryOpConversionPattern<ttir::SubtractOp, linalg::SubOp>>(
      typeConverter, ctx);
}

} // namespace mlir::tt
