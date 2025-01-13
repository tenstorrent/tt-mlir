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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::tt;

namespace {

using TensorRanks = SmallVector<int64_t, 2>;

// Helper func to check which dims need to be broadcast and which need to be
// collapsed.  Assumes that inputShape is broadcast-able to targetShape.
static void getDimsToBroadcastAndCollapse(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> targetShape,
    TensorRanks &broadcastDims, SmallVector<TensorRanks, 2> &reassocIndices) {

  broadcastDims.clear();
  reassocIndices.clear();

  // Identify what needs broadcasting, aligning from right
  int targetIdx = targetShape.size() - 1;
  int inputIdx = inputShape.size() - 1;

  while (targetIdx >= 0) {
    if (inputIdx >= 0) {
      // This should be impossible since we verify input while computing
      // targetShape.
      assert(
          (inputShape[inputIdx] == targetShape[targetIdx] ||
           inputShape[inputIdx] == 1) &&
          "attempting to broadcast shape which does not broadcast to target!");
      if (inputShape[inputIdx] == 1 && targetShape[targetIdx] != 1) {
        broadcastDims.push_back(inputIdx);
      }
      inputIdx--;
    } else {
      // Input exhausted, we need to broadcast remaining dimensions.
      broadcastDims.push_back(targetIdx);
    }
    targetIdx--;
  }

  // Group non-broadcast dimensions together for collapse.
  TensorRanks currentGroup;
  size_t nextBroadcastDimIdx = 0;
  bool fullDimInGroup = false;
  for (size_t i = 0; i < inputShape.size(); ++i) {
    if (nextBroadcastDimIdx < broadcastDims.size() &&
        static_cast<int64_t>(i) == broadcastDims[nextBroadcastDimIdx]) {
      nextBroadcastDimIdx++;
    } else {
      if (fullDimInGroup) {
        // Non-broadcast dimensions end the current group.
        reassocIndices.push_back(currentGroup);
        currentGroup.clear();
      }
      fullDimInGroup = true;
    }
    currentGroup.push_back(i);
  }

  // Add any remaining dimensions in the current group.
  if (!currentGroup.empty()) {
    reassocIndices.push_back(currentGroup);
  }
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
    SmallVector<Value, 3> inputs = adaptor.getInputs();
    assert(inputs.size() == 2 &&
           "binary element-wise operations must have 2 inputs!");
    ArrayRef<int64_t> input0Shape =
        dyn_cast<RankedTensorType>(inputs[0].getType()).getShape();
    ArrayRef<int64_t> input1Shape =
        dyn_cast<RankedTensorType>(inputs[1].getType()).getShape();

    SmallVector<int64_t, 4> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(input0Shape, input1Shape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(
          op, "Operands are not broadcastable--this should be impossible!");
    }

    // Replace any inputs which aren't in target shape with broadcast results
    // which are.
    SmallVector<Value, 4> broadcastedInputs;
    for (Value input : inputs) {
      auto inputRankedTensorType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputRankedTensorType) {
        continue;
      }
      Type elementType = inputRankedTensorType.getElementType();

      // Insert and use a broadcast op if input does not perfectly match target
      // shape.
      TensorRanks broadCastDims;
      SmallVector<TensorRanks, 2> reassocIndexes;
      getDimsToBroadcastAndCollapse(inputRankedTensorType.getShape(),
                                    broadcastedShape, broadCastDims,
                                    reassocIndexes);
      if (!broadCastDims.empty()) {
        Value broadcastInput = input;
        // The broadcast op requires we actually collapse any dimensions with
        // size 1 we want to broadcast along.
        if (reassocIndexes.size() != inputRankedTensorType.getShape().size()) {
          auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
              loc, input, reassocIndexes);
          broadcastInput = collapseOp.getResult();
        }
        auto initTensor = rewriter.create<tensor::EmptyOp>(
            loc, broadcastedShape, elementType);
        auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
            loc, broadcastInput, initTensor.getResult(), broadCastDims);
        for (auto result : broadcastOp.getResults()) {
          broadcastedInputs.push_back(result);
        }
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
