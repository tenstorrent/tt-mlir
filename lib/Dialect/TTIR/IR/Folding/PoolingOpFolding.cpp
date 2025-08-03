// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/FoldingUtils.h"

namespace mlir::tt::ttir {
using namespace folding_utils;

// This struct represents a window, which is a slice of a tensor with starts =
// starts, ends = ends, and steps = dilations.
struct Window {
  SmallVector<int64_t> starts;
  SmallVector<int64_t> ends;
  SmallVector<int64_t> dilations;

  Window(ArrayRef<int64_t> starts, ArrayRef<int64_t> ends,
         ArrayRef<int64_t> dilations)
      : starts(starts), ends(ends), dilations(dilations) {
    assert(starts.size() == ends.size() &&
           "starts and ends must have the same size");
    assert(starts.size() == dilations.size() &&
           "starts and dilations must have the same size");
    for (int64_t i = 0; i < static_cast<int64_t>(starts.size()); i++) {
      assert(starts[i] >= 0 && starts[i] < ends[i] &&
             "starts must be within ends");
    }
    for (int64_t i = 0; i < static_cast<int64_t>(dilations.size()); i++) {
      assert(dilations[i] > 0 && "dilations must be positive");
    }
  }

  // This will generate the next window by traversing in row-major order.
  std::optional<Window> next(ArrayRef<int64_t> shape,
                             ArrayRef<int64_t> stride) const {
    assert(shape.size() == starts.size() &&
           "shape must have the same size as starts");
    assert(shape.size() == ends.size() &&
           "shape must have the same size as ends");
    assert(shape.size() == dilations.size() &&
           "shape must have the same size as steps");
    assert(stride.size() == shape.size() &&
           "stride must have the same size as shape");

    SmallVector<int64_t> nextStarts(starts);
    SmallVector<int64_t> nextEnds(ends);

    int64_t currentDim = shape.size() - 1;
    while (currentDim >= 0) {
      if (nextEnds[currentDim] + stride[currentDim] <= shape[currentDim]) {
        nextStarts[currentDim] += stride[currentDim];
        nextEnds[currentDim] += stride[currentDim];
        break;
      }
      nextEnds[currentDim] -= nextStarts[currentDim];
      nextStarts[currentDim] = 0;
      currentDim--;
    }

    if (currentDim < 0) {
      return std::nullopt;
    }

    return std::make_optional(Window(nextStarts, nextEnds, dilations));
  }
};

template <typename NumericType>
Tensor<NumericType> calculatePooling(PoolingOp op,
                                     const Tensor<NumericType> &inputTensor) {
  ArrayRef<int64_t> outputShape =
      cast<RankedTensorType>(op.getResult(0).getType()).getShape();

  // Depending on the numeric type (APInt or APFloat), we need to initialize the
  // zero value differently.
  NumericType zero = [&]() -> NumericType {
    if constexpr (std::is_same_v<NumericType, llvm::APFloat>) {
      return llvm::APFloat::getZero(inputTensor.getFloatSemantics());
    } else if constexpr (std::is_same_v<NumericType, llvm::APInt>) {
      return llvm::APInt::getZero(inputTensor.getIntBitWidth());
    } else {
      llvm_unreachable("Unsupported numeric type");
    }
  }();

  // Create an empty tensor for the output
  Tensor<NumericType> outputTensor =
      Tensor<NumericType>::getEmptyTensor(outputShape, zero);

  // We will keep track of the current output index using a flat index since we
  // are traversing the input tensor in row-major order.
  int64_t currentOutputIndex = 0;

  // Create the first window, starting at the least significant location of
  // the input tensor. For a 4D tensor, the first window has
  //     starts = [0, 0, 0, 0]
  //     ends = [window_dimensions[0], window_dimensions[1],
  //     window_dimensions[2], window_dimensions[3]] dilations =
  //     [window_dilations[0], window_dilations[1], window_dilations[2],
  //     window_dilations[3]]
  std::optional<Window> window = std::make_optional(
      Window(SmallVector<int64_t>(op.getWindowDimensions().size(), 0),
             op.getWindowDimensions(), op.getWindowDilations()));
  do {
    Tensor<NumericType> windowSlice =
        inputTensor.slice(window->starts, window->ends, window->dilations);
    NumericType result = [&]() -> NumericType {
      if (op.getPoolingMethod() == PoolingMethod::Max) {
        return windowSlice.max();
      }
      if (op.getPoolingMethod() == PoolingMethod::Average) {
        return windowSlice.mean();
      }
      if (op.getPoolingMethod() == PoolingMethod::Sum) {
        return windowSlice.sum();
      }
      llvm_unreachable("Unsupported pooling method");
    }();

    outputTensor[currentOutputIndex] = result;
    currentOutputIndex++; // On the very last iteration, this will move
                          // currentOutputIndex out of bounds, thus being equal
                          // to the output tensor volume

    window = window.value().next(inputTensor.getShape(), op.getWindowStrides());
  } while (window.has_value());

  assert(currentOutputIndex == outputTensor.getVolume() &&
         "The output tensor was not filled.");
  return outputTensor;
}

::mlir::LogicalResult
mlir::tt::ttir::PoolingOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {

  // Cannot fold if there are dilations in the base as this is not implemented.
  if (std::any_of(getBaseDilations().begin(), getBaseDilations().end(),
                  [](int64_t dilation) { return dilation != 1; })) {
    return mlir::failure();
  }

  // Cannot fold if there is padding as this is not implemented.
  if (std::any_of(getPadding().begin(), getPadding().end(),
                  [](int64_t padding) { return padding != 0; })) {
    return mlir::failure();
  }

  // Cannot fold if there is more than one input as this is not implemented.
  if (adaptor.getInputs().size() > 1) {
    return mlir::failure();
  }

  auto input = adaptor.getInputs()[0];
  if (!input) {
    return mlir::failure();
  }
  if (!isa<ElementsAttr>(input)) {
    return mlir::failure();
  }

  RankedTensorType resultType = cast<RankedTensorType>(getResult(0).getType());

  ElementsAttr elements = cast<ElementsAttr>(input);
  auto inputShape = elements.getShapedType().getShape();

  if (isa<FloatType>(elements.getElementType())) {
    Tensor<llvm::APFloat> inputTensor(elements.getValues<llvm::APFloat>(),
                                      inputShape);
    Tensor<llvm::APFloat> outputTensor = calculatePooling(*this, inputTensor);
    results.push_back(
        outputTensor.getAsDenseElementsAttr(resultType.getElementType()));
  } else if (isa<IntegerType>(elements.getElementType())) {
    Tensor<llvm::APInt> inputTensor(elements.getValues<llvm::APInt>(),
                                    inputShape);
    Tensor<llvm::APInt> outputTensor = calculatePooling(*this, inputTensor);
    results.push_back(
        outputTensor.getAsDenseElementsAttr(resultType.getElementType()));
  } else {
    llvm_unreachable("Unsupported element type");
  }
  return mlir::success();
}

} // namespace mlir::tt::ttir
