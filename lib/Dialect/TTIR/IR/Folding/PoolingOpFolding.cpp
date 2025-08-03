// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/FoldingUtils.h"

namespace mlir::tt::ttir {
using namespace folding_utils;

// This function will generate a window of indices within a given shape,
// starting at a given least significant corner.
SmallVector<SmallVector<int64_t>>
getWindow(ArrayRef<int64_t> leastSignificantCorner,
          ArrayRef<int64_t> windowDimensions) {
  SmallVector<SmallVector<int64_t>> window;
  int64_t windowVolume =
      std::accumulate(windowDimensions.begin(), windowDimensions.end(), 1,
                      std::multiplies<int64_t>());

  SmallVector<SmallVector<int64_t>> offsets;

  // We iterate over each flat index within the window and generate an offset
  // for each index.
  for (int64_t windowIndex = 0; windowIndex < windowVolume; windowIndex++) {
    auto offset = getShapeIndexFromFlatIndex(windowIndex, windowDimensions);
    offsets.push_back(offset);
  }

  // We create a window by adding each offset to the least significant corner.
  for (auto offset : offsets) {
    SmallVector<int64_t> index(leastSignificantCorner);
    for (int64_t i = 0; i < static_cast<int64_t>(offset.size()); i++) {
      index[i] += offset[i];
    }
    window.push_back(index);
  }

  return window;
}

inline bool indexWithinShape(ArrayRef<int64_t> index, ArrayRef<int64_t> shape) {
  for (size_t i = 0; i < index.size(); i++) {
    if (index[i] < 0 || index[i] >= shape[i]) {
      return false;
    }
  }
  return true;
}

inline bool windowWithinShape(SmallVector<SmallVector<int64_t>> window,
                              ArrayRef<int64_t> shape) {
  for (auto index : window) {
    if (!indexWithinShape(index, shape)) {
      return false;
    }
  }
  return true;
}

// This function will generate the next index within a given shape, according to
// a given movement stride (not to be confused with the strides of a shape).
// This function will return the next index in row-major order, or nullopt if
// there is no next index, implying that the current index is the last index in
// the shape.
std::optional<SmallVector<int64_t>> getNextIndex(SmallVector<int64_t> index,
                                                 ArrayRef<int64_t> shape,
                                                 ArrayRef<int64_t> stride) {
  SmallVector<int64_t> nextIndex(index);
  int64_t currentDim = shape.size() - 1;

  // Starting at the innermost dimension, we will increment the index by the
  // stride to produce a new index. If this is out of bounds, we will reset the
  // current dimension to 0, and try to increment the next spatial dimension.
  while (currentDim >= 0) {

    if (nextIndex[currentDim] + stride[currentDim] < shape[currentDim]) {
      nextIndex[currentDim] += stride[currentDim];
      return std::make_optional(nextIndex);
    }
    // If the new index is out of bounds, we will reset the current dimension to
    // 0, and try to increment the next spatial dimension.
    nextIndex[currentDim] = 0;
    currentDim--;
  }
  // If the current dimension we are trying to increment is < 0, that means we
  // have exhausted all dimensions, and there is no next index.
  return std::nullopt;
}

// This function will generate the next window to preform reduction over for a
// given input tensor. The next window is chosen via row-major traversal of the
// input tensor.
template <typename NumericType>
std::optional<SmallVector<SmallVector<int64_t>>>
getNextWindow(SmallVector<SmallVector<int64_t>> window,
              ArrayRef<int64_t> windowStrides,
              ArrayRef<int64_t> windowDimensions,
              const Tensor<NumericType> &inputTensor) {
  // Get the least significant corner of the current window.
  // The least significant corner is the one which has the lowest flat index
  // within the window.
  SmallVector<int64_t> currentleastSignificantCorner;
  int64_t minFlatIndex = std::numeric_limits<int64_t>::max();
  for (auto index : window) {
    int64_t flatIndex = getFlatIndexFromShape(index, windowDimensions);
    if (flatIndex < minFlatIndex) {
      minFlatIndex = flatIndex;
      currentleastSignificantCorner = index;
    }
  }

  // Retrieve the next index within the input shape after the current least
  // significant corner, according to the window strides.
  auto nextLeastSignificantCorner = getNextIndex(
      currentleastSignificantCorner, inputTensor.getShape(), windowStrides);
  // If the next index is out of bounds, there is no next window. Return
  // nullopt.
  if (!nextLeastSignificantCorner.has_value()) {
    return std::nullopt;
  }

  // Generate the next window from the next least significant corner.
  SmallVector<SmallVector<int64_t>> nextWindow =
      getWindow(nextLeastSignificantCorner.value(), windowDimensions);

  // It is possible that one or more of the indices in the newly generated
  // window are out of bounds, even if the next least significant corner is
  // within bounds. This indicates that the next legal window begins in a
  // different location within a higher spatial dimension. We naively increase
  // the least significant corner until the window we generate is within bounds.
  while (!windowWithinShape(nextWindow, inputTensor.getShape())) {

    // If at any point during our search we cannot generate a legal least
    // significant corner, it means that there is no next window. Return
    // nullopt.
    auto currentleastSignificantCorner_ = getNextIndex(
        currentleastSignificantCorner, inputTensor.getShape(), windowStrides);
    if (!currentleastSignificantCorner_.has_value()) {
      return std::nullopt;
    }
    currentleastSignificantCorner = currentleastSignificantCorner_.value();
    nextWindow = getWindow(currentleastSignificantCorner, windowDimensions);
  }

  return std::make_optional(nextWindow);
}

llvm::APFloat
applyReductionWindow(PoolingOp op, const Tensor<llvm::APFloat> &inputTensor,
                     const SmallVector<SmallVector<int64_t>> &window) {
  llvm::APFloat result(
      op.getPoolingMethod() == PoolingMethod::Max
          ? llvm::APFloat::getInf(inputTensor.getFloatSemantics(), true)
          : llvm::APFloat::getZero(inputTensor.getFloatSemantics()));
  for (auto index : window) {
    if (op.getPoolingMethod() == PoolingMethod::Max) {
      llvm::APFloat element = inputTensor[index];
      if (element > result) {
        result = element;
      }
    } else if (op.getPoolingMethod() == PoolingMethod::Average ||
               op.getPoolingMethod() == PoolingMethod::Sum) {
      result = result + inputTensor[index];
    } else {
      llvm_unreachable("Unsupported pooling method");
    }
  }

  if (op.getPoolingMethod() == PoolingMethod::Average) {
    float windowVolume = std::accumulate(op.getWindowDimensions().begin(),
                                         op.getWindowDimensions().end(), 1,
                                         std::multiplies<int64_t>());
    llvm::APFloat windowVolumeAPFloat(windowVolume);
    bool losesInfo = false;
    windowVolumeAPFloat.convert(result.getSemantics(),
                                llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    result = result / windowVolumeAPFloat;
  }
  return result;
}

llvm::APInt
applyReductionWindow(PoolingOp op, const Tensor<llvm::APInt> &inputTensor,
                     const SmallVector<SmallVector<int64_t>> &window) {
  llvm::APInt result(
      op.getPoolingMethod() == PoolingMethod::Max
          ? llvm::APInt::getSignedMinValue(inputTensor.getIntBitWidth())
          : llvm::APInt::getZero(inputTensor.getIntBitWidth()));
  for (auto index : window) {
    if (op.getPoolingMethod() == PoolingMethod::Max) {
      llvm::APInt element = inputTensor[index];
      if (element.sgt(result)) {
        result = element;
      }
    } else if (op.getPoolingMethod() == PoolingMethod::Average ||
               op.getPoolingMethod() == PoolingMethod::Sum) {
      result += inputTensor[index];
    } else {
      llvm_unreachable("Unsupported pooling method");
    }
  }

  if (op.getPoolingMethod() == PoolingMethod::Average) {
    llvm::APInt windowVolume(result.getBitWidth(),
                             std::accumulate(op.getWindowDimensions().begin(),
                                             op.getWindowDimensions().end(), 1,
                                             std::multiplies<int64_t>()));
    result = result.sdiv(windowVolume);
  }
  return result;
}

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

  // Begin the sliding window in the least significant location of the input
  // tensor. i.e for a 4D tensor, the least significant corner is located at (0,
  // 0, 0, 0).
  SmallVector<int64_t> firstWindowCorner(inputTensor.getRank(), 0);

  // We will keep track of the current output index using a flat index since we
  // are traversing the input tensor in row-major order.
  int64_t currentOutputIndex = 0;

  // Generate the first window, starting at the least significant location of
  // the input tensor.
  auto window = std::make_optional(
      getWindow(firstWindowCorner, op.getWindowDimensions()));
  do {
    NumericType result = applyReductionWindow(op, inputTensor, window.value());

    outputTensor[currentOutputIndex] = result;
    currentOutputIndex++; // On the very last iteration, this will move
                          // currentOutputIndex out of bounds, thus being equal
                          // to the output tensor volume

    window = getNextWindow(window.value(), op.getWindowStrides(),
                           op.getWindowDimensions(), inputTensor);
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

  // Cannot fold if there are dilations in the window as this is not
  // implemented.
  if (std::any_of(getWindowDilations().begin(), getWindowDilations().end(),
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
