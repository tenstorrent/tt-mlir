// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_FOLDINGUTILS_H
#define TTMLIR_DIALECT_TTIR_UTILS_FOLDINGUTILS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir::folding_utils {

// This function will calculate the strides for a given shape. That is, for each
// dimension it will calculate the number of elements along the flat data buffer
// you must traverse to reach the next element in that dimension.
inline SmallVector<int64_t> calculateStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size());
  strides.back() = 1;

  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

// Calculate the flat index of a given shape index given the strides of a shape.
inline int64_t getFlatIndexFromStride(ArrayRef<int64_t> index,
                                      ArrayRef<int64_t> stride) {
  assert(index.size() == stride.size() &&
         "index and stride must have the same size");
  int64_t flatIndex = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(index.size()); i++) {
    flatIndex += index[i] * stride[i];
  }
  return flatIndex;
}

// Calculate the flat index within a given shape given the shape index.
inline int64_t getFlatIndexFromShape(ArrayRef<int64_t> index,
                                     ArrayRef<int64_t> shape) {
  assert(index.size() == shape.size() &&
         "index and shape must have the same size");
  return getFlatIndexFromStride(index, calculateStrides(shape));
}

// Calculate the shape index of a given flat index and shape.
inline SmallVector<int64_t>
getShapeIndexFromFlatIndex(int64_t flatIndex, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> index(shape.size());
  for (int64_t i = shape.size() - 1; i >= 0; i--) {
    index[i] = (flatIndex % shape[i]);
    flatIndex /= shape[i];
  }
  return index;
}

// Basic tensor class to use for folding. Holds the data as a std::vector of
// NumericType which must be eithet APInt or APFloat.
template <typename NumericType>
struct Tensor {
  static_assert(std::is_same_v<NumericType, llvm::APInt> ||
                    std::is_same_v<NumericType, llvm::APFloat>,
                "NumericType must be either APInt or APFloat");

  template <typename Iterable>
  Tensor(const Iterable &data, ArrayRef<int64_t> shape)
      : shape(shape), strides(calculateStrides(shape)) {
    volume = std::accumulate(shape.begin(), shape.end(), 1,
                             std::multiplies<int64_t>());
    this->data.reserve(volume);
    int64_t index = 0;
    for (const NumericType &element : data) {
      this->data.push_back(element);
      index++;
    }
    isSplat = this->data.size() == 1;
    assert((index == volume || isSplat) &&
           "size of data does not match volume of shape, and the data is not "
           "splat.");
  }

  // Creates a tensor with shape <shape> and all elements set to <zeroValue>.
  static Tensor<NumericType> getEmptyTensor(ArrayRef<int64_t> shape,
                                            NumericType zeroValue) {
    auto data =
        std::vector<NumericType>(std::accumulate(shape.begin(), shape.end(), 1,
                                                 std::multiplies<int64_t>()),
                                 zeroValue);

    return Tensor<NumericType>(data, shape);
  }

  int64_t getRank() const { return shape.size(); }

  const SmallVector<int64_t> &getShape() const { return shape; }

  int64_t getVolume() const { return volume; }

  // Returnst a mutable reference to the element at a given shape index
  NumericType &operator[](ArrayRef<int64_t> index) {
    if (isSplat) {
      return data[0];
    }
    return operator[](getFlatIndexFromStride(index, strides));
  }

  // Returns a const reference to the element at a given shape index
  const NumericType &operator[](ArrayRef<int64_t> index) const {
    if (isSplat) {
      return data[0];
    }
    return operator[](getFlatIndexFromStride(index, strides));
  }

  NumericType &operator[](int64_t flatIndex) {
    assert(flatIndex < volume && "Index out of bounds");
    if (isSplat) {
      return data[0];
    }
    return data[flatIndex];
  }

  const NumericType &operator[](int64_t flatIndex) const {
    assert(flatIndex < volume && "Index out of bounds");
    if (isSplat) {
      return data[0];
    }
    return data[flatIndex];
  }

  // Returns the float semantics of the tensor if the tensor is an APFloat
  // tensor.
  const llvm::fltSemantics &getFloatSemantics() const {
    static_assert(std::is_same_v<NumericType, llvm::APFloat> &&
                  "getFloatSemantics is only valid for APFloat tensors");
    return cast<llvm::APFloat>(data[0]).getSemantics();
  }

  // Returns the bit width of the tensor if the tensor is an APInt tensor.
  unsigned getIntBitWidth() const {
    static_assert(std::is_same_v<NumericType, llvm::APInt> &&
                  "getIntBitWidth is only valid for APInt tensors");
    return cast<llvm::APInt>(data[0]).getBitWidth();
  }

  // Returns true if the tensor is an APInt tensor and the element is signed.
  bool isSigned() const {
    static_assert(std::is_same_v<NumericType, llvm::APInt> &&
                  "isSigned is only valid for APInt tensors");
    return cast<llvm::APInt>(data[0]).isSigned();
  }

  // Returns a DenseElementsAttr with the same shape and data as the tensor
  // with element type ElementType.
  const DenseElementsAttr getAsDenseElementsAttr(Type elementType) const {
    return DenseElementsAttr::get(RankedTensorType::get(shape, elementType),
                                  data);
  }

private:
  std::vector<NumericType> data;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  int64_t volume;
  bool isSplat;
};

} // namespace mlir::tt::ttir::folding_utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_FOLDINGUTILS_H
