// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_FOLDINGUTILS_H
#define TTMLIR_DIALECT_TTIR_UTILS_FOLDINGUTILS_H

#include "ttmlir/Utils.h"

#include <numeric>

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

// Basic tensor class to use for folding. Holds the data as a std::vector of
// NumericType which must be eithet APInt or APFloat.
template <typename NumericType>
struct Tensor {
  static_assert(std::is_same_v<NumericType, llvm::APInt> ||
                    std::is_same_v<NumericType, llvm::APFloat>,
                "NumericType must be either APInt or APFloat");

  template <typename Iterable>
  Tensor(const Iterable &data, ArrayRef<int64_t> shape)
      : shape(shape), strides(calculateStrides(shape)),
        volume(std::accumulate(shape.begin(), shape.end(), 1,
                               std::multiplies<int64_t>())) {
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

  // Returns a compatible APInt or APFloat which can be operated with the tensor
  // data
  template <typename NativeNumericType>
  NumericType getTensorCompatibleAPValue(NativeNumericType value) const {
    if constexpr (std::is_same_v<float, NativeNumericType>) {
      static_assert(
          std::is_same_v<NumericType, llvm::APFloat>,
          "NativeNumericType is float, but Tensor NumericType is not APFloat");
      bool losesInfo = false;
      llvm::APFloat apFloat(value);
      apFloat.convert(getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
                      &losesInfo);
      return apFloat;
    } else if constexpr (std::is_same_v<int64_t, NativeNumericType>) {
      static_assert(
          std::is_same_v<NumericType, llvm::APInt>,
          "NativeNumericType is int64_t, but Tensor NumericType is not APInt");
      return llvm::APInt(getIntBitWidth(), value);
    } else {
      llvm_unreachable("NativeNumericType must be either float or int64_t");
    }
  }

  int64_t getRank() const { return shape.size(); }

  const SmallVector<int64_t> &getShape() const { return shape; }

  int64_t getVolume() const { return volume; }

  // Returnst a mutable reference to the element at a given shape index
  NumericType &operator[](ArrayRef<int64_t> index) {
    return operator[](getFlatIndexFromStride(index, strides));
  }

  // Returns a const reference to the element at a given shape index
  const NumericType &operator[](ArrayRef<int64_t> index) const {
    if (isSplat) {
      return data[0];
    }
    return operator[](getFlatIndexFromStride(index, strides));
  }

  // Index a tensor by a flat index, will directly index the data vector.
  NumericType &operator[](int64_t flatIndex) {
    assert(flatIndex == 0 ||
           !isSplat && "Cannot mutate splat tensor unless index is 0.");
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

  Tensor<NumericType> slice(ArrayRef<int64_t> starts, ArrayRef<int64_t> ends,
                            ArrayRef<int64_t> steps) const {
    assert(starts.size() == shape.size() &&
           "starts must have the same size as shape");
    assert(ends.size() == shape.size() &&
           "ends must have the same size as shape");
    assert(steps.size() == shape.size() &&
           "steps must have the same size as shape");
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); i++) {
      assert(starts[i] >= 0 && starts[i] < shape[i] &&
             "starts must be within shape");
      assert(ends[i] >= 0 && ends[i] <= shape[i] &&
             "ends must be within shape");
      assert(steps[i] > 0 && "steps must be positive");
    }

    // Calculate the shape of the output tensor.
    SmallVector<int64_t> newShape;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); i++) {
      newShape.push_back((ends[i] - starts[i]) / steps[i]);
    }

    int64_t newVolume = std::accumulate(newShape.begin(), newShape.end(), 1,
                                        std::multiplies<int64_t>());

    SmallVector<int64_t> currentIndex(starts);
    std::vector<NumericType> newData;
    newData.reserve(newVolume);
    newData.push_back(operator[](currentIndex));

    while (static_cast<int64_t>(newData.size()) < newVolume) {
      int64_t currentDim = shape.size() - 1;
      while (currentDim >= 0) {
        currentIndex[currentDim] += steps[currentDim];
        if (currentIndex[currentDim] < ends[currentDim]) {
          break;
        }
        currentIndex[currentDim] = starts[currentDim];
        currentDim--;
      }
      newData.push_back(operator[](currentIndex));
    }
    return Tensor<NumericType>(newData, newShape);
  }

  NumericType sum() const {
    if (isSplat) {
      // Constructing a valid multiplier for APInt and APFloat requires
      // different logic, so we use if constexpr to handle the different cases.
      if constexpr (std::is_same_v<NumericType, llvm::APInt>) {
        return data[0] * getTensorCompatibleAPValue(volume);
      } else if constexpr (std::is_same_v<NumericType, llvm::APFloat>) {
        return data[0] * getTensorCompatibleAPValue(static_cast<float>(volume));
      } else {
        llvm_unreachable("Unsupported numeric type");
      }
    }
    NumericType zero = [&]() -> NumericType {
      if constexpr (std::is_same_v<NumericType, llvm::APInt>) {
        return getTensorCompatibleAPValue(0l);
      } else if constexpr (std::is_same_v<NumericType, llvm::APFloat>) {
        return getTensorCompatibleAPValue(0.0f);
      } else {
        llvm_unreachable("Unsupported numeric type");
      }
    }();

    // Both APFloat and APInt implement operator+. So we can use std::accumulate
    // to sum the elements as long as we provide the correctly typed zero value.
    return std::accumulate(data.begin(), data.end(), zero);
  }

  NumericType max() const {
    if (isSplat) {
      return data[0];
    }

    // Using std::reduce instead of std::max_element because APFloat and APInt
    // have different interfaces for comparison.
    return std::reduce(
        data.begin(), data.end(), data[0],
        [](NumericType a, NumericType b) -> NumericType {
          if constexpr (std::is_same_v<NumericType, llvm::APInt>) {
            return a.sgt(b) ? a : b;
          } else if constexpr (std::is_same_v<NumericType, llvm::APFloat>) {
            return a > b ? a : b;
          } else {
            llvm_unreachable("Unsupported numeric type");
          }
        });
  }

  NumericType mean() const {
    if (isSplat) {
      return data[0];
    }

    // APFloat and APInt do not have the same interface for division, so we use
    // if constexpr to handle the different cases.
    if constexpr (std::is_same_v<NumericType, llvm::APInt>) {
      return sum().sdiv(getTensorCompatibleAPValue(volume));
    }
    if constexpr (std::is_same_v<NumericType, llvm::APFloat>) {
      return sum() / getTensorCompatibleAPValue(static_cast<float>(volume));
    }
    llvm_unreachable("Unsupported numeric type");
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
