// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_DYLIB_H
#define TT_RUNTIME_DETAIL_COMMON_DYLIB_H

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <stdint.h>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
// Linux memfd_create syscall number, if not available in <sys/mman.h>
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef SYS_memfd_create
#define SYS_memfd_create 319
#endif

namespace tt::runtime::common {
using DylibHandleMap = std::unordered_map<uint32_t, void *>;

struct WrappedTensor {
  void *start;
  void *alignedStart;
  int64_t startIdx;
  int64_t *sizesAndStrides;
};

using WrappedFunc = WrappedTensor *(*)(WrappedTensor *);

// Common function to pack tensors, using std::function for the customizable
// parts
template <typename TensorRefType>
std::vector<common::WrappedTensor> inline packTensors(
    const flatbuffers::Vector<flatbuffers::Offset<TensorRefType>> *ins,
    std::function<void *(const TensorRefType *)> getTensorDataPtr,
    std::vector<std::vector<int64_t>> &allSizesAndStrides) {

  allSizesAndStrides.reserve(ins->size() + 1);
  std::vector<common::WrappedTensor> packedTensors;
  packedTensors.reserve(ins->size());

  for (size_t i = 0; i < ins->size(); ++i) {
    auto tensorRef = ins->Get(i);
    auto shape = tensorRef->desc()->shape();
    const size_t rank = shape->size();

    std::vector<int64_t> sizes(rank);
    for (size_t j = 0; j < rank; ++j) {
      sizes[j] = shape->Get(j);
    }

    std::vector<uint32_t> strides = tt::runtime::utils::calculateStride(sizes);
    allSizesAndStrides.emplace_back(2 * rank);
    std::copy(sizes.begin(), sizes.end(), allSizesAndStrides.back().begin());
    std::transform(strides.begin(), strides.end(),
                   allSizesAndStrides.back().begin() + rank,
                   [](uint32_t s) -> int64_t { return s; });

    void *rawDataPtr = getTensorDataPtr(tensorRef);
    packedTensors.emplace_back(rawDataPtr, rawDataPtr, 0,
                               allSizesAndStrides.back().data());
  }

  return packedTensors;
}

// Extract sizes from a tensor reference (works with both BufferRef and
// TensorRef)
template <typename TensorRefType>
std::vector<int64_t> extractSizes(const TensorRefType *tensorRef) {
  auto shape = tensorRef->desc()->shape();
  const size_t rank = shape->size();
  std::vector<int64_t> sizes(rank);

  for (size_t j = 0; j < rank; ++j) {
    sizes[j] = shape->Get(j);
  }

  return sizes;
}

// Calculate strides and prepare the combined sizes+strides vector
inline void
prepareSizesAndStrides(const std::vector<int64_t> &sizes,
                       std::vector<std::vector<int64_t>> &allSizesAndStrides) {

  std::vector<uint32_t> strides = tt::runtime::utils::calculateStride(sizes);
  const size_t rank = sizes.size();

  allSizesAndStrides.emplace_back(2 * rank);
  std::copy(sizes.begin(), sizes.end(), allSizesAndStrides.back().begin());
  std::transform(strides.begin(), strides.end(),
                 allSizesAndStrides.back().begin() + rank,
                 [](uint32_t s) -> int64_t { return s; });
}

// Callback type for creating runtime tensors from the provided pre-allocated
// data buffer.
//
// The callback receives the tensor reference, and a shared_ptr to the
// pre-allocated data buffer. The callback is expected to return a
// runtime-specific tensor.
//
template <typename TensorType, typename TensorRefType>
using CreateTensorCallbackType =
    std::function<TensorType(const TensorRefType *, std::shared_ptr<void>)>;

// Unpack tensors returned from a CPU-hoisted function.
//
// Takes the returned WrappedTensor array and a target-specific callback to
// create tensors. Returns a vector of created tensors.
//
template <typename TensorType, typename TensorRefType>
std::vector<TensorType> inline unpackTensors(
    WrappedTensor *outputArray, size_t numOutputs,
    const flatbuffers::Vector<flatbuffers::Offset<TensorRefType>> *outs,
    CreateTensorCallbackType<TensorType, TensorRefType> createTensorCallback) {
  std::vector<TensorType> results;
  results.reserve(numOutputs);

  // Track already-seen allocations so that outputs aliasing the same buffer
  // share same ttnn::Tensor instance, in order to prevent double-free issues.
  std::unordered_map<void *, std::shared_ptr<void>> seenPtrs;

  // Create tensors using the provided callback.
  for (size_t i = 0; i < numOutputs; ++i) {
    const WrappedTensor &wrappedTensor = outputArray[i];

    // Set up a shared_ptr with custom deleter to free the tensor buffer
    // allocated by the CPU-hoisted function once the runtime tensor is
    // destroyed.
    // Note: we use the alignedStart pointer for the tensor data, but
    // we need to free the original start pointer.
    auto it = seenPtrs.find(wrappedTensor.start);
    std::shared_ptr<void> dataPtr;
    if (it != seenPtrs.end()) {
      dataPtr = it->second;
    } else {
      dataPtr = std::shared_ptr<void>(
          wrappedTensor.alignedStart,
          [data = wrappedTensor.start](void *) { std::free(data); });
      seenPtrs[wrappedTensor.start] = dataPtr;
    }

    results.push_back(createTensorCallback(outs->Get(i), dataPtr));
  }

  // Free the sizesAndStrides buffer allocated inside the CPU-hoisted function
  // (all outputs share one allocation).
  if (numOutputs > 0) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    std::free(outputArray[0].sizesAndStrides);
  }

  // Free the output WrappedTensor array allocated inside the CPU-hoisted
  // function.
  //  NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  std::free(outputArray);

  return results;
}

class DylibManager {
public:
  // Constructor takes dylibs and loads them
  DylibManager(
      const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::DynamicLib>>
          *dylibs);

  // Destructor cleans up handles
  ~DylibManager();

  // Disable copy operations
  DylibManager(const DylibManager &) = delete;
  DylibManager &operator=(const DylibManager &) = delete;

  // Allow move operations
  DylibManager(DylibManager &&other) noexcept;
  DylibManager &operator=(DylibManager &&other) noexcept;

  // Access the handle map
  const DylibHandleMap &getHandles() const { return handles; }

  void *getHandle(const uint32_t key) const {
    const auto it = handles.find(key);
    return (it == handles.end()) ? nullptr : it->second;
  }

  WrappedFunc getFunc(const uint32_t key, const std::string &funcName) const;

private:
  DylibHandleMap handles;
};
} // namespace tt::runtime::common

#endif
