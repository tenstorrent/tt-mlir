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

using WrappedFunc = void (*)(WrappedTensor *);

// Common function to pack tensors, using std::function for the customizable
// parts
template <typename TensorRefType>
std::vector<common::WrappedTensor> inline packTensors(
    const flatbuffers::Vector<flatbuffers::Offset<TensorRefType>> *ins,
    const TensorRefType *out,
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
